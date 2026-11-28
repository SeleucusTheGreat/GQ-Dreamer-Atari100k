"""Streaming, episode-aware replay buffer (one ring per training env, rows written while they are played; life-loss,
recent, reward and surprise sampling) and a background batch prefetcher."""
import json
import os
import queue
import threading

import numpy as np
import torch


class ReplayBuffer:
    """Row t of an episode = observation t, the action taken at t, the reward of t -> t+1 and the continue flag of
    observation t (0 = a terminal frame: a life loss or the game over). Every training env writes into its own ring
    of capacity // num_streams rows, so the rows of an episode are contiguous in time even while several envs play.
    An "episode" here is a replay segment: with life_loss_terminal it ends at every life loss (the game goes on in a
    new segment). Training windows never cross a segment; short segments are padded with their last row.

    Sampling mixes uniform full windows with (probabilities in the constructor):
    - death windows: windows that contain a terminal row at a uniform index (life losses);
    - recent windows: full windows written in the last `recent_steps` rows;
    - reward windows: windows that contain a non-zero reward at a uniform index;
    - surprise windows: windows around rows drawn with probability ~ the world model's KL on them (Curious Replay).

    Thread-safe: the prefetcher samples while the collector appends."""
    FORMAT = "gq_atari_v1"
    FILE_MAGIC = b"GQATARI1"
    META = ("episode_ids", "ep_steps", "ep_lens", "write_ids")
    _ALIGN = 64

    def __init__(self, capacity=120_000, num_streams=1, img_size=64, channels=3, death_sample_prob=0.0,
                 recent_sample_prob=0.0, recent_steps=5_000, reward_sample_prob=0.0, loss_sample_prob=0.0,
                 loss_priority_alpha=1.0):
        self.num_streams = max(1, int(num_streams))
        self.stream_cap = int(capacity) // self.num_streams
        self.capacity = self.stream_cap * self.num_streams
        self.img_size, self.channels = int(img_size), int(channels)
        self.death_sample_prob, self.recent_sample_prob = death_sample_prob, recent_sample_prob
        self.recent_steps, self.reward_sample_prob = int(recent_steps), reward_sample_prob
        self.loss_sample_prob, self.loss_priority_alpha = loss_sample_prob, loss_priority_alpha
        self.FIELDS = {"images": ((self.channels, self.img_size, self.img_size), torch.uint8),
                       "actions": ((), torch.uint8), "rewards": ((), torch.float32),
                       "continues": ((), torch.float32)}
        C = self.capacity
        for name, (shape, dtype) in self.FIELDS.items():
            setattr(self, name, torch.zeros((C, *shape), dtype=dtype))
        self.episode_ids = torch.full((C,), -1, dtype=torch.long)
        self.ep_steps = torch.zeros((C,), dtype=torch.int32)
        self.ep_lens = torch.full((C,), -1, dtype=torch.int32)         # -1: the episode is still open
        self.write_ids = torch.full((C,), -1, dtype=torch.long)        # global write counter (recency)
        self.priorities = torch.full((C,), -1.0, dtype=torch.float32)  # world-model KL, -1 = not scored yet
        self.default_priority = 1.0
        self._head = [0] * self.num_streams                            # next local index of each ring
        self._count = [0] * self.num_streams                           # rows held by each ring
        self._open = [None] * self.num_streams                         # open episode {id, start, length}
        self._ep_start = {}                                            # episode id -> (stream, local start); stale
                                                                       # entries are caught by locate()
        self._episode_counter = 0
        self.total_written = 0
        self.episodes_closed = 0
        self.avg_episode_length = 0.0
        self._lock = threading.RLock()
        self._pools_T = None
        self._dirty = True

    def __len__(self):
        return sum(self._count)

    # --- Index helpers ----------------------------------------------------------------------------------------------
    def _wrap(self, rows, delta):
        """Row `delta` steps after (or before) `rows` in the same ring."""
        cap = self.stream_cap
        return (rows // cap) * cap + (rows % cap + delta) % cap

    def _rows(self, stream, local_start, n):
        return stream * self.stream_cap + (local_start + torch.arange(n)) % self.stream_cap

    # --- Writing ----------------------------------------------------------------------------------------------------
    def add_rows(self, stream, images, actions, rewards, continues, new_episode=False, end_episode=False):
        """Appends rows to the ring of `stream`: images uint8 [n, C, S, S], actions [n], rewards [n], continues [n]
        (numpy or tensors). new_episode: they start a new episode (an open one is closed first, as truncated);
        end_episode: the episode is complete after them."""
        n = len(images)
        with self._lock:
            if new_episode or self._open[stream] is None:
                self.begin_episode(stream)
            ep = self._open[stream]
            if n > 0:
                rows = self._rows(stream, self._head[stream], n)
                self.images[rows] = torch.as_tensor(np.asarray(images), dtype=torch.uint8)
                self.actions[rows] = torch.as_tensor(np.asarray(actions), dtype=torch.long).to(torch.uint8)
                self.rewards[rows] = torch.as_tensor(np.asarray(rewards, dtype=np.float32))
                self.continues[rows] = torch.as_tensor(np.asarray(continues, dtype=np.float32))
                self.episode_ids[rows] = ep["id"]
                self.ep_steps[rows] = ep["length"] + torch.arange(n, dtype=torch.int32)
                self.ep_lens[rows] = -1
                self.write_ids[rows] = self.total_written + torch.arange(n)
                self.priorities[rows] = -1.0
                self._head[stream] = (self._head[stream] + n) % self.stream_cap
                self._count[stream] = min(self.stream_cap, self._count[stream] + n)
                ep["length"] += n
                self.total_written += n
            if end_episode:
                self._close(stream)
            self._dirty = True

    def _close(self, stream):
        """Marks the open episode of `stream` complete (its length on every row it still holds)."""
        ep = self._open[stream]
        self._open[stream] = None
        if ep is None or ep["length"] == 0:
            return
        L, cap = ep["length"], self.stream_cap
        kept = min(L, cap)
        rows = self._rows(stream, ep["start"] + (L - kept), kept)
        self.ep_lens[rows] = L
        self.episodes_closed += 1
        d = 0.9
        self.avg_episode_length = float(L) if self.avg_episode_length == 0.0 else \
            d * self.avg_episode_length + (1.0 - d) * L

    def begin_episode(self, stream):
        """Starts a new (empty) episode in the ring of `stream` (an open one is closed first, as truncated) -> its id;
        add_rows then appends to it."""
        with self._lock:
            if self._open[stream] is not None:
                self._close(stream)
            eid = self._episode_counter
            self._episode_counter += 1
            self._open[stream] = {"id": eid, "start": self._head[stream], "length": 0}
            self._ep_start[eid] = (stream, self._head[stream])
            return eid

    def close_all(self):
        """Closes every open episode (as truncated)."""
        with self._lock:
            for s in range(self.num_streams):
                if self._open[s] is not None:
                    self._close(s)
            self._dirty = True

    def open_episode(self, stream):
        """(episode id, rows so far) of the open episode of `stream`, or None."""
        ep = self._open[stream]
        return None if ep is None else (ep["id"], ep["length"])

    def locate(self, episode_id, step):
        """Buffer row of step `step` of episode `episode_id`, or -1 when it is no longer held."""
        loc = self._ep_start.get(int(episode_id))
        if loc is None:
            return -1
        s, start = loc
        row = s * self.stream_cap + (start + int(step)) % self.stream_cap
        if int(self.episode_ids[row]) != int(episode_id) or int(self.ep_steps[row]) != int(step):
            return -1
        return row

    def update_priorities(self, rows, values):
        """Stores the world model's KL of these rows as their sampling priority."""
        if rows.numel() == 0:
            return
        values = values.float().clamp_min(0.0)
        with self._lock:
            self.priorities[rows] = values
            self.default_priority = 0.99 * self.default_priority + 0.01 * float(values.mean())

    # --- Sampling ---------------------------------------------------------------------------------------------------
    def _build_pools(self, T):
        """Window-start pools for windows of length T."""
        C = self.capacity
        pos = torch.arange(C)
        ids, steps, lens = self.episode_ids, self.ep_steps.long(), self.ep_lens.long()
        valid = ids >= 0
        end = self._wrap(pos, T - 1)
        full = valid & (ids[end] == ids) & (steps[end] == steps + T - 1)
        short = valid & (steps == 0) & (lens > 0) & (lens < T)
        starts = pos[full | short]
        deaths = pos[valid & (self.continues <= 0.0)]                  # terminal rows (life loss / game over)
        back = torch.clamp(steps[deaths], max=T - 1)
        dstart = self._wrap(deaths, -back)
        ok = (ids[dstart] == ids[deaths]) & (back > 0)
        rewards = pos[valid & (self.rewards != 0.0)]
        recent = starts[self.write_ids[starts] >= self.total_written - self.recent_steps]
        self._pools = {"starts": starts, "deaths": deaths[ok], "death_back": back[ok], "rewards": rewards,
                       "recent": recent}
        self._pools_T = T
        self._dirty = False

    def _death_starts(self, n):
        """Starts of n windows that contain a terminal row, at a uniform index 1 .. T-1 of the window."""
        pools = self._pools
        j = torch.randint(0, pools["deaths"].numel(), (n,))
        db = pools["death_back"][j]
        back = 1 + (torch.rand(n) * db).long().clamp(max=db - 1)
        return self._wrap(pools["deaths"][j], -back)

    def _anchored_starts(self, r, T):
        """Window starts that put rows r at a uniform index of their window (same episode)."""
        off = (torch.rand(r.shape[0]) * (torch.clamp(self.ep_steps[r].long(), max=T - 1) + 1)).long()
        s = self._wrap(r, -off)
        return torch.where(self.episode_ids[s] == self.episode_ids[r], s, r)

    def _priority_rows(self, n):
        """n rows drawn with probability ~ priority ** alpha (rows not scored yet: the running mean)."""
        p = self.priorities
        w = torch.where(p < 0, torch.full_like(p, self.default_priority), p).clamp_min(1e-3)
        if self.loss_priority_alpha != 1.0:
            w = w ** self.loss_priority_alpha
        w = w * (self.episode_ids >= 0).float()
        return torch.multinomial(w, n, replacement=True)

    def sample(self, batch_size, seq_len):
        """Batch of windows {field: [B, T, ...]} + valid mask, episode ends and buffer rows; None while no window
        exists."""
        T = seq_len
        with self._lock:
            if len(self) == 0:
                return None
            if self._dirty or self._pools_T != T:
                self._build_pools(T)
            pools = self._pools
            if pools["starts"].numel() == 0 and pools["deaths"].numel() == 0:
                return None

            def pick(pool, n):
                return pool[torch.randint(0, pool.numel(), (n,))]

            starts = (pick(pools["starts"], batch_size) if pools["starts"].numel() > 0
                      else self._death_starts(batch_size))
            u = torch.rand(batch_size)
            p_d = self.death_sample_prob if pools["deaths"].numel() > 0 else 0.0
            p_r = self.reward_sample_prob if pools["rewards"].numel() > 0 else 0.0
            p_c = self.recent_sample_prob if pools["recent"].numel() > 0 else 0.0
            p_l = self.loss_sample_prob
            death_m = u < p_d
            reward_m = (u >= p_d) & (u < p_d + p_r)
            recent_m = (u >= p_d + p_r) & (u < p_d + p_r + p_c)
            loss_m = (u >= p_d + p_r + p_c) & (u < p_d + p_r + p_c + p_l)
            if death_m.any():
                starts[death_m] = self._death_starts(int(death_m.sum()))
            if recent_m.any():
                starts[recent_m] = pick(pools["recent"], int(recent_m.sum()))
            if reward_m.any():
                starts[reward_m] = self._anchored_starts(pick(pools["rewards"], int(reward_m.sum())), T)
            if loss_m.any():
                starts[loss_m] = self._anchored_starts(self._priority_rows(int(loss_m.sum())), T)

            offs = torch.arange(T)[None]
            seq = self._wrap(starts[:, None], offs)                    # [B, T] rows if the episode continued
            same = (self.episode_ids[seq] == self.episode_ids[starts][:, None]) & \
                   (self.ep_steps[seq].long() == self.ep_steps[starts].long()[:, None] + offs)
            vlen = torch.cumprod(same.long(), dim=1).sum(1).clamp(min=1)
            rows = self._wrap(starts[:, None], torch.minimum(offs, vlen[:, None] - 1))   # pads repeat the last row
            batch = {name: getattr(self, name)[rows] for name in self.FIELDS}
            lens = self.ep_lens[rows]
            batch["valid"] = (offs < vlen[:, None]).float()
            batch["episode_end"] = (lens > 0) & (self.ep_steps[rows] == lens - 1)   # last row of a complete episode
            batch["index"] = rows
        return batch

    def sample_frames(self, n):
        """n random stored frames {"image": uint8 [n, C, S, S]} (encoder-drift probe)."""
        with self._lock:
            held = torch.nonzero(self.episode_ids >= 0).squeeze(1)
            idx = held[torch.randint(0, held.numel(), (n,))]
            return {"image": self.images[idx].clone()}

    # --- Persistence (JSON header + raw field bytes) ----------------------------------------------------------------
    def save(self, path):
        """Writes the held rows to one file, episode by episode in write order (atomic replace)."""
        with self._lock:
            held = torch.nonzero(self.episode_ids >= 0).squeeze(1).numpy()
            ids, wid = self.episode_ids.numpy()[held], self.write_ids.numpy()[held]
            order = held[np.lexsort((wid, ids))]
            names = list(self.FIELDS) + list(self.META)
            arrays = {name: getattr(self, name).numpy()[order] for name in names}
            open_ids = [ep["id"] for ep in self._open if ep is not None]
            meta = {"format": self.FORMAT, "n": int(len(order)), "img_size": self.img_size, "channels": self.channels,
                    "total_written": self.total_written, "episode_counter": self._episode_counter,
                    "open_episode_ids": open_ids, "avg_episode_length": self.avg_episode_length}
        fields, off = {}, 0
        for name, a in arrays.items():
            fields[name] = {"dtype": a.dtype.str, "shape": list(a.shape), "offset": off}
            off += -(-a.nbytes // self._ALIGN) * self._ALIGN
        meta["fields"] = fields
        header = json.dumps(meta).encode()
        data_start = -(-(len(self.FILE_MAGIC) + 8 + len(header)) // self._ALIGN) * self._ALIGN
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(self.FILE_MAGIC + np.uint64(len(header)).tobytes() + header)
            for name, a in arrays.items():
                f.seek(data_start + fields[name]["offset"])
                f.write(np.ascontiguousarray(a).data)
            f.truncate(data_start + off)
        os.replace(tmp, path)

    def load(self, path):
        """Restores a saved buffer: its episodes are appended again (round-robin over this buffer's rings), the
        episodes that were open when it was saved as truncated ones."""
        try:
            with open(path, "rb") as f:
                if f.read(len(self.FILE_MAGIC)) != self.FILE_MAGIC:
                    raise ValueError("not a replay buffer file")
                hlen = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
                meta = json.loads(f.read(hlen))
        except (OSError, ValueError) as e:
            print(f"[!] could not read replay buffer {path} ({e}); ignoring it.")
            return
        if meta.get("format") != self.FORMAT or meta.get("img_size") != self.img_size:
            print(f"[!] {path} is not a {self.FORMAT} buffer of {self.img_size}x{self.img_size} frames; ignoring it.")
            return
        data_start = -(-(len(self.FILE_MAGIC) + 8 + hlen) // self._ALIGN) * self._ALIGN
        arrays = {}
        for name, fd in meta["fields"].items():
            shape = tuple(fd["shape"])
            arrays[name] = (np.memmap(path, dtype=np.dtype(fd["dtype"]), mode="r", offset=data_start + fd["offset"],
                                      shape=shape) if shape[0] > 0 else np.zeros(shape, dtype=np.dtype(fd["dtype"])))
        ids = np.asarray(arrays["episode_ids"])
        n = len(ids)
        bounds = np.flatnonzero(np.diff(ids)) + 1 if n else np.zeros(0, dtype=np.int64)
        starts = np.concatenate([[0], bounds]) if n else np.zeros(0, dtype=np.int64)
        ends = np.concatenate([bounds, [n]]) if n else np.zeros(0, dtype=np.int64)
        first = max(0, n - self.capacity)                              # the newest rows if it does not fit
        k = 0
        for a, b in zip(starts, ends):
            if b <= first:
                continue
            seg = slice(max(int(a), first), int(b))
            self.add_rows(k % self.num_streams, np.array(arrays["images"][seg]), np.array(arrays["actions"][seg]),
                          np.array(arrays["rewards"][seg]), np.array(arrays["continues"][seg]),
                          new_episode=True, end_episode=True)
            k += 1
        del arrays
        self.avg_episode_length = float(meta.get("avg_episode_length") or self.avg_episode_length)
        print(f"\033[90m    [setup] replay buffer: {len(self):,} transitions in {k} episodes restored "
              f"({100.0 * len(self) / self.capacity:.0f}% full)\033[0m")

    def print_diagnostics(self):
        """Prints fill level and sampling statistics."""
        with self._lock:
            n = len(self)
            held = self.episode_ids >= 0
            terminals = int((held & (self.continues <= 0.0)).sum())
            rewards = int((held & (self.rewards != 0.0)).sum())
        print("\n" + "=" * 50)
        print("          REPLAY BUFFER DIAGNOSTICS")
        print("=" * 50)
        print(f"  Buffer fill : {n:,} / {self.capacity:,} ({100.0 * n / self.capacity:.1f}%) in "
              f"{self.num_streams} ring(s)")
        print(f"  Segments : {self.episodes_closed:,} closed | avg length {self.avg_episode_length:.0f}")
        print(f"  Terminal windows : p={self.death_sample_prob} | {terminals:,} terminal rows (life losses + game "
              f"overs)")
        print(f"  Recent windows : p={self.recent_sample_prob} | last {self.recent_steps:,} rows")
        print(f"  Reward windows : p={self.reward_sample_prob} | {rewards:,} rows with a non-zero reward")
        if self.loss_sample_prob > 0:
            print(f"  Surprise windows : p={self.loss_sample_prob} | priority = world-model KL of the frame "
                  f"(alpha {self.loss_priority_alpha}), running mean {self.default_priority:.2f} nats")
        print("=" * 50 + "\n")


def with_batch_extras(cpu):
    """CPU batch -> (batch + real-frame gather indices, CPU-side extras). Every row's real frames come first and its
    pads repeat the last real one: real_idx lists the real frames of the flattened [B, T] batch, real_gather maps
    every position to its frame in real_idx (skip_pad_frames). The extras stay on the CPU: the number of real frames
    with t >= 1 and the buffer rows / valid mask for the surprise priorities (no host sync on the GPU side)."""
    valid = cpu["valid"]
    B, T = valid.shape
    vlen = valid.sum(1).long().clamp(min=1)
    t = torch.arange(T)
    real = (t[None] < vlen[:, None]).flatten().nonzero().squeeze(1)
    pos = torch.full((B * T,), -1, dtype=torch.long)
    pos[real] = torch.arange(real.numel())
    src = (torch.arange(B)[:, None] * T + torch.minimum(t[None], vlen[:, None] - 1)).flatten()
    extra = {"n_valid1": int(valid[:, 1:].sum()), "index_cpu": cpu["index"], "valid_cpu": valid}
    return {**cpu, "real_idx": real, "real_gather": pos[src]}, extra


class BatchPrefetcher:
    """Background thread that keeps the next replay batches ready in (pinned) CPU memory, or already on the GPU
    (device: copied on the thread's own CUDA stream while the current update runs). The buffer may grow meanwhile
    (it is locked while a batch is drawn)."""

    def __init__(self, buffer, batch_size, seq_len, pin, depth=2, device=None):
        self.batch_size, self.seq_len = batch_size, seq_len
        self.device = device
        self._queue = queue.Queue(maxsize=1 if device is not None else depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(buffer, pin), daemon=True)
        self._thread.start()

    def _run(self, buffer, pin):
        """Producer loop; errors are passed to get()."""
        stream = torch.cuda.Stream(self.device) if self.device is not None else None
        while not self._stop.is_set():
            try:
                item = buffer.sample(self.batch_size, self.seq_len)
                if item is not None:
                    cpu, extra = with_batch_extras(item)
                    if pin:
                        cpu = {k: v.pin_memory() for k, v in cpu.items()}
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            gpu = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
                            ev = torch.cuda.Event()
                            ev.record(stream)
                        item = {"__gpu__": gpu, "__event__": ev, "__cpu__": extra}
                    else:
                        item = {"__batch__": cpu, "__cpu__": extra}
            except Exception as e:
                item = e
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is None or isinstance(item, Exception):
                return

    def get(self):
        """Next batch (re-raises producer errors)."""
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """Stops the thread and drops queued batches."""
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5)
