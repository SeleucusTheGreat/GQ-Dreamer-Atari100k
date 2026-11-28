"""ALE environment with the Atari 100k preprocessing, and its vectorised wrappers (in-process or one process per env).

AtariEnv drives the emulator directly (`ale.act`), not through gymnasium's step(), so only the two frames that are
max-pooled are ever copied out of the emulator. Per agent step: the action is repeated `frameskip` times, the rewards
are summed, the last two frames are max-pooled (flicker), and the result is resized to img_size x img_size RGB (area
interpolation). A reset plays 0..noop_max random no-ops. A life loss is reported in info["life_loss"] (the game
goes on; the agent stores it as a terminal of its own replay segment), a game over is `terminated`, and the
108k-frame cap is `truncated`."""
import multiprocessing as mp

import numpy as np
from gymnasium import spaces

from .atari import FRAMESKIP, IMG_SIZE, MAX_EPISODE_FRAMES, NOOP_MAX, STICKY_ACTIONS, canonical_game, env_id


def _make_ale(game, sticky_actions, max_episode_frames):
    """Gymnasium's ALE v5 env of a game at frame skip 1 (the repeat is done here), minimal action set."""
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    return gym.make(env_id(game), frameskip=1, repeat_action_probability=float(sticky_actions),
                    full_action_space=False, max_num_frames_per_episode=int(max_episode_frames))


class AtariEnv:
    """One ALE game with the Atari 100k preprocessing; observations are {"image": uint8 [3, S, S]}."""

    def __init__(self, game, img_size=IMG_SIZE, frameskip=FRAMESKIP, noop_max=NOOP_MAX,
                 sticky_actions=STICKY_ACTIONS, max_episode_frames=MAX_EPISODE_FRAMES, seed=None):
        import cv2
        self._cv2 = cv2
        self.game = canonical_game(game)
        self._gym = _make_ale(self.game, sticky_actions, max_episode_frames)
        self._gym.reset(seed=None if seed is None else int(seed))       # loads the ROM, seeds the emulator RNG
        u = self._gym.unwrapped
        self.ale = u.ale
        self.action_set = list(u._action_set)
        self.action_names = tuple(u.get_action_meanings())
        self.action_space = spaces.Discrete(len(self.action_set))
        self.img_size, self.frameskip, self.noop_max = int(img_size), int(frameskip), int(noop_max)
        self.max_episode_steps = int(max_episode_frames) // self.frameskip
        self._rng = np.random.default_rng(seed)
        h, w = self.ale.getScreenDims()
        self._buf = np.zeros((2, h, w, 3), dtype=np.uint8)               # the two frames that are max-pooled
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)                 # last max-pooled full-resolution frame
        self._capture = False
        self.lives, self.steps, self.episode_return = 0, 0, 0.0

    # --- Observations -----------------------------------------------------------------------------------------------
    def _observe(self):
        """Max-pool of the two grabbed frames -> {"image": uint8 [3, S, S]} (area-resized RGB)."""
        np.maximum(self._buf[0], self._buf[1], out=self._frame)
        img = self._cv2.resize(self._frame, (self.img_size, self.img_size), interpolation=self._cv2.INTER_AREA)
        return {"image": np.ascontiguousarray(img.transpose(2, 0, 1))}

    def _info(self, life_loss=False, frames=None):
        info = {"lives": self.lives, "life_loss": life_loss, "steps": self.steps, "limit": self.max_episode_steps,
                "episode_return": self.episode_return}
        if frames is not None:
            info["frames"] = frames
        return info

    def rgb_frame(self):
        """The last max-pooled frame at full resolution [210, 160, 3] (for videos)."""
        return self._frame.copy()

    def set_capture(self, on):
        """on: every step's info carries all its emulator frames (info["frames"], for real-speed videos)."""
        self._capture = bool(on)

    # --- Gym-style API ----------------------------------------------------------------------------------------------
    def reset(self):
        """New game with 0..noop_max random no-ops -> (obs, info)."""
        ale, noop = self.ale, self.action_set[0]
        ale.reset_game()
        for _ in range(int(self._rng.integers(0, self.noop_max + 1))):
            ale.act(noop)
            if ale.game_over():
                ale.reset_game()
        ale.getScreenRGB(self._buf[0])
        self._buf[1] = self._buf[0]
        self.lives, self.steps, self.episode_return = ale.lives(), 0, 0.0
        frames = [self._buf[0].copy()] if self._capture else None
        return self._observe(), self._info(frames=frames)

    def step(self, action):
        """One agent step (frameskip emulator frames) -> (obs, reward, terminated, truncated, info)."""
        ale, a = self.ale, self.action_set[int(action)]
        reward, got, fs = 0.0, 0, self.frameskip
        frames = [] if self._capture else None
        for k in range(fs):
            reward += ale.act(a)
            over = ale.game_over()                                     # game over or the frame cap
            if frames is not None:
                frames.append(ale.getScreenRGB())
            if k >= fs - 2 or over:
                ale.getScreenRGB(self._buf[got % 2])
                got += 1
            if over:
                break
        if got == 1:                                                   # ended before the second grab
            self._buf[1] = self._buf[0]
        self.steps += 1
        self.episode_return += reward
        terminated = ale.game_over(with_truncation=False)
        truncated = not terminated and (ale.game_truncated() or self.steps >= self.max_episode_steps)
        lives = ale.lives()
        life_loss = lives < self.lives and not terminated
        self.lives = lives
        return self._observe(), float(reward), terminated, truncated, self._info(life_loss, frames)

    # --- Snapshots (dream audit) ------------------------------------------------------------------------------------
    def get_snapshot(self):
        """Complete state after the last step (emulator incl. its RNG, frame buffers, counters)."""
        return {"ale": self.ale.cloneState(include_rng=True), "buf": self._buf.copy(), "lives": self.lives,
                "steps": self.steps, "episode_return": self.episode_return}

    def restore_snapshot(self, snap):
        """Continues from get_snapshot() of any env of the same game."""
        self.ale.restoreState(snap["ale"])
        self._buf[:] = snap["buf"]
        self.lives, self.steps, self.episode_return = snap["lives"], snap["steps"], snap["episode_return"]
        return self._observe()

    def close(self):
        self._gym.close()


# --- Vectorised environments ----------------------------------------------------------------------------------------
def _step_env(env, action, snapshot=False):
    """Steps env; with snapshot the info also carries env.get_snapshot() of the state after the step."""
    out = env.step(action)
    if snapshot:
        out[4]["snapshot"] = env.get_snapshot()
    return out


class _VecBase:
    """Shared attributes of the vectorised wrappers."""

    def __len__(self):
        return self.num_envs

    def _set_spec(self, spec):
        self.game, self.action_names = spec["game"], tuple(spec["action_names"])
        self.action_space = spaces.Discrete(len(self.action_names))
        self.frameskip, self.img_size = spec["frameskip"], spec["img_size"]
        self.max_episode_steps = spec["max_episode_steps"]


def _spec(env):
    return {"game": env.game, "action_names": env.action_names, "frameskip": env.frameskip, "img_size": env.img_size,
            "max_episode_steps": env.max_episode_steps}


class SequentialEnvs(_VecBase):
    """Several AtariEnvs stepped one after another in this process (ALE is fast: ~15k frames/s)."""

    def __init__(self, envs):
        self._envs = list(envs)
        self.num_envs = len(self._envs)
        self._set_spec(_spec(self._envs[0]))

    def reset(self):
        """Resets every env -> [(obs, info)]."""
        return [e.reset() for e in self._envs]

    def reset_one(self, i):
        """Resets env i -> (obs, info)."""
        return self._envs[i].reset()

    def step(self, action_idxs, active=None, snapshot=None):
        """Steps the active envs -> list of step results (None for inactive envs); envs flagged in `snapshot` also put
        their post-step snapshot into the info."""
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        return [_step_env(e, int(action_idxs[i]), snapshot[i]) if active[i] else None
                for i, e in enumerate(self._envs)]

    def restore(self, snapshots):
        """Env i continues from snapshots[i] (None = left as it is) -> [obs or None]."""
        return [e.restore_snapshot(s) if s is not None else None for e, s in zip(self._envs, snapshots)]

    def set_capture(self, on):
        for e in self._envs:
            e.set_capture(on)

    def rgb_frame(self, i):
        return self._envs[i].rgb_frame()

    def close(self):
        for e in self._envs:
            e.close()


def _env_worker(conn, kwargs):
    """Subprocess loop serving one AtariEnv over a pipe."""
    try:
        env = AtariEnv(**kwargs)
        conn.send(("__spec__", _spec(env)))
    except Exception as e:
        try:
            conn.send(("__error__", repr(e)))
        finally:
            conn.close()
        return
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                conn.send(_step_env(env, *data))
            elif cmd == "reset":
                conn.send(env.reset())
            elif cmd == "restore":
                conn.send(env.restore_snapshot(data))
            elif cmd == "rgb_frame":
                conn.send(env.rgb_frame())
            elif cmd == "capture":
                env.set_capture(data)
                conn.send(True)
            elif cmd == "close":
                break
            else:
                conn.send(("__error__", f"unknown command {cmd!r}"))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        conn.close()


class ParallelEnvs(_VecBase):
    """One AtariEnv per process (spawn); step() sends every command before reading any reply. Worth it for many
    evaluation envs; the training default (1 env) runs in-process."""

    def __init__(self, env_kwargs):
        ctx = mp.get_context("spawn")
        self.num_envs = len(env_kwargs)
        self._parents, self._procs = [], []
        for kw in env_kwargs:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_env_worker, args=(child, kw), daemon=True)
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)
        specs = [self._check(p.recv()) for p in self._parents]
        self._set_spec(specs[0][1])

    @staticmethod
    def _check(result):
        """Raises a worker's error in the main process."""
        if isinstance(result, tuple) and len(result) == 2 and result[0] == "__error__":
            raise RuntimeError(f"Env worker error: {result[1]}")
        return result

    def reset(self):
        for p in self._parents:
            p.send(("reset", None))
        return [self._check(p.recv()) for p in self._parents]

    def reset_one(self, i):
        self._parents[i].send(("reset", None))
        return self._check(self._parents[i].recv())

    def step(self, action_idxs, active=None, snapshot=None):
        active = active or [True] * self.num_envs
        snapshot = snapshot or [False] * self.num_envs
        for i, p in enumerate(self._parents):
            if active[i]:
                p.send(("step", (int(action_idxs[i]), bool(snapshot[i]))))
        results = [None] * self.num_envs
        for i, p in enumerate(self._parents):
            if active[i]:
                results[i] = self._check(p.recv())
        return results

    def restore(self, snapshots):
        sent = [i for i, s in enumerate(snapshots) if s is not None]
        for i in sent:
            self._parents[i].send(("restore", snapshots[i]))
        out = [None] * self.num_envs
        for i in sent:
            out[i] = self._check(self._parents[i].recv())
        return out

    def set_capture(self, on):
        for p in self._parents:
            p.send(("capture", bool(on)))
        for p in self._parents:
            self._check(p.recv())

    def rgb_frame(self, i):
        self._parents[i].send(("rgb_frame", None))
        return self._check(self._parents[i].recv())

    def close(self):
        for p in self._parents:
            try:
                p.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def env_seeds(seed, n, offset=0):
    """Distinct emulator seeds of n envs of run seed `seed` (offset separates training, evaluation and audit envs)."""
    return [int(seed) * 100_003 + int(offset) + i for i in range(n)]


def create_envs(game, num_envs=1, seed=0, parallel=False, seed_offset=0, img_size=IMG_SIZE, frameskip=FRAMESKIP,
                noop_max=NOOP_MAX, sticky_actions=STICKY_ACTIONS, max_episode_frames=MAX_EPISODE_FRAMES,
                verbose=True):
    """`num_envs` AtariEnvs of one game (in-process, or one process each with parallel=True)."""
    game = canonical_game(game)
    kwargs = [dict(game=game, img_size=img_size, frameskip=frameskip, noop_max=noop_max,
                   sticky_actions=sticky_actions, max_episode_frames=max_episode_frames, seed=s)
              for s in env_seeds(seed, num_envs, seed_offset)]
    if verbose:
        print(f"\033[90m    [setup] {num_envs} x {env_id(game)} ({'one process each' if parallel else 'in-process'}"
              f", frameskip {frameskip}, sticky {sticky_actions}, noops 0-{noop_max}, {img_size}x{img_size} RGB, "
              f"cap {max_episode_frames:,} frames)\033[0m")
    if parallel and num_envs > 1:
        return ParallelEnvs(kwargs)
    return SequentialEnvs([AtariEnv(**kw) for kw in kwargs])
