"""Checkpointing of GQ-Dreamer: save, find and partially load checkpoints (every part whose names and shapes match).
Checkpoints live in the run directory: runs/<Game>/<config>_s<seed>/checkpoints/."""
import glob
import os
import re
import threading

import torch


def _cpu_copy(obj):
    """Nested dicts / lists of tensors -> the same structure with CPU copies of the tensors."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        out = type(obj)((k, _cpu_copy(v)) for k, v in obj.items())
        if hasattr(obj, "_metadata"):
            out._metadata = obj._metadata
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu_copy(v) for v in obj)
    return obj


def _short_list(keys, n=4):
    keys = list(keys)
    return ", ".join(keys[:n]) + (f", ... (+{len(keys) - n})" if len(keys) > n else "")


class CheckpointMixin:
    """Save / load of every GQDreamer module, optimizer and run counter."""
    ARCH_VERSION = "atari-v1"      # architecture id (pixels only, one reward / value stream)
    MODULES = ("tokenizer", "tssm", "prior", "reward_head", "continue_head", "actor", "critic", "critic_target",
               "harmony", "return_scale")
    OPTIMIZERS = ("wm_optimizer", "actor_optimizer", "critic_optimizer", "harmony_optimizer")
    OPTIMIZER_MODULES = {                                              # parameter order of each optimizer
        "wm_optimizer": ("tokenizer", "tssm", "prior", "reward_head", "continue_head"),
        "actor_optimizer": ("actor",),
        "critic_optimizer": ("critic",),
        "harmony_optimizer": ("harmony",),
    }
    ACTOR_CRITIC_PARTS = ("actor", "critic", "critic_target", "return_scale", "actor_optimizer", "critic_optimizer")
    ARCH_KEYS = ("arch_version", "action_dim", "img_size", "n_latents", "classes", "enc_stem", "enc_patch", "enc_dim",
                 "enc_depth", "enc_heads", "enc_pos_inject", "enc_swiglu", "enc_ffn", "decoder", "dec_dim",
                 "dec_depth", "dec_heads", "dec_patch", "dec_chans", "tssm_dim", "tssm_layers", "tssm_heads",
                 "tssm_kv_heads", "tssm_ffn", "prior_layers", "mlp_dim", "attn_gate", "enc_logit_norm")
    COMPAT_KEYS = ("game", "frameskip", "img_size", "action_dim")      # a mismatch blocks loading

    # --- Names ------------------------------------------------------------------------------------------------------
    def _set_file_names(self, run_dir):
        """Checkpoint / buffer paths of a run directory."""
        self.run_dir = run_dir
        self.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
        self.BUFFER_NAME = "replay.buffer"
        self.CHECKPOINT_GLOB = "ckpt_S*_G*.pt"

    def checkpoint_path(self):
        """Path of a new checkpoint at the current counters."""
        return os.path.join(self.CHECKPOINT_DIR, f"ckpt_S{self.total_num_steps}_G{self.total_num_updates}.pt")

    def optimizer_named_params(self, name):
        """[(qualified name, parameter)] updated by optimizer `name`."""
        return [(f"{m}.{n}", p) for m in self.OPTIMIZER_MODULES[name] for n, p in getattr(self, m).named_parameters()]

    def optimizer_params(self, name):
        return [p for _, p in self.optimizer_named_params(name)]

    def find_checkpoint(self):
        """Newest checkpoint (highest env step, then gradient step) of this run, or None."""
        files = glob.glob(os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_GLOB))

        def key(p):
            m = re.search(r"ckpt_S(\d+)_G(\d+)\.pt$", p)
            return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
        return max(files, key=key) if files else None

    # --- Save -------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, path=None, save_buffer=True, background=False, after=None):
        """Writes modules, optimizers and counters to `path` (default: checkpoint_path()) and the replay buffer next
        to it. background: the state is copied to CPU memory here and a thread writes the files (returned; join it
        before the buffer is changed by anything else than the collector); after: called once the files are written."""
        path = path or self.checkpoint_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {name: getattr(self, name).state_dict() for name in self.MODULES + self.OPTIMIZERS}
        ckpt.update({
            "wm_clipper": self.wm_clipper.state_dict(),
            "total_num_episodes": self.total_num_episodes,
            "total_num_steps": self.total_num_steps,
            "total_num_updates": self.total_num_updates,
            "updates_owed": self.updates_owed,
            "config": self.config,
            "optimizer_param_names": {name: [n for n, _ in self.optimizer_named_params(name)]
                                      for name in self.OPTIMIZERS},
        })
        if not background:
            self._write_checkpoint(ckpt, path, save_buffer, after)
            return None
        ckpt = _cpu_copy(ckpt)
        thread = threading.Thread(target=self._write_checkpoint, args=(ckpt, path, save_buffer, after),
                                  name="checkpoint-writer")
        thread.start()
        return thread

    def _write_checkpoint(self, ckpt, path, save_buffer, after):
        """Writes a checkpoint dict (+ the replay buffer); errors are reported, not raised."""
        try:
            torch.save(ckpt, path)
            if save_buffer:
                self.buffer.save(os.path.join(os.path.dirname(path) or ".", self.BUFFER_NAME))
            print(f"\033[90m    [ckpt] saved {path}{' (+ replay buffer)' if save_buffer else ''}\033[0m")
            if after is not None:
                after()
        except Exception as e:
            print(f"[!] Warning: checkpoint {path} not written completely: {type(e).__name__}: {e}")

    # --- Load -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _load_module_partial(name, module, saved):
        """Loads every tensor whose name and shape match; the rest keeps its fresh init. -> True if all matched."""
        own = module.state_dict()
        fits = {k: v for k, v in saved.items() if k in own and own[k].shape == v.shape}
        reshaped = [k for k, v in saved.items() if k in own and own[k].shape != v.shape]
        new = [k for k in own if k not in saved]
        dropped = [k for k in saved if k not in own]
        module.load_state_dict(fits, strict=False)
        if not (reshaped or new or dropped):
            return True
        msg = f"[load] '{name}': {len(fits)}/{len(own)} tensors restored"
        if reshaped:
            msg += f" | shape changed -> fresh: {_short_list(reshaped)}"
        if new:
            msg += f" | new -> fresh: {_short_list(new)}"
        if dropped:
            msg += f" | no longer in the model -> dropped: {_short_list(dropped)}"
        print(msg)
        return False

    def _load_optimizer_partial(self, name, ckpt):
        """Restores optimizer moments by parameter name (same shape); others start fresh."""
        opt = getattr(self, name)
        saved = ckpt[name]
        old_idx = [i for g in saved["param_groups"] for i in g["params"]]
        old_state = saved.get("state", {})
        names = ckpt.get("optimizer_param_names", {}).get(name)
        if names is None or len(names) != len(old_idx):
            print(f"[load] '{name}': no usable parameter names -> fresh optimizer state")
            return
        by_name = {n: old_state[i] for n, i in zip(names, old_idx) if i in old_state}
        current = self.optimizer_named_params(name)
        new_state = {}
        for idx, (n, p) in enumerate(current):
            st = by_name.get(n)
            if st is not None and all(tuple(t.shape) == tuple(p.shape)
                                      for t in st.values() if torch.is_tensor(t) and t.dim() > 0):
                new_state[idx] = st
        sd = opt.state_dict()
        sd["state"] = new_state
        opt.load_state_dict(sd)
        if old_state and len(new_state) < len(current):
            print(f"[load] '{name}': moments restored for {len(new_state)}/{len(current)} parameters "
                  f"(the others start fresh)")

    def load_checkpoint(self, path=None, reset_actor_critic=False, load_buffer=True, weights_only=False):
        """Loads the newest (or given) checkpoint part by part -> True if one was loaded. reset_actor_critic keeps
        only the world model; weights_only (evaluation) skips optimizers, counters' consequences and the buffer."""
        if path is None:
            path = self.find_checkpoint()
            if path is None:
                print(f"\033[90m    [setup] no checkpoint in {self.CHECKPOINT_DIR} -- the model starts from "
                      f"scratch\033[0m")
                return False
        if not os.path.exists(path):
            print(f"[!] no checkpoint at {path} -- starting from scratch")
            return False
        print(f"\033[90m    [setup] loading checkpoint {path}\033[0m")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        old_cfg, new_cfg = ckpt.get("config", {}), self.config

        def diff(keys):
            return {k: (old_cfg.get(k), new_cfg.get(k)) for k in keys
                    if old_cfg.get(k) != new_cfg.get(k) and not (isinstance(old_cfg.get(k), (list, tuple)) and
                                                                 tuple(old_cfg.get(k)) == tuple(new_cfg.get(k) or ()))}

        def fmt(d):
            return ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in d.items())

        incompatible = diff(self.COMPAT_KEYS)
        if incompatible:
            print(f"[load] {os.path.basename(path)} was trained with {fmt(incompatible)}: not loaded.")
            return False
        changed = diff(self.ARCH_KEYS)
        if changed:
            print(f"[load] architecture changed since {os.path.basename(path)} ({fmt(changed)}): loading every part "
                  f"that still fits, the rest starts fresh.")
        skip = set(self.ACTOR_CRITIC_PARTS) if reset_actor_critic else set()
        exact = True
        for name in self.MODULES:
            if name in skip:
                continue
            if name not in ckpt:
                print(f"[load] '{name}' not in checkpoint -> fresh init.")
                exact = False
                continue
            exact &= self._load_module_partial(name, getattr(self, name), ckpt[name])
        if reset_actor_critic:
            self.critic_target.load_state_dict(self.critic.state_dict())
        self.total_num_episodes = ckpt.get("total_num_episodes", 0)
        self.total_num_steps = ckpt.get("total_num_steps", 0)
        self.total_num_updates = ckpt.get("total_num_updates", 0)
        self.updates_owed = ckpt.get("updates_owed", 0.0)
        self._resumed = True
        if not weights_only:
            for name in self.OPTIMIZERS:
                if name in skip or name not in ckpt:
                    continue
                try:
                    self._load_optimizer_partial(name, ckpt)
                except Exception as e:
                    print(f"[load] '{name}' could not be restored -> fresh optimizer state ({e}).")
            self.wm_clipper.load_state_dict(ckpt.get("wm_clipper", {}))
        if exact:
            print(f"\033[90m    [setup] checkpoint restored completely ({self.total_num_steps:,} env steps, "
                  f"{self.total_num_updates:,} updates)\033[0m")
        del ckpt
        if load_buffer and not weights_only:
            p = os.path.join(os.path.dirname(path) or ".", self.BUFFER_NAME)
            if os.path.exists(p):
                self.buffer.load(p)
        return True
