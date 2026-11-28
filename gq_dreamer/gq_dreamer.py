"""GQ-Dreamer agent for Atari 100k: world-model training, imagination (actor-critic), acting, streaming collection,
evaluation and the dream audit.

The networks it builds are in model.py; this file runs and trains them.
"""
import contextlib
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audit import AUDIT_KINDS, SnapshotPool, audit_enabled, audit_metrics, rmse, select_starts
from .checkpoint import CheckpointMixin
from .config import GQ_CONFIG, config_to_dict
from .model import (TSSM, Actor, ContinueHead, Critic, GraphedDreamStep, PixelDecoder, PriorHead, Tokenizer,
                    TwoHotEncoding, TwoHotHead, class_usage, entropy_categorical, gumbel_like, kl_categorical,
                    sample_onehot, unimix_logprobs)
from .observations import OBS_KEYS, image_target, obs_to_tensors, sample_hide
from .replay import BatchPrefetcher, ReplayBuffer, with_batch_extras
from .train_utils import (DevicePercentileClipper, LossHarmonizer, PercentileClipper, PhaseTimer, ReturnScale,
                          actor_metrics, compile_available, compile_module, eager_modules, explained_variance,
                          lambda_returns, make_grad_scaler, masked_mean, reinforce_loss, resolve_amp_dtype,
                          setup_compile, weighted_moments)

torch.set_float32_matmul_precision("high")

ADV_STD_MIN = 1e-3                                                         # floor of the advantage std ("std" mode)


class GQDreamer(CheckpointMixin):
    """GroundQuery-Dreamer: query-fusion tokenizer + TSSM world model, actor-critic trained in imagination."""
    DEATH_PROBE_STEPS = 4                                                  # open-loop steps of the terminal probe
    DEATH_PROBE_EVERY = 5                                                  # probe every n-th update (round average)
    # world-model loss terms, each multiplied by its fixed weight (they sum to world_model_loss); round averages
    WM_LOSS_TERMS = ("pixel", "kl_dyn", "kl_rep", "rollout", "reward", "continue", "usage")

    def __init__(self, cfg=GQ_CONFIG, device="cuda", envs=None, run_dir="runs/tmp", verbose=True):
        c = self.cfg = cfg
        assert c.seq_len > c.context_length, "training sequences must be longer than the context window"
        assert c.rollout_steps == 0 or c.seq_len >= c.rollout_steps + 2, "the rollout needs rollout_steps + 2 frames"
        if c.adv_norm not in ("percentile", "std"):
            raise ValueError(f"adv_norm must be 'percentile' or 'std', not {c.adv_norm!r}")
        if c.reward_transform not in ("none", "sign"):
            raise ValueError(f"reward_transform must be 'none' or 'sign', not {c.reward_transform!r}")
        self.device = torch.device(device)
        self.envs = envs
        self.verbose = verbose
        self.action_dim = c.action_dim
        self.n_latents, self.classes = c.n_latents, c.classes
        self.latent_dim = c.n_latents * c.classes
        self.tssm_dim = c.tssm_dim
        self.state_dim = c.tssm_dim + self.latent_dim                      # s = [h, flat one-hot z]

        # run state
        self.total_num_episodes = self.total_num_steps = self.total_num_updates = 0
        self.updates_owed = 0.0                                            # fractional updates of the train ratio
        self.last_drift = float("nan")
        self._probe = self._probe_ref = None
        self._prefetcher = None
        self._death_probe_acc = None
        self._wm_loss_acc = None
        self._dream_graph = None
        self._use_dream_graph = c.dream_cuda_graph
        self._act_graphs = {}
        self._use_act_graph = c.act_cuda_graph
        self._pending_prio = None
        self._resumed = False
        self.last_grad_shares = {}
        self._grad_share_ok = True
        self._collector = None                                             # persistent state of collect()
        self.collect_stats = self._empty_collect_stats()
        self._rng = np.random.default_rng(c.seed + 17)
        self.audit_snapshots = SnapshotPool(c.audit_snapshot_every if audit_enabled(c) else 0)
        self._audit_env = None
        self.timer = PhaseTimer(self.device)
        self.two_hot = TwoHotEncoding(device=self.device, target=c.value_target, sigma_ratio=c.hl_sigma_ratio)

        self._build_networks()
        self._build_optimizers()
        self._compile_networks()
        self._set_file_names(run_dir)
        self.buffer = ReplayBuffer(
            capacity=c.buffer_size, num_streams=len(envs) if envs is not None else 1, img_size=c.img_size,
            death_sample_prob=c.death_sample_prob, recent_sample_prob=c.recent_sample_prob,
            recent_steps=c.recent_steps, reward_sample_prob=c.reward_sample_prob,
            loss_sample_prob=c.loss_sample_prob, loss_priority_alpha=c.loss_priority_alpha)
        if verbose:
            self._print_setup()

    # --- Setup ------------------------------------------------------------------------------------------------------
    def _build_networks(self):
        """World model (tokenizer, TSSM, prior, heads) and behaviour (actor, critic, EMA critic)."""
        c, dev, S = self.cfg, self.device, self.state_dim
        self.tokenizer = Tokenizer(c.n_latents, c.classes, c.img_size, c.enc_dim, c.enc_depth, c.enc_heads,
                                   enc_stem=c.enc_stem, enc_patch=c.enc_patch, enc_pos_inject=c.enc_pos_inject,
                                   enc_swiglu=c.enc_swiglu, enc_ffn=c.enc_ffn, enc_fast_attn=c.enc_fast_attn,
                                   enc_last_queries=c.enc_last_queries, grad_checkpoint=c.enc_grad_checkpoint,
                                   unimix=c.unimix, decoder=c.decoder, dec_dim=c.dec_dim, dec_depth=c.dec_depth,
                                   dec_heads=c.dec_heads, dec_patch=c.dec_patch, dec_chans=tuple(c.dec_chans),
                                   dec_sac=c.dec_sac, enc_logit_norm=c.enc_logit_norm,
                                   enc_logit_scale=c.enc_logit_scale).to(dev)
        # sink_max_pos: training sequences hold positions 0 .. seq_len - 1; acting counts higher
        self.tssm = TSSM(self.latent_dim, c.action_dim, c.tssm_dim, c.tssm_layers, c.tssm_heads, c.tssm_kv_heads,
                         c.tssm_ffn, c.context_length, attn_gate=c.attn_gate, sink_max_pos=c.seq_len - 1).to(dev)
        self.prior = PriorHead(c.tssm_dim, c.n_latents, c.classes, hidden=c.mlp_dim, layers=c.prior_layers,
                               unimix=c.unimix).to(dev)
        self.reward_head = TwoHotHead(S, hidden=c.mlp_dim).to(dev)
        self.continue_head = ContinueHead(S, hidden=c.mlp_dim).to(dev)
        self.actor = Actor(c.action_dim, S, c.mlp_dim).to(dev)
        self.critic = Critic(S, mlp_dim=c.mlp_dim).to(dev)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)
        self.harmony = LossHarmonizer().to(dev)                            # always built: one checkpoint layout
        self.return_scale = ReturnScale(c.return_scale_decay).to(dev)

    def _build_optimizers(self):
        """One AdamW for the whole world model, Adam for actor, critic and loss weights."""
        c = self.cfg
        fused = {"fused": True} if c.sync_free and self.device.type == "cuda" else {}
        self.wm_params = self.optimizer_params("wm_optimizer")
        self.wm_optimizer = torch.optim.AdamW(self.wm_params, lr=c.wm_lr, weight_decay=c.weight_decay, **fused)
        clipper = DevicePercentileClipper if c.sync_free else PercentileClipper
        self.wm_clipper = clipper(c.clip_percentile, c.clip_window, c.clip_warmup, c.clip_initial,
                                  **({"device": self.device} if c.sync_free else {}))
        self.actor_optimizer = torch.optim.Adam(self.optimizer_params("actor_optimizer"), lr=c.actor_lr, **fused)
        self.critic_optimizer = torch.optim.Adam(self.optimizer_params("critic_optimizer"), lr=c.critic_lr, **fused)
        self.harmony_optimizer = torch.optim.Adam(self.optimizer_params("harmony_optimizer"), lr=c.harmony_lr)
        self.amp_dtype = resolve_amp_dtype(self.device, c.mixed_precision)
        self.cache_dtype = self.amp_dtype or torch.float32
        self.scalers = {name: make_grad_scaler(self.device, self.amp_dtype == torch.float16)
                        for name in self.OPTIMIZERS}

    def _compile_networks(self):
        """cfg.compile: torch.compile of the encoder's checkpointed layers, the decoder's checkpointed chunks, the MLP
        heads / actor / critics / prior and the TSSM's teacher-forced blocks (state_dict keys unchanged)."""
        self.compiled = False
        if not self.cfg.compile:
            return
        ok, why = compile_available(self.device)
        if not ok:
            if self.verbose:
                print(f"\033[93m    [setup] compile=True but {why}: running eagerly\033[0m")
            return
        setup_compile()
        self.tokenizer.encoder.compile_layers()
        if isinstance(self.tokenizer.decoder, PixelDecoder):
            self.tokenizer.compile_decoder()
        for name in ("actor", "critic", "critic_target", "reward_head", "continue_head", "prior"):
            compile_module(getattr(self, name).net)
        for blk in self.tssm.blocks:
            blk.forward_train = torch.compile(blk.forward_train)
        self.compiled = True

    def parameter_counts(self):
        """Millions of parameters per part."""
        def count(*mods):
            return sum(p.numel() for m in mods for p in m.parameters()) / 1e6
        return {"encoder": count(self.tokenizer.encoder), "decoder": count(self.tokenizer.decoder),
                "tssm": count(self.tssm), "prior": count(self.prior),
                "heads": count(self.reward_head, self.continue_head), "actor": count(self.actor),
                "critic": count(self.critic),
                "world_model": count(self.tokenizer, self.tssm, self.prior, self.reward_head, self.continue_head)}

    def _print_setup(self):
        """One-line model summary."""
        c, n = self.cfg, self.parameter_counts()
        amp = str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "off (fp32)"
        print(f"\033[90m    [setup] GQ-Dreamer {self.ARCH_VERSION} \"{c.config_name}\" on {c.game} | encoder "
              f"{n['encoder']:.1f}M ({c.enc_stem} stem, {self.tokenizer.encoder.n_image_tokens} image tokens + "
              f"{c.n_latents} queries, {c.enc_dim} x {c.enc_depth}) | {c.decoder} decoder {n['decoder']:.1f}M | "
              f"TSSM {n['tssm']:.1f}M ({c.tssm_dim} x {c.tssm_layers}, window {c.context_length}) | prior "
              f"{n['prior']:.1f}M | heads {n['heads']:.1f}M | actor {n['actor']:.1f}M | critic {n['critic']:.1f}M | "
              f"world model {n['world_model']:.1f}M | z = {c.n_latents} x {c.classes} one-hot | batch "
              f"{c.batch_size} x {c.seq_len}, train ratio {c.train_ratio} | mixed precision {amp}\033[0m")
        on = [k for k in ("compile", "act_cuda_graph", "dream_cuda_graph", "dec_sac", "enc_fast_attn",
                          "skip_pad_frames", "sync_free", "h2d_stream", "async_save")
              if getattr(c, k) and (k != "compile" or self.compiled)
              and (self.device.type == "cuda" or k not in ("act_cuda_graph", "dream_cuda_graph", "h2d_stream"))]
        print(f"\033[90m    [setup] speed: {', '.join(on) or 'none'}"
              f"{' (the first updates compile: minutes, cached on disk afterwards)' if self.compiled else ''}\033[0m")

    @property
    def config(self):
        """Config as a plain dict (stored in checkpoints)."""
        return {**config_to_dict(self.cfg), "arch_version": self.ARCH_VERSION}

    # --- Batches ----------------------------------------------------------------------------------------------------
    def start_prefetch(self, batch_size=None, seq_len=None):
        """Samples batches on a background thread while the GPU trains (the buffer may grow meanwhile)."""
        self.stop_prefetch()
        cuda = self.device.type == "cuda"
        self._prefetcher = BatchPrefetcher(self.buffer, batch_size or self.cfg.batch_size,
                                           seq_len or self.cfg.seq_len, pin=cuda,
                                           device=self.device if cuda and self.cfg.h2d_stream else None)

    def stop_prefetch(self):
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None

    def transform_reward(self, r):
        """The reward the model learns: raw ("none") or its sign ("sign")."""
        return torch.sign(r) if self.cfg.reward_transform == "sign" else r

    def sample_batch(self, batch_size=None, seq_len=None):
        """Replay batch on device, decoded into model inputs and targets; None while the buffer has no window."""
        batch_size, seq_len = batch_size or self.cfg.batch_size, seq_len or self.cfg.seq_len
        pf = self._prefetcher
        use_pf = pf is not None and (pf.batch_size, pf.seq_len) == (batch_size, seq_len)
        item = pf.get() if use_pf else self.buffer.sample(batch_size, seq_len)
        if item is None:
            return None
        if "__gpu__" in item:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(item["__event__"])
            b, extra = item["__gpu__"], item["__cpu__"]
            for v in b.values():
                v.record_stream(stream)
        else:
            if "__cpu__" in item:
                extra, cpu = item["__cpu__"], item["__batch__"]
            else:
                cpu, extra = with_batch_extras(item)
            b = {k: v.to(self.device, non_blocking=True) for k, v in cpu.items()}
        return {
            "image": b["images"],
            "actions": F.one_hot(b["actions"].long(), self.action_dim).float(),
            "rewards": self.transform_reward(b["rewards"].float()),
            "continues": b["continues"],
            "valid": b["valid"],
            "episode_end": b["episode_end"],
            "index": b["index"],
            "real_idx": b["real_idx"], "real_gather": b["real_gather"],
            **extra,
        }

    @staticmethod
    def _flat_obs(batch, idx=None):
        """Encoder inputs of a batch, flattened over [B, T] (optionally indexed)."""
        out = {}
        for k in OBS_KEYS:
            v = batch[k].flatten(0, 1)
            out[k] = v if idx is None else v[idx]
        return out

    # --- Small helpers ----------------------------------------------------------------------------------------------
    def _amp(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                              enabled=self.amp_dtype is not None)

    def _optimize(self, loss, name, params, max_norm=None):
        """Backward + optional clipping + step of optimizer `name` through its GradScaler -> pre-clip norm."""
        optimizer, scaler = getattr(self, name), self.scalers[name]
        if self.cfg.sync_free and not scaler.is_enabled() and optimizer.defaults.get("fused"):
            loss.backward()
            norm, optimizer.found_inf = None, None
            if max_norm is not None:
                grads = [p.grad for p in params if p.grad is not None]
                norm = (torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads))) if grads
                        else torch.zeros((), device=self.device))
                torch._foreach_mul_(grads, (max_norm / (norm + 1e-6)).clamp(max=1.0))
                optimizer.found_inf = (~torch.isfinite(norm)).float()
            optimizer.step()
            return norm
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(params, max_norm) if max_norm is not None else None
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        elif norm is None or torch.isfinite(norm):
            optimizer.step()
        return norm

    @torch.no_grad()
    def _critic_ema_update(self):
        for pt, po in zip(self.critic_target.parameters(), self.critic.parameters()):
            pt.lerp_(po, 1.0 - self.cfg.critic_ema_tau)

    def _defer_priorities(self, batch, dyn):
        """Surprise priorities without a host sync: written into the buffer at the next update."""
        self.flush_priorities()
        val = torch.empty(dyn.shape, dtype=torch.float32, pin_memory=True)
        val.copy_(dyn.detach().float(), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self._pending_prio = (ev, batch["index_cpu"][:, 1:], batch["valid_cpu"][:, 1:] > 0, val)

    def flush_priorities(self):
        if self._pending_prio is not None:
            ev, idx, sel, val = self._pending_prio
            ev.synchronize()
            self.buffer.update_priorities(idx[sel], val[sel])
            self._pending_prio = None

    @staticmethod
    def full_state(h, zf):
        """Agent state s = [h, z]."""
        return torch.cat([h.float(), zf.float()], dim=-1)

    def alive_prob(self, s, logit=None):
        """p(continue) of states s [..., S] (or of a continue `logit`); with cfg.continue_calibrated, log of
        continue_zero_weight is added to the logit, which undoes the loss weighting of the terminal rows."""
        logit = (self.continue_head(s) if logit is None else logit).float()
        w = float(self.cfg.continue_zero_weight)
        if self.cfg.continue_calibrated and w > 0 and w != 1.0:
            logit = logit + math.log(w)
        return torch.sigmoid(logit)

    def real_rewards(self, batch):
        """Rewards of the batch's transitions aligned with the states (entry j: state j = observation j + 1 ->
        observation j + 2) [B, T-1], plus cfg.death_penalty on every transition into a terminal row (never stored, so
        the reward head does not learn it)."""
        rows = batch["rewards"].float()
        if self.cfg.death_penalty:
            nxt = (batch["continues"][:, 1:] <= 0).float() * batch["valid"][:, 1:]
            rows = rows + torch.cat([self.cfg.death_penalty * nxt, torch.zeros_like(rows[:, :1])], dim=1)
        return rows[:, 1:]

    # --- Encoder ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_frames(self, obs, chunk=1024):
        """Observations {"image": [..., 3, S, S]} -> (flat one-hot samples [..., D], posterior log-probs [..., L, K])."""
        img = obs["image"]
        lead = img.shape[:-3]
        flat = img.reshape(-1, *img.shape[-3:])
        zs, lps = [], []
        with self._amp():
            for i in range(0, flat.shape[0], chunk):
                z, lp = self.tokenizer.encode({"image": flat[i:i + chunk]})
                zs.append(z.float())
                lps.append(lp.float())
        return (torch.cat(zs).view(*lead, self.latent_dim),
                torch.cat(lps).view(*lead, self.n_latents, self.classes))

    @torch.no_grad()
    def _update_drift(self):
        """Every drift_interval updates: share of categoricals whose most likely class changed on fixed probe frames."""
        c = self.cfg
        if c.drift_interval <= 0 or self.total_num_updates % c.drift_interval:
            return
        if self._probe is None:
            n = min(c.drift_frames, len(self.buffer))
            if n < 2:
                return
            self._probe = {"image": self.buffer.sample_frames(n)["image"].to(self.device)}
        mode = self.encode_frames(self._probe)[1].argmax(-1)
        if self._probe_ref is not None:
            self.last_drift = (mode != self._probe_ref).float().mean().item()
        self._probe_ref = mode

    # --- World model: losses ----------------------------------------------------------------------------------------
    def _recon_branch(self, batch, z):
        """Pixel decoder on z only. Clean part: the posterior samples of recon_frames real frames t >= 1 (0 = every
        valid frame); their previous frames give the motion mask / weights. Augmented part (mask_frames > 0): some of
        them re-encoded with random patch masking, weighted mask_weight in the loss."""
        c, dev = self.cfg, self.device
        T = z.shape[1]
        valid1 = batch["valid"][:, 1:].flatten()
        n_valid = batch["n_valid1"] if "n_valid1" in batch else int(valid1.sum().item())
        if c.recon_frames <= 0 or c.recon_frames >= n_valid:
            sel1 = valid1.nonzero().squeeze(1)
        else:
            sel1 = torch.multinomial(valid1, c.recon_frames, replacement=False)
        sel = (sel1 // (T - 1)) * T + sel1 % (T - 1) + 1                   # the same frames, index into [B, T]
        imgs = batch["image"].flatten(0, 1)
        o_img = imgs[sel]
        tgt = self.tokenizer.recon_targets(o_img, imgs[sel - 1], c.motion_weight, c.motion_threshold)
        zc = z.flatten(0, 1)[sel]
        pix_loss, stats = self.tokenizer.recon_loss(zc, tgt)
        n_aug = min(c.mask_frames, sel.shape[0])
        hide = None
        if n_aug > 0:
            pick = torch.randperm(sel.shape[0], device=dev)[:n_aug]
            hide = sample_hide(n_aug, self.tokenizer.encoder.n_image_tokens, c.mask_max, dev)
            za = sample_onehot(self.tokenizer.posterior({"image": o_img[pick]}, hide)).flatten(1)
            pix_a, _ = self.tokenizer.recon_loss(za, self.tokenizer.subset_targets(tgt, pick))
            pix_loss = (1.0 - c.mask_weight) * pix_loss + c.mask_weight * pix_a
        parts = {"pixel": c.w_recon * pix_loss}
        with torch.no_grad():
            stats.update({"pix_loss": pix_loss.detach(),
                          "mask_ratio": hide.float().mean() if hide is not None else torch.zeros((), device=dev)})
        return stats, parts

    def _usage_loss(self, post, valid):
        """Class-usage regulariser, MAGVIT-v2's LFQ entropy loss applied to every categorical: the mean entropy of the
        single posteriors plus (log K minus the entropy of the batch-averaged posterior), summed over the categoricals.
        It is 0 only when each frame's posterior is certain AND different frames use different classes, i.e. when z
        depends on the image; both degenerate solutions (every posterior uniform, or every frame mapped to the same
        class) cost log K per categorical; it equals L log K minus the mutual information between frames and classes
        estimated on the batch. Optional (cfg.w_usage, 0 = off, the loss is then only logged): at a posterior that
        does not depend on the image its gradient vanishes, so it cannot pull a collapsed latent out by itself; it
        sharpens image dependence once there is some. -> (loss, {"usage_sample_entropy", "usage_marginal_gap"}),
        the parts per categorical, averaged."""
        w = valid.float()[..., None]
        logp = post.float()
        p = logp.exp()
        h_sample = (-(p * logp).sum(-1) * w).sum((0, 1)) / w.sum().clamp(min=1.0)           # [L]
        pbar = (p * w[..., None]).sum((0, 1)) / w.sum().clamp(min=1.0)                      # [L, K]
        gap = math.log(self.classes) + (pbar * pbar.clamp_min(1e-8).log()).sum(-1)          # [L], >= 0
        return (h_sample + gap).sum(), {"usage_sample_entropy": h_sample.mean().detach(),
                                        "usage_marginal_gap": gap.mean().detach()}

    @staticmethod
    def _targets(batch, rows, t):
        """Targets of the state at observation t: reward of t-1 -> t and the continue flag of t."""
        return {"reward": batch["rewards"][rows, t - 1], "cont": batch["continues"][rows, t],
                "mask": batch["valid"][rows, t]}

    def _head_losses(self, s, tgt):
        """Reward and continue losses on states s [..., S]; pads are ignored."""
        c, m = self.cfg, tgt["mask"]
        reward_loss = self.two_hot.loss(self.reward_head(s), tgt["reward"], m)
        cont_logit = self.continue_head(s).float()
        cont = tgt["cont"]
        cw = (1.0 + (c.continue_zero_weight - 1.0) * (1.0 - cont)) * m     # rare terminal rows up-weighted
        continue_loss = (F.binary_cross_entropy_with_logits(cont_logit, cont, reduction="none") * cw).sum() \
            / cw.sum().clamp(min=1.0)
        with torch.no_grad():
            p_alive = self.alive_prob(None, logit=cont_logit)
            dead = (cont < 0.5).float() * m
            alive = (cont >= 0.5).float() * m
            nan = torch.tensor(float("nan"), device=p_alive.device)
            at_death = (p_alive * dead).sum() / dead.sum().clamp(min=1.0)
            stats = {"p_alive_at_death": torch.where(dead.sum() > 0, at_death, nan),
                     "p_alive_alive": (p_alive * alive).sum() / alive.sum().clamp(min=1.0),
                     "n_deaths": dead.sum()}
        return {"reward": reward_loss, "cont": continue_loss}, stats

    def _rollout(self, kv, z, post, batch):
        """Open-loop rollout from a random split point, fed its own prior samples: KL to real posteriors + states."""
        c, dev = self.cfg, self.device
        B, T = z.shape[:2]
        K = c.rollout_steps
        rows = torch.arange(B, device=dev).repeat_interleave(c.rollout_splits)
        R = rows.shape[0]
        n_real = batch["valid"].sum(1).long()[rows]
        p_max = torch.clamp(n_real - 1 - K, min=1, max=T - 1 - K)
        p = 1 + torch.minimum((torch.rand(R, device=dev) * p_max).long(), p_max - 1)
        state = self.tssm.open_loop_state(kv, rows, p)
        zin = z[rows, p].detach()
        kls, states, match = [], [], []
        for j in range(K):
            t = p + j
            h = self.tssm.step_append(zin, batch["actions"][rows, t], state).float()
            raw = self.prior(h)
            target = post[rows, t + 1].detach()
            kls.append(kl_categorical(target, unimix_logprobs(raw, self.prior.unimix)))
            with torch.no_grad():
                zs = sample_onehot(unimix_logprobs(raw, c.dream_unimix), straight_through=False)
                match.append((zs.argmax(-1) == target.argmax(-1)).float().mean(-1))
                zin = zs.flatten(1)
            states.append(self.full_state(h, zin))
        t_idx = p[:, None] + 1 + torch.arange(K, device=dev)[None]
        M = batch["valid"][rows[:, None], t_idx]
        w = c.rollout_decay ** torch.arange(K, device=dev, dtype=torch.float32)
        kl = torch.stack(kls, dim=1)
        loss = (kl.clamp(min=c.free_nats) * M * w).sum() / (M * w).sum().clamp(min=1e-6)
        return {"loss": loss, "states": torch.stack(states, dim=1), "rows": rows, "t": t_idx, "mask": M,
                "match": torch.stack(match, dim=1), "kl": kl.detach()}

    # --- World model: update ----------------------------------------------------------------------------------------
    def train_world_model(self, batch, compute_metrics=True):
        """One end-to-end world-model update -> (states [B, T-1, S], kv context, start mask, metrics, start info)."""
        c = self.cfg
        B, T = batch["image"].shape[:2]
        valid = batch["valid"]
        m1 = valid[:, 1:]
        tm = self.timer
        tm.start(compute_metrics)
        zero = torch.zeros((), device=self.device)
        self.wm_optimizer.zero_grad(set_to_none=True)
        measure = (c.grad_share_interval > 0 and self._grad_share_ok
                   and self.total_num_updates % c.grad_share_interval == 0)
        eager = eager_modules() if measure else contextlib.nullcontext()
        with eager, self._amp():
            # 1. clean posteriors of every frame, straight-through one-hot samples
            flat = self._flat_obs(batch)
            if c.skip_pad_frames and "real_idx" in batch:                 # pads repeat their row's last real frame
                ri = batch["real_idx"]
                post = self.tokenizer.posterior({k: flat[k][ri] for k in OBS_KEYS})[batch["real_gather"]]
            else:
                post = self.tokenizer.posterior(flat)
            post = post.view(B, T, self.n_latents, self.classes)
            z = sample_onehot(post).flatten(-2)
            tm.mark("encode")

            # 2. dynamics: TSSM (z_t, a_t) -> h_{t+1} -> prior of z_{t+1}; balanced KL with free nats
            h, kv = self.tssm.forward_sequence(z[:, :-1], batch["actions"][:, :-1], return_kv=True, detach_kv=False)
            h = h.float()
            prior_raw = self.prior(h)
            prior = unimix_logprobs(prior_raw, self.prior.unimix)
            post1 = post[:, 1:]
            dyn = kl_categorical(post1.detach(), prior)
            rep = kl_categorical(post1, prior.detach())
            kl_dyn_term = c.w_dyn * masked_mean(dyn.clamp(min=c.free_nats), m1)
            kl_rep_term = c.w_rep * masked_mean(rep.clamp(min=c.free_nats), m1)
            kl_loss = kl_dyn_term + kl_rep_term
            usage_loss, usage_stats = self._usage_loss(post, valid)
            if c.w_usage <= 0:
                usage_loss = usage_loss.detach()
            tm.mark("TSSM + prior fwd")

            # 3. heads on real states [h_{t+1}, z_{t+1}] and on dream-like states (rollout or one-step prior samples)
            s_real = self.full_state(h, z[:, 1:])
            rows = torch.arange(B, device=self.device)[:, None]
            t_real = torch.arange(1, T, device=self.device)[None]
            tgt_real = self._targets(batch, rows, t_real)
            head_r, head_stats = self._head_losses(s_real, tgt_real)
            ro = None
            if c.rollout_steps > 0:
                ro = self._rollout(kv, z, post, batch)
                d_states, d_tgt = ro["states"], self._targets(batch, ro["rows"][:, None], ro["t"])
                rollout_loss = ro["loss"]
            else:
                with torch.no_grad():
                    z1 = sample_onehot(unimix_logprobs(prior_raw, c.dream_unimix), straight_through=False)
                d_states, d_tgt = self.full_state(h, z1.flatten(-2)), tgt_real
                rollout_loss = zero
            head_o, _ = self._head_losses(d_states, d_tgt)
            head = {k: (1.0 - c.w_dreamlike) * head_r[k] + c.w_dreamlike * head_o[k] for k in head_r}
            tm.mark("rollout + heads fwd")

            # 4. reconstruction: the decoder reads z only
            rec_stats, rec_parts = self._recon_branch(batch, z)
            tm.mark("decode")

        terms = {"pixel": rec_parts["pixel"], "kl_dyn": kl_dyn_term, "kl_rep": kl_rep_term,
                 "rollout": c.w_rollout * rollout_loss, "reward": c.w_reward * head["reward"],
                 "continue": c.w_continue * head["cont"], "usage": c.w_usage * usage_loss}
        world_model_loss = sum(terms.values())
        self._accumulate_wm_losses(terms)
        if measure:
            with eager_modules():
                self._measure_grad_shares(terms)
        groups = {"pixel": terms["pixel"], "kl": terms["kl_dyn"] + terms["kl_rep"], "rollout": terms["rollout"],
                  "reward": terms["reward"], "cont": terms["continue"]}
        hw = None
        if c.harmony:
            hw = self.harmony.weights()
            wm_opt_loss = sum(hw[k] * v for k, v in groups.items()) + terms["usage"]
        else:
            wm_opt_loss = world_model_loss
        clip = self.wm_clipper.threshold()
        grad_norm = self._optimize(wm_opt_loss, "wm_optimizer", self.wm_params, clip)
        if not c.sync_free:
            grad_norm = float(grad_norm)
        self.wm_clipper.record(grad_norm, clip)
        if c.harmony:
            self.harmony_optimizer.zero_grad(set_to_none=True)
            self.harmony.scale_loss({k: v.detach() for k, v in groups.items()}).backward()
            self.harmony_optimizer.step()
        tm.mark("WM backward + step")
        self._update_drift()

        start_mask = (m1 * (batch["continues"][:, 1:] > 0).float()).detach()
        with torch.no_grad():                                              # distance to the next terminal + surprise
            cont, idx = batch["continues"], torch.arange(T, device=self.device)[None].expand(B, T)
            ends = batch.get("episode_end")
            if ends is None:
                ends = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1) <= 0
            death = (cont <= 0) & (valid > 0) & ends
            pos = torch.where(death, idx, torch.full_like(idx, 4 * T))
            next_death = torch.flip(torch.cummin(torch.flip(pos, [1]), dim=1).values, [1])
            start_info = {"death_dist": (next_death - idx)[:, 1:], "surprise": dyn.detach().float() * m1}
            if c.loss_sample_prob > 0 and "index" in batch:
                if c.sync_free and "index_cpu" in batch and self.device.type == "cuda":
                    self._defer_priorities(batch, dyn)
                else:
                    sel = (m1 > 0).cpu()
                    self.buffer.update_priorities(batch["index"][:, 1:].cpu()[sel], dyn.detach().float().cpu()[sel])
        if compute_metrics or self.total_num_updates % self.DEATH_PROBE_EVERY == 0:
            self._accumulate_death_probes(h, prior_raw, z, kv, batch, death)

        metrics = {}
        if compute_metrics:
            with torch.no_grad():
                metrics = {"world_model_loss_last": world_model_loss.item(), "kl_dyn": masked_mean(dyn, m1).item(),
                           "kl_rep": masked_mean(rep, m1).item(), "kl_loss": kl_loss.item(),
                           "rollout_loss": rollout_loss.item(), "wm_grad_norm": float(grad_norm),
                           "padded_fraction": 1.0 - valid.mean().item(), "encoder_drift": self.last_drift}
                metrics.update({k: float(v) for k, v in rec_stats.items()})
                metrics.update({"reward_loss": head["reward"].item(), "continue_loss": head["cont"].item(),
                                "usage_loss": float(usage_loss)})
                metrics.update({k: float(v) for k, v in usage_stats.items()})
                metrics.update({k: v.item() for k, v in head_stats.items()})
                mode_post = post.argmax(-1)
                metrics["post_entropy"] = masked_mean(entropy_categorical(post), valid).item()
                metrics["prior_entropy"] = masked_mean(entropy_categorical(prior), m1).item()
                metrics["latent_usage"] = float(class_usage(z.detach(), self.n_latents, self.classes, valid))
                metrics["pred_match"] = masked_mean((prior.argmax(-1) == mode_post[:, 1:]).float().mean(-1), m1).item()
                metrics["copy_match"] = masked_mean((mode_post[:, :-1] == mode_post[:, 1:]).float().mean(-1), m1).item()
                if ro is not None:
                    k = ro["match"].shape[1] - 1
                    metrics["rollout_match_k"] = masked_mean(ro["match"][:, k], ro["mask"][:, k]).item()
                    metrics["rollout_kl_k"] = masked_mean(ro["kl"][:, k], ro["mask"][:, k]).item()
                clip_threshold, clip_fraction = self.wm_clipper.stats()
                metrics.update({"wm_clip_threshold": clip_threshold, "wm_clip_fraction": clip_fraction})
                if hw is not None:
                    metrics.update({f"harmony_{k}": float(v) for k, v in hw.items()})
                lo, hi = c.death_dream_window
                dd = start_info["death_dist"]
                metrics["pre_death_states"] = float(((dd >= lo) & (dd <= hi)).float().mul(start_mask).sum())
                metrics.update(self._death_probe_means())
                metrics.update(self._wm_loss_means())
                metrics.update(self.last_grad_shares)
            tm.mark("metrics")
        kv_context = [(k.detach(), v.detach()) for k, v in kv]
        return s_real.detach(), kv_context, start_mask, metrics, start_info

    GRAD_SHARE_GROUPS = ("tssm", "encoder")

    def _measure_grad_shares(self, terms):
        """Diagnostic (no training effect): the gradient norm of every weighted loss term on the TSSM and on the
        encoder parameters, and each term's share -> self.last_grad_shares."""
        groups = {"tssm": [p for p in self.tssm.parameters() if p.requires_grad],
                  "encoder": [p for p in self.tokenizer.encoder.parameters() if p.requires_grad]}
        params = groups["tssm"] + groups["encoder"]
        n_t = len(groups["tssm"])
        zero = torch.zeros((), device=self.device)
        try:
            norms = {}
            for name in self.WM_LOSS_TERMS:
                loss = terms[name]
                if not (torch.is_tensor(loss) and loss.requires_grad):
                    norms[name] = (zero, zero)
                    continue
                grads = torch.autograd.grad(loss.float(), params, retain_graph=True, allow_unused=True)

                def norm(gs):
                    sq = [g.float().square().sum() for g in gs if g is not None]
                    return torch.stack(sq).sum().sqrt() if sq else zero
                norms[name] = (norm(grads[:n_t]), norm(grads[n_t:]))
                del grads
            vals = torch.stack([torch.stack(v) for v in norms.values()]).tolist()
        except Exception as e:
            print(f"[!] gradient-share log disabled ({type(e).__name__}: {e})")
            self._grad_share_ok = False
            return
        out = {"grad_share_update": float(self.total_num_updates)}
        for gi, group in enumerate(self.GRAD_SHARE_GROUPS):
            total = sum(v[gi] for v in vals)
            for name, v in zip(norms, vals):
                out[f"gnorm_{group}_{name}"] = v[gi]
                out[f"gshare_{group}_{name}"] = v[gi] / total if total > 0 else 0.0
        self.last_grad_shares = out

    @torch.no_grad()
    def _accumulate_wm_losses(self, terms):
        vals = torch.stack([terms[k].detach().float() for k in self.WM_LOSS_TERMS])
        if self._wm_loss_acc is None:
            self._wm_loss_acc = torch.zeros(len(self.WM_LOSS_TERMS) + 1, device=vals.device)
        self._wm_loss_acc[:-1] += vals
        self._wm_loss_acc[-1] += 1.0

    def _wm_loss_means(self):
        """Round averages of the weighted loss terms ("wm_<term>") and of their sum ("world_model_loss"); resets."""
        if self._wm_loss_acc is None or float(self._wm_loss_acc[-1]) == 0.0:
            return {}
        acc = self._wm_loss_acc.tolist()
        n = acc[-1]
        out = {f"wm_{k}": v / n for k, v in zip(self.WM_LOSS_TERMS, acc[:-1])}
        out["world_model_loss"] = sum(acc[:-1]) / n
        out["wm_updates"] = n
        self._wm_loss_acc.zero_()
        return out

    # --- Open-loop probes (diagnostics) -----------------------------------------------------------------------------
    @torch.no_grad()
    def _accumulate_death_probes(self, h, prior_raw, z, kv, batch, death):
        """Adds p(continue) at the real terminals of a batch, read on dream-side states, to the round sums: one dream
        step (pred1) and DEATH_PROBE_STEPS open-loop steps on the model's own samples with the real actions (dream,
        and the survival product). A model whose dreams end where the real game did gives values near 0."""
        c, dev = self.cfg, self.device
        if self._death_probe_acc is None:
            self._death_probe_acc = torch.zeros(5, device=dev)
        acc = self._death_probe_acc
        rows, t = torch.nonzero(death[:, 1:], as_tuple=True)
        if rows.numel() == 0:
            return
        t = t + 1
        alive = self.alive_prob
        sample = lambda raw: sample_onehot(unimix_logprobs(raw, c.dream_unimix),  # noqa: E731
                                           straight_through=False).flatten(-2)
        with self._amp():
            z1 = sample(prior_raw[rows, t - 1].detach().float())
            pa1 = alive(self.full_state(h[rows, t - 1].detach(), z1))
            acc[0] += pa1.sum()
            acc[1] += pa1.numel()
            k = self.DEATH_PROBE_STEPS
            ok = t - k >= 1
            if bool(ok.any()):
                r, p = rows[ok], t[ok] - k
                state = self.tssm.open_loop_state([(a.detach(), b.detach()) for a, b in kv], r, p)
                zz = z[r, p].detach()
                surv = torch.ones(r.shape[0], device=dev)
                for j in range(k):
                    hj = self.tssm.step_append(zz, batch["actions"][r, p + j], state).float()
                    zz = sample(self.prior(hj))
                    pa = alive(self.full_state(hj, zz))
                    surv = surv * pa
                acc[2] += pa.sum()
                acc[3] += surv.sum()
                acc[4] += pa.numel()

    def _death_probe_means(self):
        nan = float("nan")
        out = {"p_alive_death_pred1": nan, "p_alive_death_dream": nan, "dream_death_survival": nan,
               "death_probe_deaths": 0.0}
        if self._death_probe_acc is not None:
            s = self._death_probe_acc.tolist()
            if s[1] > 0:
                out["p_alive_death_pred1"] = s[0] / s[1]
            if s[4] > 0:
                out["p_alive_death_dream"] = s[2] / s[4]
                out["dream_death_survival"] = s[3] / s[4]
            out["death_probe_deaths"] = s[1]
            self._death_probe_acc.zero_()
        return out

    @torch.no_grad()
    def context_kv(self, zf, actions):
        """Teacher-forced pass over real latents -> (h [B, T-1, d], kv)."""
        with self._amp():
            h, kv = self.tssm.forward_sequence(zf[:, :-1], actions[:, :-1], return_kv=True)
        return h.float(), kv

    @torch.no_grad()
    def dream_step(self, z, action, cache):
        """One eager imagined step -> (h, one-hot sample z, prior log-probs)."""
        h = self.tssm.forward_step(z, action, cache).float()
        logp = unimix_logprobs(self.prior(h), self.cfg.dream_unimix)
        return h, sample_onehot(logp, straight_through=False).flatten(1), logp

    @torch.no_grad()
    def open_loop(self, zf, actions, kv, valid, n_rows=16, horizon=15, t0=None, rows=None):
        """Open-loop generations from real contexts: the recorded actions (two seeds), NOOP held, random actions."""
        B, T = valid.shape
        lens = valid.sum(1).long()
        if rows is None:
            cand = torch.nonzero(lens >= horizon + 2).flatten()
            if cand.numel() == 0:
                return None
            rows = cand[torch.randperm(cand.numel(), device=cand.device)[:n_rows]]
        n = rows.shape[0]
        if t0 is None:
            span = (lens[rows] - 1 - horizon).clamp(min=1)
            t0 = 1 + (torch.rand(n, device=self.device) * span).long().clamp(max=span - 1)
        steps = t0[:, None] + torch.arange(horizon + 1, device=self.device)[None]
        real = zf[rows[:, None], steps]
        runs = {}
        with self._amp():
            for name in ("teacher", "teacher2", "noop", "random"):
                cache = self.tssm.make_cache(n, self.device, self.cache_dtype)
                self.tssm.prefill(cache, kv, rows, t0 - 1)
                z = zf[rows, t0]
                out = []
                for k in range(horizon):
                    if name == "noop":
                        a = F.one_hot(torch.zeros(n, dtype=torch.long, device=self.device), self.action_dim).float()
                    elif name == "random":
                        a = F.one_hot(torch.randint(0, self.action_dim, (n,), device=self.device),
                                      self.action_dim).float()
                    else:
                        a = actions[rows, t0 + k]
                    _, z, _ = self.dream_step(z, a, cache)
                    out.append(z.float())
                runs[name] = torch.stack(out, dim=1)
        return {"rows": rows, "t0": t0, "real": real, **runs}

    # --- Imagination ------------------------------------------------------------------------------------------------
    def _dream_graph_for(self, n):
        """CUDA graph of one imagined step for n dreams, or None to run eagerly."""
        if not self._use_dream_graph or self.device.type != "cuda":
            return None
        g = self._dream_graph
        if g is not None and g.n == n:
            return g
        self._dream_graph = None
        try:
            g = GraphedDreamStep(self.tssm, self.prior, n, self.cache_dtype, self.amp_dtype, self.device)
        except Exception as e:
            print(f"[!] dream CUDA graph disabled ({type(e).__name__}: {e}); dreams run eagerly.")
            self._use_dream_graph = False
            return None
        self._dream_graph = g
        return g

    def _dream_starts(self, start_mask, start_info, B, T, dev):
        """Start (row, t0) of every dream: uniform, pre-terminal (each dream with probability death_dream_prob),
        surprising, or first-action branch groups."""
        c, K, A = self.cfg, self.cfg.dreams_per_sequence, self.action_dim
        N = B * K
        rows = torch.arange(B, device=dev).repeat_interleave(K)
        elig = start_mask.float() if start_mask is not None else torch.ones(B, T, device=dev)
        e = elig[rows]
        counts = e.sum(1, keepdim=True)
        t0 = torch.multinomial(torch.where(counts > 0, e, torch.ones_like(e)), 1).squeeze(1)
        forced = torch.full((N,), -1, dtype=torch.long, device=dev)
        pre = torch.zeros(B, T, device=dev)
        G, n_norm = 0, N
        if start_info is not None:
            lo, hi = c.death_dream_window
            pre = ((start_info["death_dist"] >= lo) & (start_info["death_dist"] <= hi)).float() * elig
            floor = 1e-6 * elig.flatten() + 1e-12                          # empty pool -> uniform over eligible
            G = min(int(c.branch_groups), (N // 2) // A)
            n_norm = N - G * A
            if c.death_dream_prob > 0:
                death_m = torch.rand(n_norm, device=dev) < c.death_dream_prob
                idx = torch.multinomial(pre.flatten() + floor, n_norm, replacement=True)
                rows[:n_norm] = torch.where(death_m, idx // T, rows[:n_norm])
                t0[:n_norm] = torch.where(death_m, idx % T, t0[:n_norm])
            else:
                death_m = torch.zeros(n_norm, dtype=torch.bool, device=dev)
            n_l = min(int(round(c.dream_loss_frac * n_norm)), n_norm)
            if n_l > 0:
                sur = start_info["surprise"].float().clamp_min(0.0) * elig
                idx = torch.multinomial(sur.flatten() + floor, n_l, replacement=True)
                rank = torch.cumsum((~death_m).long(), 0) - 1
                loss_m = ~death_m & (rank < n_l)
                pick = idx[rank.clamp(0, n_l - 1)]
                rows[:n_norm] = torch.where(loss_m, pick // T, rows[:n_norm])
                t0[:n_norm] = torch.where(loss_m, pick % T, t0[:n_norm])
            if G > 0:
                idx = torch.multinomial(pre.flatten() + floor, G, replacement=False)
                rows[n_norm:] = (idx // T).repeat_interleave(A)
                t0[n_norm:] = (idx % T).repeat_interleave(A)
                forced[n_norm:] = torch.arange(A, device=dev).repeat(G)
        start_ok = (elig[rows, t0] > 0).float()
        return rows, t0, start_ok, forced, n_norm, G, pre

    def imagine(self, full_state, kv_context, start_mask=None, start_info=None, real=None, horizon=None,
                compute_metrics=True, pack_dreams=True):
        """Actor-critic update on dreams started from real states -> (metrics, best dream, random dream)."""
        c = self.cfg
        horizon = horizon or c.dream_horizon
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        tm = self.timer
        tm.start(compute_metrics, reset=False)

        all_states = full_state.detach()
        B, T = all_states.shape[:2]
        dev = all_states.device
        A = self.action_dim
        rows, t0, start_ok, forced, n_norm, G, pre = self._dream_starts(start_mask, start_info, B, T, dev)
        num_dreams = rows.shape[0]
        start_states = all_states[rows, t0]
        graph = self._dream_graph_for(num_dreams)
        cache = graph.cache if graph is not None else self.tssm.make_cache(num_dreams, dev, self.cache_dtype)
        self.tssm.prefill(cache, kv_context, rows, t0)

        with self._amp():
            # --- roll out the policy in the world model
            states = [start_states]
            log_probs, entropies, acts = [], [], []
            curr = start_states
            z = start_states[:, self.tssm_dim:]
            probs0 = None
            for i in range(horizon):
                action, logprob, entropy, probs = self.actor(curr, return_probs=True)
                if i == 0:
                    probs0 = probs.detach()
                    if G > 0:                                              # branch dreams: given first action
                        a_f = F.one_hot(forced.clamp(min=0), A).float()
                        is_b = forced >= 0
                        action = torch.where(is_b[:, None], a_f, action)
                        logprob = torch.where(is_b, (probs.clamp_min(1e-8).log() * a_f).sum(-1), logprob)
                with torch.no_grad():
                    if graph is not None:
                        h, z, _ = graph(action.detach(), z, unimix=c.dream_unimix)
                    else:
                        h, z, _ = self.dream_step(z, action.detach(), cache)
                    curr = torch.nan_to_num(self.full_state(h, z), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
                log_probs.append(logprob)
                entropies.append(entropy)
                acts.append(action)
            states = torch.stack(states, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            entropies = torch.stack(entropies, dim=1)
            acts = torch.stack(acts, dim=1)

            # --- predicted rewards and survival
            with torch.no_grad():
                nxt = states[:, 1:]
                pred_rewards = self.two_hot.decode(self.reward_head(nxt)).squeeze(-1)
                pred_alive = self.alive_prob(nxt)
                weights = torch.cat([torch.ones_like(pred_alive[:, :1]),
                                     torch.cumprod(pred_alive, dim=1)[:, :-1]], dim=1) * start_ok[:, None]
                cont = c.continue_discount * pred_alive
                r_death = c.death_penalty * (1.0 - pred_alive) if c.death_penalty else torch.zeros_like(pred_alive)
                pred_rewards = pred_rewards + r_death

            # --- returns and advantages
            imagined = states.detach()
            critic_logits = self.critic(imagined)
            with torch.no_grad():
                values = self.two_hot.decode(self.critic_target(imagined)).squeeze(-1)
                lam = lambda_returns(pred_rewards, values, cont, c.lambda_)
                adv = lam - values[:, :-1]
                crit_w = weights
                if G > 0:                                                  # branch step 0: group-mean baseline
                    g_ = lam[n_norm:, 0].view(G, A)
                    adv[n_norm:, 0] = (g_ - g_.mean(1, keepdim=True)).flatten()
                    crit_w = weights.clone()
                    crit_w[n_norm:, 0] = 0.0
                if c.adv_norm == "percentile":                             # DreamerV3: A / max(1, EMA(P95 - P5))
                    self.return_scale.update(lam, weights)
                    scale = self.return_scale.divisor()
                    norm_adv = adv / scale
                else:                                                      # centred, unit std
                    m_a, scale = weighted_moments(adv, weights)
                    norm_adv = (adv - m_a) / scale.clamp(min=ADV_STD_MIN)
                forced_first = None
                if G > 0:
                    forced_first = (n_norm, probs0[n_norm:].gather(1, forced[n_norm:, None]).squeeze(1))

            # --- losses (REINFORCE actor)
            actor_loss, _ = reinforce_loss(c, norm_adv, log_probs, entropies, weights, forced_first)
            critic_loss = self.two_hot.loss(critic_logits[:, :-1], lam, crit_w)

            # replay value loss: the critic also regresses lambda-returns of real rewards along the real sequence
            use_repval = real is not None and c.w_repval > 0 and T > 1
            if use_repval:
                rc, rv = real["continues"].float(), real["valid"].float()
                with torch.no_grad():
                    real_values = self.two_hot.decode(self.critic_target(all_states)).squeeze(-1)
                    real_rew = real["rewards"][:, :T - 1].float()
                    step_ok = rv[:, 1:] * (rc[:, :T - 1] > 0).float()
                    real_lam = lambda_returns(real_rew, real_values, c.continue_discount * rc[:, 1:], c.lambda_,
                                              valid=step_ok)
                real_logits = self.critic(all_states[:, :-1])
                repval_loss = self.two_hot.loss(real_logits, real_lam, step_ok)
            else:
                repval_loss = torch.zeros((), device=dev)

        self._optimize(critic_loss + c.w_repval * repval_loss, "critic_optimizer", self.critic.parameters(), 1.0)
        self._optimize(actor_loss, "actor_optimizer", self.actor.parameters(), 1.0)
        self._critic_ema_update()
        tm.mark("dream + actor/critic")

        metrics = {}
        if compute_metrics:
            metrics = {
                "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "entropies": entropies.mean().item(),
                "advantages": adv.mean().item(), "return_scale": float(scale),
                "critic_values": self.two_hot.decode(critic_logits.detach()).mean().item(),
                "target_critic_values": values.mean().item(),
                "advantage_pos_fraction": (norm_adv >= 0).float().mean().item(),
                "dream_mean_reward": pred_rewards.mean().item(),
                "dream_mean_alive": pred_alive.mean().item(),
                "dream_death_fraction": (weights[:, -1] < 0.5).float().mean().item(),
                "replay_value_loss": repval_loss.item(),
                "replay_critic_values": self.two_hot.decode(real_logits.detach()).mean().item() if use_repval else 0.0,
                "replay_returns": masked_mean(real_lam, step_ok).item() if use_repval else 0.0,
                **actor_metrics(norm_adv, weights),
                "dream_start_pre_death": pre[rows, t0].mean().item(),
                "dream_death_penalty": masked_mean(r_death, weights).item(),
            }
            with torch.no_grad():
                pred = self.two_hot.decode(critic_logits[:, :-1].detach()).squeeze(-1)
                metrics["critic_ev"] = explained_variance(pred, lam, crit_w)
                r = torch.quantile(lam.flatten().float(), torch.tensor([0.05, 0.95], device=dev))
                metrics["return_range"] = float(r[1] - r[0])
            if G > 0:
                c0 = norm_adv[n_norm:, 0].view(G, A)
                pol = probs0[n_norm::A].argmax(-1)
                metrics["branch_agree"] = (c0.argmax(1) == pol).float().mean().item()
                metrics["branch_spread"] = (c0.max(1).values - c0.min(1).values).mean().item()
        if not pack_dreams:
            return metrics, None, None

        traj = (norm_adv * weights).sum(dim=1)
        fields = (states, pred_rewards, lam, acts, adv, norm_adv)
        if c.sync_free:                                                    # GPU copies; to CPU when the round ends
            def keep(i, metric, label):
                return (metric,) + tuple(x[i].detach().clone() for x in fields) + (label,)
            best_idx = torch.argmax(traj)
            rand_idx = torch.randint(0, traj.shape[0], (1,)).item()
            return metrics, keep(best_idx, traj[best_idx].detach().clone(), "MaxAdvantage"), \
                keep(rand_idx, traj[rand_idx].detach().clone(), "Random")

        def pack(i, metric, label):
            return (metric,) + tuple(x[i].detach().float().cpu() for x in fields) + (label,)
        best_idx = torch.argmax(traj).item()
        rand_idx = torch.randint(0, traj.shape[0], (1,)).item()
        return metrics, pack(best_idx, traj[best_idx].item(), "MaxAdvantage"), \
            pack(rand_idx, traj[rand_idx].item(), "Random")

    def update(self, batch, compute_metrics=False, pack_dreams=False):
        """One world-model + imagination update -> (wm metrics, dream metrics, best dream, random dream)."""
        states, kv_context, start_mask, m, start_info = self.train_world_model(batch, compute_metrics=compute_metrics)
        real = {"rewards": self.real_rewards(batch), "continues": batch["continues"][:, 1:],
                "valid": batch["valid"][:, 1:]}
        d, best, rnd = self.imagine(states, kv_context, start_mask=start_mask, start_info=start_info, real=real,
                                    compute_metrics=compute_metrics, pack_dreams=pack_dreams)
        self.total_num_updates += 1
        return m, d, best, rnd

    # --- Acting -----------------------------------------------------------------------------------------------------
    def new_act_context(self, n, key="train"):
        """Per-env acting state: TSSM cache and h (the static buffers of the acting CUDA graph when it is used; one
        graph per (n, key), so evaluation never resets the collector's)."""
        g = self._act_graph_for(n, key)
        if g is not None:
            g.reset()
            return {"cache": g.cache, "h": g.h, "graph": g}
        return {"cache": self.tssm.make_cache(n, self.device, self.cache_dtype),
                "h": torch.zeros(n, self.tssm_dim, device=self.device)}

    def _act_graph_for(self, n, key="train"):
        if not self._use_act_graph or self.device.type != "cuda":
            return None
        g = self._act_graphs.get((n, key))
        if g is None:
            try:
                t = time.perf_counter()
                g = GraphedActStep(self, n)
            except Exception as e:
                print(f"[!] acting CUDA graph disabled ({type(e).__name__}: {e}); acting runs eagerly.")
                self._use_act_graph = False
                return None
            self._act_graphs[(n, key)] = g
            if self.verbose:
                print(f"\033[90m    [setup] acting: CUDA graph of one acting step captured and checked for {n} envs "
                      f"in {time.perf_counter() - t:.1f}s\033[0m")
        return g

    @staticmethod
    def reset_act_rows(ctx, rows):
        """Clears the acting state of envs that start a new segment."""
        ctx["cache"].reset_rows(rows)
        ctx["h"][rows] = 0.0

    @torch.no_grad()
    def act(self, observations, ctx, epsilon=0.0, greedy=False):
        """Encode observations, sample actions from [h_t, z_t], advance the TSSM -> one-hot actions [N, A]."""
        g = ctx.get("graph")
        if g is not None and epsilon == 0.0 and not greedy:
            return g(observations)
        obs = obs_to_tensors(observations, self.device)
        with self._amp():
            z, _ = self.tokenizer.encode(obs)
            z = z.float()
            action, _, _ = self.actor(self.full_state(ctx["h"], z), greedy=greedy, unimix=self.cfg.play_unimix)
            if epsilon > 0.0:
                override = torch.rand(action.shape[0], device=self.device) < epsilon
                if override.any():
                    ra = torch.randint(0, self.action_dim, (int(override.sum().item()),), device=self.device)
                    action[override] = F.one_hot(ra, self.action_dim).float()
            h = self.tssm.forward_step(z, action, ctx["cache"]).float()
            if g is not None:
                ctx["h"].copy_(h)
            else:
                ctx["h"] = h
        return action

    # --- Streaming collection ---------------------------------------------------------------------------------------
    @staticmethod
    def _empty_collect_stats():
        return {"steps": 0, "returns": [], "lengths": [], "life_losses": 0, "segments": 0, "t_model": 0.0,
                "t_env": 0.0, "t_buffer": 0.0, "t_total": 0.0}

    def pop_collect_stats(self):
        """The collection statistics since the last call (finished games' returns / lengths, timings); resets."""
        s, self.collect_stats = self.collect_stats, self._empty_collect_stats()
        return s

    def _start_collector(self):
        """Resets every env and opens one replay segment per env."""
        n = len(self.envs)
        obs = [o for o, _ in self.envs.reset()]
        self._collector = {"ctx": self.new_act_context(n), "obs": obs, "pending": [[] for _ in range(n)],
                           "seg_id": [self.buffer.begin_episode(i) for i in range(n)], "seg_rows": [0] * n,
                           "ep_return": [0.0] * n, "ep_len": [0] * n}
        return self._collector

    def _flush(self, i, end=False, died=False):
        """Writes env i's pending rows into its ring; end: the segment is complete."""
        st = self._collector
        pend = st["pending"][i]
        if pend:
            self.buffer.add_rows(i, np.stack([p[0] for p in pend]), np.array([p[1] for p in pend]),
                                 np.array([p[2] for p in pend], dtype=np.float32),
                                 np.array([p[3] for p in pend], dtype=np.float32), end_episode=end)
            st["pending"][i] = []
        elif end:
            self.buffer.add_rows(i, np.zeros((0, 3, self.cfg.img_size, self.cfg.img_size), np.uint8), [], [], [],
                                 end_episode=True)
        if end:
            self.audit_snapshots.close_episode(i, st["seg_rows"][i], died, self.cfg.death_dream_window)
            self.collect_stats["segments"] += 1

    @torch.no_grad()
    def collect(self, n_steps, random_policy=False):
        """Plays n_steps env steps (all training envs together; episodes continue across calls) with the current
        policy or uniformly random actions, and streams them into the replay buffer. A life loss (with
        life_loss_terminal) or a game over closes the env's replay segment with a terminal row (cont = 0) and
        clears its acting memory; the 27k-step cap closes it without one. -> number of steps played."""
        c, envs = self.cfg, self.envs
        st = self._collector or self._start_collector()
        n = len(envs)
        pool = self.audit_snapshots
        stats = self.collect_stats
        t_start = time.perf_counter()
        steps = 0
        while steps < n_steps:
            k = min(n, n_steps - steps)
            active = [i < k for i in range(n)]
            t0 = time.perf_counter()
            if random_policy:
                acts = self._rng.integers(0, self.action_dim, n)
            else:
                acts = torch.argmax(self.act(st["obs"], st["ctx"], epsilon=c.play_epsilon), dim=-1).cpu().numpy()
            t1 = time.perf_counter()
            flags = [active[i] and pool.due(st["seg_rows"][i] + 1) for i in range(n)] if pool.every > 0 else None
            results = envs.step(acts, active=active, snapshot=flags)
            t2 = time.perf_counter()
            stats["t_model"] += t1 - t0
            stats["t_env"] += t2 - t1
            for i in range(k):
                o2, r, term, trunc, info = results[i]
                st["pending"][i].append((st["obs"][i]["image"], int(acts[i]), float(r), 1.0))
                st["seg_rows"][i] += 1
                st["ep_return"][i] += r
                st["ep_len"][i] += 1
                self.total_num_steps += 1
                steps += 1
                if "snapshot" in info:
                    pool.add(i, st["seg_id"][i], st["seg_rows"][i], info.pop("snapshot"))
                life_end = bool(info.get("life_loss")) and c.life_loss_terminal
                stats["life_losses"] += int(bool(info.get("life_loss")))
                if term or life_end:                                       # terminal frame: its own cont = 0 row
                    st["pending"][i].append((o2["image"], 0, 0.0, 0.0))
                    st["seg_rows"][i] += 1
                    self._flush(i, end=True, died=True)
                elif trunc:                                                # the cap: the last frame, no terminal
                    st["pending"][i].append((o2["image"], 0, 0.0, 1.0))
                    st["seg_rows"][i] += 1
                    self._flush(i, end=True, died=False)
                if term or trunc or life_end:
                    self.reset_act_rows(st["ctx"], i)
                    st["seg_id"][i] = self.buffer.begin_episode(i)
                    st["seg_rows"][i] = 0
                if term or trunc:
                    stats["returns"].append(st["ep_return"][i])
                    stats["lengths"].append(st["ep_len"][i])
                    self.total_num_episodes += 1
                    if self.verbose:
                        print(f"\033[90m    [collect] episode {self.total_num_episodes} done ({'game over' if term else 'cap'})"
                              f" | return {st['ep_return'][i]:.1f} | {st['ep_len'][i]} steps | "
                              f"{self.total_num_steps:,} env steps\033[0m")
                    st["ep_return"][i], st["ep_len"][i] = 0.0, 0
                    st["obs"][i] = envs.reset_one(i)[0]
                else:
                    st["obs"][i] = o2
        tb = time.perf_counter()
        for i in range(n):                                                 # the open segments' rows so far
            self._flush(i, end=False)
        stats["t_buffer"] += time.perf_counter() - tb
        stats["steps"] += steps
        stats["t_total"] += time.perf_counter() - t_start
        return steps

    def close_collector(self):
        """Writes the pending rows and closes every open segment (end of training)."""
        if self._collector is not None:
            for i in range(len(self.envs)):
                self._flush(i, end=False)
            self.buffer.close_all()
            self._collector = None

    # --- Evaluation -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, envs, num_episodes, greedy=False, recorder=None, progress=True):
        """Plays num_episodes full games (each env plays a fixed share, so short episodes are not over-represented)
        with the current policy; nothing is stored. Acting memory is cleared at life losses as in training.
        recorder(i, info) gets every step's info (videos). -> list of (return, length)."""
        n = len(envs)
        quota = [num_episodes // n + (1 if i < num_episodes % n else 0) for i in range(n)]
        done = [0] * n
        ret, length = [0.0] * n, [0] * n
        results_out = []
        ctx = self.new_act_context(n, key="eval")
        obs = []
        for i, (o, info) in enumerate(envs.reset()):
            obs.append(o)
            if recorder is not None:
                recorder(i, info)
        t0 = time.perf_counter()
        while any(done[i] < quota[i] for i in range(n)):
            active = [done[i] < quota[i] for i in range(n)]
            acts = torch.argmax(self.act(obs, ctx, greedy=greedy), dim=-1).cpu().numpy()
            results = envs.step(acts, active=active)
            for i in range(n):
                if not active[i]:
                    continue
                o2, r, term, trunc, info = results[i]
                ret[i] += r
                length[i] += 1
                if recorder is not None:
                    recorder(i, info)
                if term or trunc or (info.get("life_loss") and self.cfg.life_loss_terminal):
                    self.reset_act_rows(ctx, i)
                if term or trunc:
                    results_out.append((ret[i], length[i]))
                    done[i] += 1
                    if progress and self.verbose:
                        print(f"\033[90m    [eval] episode {len(results_out)}/{num_episodes}: return {ret[i]:.1f} "
                              f"({length[i]} steps, {time.perf_counter() - t0:.0f} s)\033[0m")
                    ret[i], length[i] = 0.0, 0
                    if done[i] < quota[i]:
                        o2 = envs.reset_one(i)[0]
                obs[i] = o2
        return results_out

    # --- Dream audit (diagnostics) ----------------------------------------------------------------------------------
    def audit(self, seed=None):
        """Dream audit: snapshots taken while collecting are played audit_horizon steps in the dream and in the real
        game (the audit's own emulators) with the actor's greedy action -> metrics ({} without usable snapshots)."""
        c, t_start = self.cfg, time.perf_counter()
        if c.audit_horizon <= 0:
            return {}
        if seed is None:
            seed = int(np.random.SeedSequence([c.seed, 0xA4D17, self.total_num_updates]).generate_state(1)[0])
        items = self.audit_snapshots.usable(self.buffer, c.context_length)
        picks = select_starts(items, c.audit_starts, np.random.default_rng(seed))
        self.audit_snapshots.clear()
        if not picks:
            return {}
        starts = [items[i] for i, _ in picks]
        kinds = [AUDIT_KINDS.index(k) for _, k in picks]
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        with torch.no_grad(), eager_modules():
            ctx = self.audit_context([s["row"] for s in starts], [s["step"] for s in starts], gen)
            dream = self.audit_dream(ctx, c.audit_horizon, gen)
            real = self.audit_real(ctx, [s["snapshot"] for s in starts], c.audit_horizon, gen)
            dream.update(self._audit_compare(dream, real, ctx))
        metrics = audit_metrics(dream, real, kinds, c.continue_discount)
        metrics["seconds"] = time.perf_counter() - t_start
        return metrics

    def _audit_sample(self, logp, gen):
        u = torch.rand(logp.shape, generator=gen, device=logp.device).clamp_(1e-6, 1.0 - 1e-6)
        return sample_onehot(logp.float(), straight_through=False, gumbel=-torch.log(-torch.log(u))).flatten(-2)

    def _audit_encode(self, images, gen, chunk=1024):
        zs = []
        for i in range(0, images.shape[0], chunk):
            with self._amp():
                logp = self.tokenizer.posterior({"image": images[i:i + chunk]})
            zs.append(self._audit_sample(logp, gen))
        return torch.cat(zs)

    def _audit_action(self, s):
        return self.actor(s, greedy=True)[0].float()

    @torch.no_grad()
    def audit_context(self, rows, steps, gen):
        """Real context of N starts (buffer rows, segment steps) -> {"kv", "t0", "start" [N, S], "img0"}: the
        context_length frames before each start encoded and run through the TSSM, and the start frame."""
        c, buf, dev = self.cfg, self.buffer, self.device
        W = c.context_length
        rows = torch.as_tensor(rows, dtype=torch.long)
        N = rows.shape[0]
        n_ctx = torch.as_tensor(steps, dtype=torch.long).clamp(min=1, max=W)
        j = torch.arange(W + 1)[None]
        src = buf._wrap(rows[:, None], -n_ctx[:, None] + torch.minimum(j, n_ctx[:, None]))   # pads: the start
        zf = self._audit_encode(buf.images[src.flatten()].to(dev), gen).view(N, W + 1, self.latent_dim)
        actions = F.one_hot(buf.actions[src].long(), self.action_dim).float().to(dev)
        h, kv = self.context_kv(zf, actions)
        idx = torch.arange(N, device=dev)
        t0 = (n_ctx - 1).to(dev)
        start = torch.nan_to_num(self.full_state(h[idx, t0], zf[idx, t0 + 1]), nan=0.0, posinf=1e4, neginf=-1e4)
        return {"kv": kv, "t0": t0, "start": start, "img0": buf.images[rows].to(dev)}

    def _audit_heads(self, states):
        """Rewards [N, H, 2] (reward, death penalty) and calibrated p(continue) [N, H] of states[:, 1:]."""
        c, nxt = self.cfg, states[:, 1:]
        reward = self.two_hot.decode(self.reward_head(nxt)).squeeze(-1)
        alive = self.alive_prob(nxt)
        return torch.stack([reward, c.death_penalty * (1.0 - alive)], dim=-1).float(), alive

    @torch.no_grad()
    def audit_dream(self, ctx, horizon, gen):
        c, dev, N = self.cfg, self.device, ctx["start"].shape[0]
        cache = self.tssm.make_cache(N, dev, self.cache_dtype)
        self.tssm.prefill(cache, ctx["kv"], torch.arange(N, device=dev), ctx["t0"])
        curr = ctx["start"]
        z = curr[:, self.tssm_dim:]
        states = [curr]
        with self._amp():
            for _ in range(horizon):
                h = self.tssm.forward_step(z, self._audit_action(curr), cache).float()
                z = self._audit_sample(unimix_logprobs(self.prior(h), c.dream_unimix), gen)
                curr = torch.nan_to_num(self.full_state(h, z), nan=0.0, posinf=1e4, neginf=-1e4)
                states.append(curr)
            states = torch.stack(states, dim=1)
            r, alive = self._audit_heads(states)
            zs = states[..., self.tssm_dim:]
            S = c.img_size
            frames = torch.cat([torch.sigmoid(self.tokenizer.decode(zz).float())
                                for zz in zs.flatten(0, 1).split(512)]).view(N, horizon + 1, 3, S, S)
        return {"r": r, "alive": alive, "frames": frames}

    def _audit_envs(self):
        """The audit's own emulators (created at the first audit)."""
        if self._audit_env is None:
            from .env import create_envs
            c = self.cfg
            self._audit_env = create_envs(c.game, c.audit_envs, seed=c.seed, seed_offset=90_000, img_size=c.img_size,
                                          frameskip=c.frameskip, noop_max=c.noop_max, sticky_actions=c.sticky_actions,
                                          max_episode_frames=c.max_episode_frames, verbose=False)
        return self._audit_env

    @torch.no_grad()
    def audit_real(self, ctx, snapshots, horizon, gen):
        """The real rollouts: each audit env restores one start's snapshot and the batch is played in lockstep with the
        greedy action of the acting loop until the segment ends -> {"r" [N, H, 2], "steps" [N], "died" [N], "img"
        [N, H, 3, S, S]}."""
        c, dev, N, H = self.cfg, self.device, len(snapshots), int(horizon)
        envs = self._audit_envs()
        E = len(envs)
        r = np.zeros((N, H, 2), dtype=np.float32)
        steps, died = np.full(N, H, dtype=np.int64), np.zeros(N, dtype=bool)
        imgs = np.zeros((N, H, 3, c.img_size, c.img_size), dtype=np.uint8)
        for b0 in range(0, N, E):
            sel = list(range(b0, min(b0 + E, N)))
            n = len(sel)
            envs.restore([snapshots[i] for i in sel] + [None] * (E - n))
            idx_t = torch.as_tensor(sel, device=dev)
            cache = self.tssm.make_cache(n, dev, self.cache_dtype)
            self.tssm.prefill(cache, [(k[idx_t], v[idx_t]) for k, v in ctx["kv"]], torch.arange(n, device=dev),
                              ctx["t0"][idx_t])
            s = ctx["start"][idx_t]
            alive, last = [True] * n, [None] * n
            for k in range(H):
                with self._amp():
                    action = self._audit_action(s)
                    h = self.tssm.forward_step(s[:, self.tssm_dim:], action, cache).float()
                act = action.argmax(-1).cpu().numpy()
                results = envs.step(np.concatenate([act, np.zeros(E - n, dtype=act.dtype)]),
                                    active=alive + [False] * (E - n))
                for j, i in enumerate(sel):
                    if not alive[j]:
                        continue
                    o, rew, term, trunc, info = results[j]
                    end = term or (info.get("life_loss") and c.life_loss_terminal)
                    rew_t = float(self.transform_reward(torch.tensor(rew)))
                    r[i, k] = (rew_t, c.death_penalty if end else 0.0)
                    last[j] = o["image"]
                    if end or trunc:
                        alive[j], steps[i], died[i] = False, k + 1, bool(end)
                for j, i in enumerate(sel):
                    imgs[i, k] = last[j]
                if not any(alive):
                    for kk in range(k + 1, H):
                        for j, i in enumerate(sel):
                            imgs[i, kk] = last[j]
                    break
                s = self.full_state(h, self._audit_encode(torch.from_numpy(np.stack(last)).to(dev), gen))
        return {"r": torch.as_tensor(r), "steps": steps, "died": died, "img": torch.from_numpy(imgs)}

    @torch.no_grad()
    def _audit_compare(self, dream, real, ctx):
        """Dream vs real frames -> {"frame_err" [N, H + 1]} (RMSE; step 0 = the start frame's reconstruction)."""
        frames = dream["frames"]
        N, H1 = frames.shape[:2]
        imgs = torch.cat([ctx["img0"][:, None].cpu(), real["img"]], dim=1)          # [N, H + 1, 3, S, S]
        rgb = image_target(imgs.flatten(0, 1)).to(frames.device)
        return {"frame_err": rmse(frames.flatten(0, 1), rgb).view(N, H1)}


class GraphedActStep:
    """agent.act for n envs (epsilon 0, sampled actions) as one CUDA graph: encoder posterior -> z sample -> actor ->
    action sample -> TSSM step on a static cache. The Gumbel noise of both samples is drawn outside the graph, so the
    graph itself is deterministic and is checked against the eager computation at capture."""

    def __init__(self, agent, n, warmup=3):
        self.agent, self.n = agent, n
        dev = self.device = agent.device
        S = agent.cfg.img_size
        self.cache = agent.tssm.make_cache(n, dev, agent.cache_dtype)
        self.h = torch.zeros(n, agent.tssm_dim, device=dev)
        self.obs = {"image": torch.zeros((n, 3, S, S), dtype=torch.uint8, device=dev)}
        self.staging = {"image": torch.zeros((n, 3, S, S), dtype=torch.uint8, pin_memory=True)}
        self.gz = torch.zeros(n, agent.n_latents, agent.classes, device=dev)
        self.ga = torch.zeros(n, agent.action_dim, device=dev)
        self.graph = None
        self._capture(warmup)

    @torch.no_grad()
    def _body(self):
        ag = self.agent
        with eager_modules(), torch.autocast(self.device.type, dtype=ag.amp_dtype or torch.bfloat16,
                                             enabled=ag.amp_dtype is not None, cache_enabled=False):
            logp = ag.tokenizer.posterior(self.obs)
            z = torch.zeros_like(logp).scatter_(-1, (logp + self.gz).argmax(-1, keepdim=True), 1.0).flatten(1)
            probs = ag.actor.probs(ag.full_state(self.h, z), unimix=ag.cfg.play_unimix)
            a = torch.zeros_like(probs).scatter_(-1, (probs.clamp_min(1e-12).log() + self.ga).argmax(-1, keepdim=True),
                                                 1.0)
            self.h.copy_(ag.tssm.forward_step(z, a, self.cache).float())
        return a

    def _noise(self):
        self.gz.copy_(gumbel_like(self.gz))
        self.ga.copy_(gumbel_like(self.ga))

    def _randomize(self):
        self.obs["image"].random_(0, 256)
        self._noise()

    def _capture(self, warmup):
        snap, h0 = self.cache.snapshot(), self.h.clone()

        def restore():
            self.cache.restore(snap)
            self.h.copy_(h0)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._randomize()
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.a_out = self._body()
        restore()
        self._randomize()
        graph.replay()
        a_g, h_g = self.a_out.clone(), self.h.clone()
        restore()
        a_e = self._body()
        err = (self.h - h_g).abs().max().item()
        restore()
        tol = 5e-2 if self.agent.amp_dtype is not None else 1e-3
        if not (bool((a_e == a_g).all()) and err <= tol):
            raise RuntimeError(f"graph replay differs from the eager step (h error {err:.3g}, tolerance {tol})")
        self.graph = graph

    def reset(self):
        self.cache.reset_rows(slice(None))
        self.h.zero_()

    def __call__(self, observations):
        """Env observations -> one-hot actions [n, A] (a static tensor: read it before the next call)."""
        self.staging["image"].numpy()[:] = np.stack([o["image"] for o in observations])
        self.obs["image"].copy_(self.staging["image"], non_blocking=True)
        self._noise()
        self.graph.replay()
        return self.a_out
