"""Atari 100k training / evaluation of GQ-Dreamer: rounds of streaming collection and updates at a fixed train ratio
until the interaction budget is spent, CSV metrics, plots, dream PDFs, checkpoints, the final evaluation, and the
notebook / command-line entry points run() and run_suite()."""
import csv
import gc
import glob
import json
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import torch

from . import baselines
from .gq_dreamer import GQDreamer
from .atari import canonical_game, resolve_games
from .audit import AUDIT_COLUMNS, audit_enabled, audit_lines
from .config import config_to_dict, model_config
from .env import create_envs
from .plots import plot_game_runs, plot_training
from .visualize import save_dreams_pdf

_B, _DIM, _CYA, _GRN = "\033[1m", "\033[90m", "\033[96m", "\033[92m"
_MAG, _YEL, _RST = "\033[95m", "\033[93m", "\033[0m"


def seed_everything(seed):
    """Seeds python, numpy and torch (cuDNN autotuning stays on)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _dreams_to_cpu(dreams):
    return [(float(d[0]),) + tuple(x.cpu() if torch.is_tensor(x) else x for x in d[1:-1]) + (d[-1],) for d in dreams]


def _hns(game, score):
    try:
        return baselines.human_normalized(game, score)
    except KeyError:
        return float("nan")


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


class Trainer:
    """Runs GQ-Dreamer on one Atari game for one seed: training within the 100k budget, then the final evaluation.
    Outputs go to runs/<Game>/<config>_s<seed>/ (config.json, metrics.csv, checkpoints/, plots, dreams/,
    final_eval.json)."""

    BASE_HEADERS = ["envSteps", "frames", "gradientSteps", "wallHours", "trainEpisodes", "trainReturn",
                    "trainReturnMax", "trainHNS", "episodeLength", "lifeLosses", "evalReturn", "evalReturnStd",
                    "evalHNS"]
    # CSV column -> (metrics source, metric key)
    COLUMNS = [
        ("worldModelLoss", "wm", "world_model_loss"),            # round mean = the wm* columns below
        ("wmPixel", "wm", "wm_pixel"), ("wmKlDyn", "wm", "wm_kl_dyn"), ("wmKlRep", "wm", "wm_kl_rep"),
        ("wmRollout", "wm", "wm_rollout"), ("wmReward", "wm", "wm_reward"), ("wmContinue", "wm", "wm_continue"),
        ("wmUsage", "wm", "wm_usage"),
        ("latentUsage", "wm", "latent_usage"),                   # batch class perplexity / K (1/K = collapsed)
        ("reconLoss", "wm", "recon_excess"),                     # pixel BCE minus its entropy floor (0 = perfect)
        ("reconMotionLoss", "wm", "recon_motion_excess"),        # the same on the pixels that moved
        ("klDyn", "wm", "kl_dyn"),
        ("rewardLoss", "wm", "reward_loss"), ("continueLoss", "wm", "continue_loss"),
        ("pAliveAtDeath", "wm", "p_alive_at_death"),             # p(continue) at real terminals (life losses)
        ("pAliveDeathPred", "wm", "p_alive_death_pred1"),
        ("pAliveDeathDream", "wm", "p_alive_death_dream"),
        ("dreamDeathSurvival", "wm", "dream_death_survival"),
        ("actorLoss", "dream", "actor_loss"), ("criticLoss", "dream", "critic_loss"),
        ("entropies", "dream", "entropies"), ("advantages", "dream", "advantages"),
        ("returnScale", "dream", "return_scale"), ("criticValues", "dream", "critic_values"),
        ("criticEV", "dream", "critic_ev"), ("replayReturns", "dream", "replay_returns"),
        ("secPerUpdate", "time", "per_update"), ("updates", "time", "updates"),
        ("collectSeconds", "time", "collect"), ("trainSeconds", "time", "train"),
    ]
    EXTRA_COLUMNS = [
        ("postEntropy", "wm", "post_entropy"), ("priorEntropy", "wm", "prior_entropy"),
        ("predMatch", "wm", "pred_match"), ("copyMatch", "wm", "copy_match"),
        ("encoderDrift", "wm", "encoder_drift"), ("klRep", "wm", "kl_rep"), ("rolloutLoss", "wm", "rollout_loss"),
        ("rolloutMatchK", "wm", "rollout_match_k"), ("rolloutKlK", "wm", "rollout_kl_k"),
        ("maskRatio", "wm", "mask_ratio"), ("wmGradNorm", "wm", "wm_grad_norm"),
        ("wmClipThreshold", "wm", "wm_clip_threshold"), ("wmClipFraction", "wm", "wm_clip_fraction"),
        ("pAliveAlive", "wm", "p_alive_alive"), ("paddedFraction", "wm", "padded_fraction"),
        ("preDeathStates", "wm", "pre_death_states"),
        ("targetCriticValues", "dream", "target_critic_values"),
        ("advantagePosFraction", "dream", "advantage_pos_fraction"), ("advAbsMean", "dream", "adv_abs_mean"),
        ("returnRange", "dream", "return_range"), ("dreamMeanReward", "dream", "dream_mean_reward"),
        ("dreamMeanAlive", "dream", "dream_mean_alive"), ("dreamDeathFraction", "dream", "dream_death_fraction"),
        ("replayValueLoss", "dream", "replay_value_loss"), ("replayCriticValues", "dream", "replay_critic_values"),
        ("dreamStartPreDeath", "dream", "dream_start_pre_death"), ("dreamDeathPenalty", "dream", "dream_death_penalty"),
        ("branchAgree", "dream", "branch_agree"), ("branchSpread", "dream", "branch_spread"),
        ("actSecondsPerStep", "time", "act_per_step"), ("envSecondsPerStep", "time", "env_per_step"),
        ("batchWaitSeconds", "time", "wait"), ("gpuPeakGB", "mem", "gpu_peak_gb"),
    ]

    def __init__(self, game="Pong", seed=0, config="gq", run_dir=None, runs_root="runs", device=None,
                 visualize_dreams=False, verbose=True, **overrides):
        """config: a name of MODEL_CONFIGS ("gq", "gq_fast", "gq_small", "gq_debug"); overrides: GQDreamerConfig
        fields changed on top of it (e.g. train_ratio=512)."""
        game = canonical_game(game)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.visualize_dreams = visualize_dreams
        cfg = model_config(config, game=game, seed=int(seed), **overrides)
        self.envs = create_envs(game, cfg.num_envs, seed=cfg.seed, img_size=cfg.img_size, frameskip=cfg.frameskip,
                                noop_max=cfg.noop_max, sticky_actions=cfg.sticky_actions,
                                max_episode_frames=cfg.max_episode_frames, verbose=verbose)
        self.cfg = cfg = replace(cfg, action_dim=int(self.envs.action_space.n))
        self.game, self.seed = game, cfg.seed
        self.run_dir = run_dir or os.path.join(runs_root, game, f"{cfg.config_name}_s{cfg.seed}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_filename = os.path.join(self.run_dir, "metrics.csv")
        self.audit_filename = os.path.join(self.run_dir, "dream_audit.csv")
        self.plot_filename = os.path.join(self.run_dir, "training.png")
        self.dreams_dir = os.path.join(self.run_dir, "dreams")
        seed_everything(cfg.seed)
        self.agent = GQDreamer(cfg, device=self.device, envs=self.envs, run_dir=self.run_dir, verbose=verbose)
        self._save_thread = None
        self._pdf_pool = None
        self._pdf_async_ok = True
        self._eval_envs = None
        self._t_start = None
        self._wall_before = 0.0                              # wall hours of earlier sessions (resumed runs)
        self._last_eval = {}
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({**config_to_dict(cfg), "arch_version": GQDreamer.ARCH_VERSION,
                       "action_names": list(self.envs.action_names)}, f, indent=1)

    @property
    def columns(self):
        return self.COLUMNS + (self.EXTRA_COLUMNS if self.cfg.csv_extra_columns else [])

    @property
    def headers(self):
        return self.BASE_HEADERS + [c[0] for c in self.columns]

    # --- Training ---------------------------------------------------------------------------------------------------
    def train(self, final_eval=True, reset_actor_critic=False):
        """Trains until cfg.env_steps env steps (resuming the newest checkpoint of the run directory), saves a final
        checkpoint and, with final_eval, runs the final evaluation -> its result dict (or None)."""
        agent, c = self.agent, self.cfg
        agent.load_checkpoint(reset_actor_critic=reset_actor_critic)
        self._prepare_csv()
        self._t_start = time.perf_counter()
        budget = int(c.env_steps)
        if agent.total_num_steps >= budget:
            print(f"{_DIM}    [setup] this run already used its {budget:,} env steps{_RST}")
        else:
            print(f"{_B}{_CYA}GQ-Dreamer · {self.game} · seed {self.seed} · {budget:,} env steps "
                  f"({budget * c.frameskip:,} frames) · train ratio {c.train_ratio}{_RST}")
            try:
                if agent.total_num_steps < c.prefill_steps:
                    n = c.prefill_steps - agent.total_num_steps
                    print(f"{_DIM}    [prefill] {n:,} steps with uniformly random actions...{_RST}")
                    agent.collect(n, random_policy=True)
                rnd = agent.total_num_steps // c.round_steps
                while agent.total_num_steps < budget:
                    end = min(budget, (agent.total_num_steps // c.round_steps + 1) * c.round_steps)
                    self._round(rnd, end)
                    rnd += 1
            finally:
                agent.stop_prefetch()
                agent.flush_priorities()
                self._wait_for_save()
                if self._pdf_pool is not None:
                    self._pdf_pool.shutdown(wait=True)
                    self._pdf_pool = None
            agent.close_collector()
            self._save_checkpoint(background=False)
            self._update_plots()
        if final_eval:
            return self.final_evaluation()
        return None

    def _updates_due(self, steps):
        """Updates owed for `steps` new env steps at the train ratio (fractions carried over)."""
        c, agent = self.cfg, self.agent
        agent.updates_owed += steps * c.train_ratio / (c.batch_size * c.seq_len)
        k = int(agent.updates_owed)
        agent.updates_owed -= k
        return k

    def _round(self, rnd, end):
        """Collects and trains until env step `end`, then logs a CSV row, prints a summary, plots, checkpoints and
        (when due) evaluates and audits."""
        agent, c = self.agent, self.cfg
        self._print_round_header(rnd, end)
        self._free_gpu_cache()
        agent.pop_collect_stats()
        wm_metrics, dream_metrics = {}, {}
        best_dreams, random_dreams = [], []
        n_updates, t_wait, t_train = 0, 0.0, 0.0
        t_round = time.perf_counter()
        while agent.total_num_steps < end:
            n = agent.collect(min(c.train_every, end - agent.total_num_steps))
            k = self._updates_due(n)
            if k and agent._prefetcher is None:
                agent.start_prefetch()
            last_chunk = agent.total_num_steps >= end
            t0 = time.perf_counter()
            for u in range(k):
                tw = time.perf_counter()
                batch = agent.sample_batch()
                t_wait += time.perf_counter() - tw
                if batch is None:
                    agent.stop_prefetch()
                    break
                metrics_now = last_chunk and u == k - 1
                m, d, best, rnd_dream = agent.update(batch, compute_metrics=metrics_now,
                                                     pack_dreams=self.visualize_dreams)
                batch = None
                if metrics_now:
                    wm_metrics, dream_metrics = m, d
                if self.visualize_dreams and best is not None:
                    best_dreams.append(best)
                    random_dreams.append(rnd_dream)
                    best_dreams, random_dreams = best_dreams[-200:], random_dreams[-200:]
                n_updates += 1
            t_train += time.perf_counter() - t0
        agent.flush_priorities()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        collect = agent.pop_collect_stats()
        timing = {"train": t_train, "per_update": t_train / max(n_updates, 1), "wait": t_wait, "updates": n_updates,
                  "collect": collect["t_total"], "act_per_step": collect["t_model"] / max(collect["steps"], 1),
                  "env_per_step": collect["t_env"] / max(collect["steps"], 1), "round": time.perf_counter() - t_round,
                  "phases": dict(agent.timer.times)}

        audit = self._audit(rnd) if audit_enabled(c) and (rnd + 1) % int(c.audit_every) == 0 else {}
        if self.visualize_dreams and best_dreams:
            self._save_dreams(best_dreams, random_dreams)
        evaluation = {}
        if c.eval_every > 0 and (agent.total_num_steps // c.eval_every >
                                 (agent.total_num_steps - collect["steps"]) // c.eval_every):
            evaluation = self.evaluate_now(c.eval_episodes, tag=f"eval_S{agent.total_num_steps}")
        play = self._play_stats(collect)
        mem = self._gpu_memory()
        self._print_summary(play, wm_metrics, dream_metrics, timing, mem, evaluation, audit)
        self._append_csv_row(play, evaluation, {"wm": wm_metrics, "dream": dream_metrics, "time": timing, "mem": mem})
        if audit:
            self._append_audit_row(audit)
        self._update_plots()
        if (rnd + 1) % c.checkpoint_every == 0 and agent.total_num_steps < c.env_steps:
            self._save_checkpoint(background=c.async_save)

    def _play_stats(self, collect):
        rets, lens = collect["returns"], collect["lengths"]
        mean = float(np.mean(rets)) if rets else float("nan")
        return {"episodes": len(rets), "return": mean, "return_max": float(max(rets)) if rets else float("nan"),
                "hns": _hns(self.game, mean) if rets else float("nan"),
                "length": float(np.mean(lens)) if lens else float("nan"), "life_losses": collect["life_losses"]}

    def _save_dreams(self, best_dreams, random_dreams):
        best_dreams, random_dreams = _dreams_to_cpu(best_dreams), _dreams_to_cpu(random_dreams)
        best_dreams.sort(key=lambda x: x[0], reverse=True)
        vis = best_dreams[:self.cfg.num_best_dreams] + random.sample(
            random_dreams, min(self.cfg.num_random_dreams, len(random_dreams)))
        try:
            self._check_pdf_worker()
            path = save_dreams_pdf(self.agent, vis, self.dreams_dir, executor=self._dream_pdf_pool(),
                                   action_names=self.envs.action_names)
            print(f"{_DIM}    [dreams] {len(vis)} dreams -> {path}{_RST}")
        except Exception as e:
            print(f"[!] Warning: could not write the dream PDF: {e}")

    def _audit(self, rnd):
        try:
            m = self.agent.audit()
            if not m:
                print(f"{_DIM}    [audit] no usable snapshot: skipped{_RST}")
            return m
        except Exception as e:
            print(f"[!] Warning: dream audit failed in round {rnd + 1}: {type(e).__name__}: {e}")
            return {}

    # --- Evaluation -------------------------------------------------------------------------------------------------
    def _get_eval_envs(self, n=None):
        c = self.cfg
        n = n or c.eval_envs
        if self._eval_envs is None or len(self._eval_envs) != n:
            if self._eval_envs is not None:
                self._eval_envs.close()
            self._eval_envs = create_envs(self.game, n, seed=c.seed, seed_offset=50_000, parallel=c.eval_parallel,
                                          img_size=c.img_size, frameskip=c.frameskip, noop_max=c.noop_max,
                                          sticky_actions=c.sticky_actions, max_episode_frames=c.max_episode_frames,
                                          verbose=self.verbose)
        return self._eval_envs

    def evaluate_now(self, num_episodes, tag="eval", greedy=None, record=False, num_envs=None):
        """Evaluates the current policy on num_episodes full games -> result dict, also written to
        <run_dir>/<tag>.json."""
        c, agent = self.cfg, self.agent
        greedy = c.eval_greedy if greedy is None else greedy
        envs = self._get_eval_envs(min(num_envs or c.eval_envs, max(num_episodes, 1)))
        recorder, writers = None, []
        if record:
            recorder, writers = self._recorder(envs, tag)
        t0 = time.perf_counter()
        try:
            results = agent.evaluate(envs, num_episodes, greedy=greedy, recorder=recorder)
        finally:
            for w in writers:
                w.release()
            if record:
                envs.set_capture(False)
        rets = np.array([r for r, _ in results], dtype=np.float64)
        lens = np.array([n for _, n in results], dtype=np.float64)
        res = {"game": self.game, "seed": self.seed, "config": c.config_name, "tag": tag,
               "env_steps": agent.total_num_steps, "frames": agent.total_num_steps * c.frameskip,
               "gradient_steps": agent.total_num_updates, "episodes": len(rets), "greedy": bool(greedy),
               "mean": float(rets.mean()), "std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
               "sem": float(rets.std(ddof=1) / math.sqrt(len(rets))) if len(rets) > 1 else 0.0,
               "median": float(np.median(rets)), "min": float(rets.min()), "max": float(rets.max()),
               "hns": _hns(self.game, rets.mean()), "mean_length": float(lens.mean()),
               "returns": rets.tolist(), "lengths": lens.astype(int).tolist(),
               "seconds": time.perf_counter() - t0, "arch_version": GQDreamer.ARCH_VERSION,
               "protocol": {k: getattr(c, k) for k in ("env_steps", "frameskip", "img_size", "noop_max",
                                                       "sticky_actions", "max_episode_frames",
                                                       "life_loss_terminal", "reward_transform", "train_ratio")},
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(os.path.join(self.run_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        self._last_eval = res
        return res

    def final_evaluation(self, num_episodes=None, record=False):
        """The benchmark score: cfg.final_eval_episodes games with the trained policy -> final_eval.json (+ printed
        against the published scores of this game)."""
        c = self.cfg
        n = num_episodes or c.final_eval_episodes
        print(f"{_B}{_CYA}[final evaluation] {n} episodes of {self.game} with the policy after "
              f"{self.agent.total_num_steps:,} env steps ({'greedy' if c.eval_greedy else 'sampled'} actions)...{_RST}")
        res = self.evaluate_now(n, tag="final_eval", record=record)
        res["wall_hours"] = self._wall_hours()
        with open(os.path.join(self.run_dir, "final_eval.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        self._print_final(res)
        if self._eval_envs is not None:
            self._eval_envs.close()
            self._eval_envs = None
        return res

    def _print_final(self, res):
        g = self.game
        print(f"\n{_B}{_GRN}  ★ {g}: {res['mean']:.1f} ± {res['sem']:.1f} (s.e.m., {res['episodes']} episodes) · "
              f"HNS {res['hns']:.3f}{_RST}")
        try:
            refs = baselines.reference_lines(g, ("DreamerV3", "DreamerV3 (2024)", "STORM", "DIAMOND", "IRIS", "TWM",
                                                 "SimPLe"))
            print(f"{_DIM}    published: " + " · ".join(f"{k} {v:,.1f}" for k, v in refs.items()) + _RST)
        except KeyError:
            pass

    def _recorder(self, envs, tag, n_videos=2, fps=60, scale=3):
        """Writes the first n_videos eval envs' games as MP4 (every emulator frame) to <run_dir>/videos/."""
        import cv2
        out_dir = os.path.join(self.run_dir, "videos")
        os.makedirs(out_dir, exist_ok=True)
        envs.set_capture(True)
        writers = []
        for i in range(min(n_videos, len(envs))):
            writers.append(cv2.VideoWriter(os.path.join(out_dir, f"{tag}_env{i}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                           fps, (160 * scale, 210 * scale)))

        def record(i, info):
            if i < len(writers):
                for frame in info.get("frames") or ():
                    big = cv2.resize(frame, (160 * scale, 210 * scale), interpolation=cv2.INTER_NEAREST)
                    writers[i].write(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print(f"{_DIM}    [rec] recording {len(writers)} env(s) to {out_dir}/{_RST}")
        return record, writers

    def evaluate_checkpoint(self, checkpoint=None, num_episodes=None, record=False, tag="eval"):
        """Loads the newest (or given) checkpoint's weights and evaluates them -> result dict."""
        if not self.agent.load_checkpoint(checkpoint, weights_only=True):
            print("[!] no checkpoint to evaluate")
            return None
        n = num_episodes or self.cfg.final_eval_episodes
        res = self.evaluate_now(n, tag=tag, record=record)
        self._print_final(res)
        return res

    # --- Checkpoints and workers ------------------------------------------------------------------------------------
    def _save_checkpoint(self, background=False):
        agent, c = self.agent, self.cfg

        def prune():
            ckpts = sorted(glob.glob(os.path.join(agent.CHECKPOINT_DIR, agent.CHECKPOINT_GLOB)),
                           key=os.path.getmtime)
            while len(ckpts) > c.max_checkpoints:
                old = ckpts.pop(0)
                try:
                    os.remove(old)
                except OSError as e:
                    print(f"[!] Warning: could not remove old checkpoint {old}: {e}")
        self._wait_for_save()
        if background:
            self._save_thread = agent.save_checkpoint(save_buffer=c.save_buffer, background=True, after=prune)
        else:
            agent.save_checkpoint(save_buffer=c.save_buffer)
            prune()

    def _wait_for_save(self):
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None

    def _dream_pdf_pool(self):
        if not (self.cfg.async_dream_pdf and self._pdf_async_ok):
            return None
        if self._pdf_pool is None:
            self._pdf_pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        return self._pdf_pool

    def _check_pdf_worker(self):
        fut = save_dreams_pdf.pending
        if fut is not None and fut.done() and fut.exception() is not None:
            print(f"[!] dream PDF worker failed ({fut.exception()!r}); PDFs are drawn in this process from now on")
            self._pdf_pool.shutdown(wait=False)
            self._pdf_pool = None
            self._pdf_async_ok = False
        save_dreams_pdf.pending = None if fut is None or fut.done() else fut

    def close(self):
        """Closes every env."""
        self.agent.stop_prefetch()
        for envs in (self.envs, self._eval_envs, self.agent._audit_env):
            if envs is not None:
                try:
                    envs.close()
                except Exception:
                    pass
        self._eval_envs = None

    # --- GPU memory -------------------------------------------------------------------------------------------------
    def _free_gpu_cache(self):
        if self.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _gpu_memory(self):
        if self.device.type != "cuda":
            return {}
        free, total = torch.cuda.mem_get_info(self.device)
        gb = 1024 ** 3
        return {"gpu_peak_gb": torch.cuda.max_memory_allocated(self.device) / gb,
                "gpu_reserved_gb": torch.cuda.memory_reserved(self.device) / gb,
                "gpu_free_gb": free / gb, "gpu_total_gb": total / gb}

    # --- Metrics CSV and plots --------------------------------------------------------------------------------------
    def _wall_hours(self):
        return self._wall_before + (time.perf_counter() - self._t_start) / 3600.0 if self._t_start else 0.0

    def _prepare_csv(self):
        """Creates the CSV, or archives it for a fresh model / keeps it for a resumed one."""
        agent, path = self.agent, self.csv_filename
        if os.path.exists(path):
            with open(path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and not agent._resumed:
                archived = f"{os.path.splitext(path)[0]}_before_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                os.replace(path, archived)
                print(f"{_DIM}    [setup] new model: previous CSV moved to {archived}{_RST}")
            elif rows:
                kept = [r for r in rows if _int(r.get("envSteps")) <= agent.total_num_steps]   # rounds after the
                try:                                                                           # checkpoint: redone
                    self._wall_before = float((kept or rows)[-1].get("wallHours") or 0.0)
                except ValueError:
                    pass
                if len(kept) != len(rows) or list(rows[0].keys()) != self.headers:
                    rows = kept
                    with open(path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=self.headers, extrasaction="ignore", restval="")
                        w.writeheader()
                        w.writerows(rows)
                return
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.headers)

    def _append_csv_row(self, play, evaluation, metrics):
        agent = self.agent
        ev = evaluation or {}
        row = [agent.total_num_steps, agent.total_num_steps * self.cfg.frameskip, agent.total_num_updates,
               self._wall_hours(), play["episodes"], play["return"], play["return_max"], play["hns"], play["length"],
               play["life_losses"], ev.get("mean", ""), ev.get("std", ""), ev.get("hns", "")]
        row += [metrics[src].get(key, "") for _, src, key in self.columns]
        row = ["" if isinstance(x, float) and not np.isfinite(x) else (round(x, 5) if isinstance(x, float) else x)
               for x in row]
        with open(self.csv_filename, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_audit_row(self, audit):
        path, agent = self.audit_filename, self.agent
        new = not os.path.exists(path)
        row = [agent.total_num_steps, agent.total_num_updates] + [audit.get(key, "") for _, _, key in AUDIT_COLUMNS]
        row = [round(x, 4) if isinstance(x, float) else x for x in row]
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["envSteps", "gradientSteps"] + [c[0] for c in AUDIT_COLUMNS])
            w.writerow(row)

    def _update_plots(self):
        for name, fn in (("training curve", lambda: plot_training(self.csv_filename, self.plot_filename, self.game,
                                                                  self.cfg.reward_ma_window,
                                                                  budget=self.cfg.env_steps)),
                         ("runs of this game", lambda: plot_game_runs(os.path.dirname(self.run_dir),
                                                                      os.path.join(os.path.dirname(self.run_dir),
                                                                                   "runs.png"),
                                                                      self.cfg.reward_ma_window,
                                                                      budget=self.cfg.env_steps))):
            try:
                fn()
            except Exception as e:
                print(f"[!] Warning: could not update the {name} plot: {e}")

    # --- Console output ---------------------------------------------------------------------------------------------
    def _print_round_header(self, rnd, end):
        c, a = self.cfg, self.agent
        n_rounds = math.ceil(c.env_steps / c.round_steps)
        bar = "═" * 72
        print(f"\n{_B}{_CYA}{bar}{_RST}")
        print(f"{_B}{_CYA}  {self.game} · ROUND {rnd + 1} / {n_rounds}{_RST}{_DIM}   env steps "
              f"{a.total_num_steps:,} → {end:,} of {c.env_steps:,} | updates {a.total_num_updates:,}{_RST}")
        print(f"{_B}{_CYA}{bar}{_RST}")

    def _print_summary(self, play, wm, dream, timing, mem, evaluation, audit):
        def f(v, d=3):
            return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"

        def row(label, text):
            print(f"{_DIM}    {label:<12}│ {text}{_RST}")
        print()
        if play["episodes"]:
            print(f"  {_B}{_GRN}★ RETURN   {play['return']:9.1f}{_RST}{_GRN}   ({play['episodes']} episode(s) finished, "
                  f"best {play['return_max']:.1f}, HNS {f(play['hns'], 3)}){_RST}"
                  f"{_DIM}   avg length {play['length']:.0f} steps · {play['life_losses']} life losses{_RST}")
        else:
            print(f"  {_GRN}★ no episode finished this round{_RST}{_DIM}   ({play['life_losses']} life losses)"
                  f"{_RST}")
        if evaluation:
            print(f"  {_B}{_MAG}◆ EVAL     {evaluation['mean']:9.1f}{_RST}{_MAG} ± {evaluation['std']:.1f} over "
                  f"{evaluation['episodes']} episodes · HNS {f(evaluation['hns'])}{_RST}")
        if wm:
            row("world model", f"loss {f(wm.get('world_model_loss'), 1)} · recon excess {f(wm.get('recon_excess'), 4)}"
                               f" (moving pixels {f(wm.get('recon_motion_excess'), 4)}) · KL {f(wm.get('kl_dyn'), 2)} "
                               f"nats · reward {f(wm.get('reward_loss'))} · continue {f(wm.get('continue_loss'))}")
            row("dynamics", f"prior = posterior class {f(wm.get('pred_match'))} vs copy {f(wm.get('copy_match'))} · "
                            f"classes used {f(wm.get('latent_usage'))} · p(cont) at terminals "
                            f"{f(wm.get('p_alive_at_death'))} / elsewhere {f(wm.get('p_alive_alive'))} · after "
                            f"{self.agent.DEATH_PROBE_STEPS} dream steps {f(wm.get('p_alive_death_dream'))}")
        if dream:
            row("behaviour", f"entropy {f(dream.get('entropies'))} · return scale {f(dream.get('return_scale'), 2)} · "
                             f"critic EV {f(dream.get('critic_ev'), 2)} · critic {f(dream.get('critic_values'), 2)} "
                             f"(real λ-return {f(dream.get('replay_returns'), 2)}) · dream reward/step "
                             f"{f(dream.get('dream_mean_reward'), 3)}")
        if timing:
            row("speed", f"{timing['updates']} updates · {timing['per_update']:.3f} s/update · acting "
                         f"{1000 * timing['act_per_step']:.1f} ms/step · env {1000 * timing['env_per_step']:.2f} "
                         f"ms/step · round {timing['round']:.0f} s · {self._wall_hours():.2f} h so far")
        if mem:
            row("gpu", f"peak {mem['gpu_peak_gb']:.1f} GB · free {mem['gpu_free_gb']:.1f} / {mem['gpu_total_gb']:.1f} GB")
        for line in audit_lines(audit or {}):
            print(f"{_DIM}    {line}{_RST}")


# --- Entry points ---------------------------------------------------------------------------------------------------
def run(game="Pong", seed=0, mode="train", config="gq", device=None, runs_root="runs", visualize_dreams=False,
        verbose=True, final_eval=True, record=False, eval_episodes=None, checkpoint=None, reset_actor_critic=False,
        **overrides):
    """Trains (mode="train", then the final evaluation) or evaluates (mode="eval": the newest checkpoint, or
    `checkpoint`) GQ-Dreamer on one Atari game -> the Trainer (its .agent, .run_dir, and final_eval.json in it).
    overrides: GQDreamerConfig fields. Ctrl+C / "interrupt kernel" stops cleanly and still returns the Trainer."""
    if mode not in ("train", "eval"):
        raise ValueError(f"invalid mode {mode!r}: use 'train' or 'eval'")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    trainer = None
    try:
        trainer = Trainer(game, seed, config, runs_root=runs_root, device=device, visualize_dreams=visualize_dreams,
                          verbose=verbose, **overrides)
        if mode == "train":
            trainer.train(final_eval=final_eval, reset_actor_critic=reset_actor_critic)
        else:
            trainer.evaluate_checkpoint(checkpoint, num_episodes=eval_episodes, record=record)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by the user (the newest checkpoint resumes the run).")
    finally:
        if trainer is not None:
            trainer.close()
    return trainer


def run_suite(games=None, seeds=(0,), config="gq", runs_root="runs", skip_done=True, report=True, **kwargs):
    """Trains + evaluates every (game, seed) in turn (default: all 26 games, seed 0), skipping runs that already have
    final_eval.json (skip_done), and writes the comparison report -> benchmark.compare() result (report=True)."""
    from .benchmark import report as write_report
    games = resolve_games(games)
    todo = [(g, s) for g in games for s in seeds]
    print(f"{_B}{_CYA}Atari 100k suite: {len(games)} games x {len(seeds)} seed(s) = {len(todo)} runs ({config}){_RST}")
    for k, (g, s) in enumerate(todo):
        done_file = os.path.join(runs_root, g, f"{config}_s{s}", "final_eval.json")
        if skip_done and os.path.exists(done_file):
            print(f"{_DIM}[suite] {k + 1}/{len(todo)} {g} seed {s}: done ({done_file}){_RST}")
            continue
        print(f"\n{_B}{_YEL}[suite] {k + 1}/{len(todo)}: {g} seed {s}{_RST}")
        trainer = run(g, s, "train", config, runs_root=runs_root, **kwargs)
        interrupted = trainer is None or not os.path.exists(done_file)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if interrupted:
            print("[suite] stopped (the run did not finish); run_suite(...) again resumes it")
            break
    return write_report(runs_root, config) if report else None
