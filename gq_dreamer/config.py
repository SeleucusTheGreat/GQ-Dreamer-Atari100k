"""GQ-Dreamer hyperparameters for the Atari 100k benchmark: one dataclass whose defaults are the main configuration
("gq"), plus named presets."""
from dataclasses import asdict, dataclass, fields, replace

from .atari import BUDGET_STEPS, FINAL_EVAL_EPISODES, FRAMESKIP, IMG_SIZE, MAX_EPISODE_FRAMES, NOOP_MAX, STICKY_ACTIONS


@dataclass(frozen=True)
class GQDreamerConfig:
    """All model, loss, replay, behaviour, protocol and run-control settings ("D3": the DreamerV3 value)."""
    config_name: str = "gq"                      # tags the run directory: runs/<Game>/<config_name>_s<seed>
    game: str = "Pong"
    seed: int = 0
    action_dim: int = 18                         # set from the env (minimal action set: 3..18)

    # --- Atari 100k protocol (atari.py) -----------------------------------------------------------------------------
    env_steps: int = BUDGET_STEPS                # interaction budget in agent steps (all training envs together)
    frameskip: int = FRAMESKIP                   # action repeat, max-pool of the last two frames
    img_size: int = IMG_SIZE                     # RGB observations img_size x img_size
    noop_max: int = NOOP_MAX
    sticky_actions: float = STICKY_ACTIONS
    max_episode_frames: int = MAX_EPISODE_FRAMES
    life_loss_terminal: bool = True              # a life loss closes the replay segment with cont = 0 (the game goes on)
    reward_transform: str = "none"               # "none" (symlog heads, D3) or "sign" (clipped, as IRIS / STORM)
    num_envs: int = 1                            # training envs (D3: 1); each env step counts towards env_steps

    # --- Tokenizer: fusion encoder -> z = n_latents one-hot categoricals x classes ----------------------------------
    n_latents: int = 32                          # 32 latent queries -> 32 categoricals (D3: 32 x 32)
    classes: int = 32
    enc_stem: str = "linear"                     # image tokens: "linear" patches or "cnn" (small CNN, same grid)
    enc_patch: int = 8                           # 64x64 -> 8x8 = 64 image tokens
    enc_dim: int = 256
    enc_depth: int = 6
    enc_heads: int = 8
    enc_pos_inject: bool = True                  # 2D sin-cos position of the image tokens in every layer
    enc_swiglu: bool = True
    enc_ffn: int = 682                           # SwiGLU hidden: 8/3 x enc_dim
    enc_grad_checkpoint: bool = True
    enc_last_queries: bool = True                # last encoder layer computes only the latent queries (same result)
    enc_logit_norm: str = "batch"                # posterior logits batch-normalised with a fixed scale (BN-VAE,
    enc_logit_scale: float = 4.0                 # model.BatchLogitNorm) against posterior collapse; "none" = off
    unimix: float = 0.01
    decoder: str = "conv"                        # pixel decoder on z only: "conv" or "transformer" (patch queries)
    dec_dim: int = 256
    dec_depth: int = 4
    dec_heads: int = 8
    dec_patch: int = 8
    dec_chans: tuple = (256, 128, 96, 64, 32)    # conv decoder: 4x4 -> 64x64 in 4 upsampling stages

    # --- Reconstruction branch --------------------------------------------------------------------------------------
    recon_frames: int = 0                        # frames decoded per update (0 = every valid frame t >= 1)
    mask_frames: int = 0                         # frames re-encoded with random patch masking and decoded (0 = off)
    mask_weight: float = 0.5                     # their share of the pixel loss
    mask_max: float = 0.75
    motion_weight: float = 1.0                   # pixel-loss weight of pixels that changed since the previous frame
    motion_threshold: float = 0.05
    w_recon: float = 1.0                         # pixel BCE summed over the frame

    # --- Dynamics: TSSM + prior head --------------------------------------------------------------------------------
    tssm_dim: int = 512
    tssm_layers: int = 6
    tssm_heads: int = 4
    tssm_kv_heads: int = 4
    tssm_ffn: int = 1365
    context_length: int = 32                     # attention window (steps); < seq_len
    attn_gate: str = "elementwise"               # gated attention: "none", "headwise", "elementwise"
    prior_layers: int = 4

    # --- World-model losses -----------------------------------------------------------------------------------------
    w_dyn: float = 1.0                           # balanced KL with free nats, dyn : rep = 10 : 1
    w_rep: float = 0.1
    free_nats: float = 1.0
    rollout_steps: int = 0                       # open-loop rollout loss (0 = off)
    rollout_splits: int = 1
    rollout_decay: float = 0.9
    w_rollout: float = 1.0
    w_usage: float = 0.0                         # class-usage regulariser (0 = off, only logged)
    w_reward: float = 1.0
    w_continue: float = 1.0
    continue_zero_weight: float = 10.0           # terminal rows up-weighted in the continue loss
    continue_calibrated: bool = True             # p(continue) read with log(continue_zero_weight) added to the logit
    w_dreamlike: float = 0.5                     # reward / continue heads: share on one-step prior samples
    harmony: bool = False                        # learned loss weights (HarmonyDream)
    harmony_lr: float = 1e-3

    # --- Optimisation -----------------------------------------------------------------------------------------------
    wm_lr: float = 1e-4                          # D3
    actor_lr: float = 3e-5                       # D3
    critic_lr: float = 3e-5                      # D3
    weight_decay: float = 0.01
    clip_percentile: float = 99.0
    clip_window: int = 2000
    clip_warmup: int = 500
    clip_initial: float = 1000.0
    mixed_precision: str = "auto"                # bf16 on RTX 30xx+, fp16 + loss scaling on older GPUs, off on CPU

    # --- Batches, replay and training ratio -------------------------------------------------------------------------
    batch_size: int = 128                        # D3 / STORM
    seq_len: int = 64
    train_ratio: int = 1024                      # replayed steps per env step (D3 Atari 100k): with 16 x 64 = 1 update
                                                 # per env step
    prefill_steps: int = 1024                    # random-policy steps before the first update (part of the budget)
    train_every: int = 16                        # env steps collected between bursts of updates
    buffer_size: int = 120_000                   # rows (> the budget: nothing is overwritten in a 100k run)
    death_sample_prob: float = 0.15              # windows that contain a terminal row (life loss / game over)
    recent_sample_prob: float = 0.3
    recent_steps: int = 5_000
    reward_sample_prob: float = 0.025            # windows that contain a non-zero reward
    loss_sample_prob: float = 0.1                # windows centred on surprising frames
    loss_priority_alpha: float = 1.0

    # --- Imagination and behaviour ----------------------------------------------------------------------------------
    mlp_dim: int = 1024
    dream_horizon: int = 20                      # D3
    dreams_per_sequence: int = 64                # 16 x 64 = 1,024 dreams per update (DreamerV3, STORM)
    dream_unimix: float = 0.0
    dream_cuda_graph: bool = True
    death_dream_prob: float = 0.3                # P(a dream starts 2-20 steps before a terminal)
    death_dream_window: tuple = (2, 20)
    dream_loss_frac: float = 0.05                # dreams starting at surprising states
    branch_groups: int = 4                       # start states that try every first action (GRPO-style)
    death_penalty: float = 0.0                   # reward of a terminal (life loss / game over); 0 = off
    continue_discount: float = 0.997             # D3
    lambda_: float = 0.95
    entropy_scale: float = 0.01                  # D3 (with adv_norm "percentile")
    adv_norm: str = "std"                        # "percentile" (D3: A / max(1, EMA(P95 - P5))) or "std" (centred, unit std)
    return_scale_decay: float = 0.99
    value_target: str = "hl_gauss"               # critic / reward-head targets: "hl_gauss" or "twohot" (D3)
    hl_sigma_ratio: float = 0.75
    critic_ema_tau: float = 0.98                 # slow critic: lerp 1 - tau per update (D3)
    w_repval: float = 0.3                        # replay value loss (D3)
    play_epsilon: float = 0.0
    play_unimix: bool = False                    # 1% uniform mix of the actor while playing (training keeps it)

    # --- Speed (same math either way) -------------------------------------------------------------------------------
    compile: bool = True                         # torch.compile (CUDA + triton only)
    act_cuda_graph: bool = True                  # acting step (encoder + actor + TSSM step) as one CUDA graph
    dec_sac: bool = True                         # decoder checkpoint keeps conv outputs
    enc_fast_attn: bool = True
    skip_pad_frames: bool = True                 # posteriors of real frames only (pads copy their row's last frame)
    sync_free: bool = True                       # fused optimizers, GPU-side clipping, no host syncs per update
    h2d_stream: bool = True                      # the prefetcher copies batches to the GPU on a side stream
    async_save: bool = True                      # checkpoint + replay buffer written by a background thread
    async_dream_pdf: bool = True                 # dream PDFs rendered by a worker process

    # --- Diagnostics ------------------------------------------------------------------------------------------------
    drift_interval: int = 1000
    drift_frames: int = 256
    grad_share_interval: int = 0                 # updates between per-term gradient norms (0 = off)
    audit: bool = False                          # dream audit against the real game (diagnostic, not in the budget)
    audit_every: int = 10                        # rounds between audits
    audit_starts: int = 32
    audit_horizon: int = 15
    audit_snapshot_every: int = 25               # agent steps between env snapshots while collecting
    audit_envs: int = 8                          # the audit's own emulators

    # --- Run control and outputs ------------------------------------------------------------------------------------
    round_steps: int = 2_000                     # env steps per round (a CSV row, console summary, plots)
    checkpoint_every: int = 2                    # rounds
    max_checkpoints: int = 2
    save_buffer: bool = True                     # the replay buffer next to the checkpoint (resumable runs)
    eval_every: int = 0                          # env steps between evaluations during training (0 = off)
    eval_episodes: int = 10                      # episodes of those evaluations
    final_eval_episodes: int = FINAL_EVAL_EPISODES
    eval_envs: int = 16                          # parallel evaluation envs
    eval_parallel: bool = False                  # evaluation envs in their own processes
    eval_greedy: bool = False                    # evaluation samples from the policy (D3 / IRIS)
    num_best_dreams: int = 3
    num_random_dreams: int = 5
    csv_extra_columns: bool = False
    reward_ma_window: int = 10


GQ_CONFIG = GQDreamerConfig()

MODEL_CONFIGS = {
    "gq": GQ_CONFIG,
    # 4x fewer updates (train ratio 256): about a quarter of the compute, for sweeps; not the benchmark setting
    "gq_fast": replace(GQ_CONFIG, config_name="gq_fast", train_ratio=256),
    # a small model (encoder 192 x 4, TSSM 512 x 3): about half the cost per update
    "gq_small": replace(GQ_CONFIG, config_name="gq_small", enc_dim=192, enc_depth=4, enc_heads=6, enc_ffn=512,
                        tssm_layers=3, mlp_dim=768, dec_chans=(192, 96, 64, 48, 32)),
    # tiny CPU smoke test: every code path in minutes, not a benchmark setting
    "gq_debug": replace(GQ_CONFIG, config_name="gq_debug", enc_dim=64, enc_depth=2, enc_heads=4, enc_ffn=128,
                        n_latents=8, classes=8, tssm_dim=64, tssm_layers=2, tssm_heads=2, tssm_kv_heads=2,
                        tssm_ffn=128, context_length=8, prior_layers=2, mlp_dim=64, dec_chans=(64, 32, 32, 16, 16),
                        dec_dim=64, dec_depth=1, dec_heads=4, batch_size=4, seq_len=16, dreams_per_sequence=4,
                        dream_horizon=5, prefill_steps=64, train_every=8, train_ratio=64, round_steps=64,
                        buffer_size=5_000, final_eval_episodes=2, eval_envs=2, drift_interval=10, compile=False,
                        audit_starts=4, audit_horizon=5, audit_envs=2, audit_every=1, audit_snapshot_every=5),
}


def model_config(name="gq", **overrides):
    """Named configuration with optional field overrides."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model configuration {name!r}; available: {sorted(MODEL_CONFIGS)}")
    known = {f.name for f in fields(GQDreamerConfig)}
    unknown = sorted(k for k in overrides if k not in known)
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
    return replace(MODEL_CONFIGS[name], **overrides)


def config_from_dict(d, base=GQ_CONFIG):
    """Config from a dict (e.g. a checkpoint's); unknown keys are ignored."""
    names = {f.name for f in fields(GQDreamerConfig)}
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names}
    return replace(base, **d)


def config_to_dict(cfg):
    """Plain dict of a config (stored in checkpoints and results)."""
    return asdict(cfg)
