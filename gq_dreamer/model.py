"""Model: every network of GQ-Dreamer in one file (latents, networks on s = [h, z], tokenizer, dynamics).

Sections, in dependency order:
  Latents     one-hot categorical latents: unimix log-probabilities, sampling, KL, entropy, class usage
  Networks    binned (two-hot / HL-Gauss) reward / value heads, continue head, actor, critic
  Tokenizer   GroundQuery fusion-transformer encoder read out by latent queries, pixel decoder (conv or transformer)
  Dynamics    transformer state-space model (TSSM), KV ring cache, prior head, CUDA-graphed dream step
How these networks run and learn is in gq_dreamer.py.
"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from .observations import (autocast_dtype, image_input, image_target, motion_mask, n_patches, patch_dim, patch_grid,
                           patch_pixel_mask, patchify, unpatchify)
from .train_utils import eager_modules, eager_only


# === Latents ========================================================================================================
# One-hot categorical latents (Dreamer): unimix log-probabilities, sampling, KL, entropy and class usage.


def unimix_logprobs(logits, unimix=0.01):
    """Logits [..., K] -> fp32 log-probabilities of (1 - unimix) softmax + unimix / K."""
    logits = logits.float()
    if not unimix:
        return torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    return torch.log(probs)


def gumbel_like(logp):
    """Gumbel(0, 1) noise shaped like logp."""
    u = torch.rand_like(logp, dtype=torch.float32).clamp_(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))


def sample_onehot(logp, straight_through=True, gumbel=None):
    """Log-probabilities [..., L, K] -> one-hot sample (straight-through gradient by default)."""
    g = gumbel_like(logp) if gumbel is None else gumbel
    onehot = F.one_hot((logp.detach().float() + g).argmax(-1), logp.shape[-1]).float()
    if not straight_through:
        return onehot
    probs = logp.float().exp()
    return onehot + probs - probs.detach()


def kl_categorical(logp, logq):
    """KL(p || q) of [..., L, K] log-probabilities, summed over classes and categoricals -> [...] nats."""
    return (logp.exp() * (logp - logq)).sum((-1, -2))


def entropy_categorical(logp):
    """Entropy of [..., L, K] log-probabilities, summed over categoricals -> [...] nats."""
    return -(logp.exp() * logp).sum((-1, -2))


@torch.no_grad()
def class_usage(z, n_latents, classes, mask=None):
    """Batch class perplexity / K, averaged over categoricals (1 = all classes used, 1/K = collapsed)."""
    z = z.float().reshape(-1, n_latents, classes)
    w = torch.ones(z.shape[0], device=z.device) if mask is None else mask.float().reshape(-1)
    p = (z * w[:, None, None]).sum(0) / w.sum().clamp_min(1.0)
    h = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return (h.exp() / classes).mean()


# === Networks =======================================================================================================
# MLP heads on the agent state s = [h, z]: binned (two-hot / HL-Gauss) reward / value heads, the continue head, actor
# and critic.


def mlp(in_dim, out_dim, hidden=1024, layers=3, act=nn.SiLU):
    """MLP of `layers` hidden Linear-LayerNorm-act layers."""
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def symlog(x):
    """sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


VALUE_TARGETS = ("twohot", "hl_gauss")


class TwoHotEncoding:
    """Symlog binned regression over `num_bins` bins: decodes the softmax-weighted mean of the bin values, trained by
    cross-entropy to a "twohot" target (DreamerV3: the two bins around symlog(y), linearly weighted) or an "hl_gauss"
    target (Farebrother et al. 2024: the mass of N(symlog(y), sigma^2) in each bin, sigma = sigma_ratio x bin
    width)."""

    def __init__(self, min=-10, max=10, num_bins=255, device="cuda", target="twohot", sigma_ratio=0.75):
        if target not in VALUE_TARGETS:
            raise ValueError(f"value target {target!r}: one of {VALUE_TARGETS}")
        self.num_bins, self.target = num_bins, target
        self.bins = torch.linspace(min, max, num_bins, device=device)
        self.bin_values = symexp(self.bins)
        self.sigma = float(sigma_ratio) * (max - min) / (num_bins - 1)
        self.edges = 0.5 * (self.bins[1:] + self.bins[:-1])           # inner bin edges [num_bins - 1]

    def encode(self, x):
        """Scalars [...] -> target distributions [..., num_bins] that each sum to 1."""
        if self.target == "hl_gauss":
            return self._hl_gauss(x)
        x = torch.clamp(symlog(x.float()), self.bins[0], self.bins[-1])
        pos = (x - self.bins[0]) / (self.bins[1] - self.bins[0])
        low = torch.clamp(torch.floor(pos).long(), 0, self.num_bins - 2)
        w_high = pos - low
        two_hot = torch.zeros(*x.shape, self.num_bins, device=x.device)
        two_hot.scatter_(-1, low.unsqueeze(-1), (1.0 - w_high).unsqueeze(-1))
        two_hot.scatter_(-1, (low + 1).unsqueeze(-1), w_high.unsqueeze(-1))
        return two_hot

    def _hl_gauss(self, x):
        """HL-Gauss target p_i = Phi((e_{i+1} - y) / sigma) - Phi((e_i - y) / sigma), y = symlog(x), outer edges open
        (e_0 = -inf, e_num_bins = +inf), normalised."""
        y = symlog(x.float())
        cdf = torch.special.ndtr((self.edges - y[..., None]) / self.sigma)            # [..., num_bins - 1]
        zero, one = torch.zeros_like(cdf[..., :1]), torch.ones_like(cdf[..., :1])
        p = (torch.cat([cdf, one], dim=-1) - torch.cat([zero, cdf], dim=-1)).clamp_min(0.0)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-12)

    def decode(self, logits):
        """Logits [..., num_bins] -> expected value [..., 1]."""
        return torch.sum(torch.softmax(logits.float(), dim=-1) * self.bin_values, dim=-1, keepdim=True)

    def loss(self, logits, target, weights=None):
        """Weighted cross-entropy against the target distribution of the scalars `target`."""
        with torch.no_grad():
            th = self.encode(target)
        ce = -torch.sum(th * torch.log_softmax(logits.float(), dim=-1), dim=-1)
        if weights is None:
            return ce.mean()
        return (ce * weights).sum() / weights.sum().clamp(min=1e-6)


class TwoHotHead(nn.Module):
    """Binned regression head (rewards), zero-initialised output."""

    def __init__(self, in_dim, num_bins=255, hidden=1024, zero_bias_bin=None):
        super().__init__()
        self.net = mlp(in_dim, num_bins, hidden=hidden, layers=2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if zero_bias_bin is not None:                                  # start confidently at the zero bin
            with torch.no_grad():
                self.net[-1].bias[zero_bias_bin] = 5.0

    def forward(self, x):
        return self.net(x)


class ContinueHead(nn.Module):
    """Bernoulli logit of 'this state is not a terminal' (cont = 0: a life loss or the game over)."""

    def __init__(self, in_dim, hidden=1024):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden=hidden, layers=2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)                      # p(continue) ~ 0.98 at init

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Actor(nn.Module):
    """Categorical policy over the actions with a 1% uniform mix."""

    def __init__(self, action_dim, in_dim, mlp_dim=1024):
        super().__init__()
        self.action_dim = action_dim
        self.net = mlp(in_dim, action_dim, hidden=mlp_dim, layers=3)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def probs(self, x, unimix=True):
        """Action probabilities [..., A]; with unimix (training) 1% of the mass is spread uniformly."""
        probs = torch.softmax(self.net(x).float(), dim=-1)
        return 0.99 * probs + 0.01 / self.action_dim if unimix else probs

    def forward(self, x, greedy=False, return_probs=False, unimix=True):
        """-> (one-hot action, log-prob, entropy[, probs])."""
        probs = self.probs(x, unimix)
        dist = OneHotCategorical(probs=probs, validate_args=False)     # validation would sync the GPU
        a = F.one_hot(probs.argmax(-1), self.action_dim).float() if greedy else dist.sample()
        if return_probs:
            return a, dist.log_prob(a), dist.entropy(), probs
        return a, dist.log_prob(a), dist.entropy()


class Critic(nn.Module):
    """Two-hot value head."""

    def __init__(self, in_dim, bins=255, mlp_dim=1024):
        super().__init__()
        self.net = mlp(in_dim, bins, hidden=mlp_dim, layers=3)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# === Tokenizer ======================================================================================================
# GroundQuery tokenizer: a fusion transformer over the image tokens read out by learned latent queries (the posterior,
# one categorical per query), and a pixel decoder (conv or transformer) that reads z only.


# --- Building blocks ------------------------------------------------------------------------------------------------
class ConvResBlock(nn.Module):
    """GroupNorm-SiLU-conv x2 residual block, identity at init."""

    def __init__(self, ch, groups=8):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


def _sac_policy(ctx, op, *args, **kwargs):
    """Selective activation checkpointing: keep conv / matmul outputs, recompute everything else (norms,
    activations, adds, the loss) in the backward pass."""
    if op in (torch.ops.aten.convolution.default, torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


SAC_CONTEXT = partial(create_selective_checkpoint_contexts, _sac_policy)


class _DropSacCompileNotice(logging.Filter):
    """Drops torch.compile's notice that context_fn (the decoder's selective checkpointing) is used under compile."""

    def filter(self, record):
        return "context_fn is passed to torch.utils.checkpoint" not in record.getMessage()


_SAC_NOTICE_FILTER = _DropSacCompileNotice()
logging.getLogger("torch._higher_order_ops.wrap").addFilter(_SAC_NOTICE_FILTER)
for _h in logging.getLogger("torch").handlers:
    _h.addFilter(_SAC_NOTICE_FILTER)


class PosInjectLayer(nn.TransformerEncoderLayer):
    """Pre-norm encoder layer (built on nn.TransformerEncoderLayer) whose attention input gets a fixed position term:
    x + SA(LN1(x) + pos), then + FF(LN2(x)). As in DETR, pos is re-added in every layer instead of once at the input,
    after the LayerNorm, so it keeps its size next to the normalised content. It enters queries, keys and values: a
    latent query that finds an object by its content also reads where the object is. pos = None: the plain layer.

    swiglu: the TSSM's feed-forward, down(silu(gate(x)) * up(x)) without biases, hidden dim_feedforward."""

    def __init__(self, d_model, nhead, dim_feedforward, *args, swiglu=False, **kwargs):
        super().__init__(d_model, nhead, dim_feedforward, *args, **kwargs)
        self.swiglu = swiglu
        self.fast_attn = False                    # set by FusionEncoder: SDPA on views instead of nn.MultiheadAttention
        if swiglu:
            del self.linear1, self.linear2
            self.gate_proj = nn.Linear(d_model, dim_feedforward, bias=False)
            self.up_proj = nn.Linear(d_model, dim_feedforward, bias=False)
            self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False)

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        """Self-attention; fast_attn: the same math with scaled_dot_product_attention on views."""
        if not self.fast_attn or attn_mask is not None or key_padding_mask is not None:
            return super()._sa_block(x, attn_mask, key_padding_mask, is_causal)
        a = self.self_attn
        n, t, e = x.shape
        qkv = F.linear(x, a.in_proj_weight, a.in_proj_bias).view(n, t, 3, a.num_heads, e // a.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, t, e)
        return self.dropout1(F.linear(o, a.out_proj.weight, a.out_proj.bias))

    def _ff_block(self, x):
        if not self.swiglu:
            return super()._ff_block(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x, pos=None):
        h = self.norm1(x)
        if pos is not None:
            h = h + pos.to(h.dtype)
        x = x + self._sa_block(h, None, None)
        return x + self._ff_block(self.norm2(x))

    def forward_first(self, x, pos, n):
        """forward(x, pos)[:, :n], computing queries, attention output and feed-forward for the first n tokens only
        (keys / values from every token); the encoder's last layer, whose other outputs are never read."""
        h = self.norm1(x)
        if pos is not None:
            h = h + pos.to(h.dtype)
        a = self.self_attn
        b, t, e = x.shape
        nh = a.num_heads
        w, bias = a.in_proj_weight, a.in_proj_bias
        q = F.linear(h[:, :n], w[:e], None if bias is None else bias[:e]).view(b, n, nh, e // nh).transpose(1, 2)
        kv = F.linear(h, w[e:], None if bias is None else bias[e:]).view(b, t, 2, nh, e // nh)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, e)
        x = x[:, :n] + self.dropout1(F.linear(o, a.out_proj.weight, a.out_proj.bias))
        return x + self._ff_block(self.norm2(x))


def _transformer(dim, depth, heads, layer_cls=nn.TransformerEncoderLayer, ffn=None, **layer_kwargs):
    """Pre-norm transformer encoder, GELU MLP 4 x dim by default (layer_cls: PosInjectLayer for position injection in
    every layer, and swiglu=True for its SwiGLU feed-forward with hidden ffn)."""
    layer = layer_cls(dim, heads, ffn or dim * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
                      **layer_kwargs)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


def sincos_2d(rows, cols, dim, min_wavelength=2.0, max_wavelength=32.0):
    """Fixed 2D sin-cos embedding [n, dim] of token centres (rows[i], cols[i]) in patch units. Half of the channels
    encode the row and half the column, each as dim / 4 (sin, cos) pairs whose wavelengths are spaced geometrically
    from min_wavelength to max_wavelength patches. Every value is in [-1, 1]; the vector's norm is sqrt(dim / 2)."""
    assert dim % 4 == 0, "sincos_2d needs dim divisible by 4"
    n = dim // 4
    wl = min_wavelength * (max_wavelength / min_wavelength) ** (torch.arange(n, dtype=torch.float64) / max(n - 1, 1))
    omega = 2.0 * math.pi / wl

    def axis(p):
        a = torch.as_tensor(p, dtype=torch.float64)[:, None] * omega[None]
        return torch.cat([a.sin(), a.cos()], dim=1)
    return torch.cat([axis(rows), axis(cols)], dim=1).float()


def image_token_centres(rows, cols):
    """(rows, cols) of the centres of a rows x cols token grid in patch units, row-major."""
    rr, cc = torch.meshgrid(torch.arange(rows, dtype=torch.float64), torch.arange(cols, dtype=torch.float64),
                            indexing="ij")
    return (rr + 0.5).flatten(), (cc + 0.5).flatten()


def bce_floor(target):
    """Entropy of the target colours: the lowest BCE a perfect reconstruction reaches (fp32)."""
    t = target.float().clamp(1e-6, 1.0 - 1e-6)
    return -(t * t.log() + (1.0 - t) * (1.0 - t).log())


# --- Encoder (posterior) --------------------------------------------------------------------------------------------
class BatchLogitNorm(nn.Module):
    """Batch normalisation of the posterior logits with a FIXED scale (BN-VAE, Zhu et al. 2020, for categoricals):
    every class logit of every categorical is centred on its mean over the frames of the batch and divided by its
    std over them, then multiplied by `scale`. The part of the logits that is the same for every frame (the latent
    queries' own embedding, the static background) is removed, so the posterior has to differ between frames from
    the first update: the logits of each class vary across frames with std `scale`, and the batch-averaged posterior
    cannot concentrate on one class. A learned scale could shrink to 0 and collapse again, hence fixed (as BN-VAE's
    gamma). Training (grad enabled, >= min_batch frames) uses the batch statistics and updates running ones; acting,
    dreams and probes use the running statistics, so a single frame's posterior does not depend on its batch."""

    def __init__(self, n_latents, classes, scale=4.0, momentum=0.01, eps=1e-5, min_batch=32):
        super().__init__()
        self.scale, self.momentum, self.eps, self.min_batch = float(scale), momentum, eps, min_batch
        self.register_buffer("running_mean", torch.zeros(n_latents, classes))
        self.register_buffer("running_var", torch.ones(n_latents, classes))

    def forward(self, logits):
        x = logits.float()
        if self.training and torch.is_grad_enabled() and x.shape[0] >= self.min_batch:
            mean, var = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(var.detach(), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(var + self.eps) * self.scale


class CNNStem(nn.Module):
    """Small CNN: frame [N, 3, S, S] -> the (S / patch)^2 token grid [N, P, dim]: two stride-2 stages (S -> S / 4),
    then a (patch / 4)-strided convolution to the patch grid (overlapping receptive fields)."""

    def __init__(self, dim=256, patch=8, chans=(32, 64), groups=8, channels=3):
        super().__init__()
        assert patch % 4 == 0 and patch >= 4, "the CNN stem needs a patch size divisible by 4"
        c1, c2 = chans
        k = patch // 4
        self.net = nn.Sequential(
            nn.Conv2d(channels, c1, 4, stride=2, padding=1), nn.GroupNorm(groups, c1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.GroupNorm(groups, c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.GroupNorm(groups, c2), nn.SiLU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.GroupNorm(groups, c2), nn.SiLU(),
            nn.Conv2d(c2, dim, kernel_size=k, stride=k))

    def forward(self, img):
        return self.net(img).flatten(2).transpose(1, 2)                  # [N, P, dim], row-major


class FusionEncoder(nn.Module):
    """One transformer over the image tokens, read out by n_latents learned queries: query i -> the logits of
    categorical i (a Linear shared by all queries). Image tokens: stem "linear" = a linear embedding of the
    patch x patch pixel patches; stem "cnn" = a small CNN to the same grid. pos_inject: a fixed 2D sin-cos position of
    every image token is added to the attention input of every layer (PosInjectLayer); the queries get none.
    last_queries: the last layer computes only the queries' outputs (the only ones read; same result).

    Atari 100k is pixels only, so the sequence is [n_latents queries, image tokens]."""
    STEM_CHUNK = 1024                                                  # frames per checkpointed CNN pass
    LAYER_CHUNK = 2048                                                 # frames per checkpointed layer pass

    def __init__(self, dim=256, depth=6, heads=8, n_latents=32, classes=32, grad_checkpoint=True, stem="linear",
                 patch=8, img_size=64, pos_inject=True, swiglu=True, ffn=682, fast_attn=False, last_queries=False,
                 channels=3, logit_norm="none", logit_scale=4.0):
        super().__init__()
        if stem not in ("cnn", "linear"):
            raise ValueError(f"unknown encoder stem {stem!r}: use 'cnn' or 'linear'")
        self.n_latents, self.classes, self.stem_type = n_latents, classes, stem
        self.patch, self.img_size = patch, img_size
        self.grad_checkpoint = grad_checkpoint
        self.last_queries = last_queries
        rows, cols = patch_grid(img_size, patch)
        self.n_image_tokens = rows * cols
        if stem == "cnn":
            self.stem = CNNStem(dim, patch, channels=channels)
        else:
            self.patch_embed = nn.Linear(patch_dim(patch, channels), dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))         # hidden patches (linear stem)
        self.image_pos = nn.Parameter(torch.randn(1, self.n_image_tokens, dim) * 0.02)
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)   # the latent queries
        self.transformer = (_transformer(dim, depth, heads, PosInjectLayer, ffn, swiglu=True) if swiglu
                            else _transformer(dim, depth, heads, PosInjectLayer))
        for layer in self.transformer.layers:
            layer.fast_attn = fast_attn
        self._ckpt_runs = None                                         # compiled checkpointed layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, classes)                       # shared by all queries
        if logit_norm not in ("none", "batch"):
            raise ValueError(f"unknown logit_norm {logit_norm!r}: use 'none' or 'batch'")
        self.logit_norm = BatchLogitNorm(n_latents, classes, logit_scale) if logit_norm == "batch" else None
        inject = None                                                  # not saved: a fixed function of the layout
        if pos_inject:
            inject = torch.zeros(1, n_latents + self.n_image_tokens, dim)
            inject[0, n_latents:] = sincos_2d(*image_token_centres(rows, cols), dim, 2.0, 4.0 * max(rows, cols))
        self.register_buffer("inject_pos", inject, persistent=False)

    def _layer_fns(self):
        """Each layer's callable (x, pos) -> x; with last_queries the last one returns the latent queries only."""
        fns = list(self.transformer.layers)
        if self.last_queries:
            fns[-1] = partial(fns[-1].forward_first, n=self.n_latents)
        return fns

    def compile_layers(self):
        """torch.compile of each checkpointed layer pass (training only)."""
        def runner(fn):
            def run(x, pos):
                return checkpoint(fn, x, pos, use_reentrant=False)
            return run
        self._ckpt_runs = [torch.compile(runner(fn)) for fn in self._layer_fns()]

    def _image_tokens(self, img, hide):
        """Frames in [-1, 1] -> image tokens [N, P, dim]; hide [N, P] hides patches (masking branch)."""
        if self.stem_type == "linear":
            tok = self.patch_embed(patchify(img, self.patch))
            if hide is not None:
                tok = torch.where(hide[..., None], self.mask_token.to(tok.dtype), tok)
            return tok
        if hide is not None:                                           # hidden patches -> neutral grey before the
            img = img.masked_fill(patch_pixel_mask(hide, self.patch, self.img_size), 0.0)   # CNN: nothing leaks
        if not (self.grad_checkpoint and torch.is_grad_enabled()):
            return self.stem(img)
        return torch.cat([checkpoint(self.stem, img[i:i + self.STEM_CHUNK], use_reentrant=False)
                          for i in range(0, img.shape[0], self.STEM_CHUNK)])

    def forward(self, img, hide=None):
        """img [N, 3, S, S] in [-1, 1] -> posterior logits [N, L, K] (fp32)."""
        N = img.shape[0]
        tok = self._image_tokens(img, hide)
        dt = tok.dtype
        tok = tok + self.image_pos.to(dt)
        x = torch.cat([self.latent_tokens.expand(N, -1, -1).to(dt), tok], dim=1)   # [N, L + P, dim]
        ckpt = self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad
        pos = self.inject_pos
        for li, fn in enumerate(self._layer_fns()):
            if not ckpt:
                x = fn(x, pos)
            else:   # each layer recomputed in the backward pass, in chunks of frames (bounded peak memory)
                run = (self._ckpt_runs[li] if self._ckpt_runs is not None and not eager_only()
                       else partial(checkpoint, fn, use_reentrant=False))
                x = torch.cat([run(x[i:i + self.LAYER_CHUNK], pos) for i in range(0, N, self.LAYER_CHUNK)])
        logits = self.to_logits(self.norm(x[:, :self.n_latents])).float()
        return self.logit_norm(logits) if self.logit_norm is not None else logits


# --- Decoders -------------------------------------------------------------------------------------------------------
class PixelDecoder(nn.Module):
    """Flat one-hot z -> RGB logits [N, 3, S, S]: Linear to c0 x 4 x 4, log2(S / 4) ConvTranspose x2 stages (with
    residual blocks after the stages in res_stages), then GroupNorm-SiLU-conv to 3 channels."""

    def __init__(self, latent_dim=1024, img_size=64, chans=(256, 128, 96, 64, 32), res_stages=(1, 2, 3), base=4,
                 channels=3):
        super().__init__()
        n_up = int(round(math.log2(img_size / base)))
        assert base * 2 ** n_up == img_size, f"the conv decoder needs img_size = 4 x 2^k, not {img_size}"
        assert len(chans) == n_up + 1, f"{n_up} upsampling stages need {n_up + 1} channel numbers, got {chans}"
        self.c0, self.base = chans[0], base
        self.fc = nn.Linear(latent_dim, chans[0] * base * base)
        layers = []
        for i in range(n_up):
            cin, cout = chans[i], chans[i + 1]
            layers += [nn.GroupNorm(8, cin), nn.SiLU(), nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)]
            if i + 1 in res_stages:
                layers.append(ConvResBlock(cout))
        layers += [nn.GroupNorm(8, chans[-1]), nn.SiLU(), nn.Conv2d(chans[-1], channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.fc(z.flatten(1)).view(-1, self.c0, self.base, self.base))

    @staticmethod
    def to_image(logits):
        """Logits [N, 3, H, W] -> image [N, H, W, 3] in [0, 1]."""
        return torch.sigmoid(logits.float()).permute(0, 2, 3, 1)


class TransformerPixelDecoder(nn.Module):
    """Flat one-hot z -> RGB logits [N, 3, S, S] with a transformer (MAE / TiTok style): each categorical becomes a
    token (its own class embeddings), one learned query per output patch attends to them and to the other queries, a
    linear layer turns each query into its patch's pixels, and a zero-initialised residual conv smooths the seams."""

    def __init__(self, n_latents=32, classes=32, dim=256, depth=4, heads=8, patch=8, img_size=64, channels=3):
        super().__init__()
        self.n_latents, self.classes, self.patch, self.img_size = n_latents, classes, patch, img_size
        self.latent_emb = nn.Parameter(torch.randn(n_latents, classes, dim) * 0.02)
        self.latent_pos = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_patches(img_size, patch), dim) * 0.02)
        self.transformer = _transformer(dim, depth, heads)
        self.norm = nn.LayerNorm(dim)
        self.to_pixels = nn.Linear(dim, patch_dim(patch, channels))
        self.refine = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.SiLU(),
                                    nn.Conv2d(32, channels, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, z):
        N = z.shape[0]
        zz = z.reshape(N, self.n_latents, self.classes)
        lat = torch.einsum("nlk,lkd->nld", zz.to(self.latent_emb.dtype), self.latent_emb) + self.latent_pos
        x = torch.cat([lat, self.queries.expand(N, -1, -1)], dim=1)
        x = self.norm(self.transformer(x)[:, self.n_latents:])
        logits = unpatchify(self.to_pixels(x), self.patch, self.img_size)
        return logits + self.refine(logits)

    @staticmethod
    def to_image(logits):
        return torch.sigmoid(logits.float()).permute(0, 2, 3, 1)


# --- Tokenizer ------------------------------------------------------------------------------------------------------
class Tokenizer(nn.Module):
    """Encoder (posterior) + pixel decoder around the n_latents x classes one-hot latent; the decoder reads z only."""
    DEC_CHUNK = 512                                                    # frames per checkpointed decoder pass

    def __init__(self, n_latents=32, classes=32, img_size=64, enc_dim=256, enc_depth=6, enc_heads=8,
                 enc_stem="linear", enc_patch=8, enc_pos_inject=True, enc_swiglu=True, enc_ffn=682,
                 enc_fast_attn=False, enc_last_queries=False, grad_checkpoint=True, unimix=0.01, decoder="conv",
                 dec_dim=256, dec_depth=4, dec_heads=8, dec_patch=8, dec_chans=(256, 128, 96, 64, 32),
                 dec_res=(1, 2, 3), dec_sac=False, enc_logit_norm="none", enc_logit_scale=4.0):
        super().__init__()
        self.n_latents, self.classes, self.unimix, self.img_size = n_latents, classes, unimix, img_size
        self.latent_dim = n_latents * classes
        self.dec_sac = dec_sac
        self._chunk_run = None                                         # compiled checkpointed decoder chunk
        self.encoder = FusionEncoder(enc_dim, enc_depth, enc_heads, n_latents, classes, grad_checkpoint, enc_stem,
                                     enc_patch, img_size, enc_pos_inject, enc_swiglu, enc_ffn, enc_fast_attn,
                                     enc_last_queries, logit_norm=enc_logit_norm, logit_scale=enc_logit_scale)
        if decoder == "transformer":
            self.decoder = TransformerPixelDecoder(n_latents, classes, dec_dim, dec_depth, dec_heads, dec_patch,
                                                   img_size)
        elif decoder == "conv":
            self.decoder = PixelDecoder(self.latent_dim, img_size, dec_chans, dec_res)
        else:
            raise ValueError(f"unknown decoder {decoder!r}: use 'transformer' or 'conv'")

    def posterior(self, obs, hide=None):
        """Observations {"image": uint8 [N, 3, S, S]} -> log-probabilities [N, L, K] of q(z | o) (fp32, unimix)."""
        img = image_input(obs["image"], autocast_dtype(obs["image"].device.type))
        return unimix_logprobs(self.encoder(img, hide), self.unimix)

    def encode(self, obs, straight_through=False):
        """Observations -> (flat one-hot sample [N, L*K], log-probabilities [N, L, K])."""
        logp = self.posterior(obs)
        return sample_onehot(logp, straight_through).flatten(1), logp

    def decode(self, z):
        """Flat latent -> RGB logits [N, 3, S, S]."""
        return self.decoder(z)

    # --- Reconstruction loss ----------------------------------------------------------------------------------------
    @torch.no_grad()
    def recon_targets(self, image, prev_image=None, motion_weight=1.0, motion_threshold=0.05):
        """Pixel-loss targets of uint8 frames [N, 3, S, S], computed once per update outside the checkpoint: fp32 RGB,
        the motion mask (pixels that changed since prev_image, when given), the per-pixel loss weight
        1 + (motion_weight - 1) x motion (None when it is 1 everywhere) and the BCE entropy floors."""
        rgb = image_target(image)
        motion = None
        if prev_image is not None:
            motion = motion_mask(rgb, image_target(prev_image), motion_threshold).float()
        w = 1.0 + (motion_weight - 1.0) * motion if motion is not None and motion_weight != 1.0 else None
        return _targets(rgb, w, motion)

    @staticmethod
    @torch.no_grad()
    def subset_targets(t, idx):
        """recon_targets of a subset of the frames."""
        pick = lambda x: None if x is None else x[idx]                 # noqa: E731
        return _targets(t["rgb"][idx], pick(t["w"]), pick(t["motion"]))

    def _chunk_ckpt(self, *args):
        """One checkpointed decoder chunk (dec_sac: conv outputs kept, the rest recomputed in the backward)."""
        sac = self.dec_sac and not eager_only()
        return checkpoint(self._chunk_recon, *args, use_reentrant=False, **({"context_fn": SAC_CONTEXT} if sac else {}))

    def compile_decoder(self):
        """torch.compile of the checkpointed chunk pass."""
        self._chunk_run = torch.compile(self._chunk_ckpt)

    def recon_loss(self, z, targets):
        """Pixel BCE of decoding flat latents z [N, L*K] against recon_targets, summed over the frame (DreamerV3 sums
        the image loss over the pixels) minus its entropy floor (0 = perfect), averaged over frames, with the motion
        weights normalised so the loss keeps the size of a per-frame sum. Chunk by chunk: each chunk's decoder pass
        and loss are recomputed in the backward pass. -> (loss, stats)."""
        t = targets
        ckpt = torch.is_grad_enabled() and z.requires_grad
        run = ((self._chunk_ckpt if eager_only() else self._chunk_run or self._chunk_ckpt) if ckpt
               else self._chunk_recon)
        num = tot = mot = 0.0
        w, m = t["w"], t["motion"]
        for i in range(0, z.shape[0], self.DEC_CHUNK):
            j = i + self.DEC_CHUNK
            out = run(z[i:j], t["rgb"][i:j], None if w is None else w[i:j], None if m is None else m[i:j])
            num, tot, mot = num + out[0], tot + out[1], mot + out[2]
        loss = ((num.double() - t["floor_w"]) / t["w_sum"] * t["hw"]).float()
        stats = {"recon_excess": ((tot.double() - t["floor"]) / (3.0 * t["n"] * t["hw"])).float()}
        if m is not None:
            stats["recon_motion_excess"] = ((mot.double() - t["floor_motion"])
                                            / (3.0 * t["motion_sum"]).clamp_min(1.0)).float()
        return loss, stats

    def _chunk_recon(self, z, rgb, w, motion):
        """One chunk -> (weighted BCE sum, BCE sum, BCE sum on moving pixels; the last two detached)."""
        logits = self.decoder(z)
        bce = F.binary_cross_entropy_with_logits(logits.float(), rgb, reduction="none").sum(1, keepdim=True)
        num = (bce * w).sum() if w is not None else bce.sum()
        b = bce.detach()
        mot = (b * motion).sum() if motion is not None else b.new_zeros(())
        return num, b.sum(), mot


def _targets(rgb, w, motion):
    """Target dict of fp32 frames rgb [N, 3, S, S]: weights w / motion mask [N, 1, S, S] (or None) and the float64
    sums of the BCE entropy floor that recon_loss subtracts."""
    n, _, H, W = rgb.shape
    floor = bce_floor(rgb).sum(1, keepdim=True).double()                 # [N, 1, S, S], channel-summed
    return {"rgb": rgb, "w": w, "motion": motion, "n": n, "hw": H * W, "floor": floor.sum(),
            "floor_w": floor.sum() if w is None else (floor * w).sum(),
            "w_sum": (torch.tensor(float(n * H * W), dtype=torch.float64, device=rgb.device) if w is None
                      else w.double().sum()),
            "floor_motion": None if motion is None else (floor * motion).sum(),
            "motion_sum": None if motion is None else motion.double().sum()}


# === Dynamics =======================================================================================================
# Dynamics: transformer state-space model (TSSM) with a KV ring cache, the prior head and a CUDA-graphed dream step.
#
# Positions: RoPE makes every query-key score depend on the relative offset only, but the attention sink's key is not
# rotated, so a query's sink score depends on its ABSOLUTE position. Training only ever sees positions 0 .. seq_len - 1,
# while acting counts up to the episode length. With `sink_max_pos` set (seq_len - 1), the sink key is rotated by
# pos - min(pos, sink_max_pos) at inference (acting, dreams, open-loop rollouts): the sink score of any query is then the
# one it had at a trained position, and every query-key score is unchanged.


# --- Primitives -----------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMS normalisation computed in fp32."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight).to(dt)


def rope_cos_sin(pos, half_dim, base=10000.0):
    """RoPE cos / sin tables for positions `pos`."""
    inv_freq = base ** (-torch.arange(half_dim, device=pos.device, dtype=torch.float32) / half_dim)
    angles = pos.to(torch.float32).unsqueeze(-1) * inv_freq
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Rotates the two halves of the last dim of x."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


# --- KV ring cache --------------------------------------------------------------------------------------------------
class TSSMCache:
    """Per-row ring of W key / value slots; updated only in place, so captured CUDA graphs keep working on it."""

    def __init__(self, n_layers, batch, window, n_kv_heads, head_dim, device, dtype=torch.float32):
        self.k = torch.zeros(n_layers, batch, window, n_kv_heads, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.key_pos = torch.full((batch, window), -1, dtype=torch.long, device=device)   # step held by a slot
        self.pos = torch.zeros(batch, dtype=torch.long, device=device)                    # position of next token
        self.window = window

    def reset_rows(self, rows):
        """Empties the given rows."""
        self.pos[rows] = 0
        self.key_pos[rows] = -1

    def snapshot(self):
        """Copy of the full cache state."""
        return [t.clone() for t in (self.k, self.v, self.key_pos, self.pos)]

    def restore(self, snap):
        """Restores a snapshot in place."""
        for t, s in zip((self.k, self.v, self.key_pos, self.pos), snap):
            t.copy_(s)


# --- Transformer block ----------------------------------------------------------------------------------------------
ATTN_GATES = ("none", "headwise", "elementwise")


class TSSMBlock(nn.Module):
    """Llama-style block: RMSNorm, QK-norm, RoPE, GQA with a learned sink, optional gated attention, SwiGLU."""

    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden, attn_gate="elementwise"):
        super().__init__()
        assert attn_gate in ATTN_GATES, f"attn_gate must be one of {ATTN_GATES}"
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = hd = d_model // n_heads
        self.groups = n_heads // n_kv_heads
        self.norm_attn, self.norm_mlp = RMSNorm(d_model), RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(n_heads * hd, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(hd), RMSNorm(hd)
        self.sink_k = nn.Parameter(torch.randn(n_kv_heads, hd) * 0.02)          # always-visible attention sink
        self.sink_v = nn.Parameter(torch.zeros(n_kv_heads, hd))
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)
        self.gate_kind = attn_gate
        if attn_gate != "none":                                                  # gate = 2 sigmoid(Wx), 1 at init
            self.attn_gate = nn.Linear(d_model, n_heads * (hd if attn_gate == "elementwise" else 1), bias=False)
            nn.init.zeros_(self.attn_gate.weight)

    def _mlp(self, x):
        """SwiGLU feed-forward."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def _project(self, x, cos, sin):
        """-> RoPE-rotated q, k, v [..., heads, head_dim] and the attention gate (None without gating)."""
        h = self.norm_attn(x)
        shape = h.shape[:-1]
        q = self.q_proj(h).view(*shape, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(*shape, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(*shape, self.n_kv_heads, self.head_dim)
        q = apply_rope(self.q_norm(q), cos, sin).to(v.dtype)
        k = apply_rope(self.k_norm(k), cos, sin).to(v.dtype)
        g = None
        if self.gate_kind != "none":
            g = (2.0 * torch.sigmoid(self.attn_gate(h).float())).view(*shape, self.n_heads, -1)
        return q, k, v, g

    def _gated(self, attn, g):
        """Multiplies the per-head attention output by the gate."""
        if g is None:
            return attn
        shape = attn.shape[:-1]
        return (attn.view(*shape, self.n_heads, self.head_dim) * g.to(attn.dtype)).reshape(*shape, -1)

    def _with_sink(self, keys, values):
        """Prepends the sink to keys / values [N, L, Hkv, hd]."""
        n = keys.shape[0]
        sk = self.k_norm(self.sink_k).to(keys.dtype).expand(n, 1, -1, -1)
        sv = self.sink_v.to(values.dtype).expand(n, 1, -1, -1)
        return torch.cat([sk, keys], dim=1), torch.cat([sv, values], dim=1)

    def _sink_scores(self, qg, sink_rot=None):
        """Returns the scores [N, Hkv, G] of grouped queries qg [N, Hkv, G, hd] on the sink key; sink_rot: per-row
        RoPE tables (cos, sin) [N, 1, hd / 2] that rotate the sink key first (TSSM.sink_rot), None: unrotated."""
        sk = self.k_norm(self.sink_k).to(qg.dtype)                               # [Hkv, hd]
        if sink_rot is None:
            return torch.einsum("nkgd,kd->nkg", qg, sk)
        skr = apply_rope(sk.expand(qg.shape[0], -1, -1), *sink_rot).to(qg.dtype)  # [N, Hkv, hd]
        return torch.einsum("nkgd,nkd->nkg", qg, skr)

    def _attend(self, q, keys, values, key_mask, sink_rot=None):
        """Single-token attention of q [N, H, hd] over sink + keys [N, L, Hkv, hd] with key_mask [N, L]. The sink's
        score and value are computed separately, so the cache is read in place and never copied."""
        N = q.shape[0]
        qg = q.view(N, self.n_kv_heads, self.groups, self.head_dim)
        sv = self.sink_v.to(q.dtype)
        key_mask = torch.cat([key_mask.new_ones(N, 1), key_mask], dim=1)
        scores = torch.cat([self._sink_scores(qg, sink_rot)[..., None],
                            torch.einsum("nkgd,nlkd->nkgl", qg, keys.to(q.dtype))], dim=-1).float()
        scores = (scores / math.sqrt(self.head_dim)).masked_fill(~key_mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        out = probs[..., :1] * sv[None, :, None, :] + torch.einsum("nkgl,nlkd->nkgd", probs[..., 1:],
                                                                   values.to(q.dtype))
        return out.reshape(N, -1)

    def forward_train(self, x, cos, sin, attn_mask):
        """Teacher-forced pass over x [B, T, D] -> (x, (k, v)) with k, v [B, T, Hkv, hd]."""
        B, T, _ = x.shape
        q, k, v, g = self._project(x, cos, sin)
        keys, values = self._with_sink(k, v)
        if self.groups > 1:
            keys = keys.repeat_interleave(self.groups, dim=2)
            values = values.repeat_interleave(self.groups, dim=2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                                              attn_mask=attn_mask)
        x = x + self.o_proj(self._gated(attn.transpose(1, 2).reshape(B, T, -1), g))
        return x + self._mlp(self.norm_mlp(x)), (k, v)

    def forward_step(self, x, k_cache, v_cache, slot, key_mask, cos, sin, sink_rot=None):
        """Inference step: writes this token's K/V into ring slot `slot`, then attends to the cache."""
        q, k, v, g = self._project(x, cos, sin)
        rows = torch.arange(x.shape[0], device=x.device)
        k_cache[rows, slot] = k.to(k_cache.dtype)
        v_cache[rows, slot] = v.to(v_cache.dtype)
        x = x + self.o_proj(self._gated(self._attend(q, k_cache, v_cache, key_mask, sink_rot), g))
        return x + self._mlp(self.norm_mlp(x))

    def prefix_kv(self, keys, values):
        """Fixed open-loop context: sink + keys / values [N, L, Hkv, hd] -> [N, Hkv, 1 + L, hd]."""
        keys, values = self._with_sink(keys, values)
        return keys.transpose(1, 2).contiguous(), values.transpose(1, 2).contiguous()

    def forward_append(self, x, prefix_k, prefix_v, prefix_mask, new_k, new_v, new_mask, cos, sin, sink_rot=None):
        """Differentiable open-loop step over the fixed prefix [N, Hkv, Lp, hd] (sink included), the tokens appended
        so far and itself; the prefix is shared, never copied. -> (x, new_k + [k], new_v + [v])."""
        q, k, v, g = self._project(x, cos, sin)
        new_k, new_v = new_k + [k], new_v + [v]
        N = x.shape[0]
        qg = q.view(N, self.n_kv_heads, self.groups, self.head_dim)
        kn = torch.stack(new_k, dim=2).to(q.dtype)                              # [N, Hkv, j+1, hd]
        vn = torch.stack(new_v, dim=2).to(q.dtype)
        pre = torch.matmul(qg, prefix_k.to(q.dtype).transpose(-1, -2))
        if sink_rot is not None:                                                 # prefix column 0 is the sink
            pre = torch.cat([self._sink_scores(qg, sink_rot)[..., None], pre[..., 1:]], dim=-1)
        scores = torch.cat([pre, torch.matmul(qg, kn.transpose(-1, -2))], dim=-1).float() / math.sqrt(self.head_dim)
        mask = torch.cat([prefix_mask, new_mask], dim=1)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        Lp = prefix_k.shape[2]
        attn = torch.matmul(probs[..., :Lp], prefix_v.to(v.dtype)) + torch.matmul(probs[..., Lp:], vn.to(v.dtype))
        x = x + self.o_proj(self._gated(attn.reshape(N, -1), g))
        return x + self._mlp(self.norm_mlp(x)), new_k, new_v


# --- TSSM -----------------------------------------------------------------------------------------------------------
class TSSM(nn.Module):
    """One token per step (z_t, a_t) -> h_{t+1}; causal sliding window of `window` steps plus a sink."""

    def __init__(self, latent_dim=1024, action_dim=18, d_model=512, n_layers=6, n_heads=4, n_kv_heads=4,
                 ffn_hidden=2730, window=32, embed_hidden=None, attn_gate="elementwise", sink_max_pos=None):
        super().__init__()
        self.latent_dim, self.action_dim = latent_dim, action_dim
        self.d_model, self.n_kv_heads = d_model, n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.sink_max_pos = None if sink_max_pos is None else int(sink_max_pos)   # last trained position
        eh = embed_hidden or ffn_hidden
        self.token_mlp = nn.Sequential(nn.Linear(latent_dim + action_dim, eh), nn.SiLU(), nn.Linear(eh, d_model))
        self.embed_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([TSSMBlock(d_model, n_heads, n_kv_heads, ffn_hidden, attn_gate)
                                     for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)

    def embed(self, z, action):
        """Token u_t = RMSNorm(MLP([z_t, a_t]))."""
        return self.embed_norm(self.token_mlp(torch.cat([z, action.to(z.dtype)], dim=-1)))

    def _rope(self, pos):
        """RoPE tables for positions `pos`."""
        return rope_cos_sin(pos, self.head_dim // 2)

    def sink_rot(self, pos):
        """Returns per-row RoPE tables (cos, sin) [N, 1, hd / 2] that rotate the sink key by pos - min(pos,
        sink_max_pos), so a query at any position gets the sink score of a trained position (None when off).
        Tensor ops only (CUDA-graph safe); identity rotation up to sink_max_pos."""
        if self.sink_max_pos is None:
            return None
        cos, sin = self._rope(pos - pos.clamp(max=self.sink_max_pos))
        return cos[:, None], sin[:, None]

    def _key_mask(self, key_pos, pos):
        """Keys that are filled and inside the window of each row."""
        return (key_pos >= 0) & (key_pos > (pos - self.window)[:, None])

    # teacher-forced pass
    def forward_sequence(self, z, actions, return_kv=False, detach_kv=True):
        """z [B, T, D], actions [B, T, A] -> (h [B, T, d] with h[:, t] = h_{t+1}, per-layer (k, v) or None)."""
        x = self.embed(z, actions)
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        cos, sin = self._rope(pos)
        cos, sin = cos[None, :, None], sin[None, :, None]
        rel = pos[:, None] - pos[None, :]
        causal = (rel >= 0) & (rel < self.window)
        attn_mask = torch.cat([causal.new_ones(T, 1), causal], dim=1)            # column 0 = the sink
        kv = []
        eager = eager_only()                             # bypass a compiled blk.forward_train
        for blk in self.blocks:
            x, (k, v) = (TSSMBlock.forward_train(blk, x, cos, sin, attn_mask) if eager
                         else blk.forward_train(x, cos, sin, attn_mask))
            if return_kv:
                kv.append((k.detach(), v.detach()) if detach_kv else (k, v))
        return self.norm_out(x), (kv if return_kv else None)

    # inference on the ring cache
    def make_cache(self, batch, device, dtype=torch.float32):
        """Empty ring cache for `batch` rows."""
        return TSSMCache(len(self.blocks), batch, self.window, self.n_kv_heads, self.head_dim, device, dtype)

    @torch.no_grad()
    def prefill(self, cache, kv, rows, t0):
        """Loads teacher-forced K/V of sequences `rows`, tokens t0-W+1 .. t0, into the ring (no host sync)."""
        N, W = rows.shape[0], cache.window
        dev = rows.device
        steps = t0[:, None] - (W - 1) + torch.arange(W, device=dev)[None]        # [N, W], oldest first
        src = steps.clamp(min=0)
        slot = steps % W
        r = torch.arange(N, device=dev)[:, None].expand(N, W)
        for li, (k, v) in enumerate(kv):
            cache.k[li][r, slot] = k[rows[:, None], src].to(cache.k.dtype)
            cache.v[li][r, slot] = v[rows[:, None], src].to(cache.v.dtype)
        cache.key_pos[r, slot] = torch.where(steps >= 0, steps, torch.full_like(steps, -1))
        cache.pos.copy_(t0 + 1)

    def forward_step(self, z, action, cache):
        """One token per row on the cache (acting, dreams) -> h [N, d]; the cache advances in place."""
        x = self.embed(z, action)
        pos = cache.pos
        slot = pos % cache.window
        rows = torch.arange(x.shape[0], device=x.device)
        cache.key_pos[rows, slot] = pos
        cos, sin = self._rope(pos)
        cos, sin = cos[:, None], sin[:, None]
        key_mask = self._key_mask(cache.key_pos, pos)
        sink_rot = self.sink_rot(pos)
        for li, blk in enumerate(self.blocks):
            x = blk.forward_step(x, cache.k[li], cache.v[li], slot, key_mask, cos, sin, sink_rot)
        cache.pos.add_(1)
        return self.norm_out(x)

    # differentiable open-loop rollouts
    def open_loop_state(self, kv, rows, prefix_len):
        """Rollout context: K/V of sequences `rows` with the first `prefix_len` tokens visible (+ sink), built once
        and shared by every open-loop step."""
        T = kv[0][0].shape[1]
        steps = torch.arange(T, device=rows.device)[None]
        key_pos = torch.where(steps < prefix_len[:, None], steps, torch.full_like(steps, -1))
        prefix = [blk.prefix_kv(k[rows], v[rows]) for blk, (k, v) in zip(self.blocks, kv)]
        return {"prefix": prefix, "new_k": [[] for _ in self.blocks], "new_v": [[] for _ in self.blocks],
                "key_pos": key_pos, "new_pos": prefix_len.new_zeros(prefix_len.shape[0], 0),
                "pos": prefix_len.clone()}

    def step_append(self, z, action, state):
        """One differentiable token per row on an open-loop state -> h [N, d]."""
        x = self.embed(z, action)
        pos = state["pos"]
        cos, sin = self._rope(pos)
        cos, sin = cos[:, None], sin[:, None]
        key_mask = self._key_mask(state["key_pos"], pos)
        prefix_mask = torch.cat([key_mask.new_ones(x.shape[0], 1), key_mask], dim=1)        # column 0 = the sink
        new_pos = torch.cat([state["new_pos"], pos[:, None]], dim=1)
        new_mask = self._key_mask(new_pos, pos)
        sink_rot = self.sink_rot(pos)
        for li, blk in enumerate(self.blocks):
            pk, pv = state["prefix"][li]
            x, state["new_k"][li], state["new_v"][li] = blk.forward_append(
                x, pk, pv, prefix_mask, state["new_k"][li], state["new_v"][li], new_mask, cos, sin, sink_rot)
        state["new_pos"] = new_pos
        state["pos"] = pos + 1
        return self.norm_out(x)


# --- Prior head -----------------------------------------------------------------------------------------------------
class PriorHead(nn.Module):
    """MLP h_{t+1} -> logits [..., L, K] of the prior p(z_{t+1} | h_{t+1})."""

    def __init__(self, in_dim, n_latents=40, classes=40, hidden=1024, layers=3, unimix=0.01):
        super().__init__()
        self.n_latents, self.classes, self.unimix = n_latents, classes, unimix
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            d = hidden
        mods.append(nn.Linear(d, n_latents * classes))
        self.net = nn.Sequential(*mods)

    def forward(self, h):
        return self.net(h).float().reshape(*h.shape[:-1], self.n_latents, self.classes)

    def logprobs(self, h, unimix=None):
        """Log-probabilities with the uniform mix (`unimix` overrides it)."""
        return unimix_logprobs(self(h), self.unimix if unimix is None else unimix)


# --- CUDA-graphed dream step ----------------------------------------------------------------------------------------
class GraphedDreamStep:
    """One imagined step (TSSM step + prior head) for n dreams, captured as a CUDA graph and checked vs eager."""

    def __init__(self, tssm, prior, n, cache_dtype=torch.float32, amp_dtype=None, device="cuda", capture=True,
                 warmup=3):
        self.tssm, self.prior, self.n = tssm, prior, n
        self.L, self.K = prior.n_latents, prior.classes
        self.amp_dtype = amp_dtype
        self.device = torch.device(device)
        self.cache = tssm.make_cache(n, self.device, cache_dtype)               # static cache read by the graph
        self.action = torch.zeros(n, tssm.action_dim, device=self.device)
        self.z_prev = torch.zeros(n, tssm.latent_dim, device=self.device)
        self.graph = None
        if capture and self.device.type == "cuda":
            self._capture(warmup)

    def _body(self):
        """Graph body; autocast's weight cache stays off so replays read the current weights; compiled modules run
        eagerly here (the graph replays the captured kernels)."""
        with torch.no_grad(), eager_modules(), torch.autocast(self.device.type, dtype=self.amp_dtype or torch.bfloat16,
                                                              enabled=self.amp_dtype is not None, cache_enabled=False):
            h = self.tssm.forward_step(self.z_prev, self.action, self.cache).float()
            logits = self.prior(h)
        return h, logits

    def _randomize(self):
        """Random one-hot inputs for warm-up and the capture check."""
        self.action.copy_(F.one_hot(torch.randint(0, self.action.shape[1], (self.n,), device=self.device),
                                    self.action.shape[1]).float())
        self.z_prev.copy_(F.one_hot(torch.randint(0, self.K, (self.n, self.L), device=self.device),
                                    self.K).float().flatten(1))

    def _capture(self, warmup):
        """Warms up, captures the graph and checks it against the eager step."""
        snap = self.cache.snapshot()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._randomize()
                self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.cache.restore(snap)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.h_out, self.logits_out = self._body()
        self.graph = graph

        self._randomize()
        self.cache.restore(snap)
        self.graph.replay()
        h_g, l_g = self.h_out.clone(), self.logits_out.clone()
        self.cache.restore(snap)
        h_e, l_e = self._body()
        self.cache.restore(snap)
        err = max((h_g - h_e).abs().max().item(), (l_g - l_e).abs().max().item())
        tol = 5e-2 if self.amp_dtype is not None else 1e-3
        if not err <= tol:
            self.graph = None
            raise RuntimeError(f"graph replay differs from the eager step by {err:.3g} (tolerance {tol})")

    def __call__(self, action, z_prev, unimix=0.0, gumbel=None):
        """action [n, A], z_prev [n, L*K] -> (h [n, d], one-hot sample z [n, L*K], prior log-probs [n, L, K])."""
        self.action.copy_(action)
        self.z_prev.copy_(z_prev)
        if self.graph is None:
            h, logits = self._body()
        else:
            self.graph.replay()
            h, logits = self.h_out.clone(), self.logits_out.clone()
        logp = unimix_logprobs(logits, unimix)
        g = gumbel_like(logp) if gumbel is None else gumbel
        z = F.one_hot((logp + g).argmax(-1), self.K).float().flatten(1)
        return h, z, logp
