"""Dream audit (diagnostic only; nothing is trained on it and its env steps are not in the 100k budget): the same real
states are played forward in the dream and in the real game with the same greedy policy, and the returns, the
terminal (life-loss) rates, the decoded frames and the reward head are compared."""
import math

import numpy as np
import torch

AUDIT_STREAMS = ("reward", "penalty")                           # penalty: the death penalty on terminals (if any)
AUDIT_KINDS = ("uniform", "pre_death")                          # pre_death: a terminal follows 2-20 steps later
PRE_DEATH_SHARE = 0.3
FRAME_STEPS = (1, 5, 10, 15)                                    # decoded-frame error at these rollout steps
HEAD_STEPS = 5                                                  # reward-head error over the first rollout steps
_KIND_LABELS = {"uniform": "uniform", "pre_death": "pre-terminal starts"}

# audit CSV columns: (header, source, key)
AUDIT_COLUMNS = [
    ("auditStarts", "audit", "starts"), ("auditSeconds", "audit", "seconds"),
    ("auditStartsPreDeath", "audit", "starts_pre_death"),
    *[(f"audit_{g}_{side}", "audit", f"{g}_{side}") for g in AUDIT_STREAMS for side in ("dream", "real", "gap")],
    *[(f"audit_{g}_pre_death_{side}", "audit", f"{g}_pre_death_{side}") for g in AUDIT_STREAMS
      for side in ("dream", "real", "gap")],
    ("audit_death_dream", "audit", "death_dream"), ("audit_death_real", "audit", "death_real"),
    ("audit_death_pre_death_dream", "audit", "death_pre_death_dream"),
    ("audit_death_pre_death_real", "audit", "death_pre_death_real"),
    ("audit_frame_err_0", "audit", "frame_err_0"),
    *[(f"audit_frame_err_{s}", "audit", f"frame_err_{s}") for s in FRAME_STEPS],
    ("audit_head_err_reward", "audit", "head_err_reward"),
]


def audit_enabled(cfg):
    """True when the dream audit runs (cfg.audit on and cfg.audit_every > 0)."""
    return bool(cfg.audit) and int(cfg.audit_every) > 0


class SnapshotPool:
    """Env snapshots taken while collecting, one every `every` agent steps of a replay segment; labelled pre-death
    when the segment ends in a terminal soon after."""

    def __init__(self, every):
        self.every = max(int(every), 0)                 # 0: no snapshots (audit off)
        self.items = []                                 # labelled snapshots of closed segments
        self._open = {}                                 # env -> [(episode id, step, snapshot)] of its open segment

    def __len__(self):
        return len(self.items) + sum(len(v) for v in self._open.values())

    def clear(self):
        """Forgets the labelled snapshots (each audit uses those taken since the previous one)."""
        self.items = []

    def due(self, step):
        """True when the observation of segment step `step` gets a snapshot."""
        return self.every > 0 and step > 0 and step % self.every == 0

    MAX_OPEN = 400                                      # snapshots kept of one open segment (the newest)

    def add(self, env, episode_id, step, snapshot):
        """Adds env `env`'s snapshot of the state whose observation is row `step` of its open segment."""
        lst = self._open.setdefault(env, [])
        lst.append((int(episode_id), int(step), snapshot))
        if len(lst) > self.MAX_OPEN:
            del lst[0]

    def close_episode(self, env, length, died, death_window):
        """Labels the snapshots of env's closed segment (`length` rows; with a terminal its last row is the terminal
        frame) as pre-death (the terminal death_window[0]..[1] steps later) or not."""
        lo, hi = death_window
        for eid, t, snap in self._open.pop(env, []):
            if t >= length:
                continue
            pre = bool(died) and lo <= (length - 1) - t <= hi
            self.items.append({"snapshot": snap, "step": t, "episode_id": eid, "pre_death": pre})

    def usable(self, buffer, context):
        """Labelled snapshots (and the open segments' ones, as uniform) whose start row and up to `context` earlier
        rows are still in the buffer; each gets its buffer row."""
        out = []
        cands = list(self.items) + [{"snapshot": s, "step": t, "episode_id": e, "pre_death": False}
                                    for v in self._open.values() for e, t, s in v]
        for it in cands:
            row = buffer.locate(it["episode_id"], it["step"])
            first = buffer.locate(it["episode_id"], it["step"] - min(it["step"], context))
            if row >= 0 and first >= 0:
                out.append({**it, "row": row})
        return out


def select_starts(items, n, rng):
    """Picks the starts -> [(index into items, kind)]: round(PRE_DEATH_SHARE x n) pre-death, the rest uniform."""
    n = min(int(n), len(items))
    taken = np.zeros(len(items), dtype=bool)
    picks = []
    for kind, flag, want in (("pre_death", "pre_death", int(round(PRE_DEATH_SHARE * n))), ("uniform", None, n)):
        cands = [i for i, it in enumerate(items) if not taken[i] and (flag is None or it[flag])]
        k = min(want, n - len(picks), len(cands))
        if k <= 0:
            continue
        for i in rng.choice(np.asarray(cands, dtype=np.int64), size=k, replace=False):
            taken[i] = True
            picks.append((int(i), kind))
    return picks


def audit_metrics(dream, real, kinds, discount):
    """Audit metrics of N starts over H steps. dream: "r" [N, H, 2] (reward, penalty), "alive" [N, H] calibrated
    p(continue), "frame_err" [N, H + 1] (step 0: the start frame's reconstruction); real: "r" [N, H, 2], "steps" [N]
    (steps taken before the segment ended), "died" [N]. Per stream the discounted return (the dream's weighted by
    its predicted survival, the real one over the steps taken) and the gap, terminal rates, frame errors and the
    reward head's error; overall and for pre-death starts."""
    rd, alive = (torch.as_tensor(dream[k]).detach().float().cpu() for k in ("r", "alive"))
    rr = torch.as_tensor(real["r"]).float().cpu()
    steps = torch.as_tensor(np.asarray(real["steps"]), dtype=torch.long)
    died = torch.as_tensor(np.asarray(real["died"]), dtype=torch.float32)
    kinds = torch.as_tensor(np.asarray(kinds), dtype=torch.long)
    N, H, G = rd.shape
    k = torch.arange(H)
    disc = float(discount) ** k.double()
    surv = torch.cat([torch.ones(N, 1), torch.cumprod(alive, dim=1)[:, :-1]], dim=1).double()
    taken = (k[None] < steps[:, None]).double()
    ret_d = ((disc * surv)[..., None] * rd.double()).sum(1)
    ret_r = ((disc * taken)[..., None] * rr.double()).sum(1)
    death_d = 1.0 - alive.double().prod(1)
    out = {"starts": float(N), "horizon": float(H)}
    for name, sel in (("", torch.ones(N, dtype=torch.bool)), ("_pre_death", kinds == AUDIT_KINDS.index("pre_death"))):
        if name:
            out[f"starts{name}"] = float(sel.sum())
        if not bool(sel.any()):
            continue
        for gi, g in enumerate(AUDIT_STREAMS[:G]):
            d, r = float(ret_d[sel, gi].mean()), float(ret_r[sel, gi].mean())
            out.update({f"{g}{name}_dream": d, f"{g}{name}_real": r, f"{g}{name}_gap": d - r})
        out[f"death{name}_dream"] = float(death_d[sel].mean())
        out[f"death{name}_real"] = float(died[sel].mean())
    fe = torch.as_tensor(dream["frame_err"]).float().cpu()
    out["frame_err_0"] = float(fe[:, 0].mean())
    for s in FRAME_STEPS:
        running = steps >= s
        if s > H or not bool(running.any()):
            continue
        out[f"frame_err_{s}"] = float(fe[running, s].mean())
    mh = taken[:, :min(HEAD_STEPS, H)]
    if float(mh.sum()) > 0:
        err = (rd[:, :mh.shape[1], 0] - rr[:, :mh.shape[1], 0]).abs().double()
        out["head_err_reward"] = float((err * mh).sum() / mh.sum())
    return out


def audit_lines(m):
    """The audit block of the round summary ([] without an audit)."""
    if not m or not m.get("starts"):
        return []

    def pct(v):
        return "–" if v is None or v != v else f"{100 * v:.0f}%"

    def val(v):
        return "–" if v is None or v != v else f"{v:+.2f}"

    def num(v, d=3):
        return "–" if v is None or v != v else f"{v:.{d}f}"

    H = int(m.get("horizon", FRAME_STEPS[-1]))
    lines = [f"DREAM AUDIT · {int(m['starts'])} real states ({int(m.get('starts_pre_death', 0))} pre-terminal), each "
             f"played {H} steps in the dream and in the game ({m.get('seconds', 0.0):.1f} s)",
             f"  {f'return over {H} steps':<24}{'dream':>8}  {'real':>8}  {'gap':>8}   gap = dream − real "
             f"(+ = the dream is too optimistic)"]
    for g in AUDIT_STREAMS:
        if f"{g}_dream" in m:
            pre = (f"   pre-terminal {val(m.get(f'{g}_pre_death_dream'))} / {val(m.get(f'{g}_pre_death_real'))}"
                   if f"{g}_pre_death_dream" in m else "")
            lines.append(f"    {g:<22}{val(m[f'{g}_dream']):>8}  {val(m.get(f'{g}_real')):>8}  "
                         f"{val(m[f'{g}_dream'] - m.get(f'{g}_real', float('nan'))):>8}{pre}")
    pre = (f"   pre-terminal {pct(m.get('death_pre_death_dream'))} / {pct(m.get('death_pre_death_real'))}"
           if "death_pre_death_dream" in m else "")
    lines.append(f"  {f'terminal within {H} steps':<24}{pct(m.get('death_dream')):>8}  {pct(m.get('death_real')):>8}"
                 f"{'':>10}{pre}")
    errs = " · ".join(f"{s} {num(m.get(f'frame_err_{s}'))}" for s in FRAME_STEPS)
    lines.append(f"  {'frame RMSE':<24}start {num(m.get('frame_err_0'))} · {errs} · reward head error (5 steps) "
                 f"{num(m.get('head_err_reward'))}")
    return lines


def rmse(a, b, mask=None):
    """Per-frame RMSE of images [N, 3, H, W] (mask [N, H, W]: only those pixels; NaN where a frame has none)."""
    sq = (a.float() - b.float()).square().mean(1)
    if mask is None:
        return sq.flatten(1).mean(1).sqrt()
    m = mask.float()
    n = m.flatten(1).sum(1)
    v = (sq * m).flatten(1).sum(1) / n.clamp(min=1.0)
    return torch.where(n > 0, v.sqrt(), torch.full_like(v, math.nan))
