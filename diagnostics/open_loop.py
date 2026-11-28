"""Open-loop check of a trained run: real replay windows vs the world model's dreams driven by the recorded actions.

    from diagnostics.open_loop import load_run, open_loop_figure
    agent = load_run("runs/Breakout/gq_s0")                    # config.json + newest checkpoint + replay buffer
    open_loop_figure(agent, "runs/Breakout/gq_s0/open_loop.png", rows=4, horizon=15)

Each pair of rows of the figure is one replay window: the real frames t0 .. t0 + H (top) and the decoded dream
(bottom), which starts from the real context up to t0 and then runs on its own prior samples with the actions that
were really taken. The title gives the frame RMSE after 1, 5 and H steps, averaged over the windows."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gq_dreamer.gq_dreamer import GQDreamer                               # noqa: E402
from gq_dreamer.config import config_from_dict                             # noqa: E402
from gq_dreamer.observations import image_target                           # noqa: E402


class _NoEnvs:
    """Stand-in for the training envs (one replay ring)."""

    def __len__(self):
        return 1


def load_run(run_dir, device=None, checkpoint=None):
    """GQDreamer of a run directory with its newest (or given) checkpoint and replay buffer."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = config_from_dict(json.load(f))
    agent = GQDreamer(cfg, device=device, envs=_NoEnvs(), run_dir=run_dir, verbose=False)
    agent.load_checkpoint(checkpoint)
    return agent


@torch.no_grad()
def open_loop(agent, rows=8, horizon=15, seed=0):
    """Real and dreamed frames of `rows` replay windows -> (real [N, H + 1, 3, S, S], dream [N, H + 1, 3, S, S],
    rmse [N, H + 1]) in [0, 1]; step 0 of the dream is the posterior reconstruction of the start frame."""
    torch.manual_seed(seed)
    c = agent.cfg
    batch = agent.sample_batch(rows * 2, c.seq_len)
    zf, _ = agent.encode_frames({"image": batch["image"]})
    _, kv = agent.context_kv(zf, batch["actions"])
    out = agent.open_loop(zf, batch["actions"], kv, batch["valid"], n_rows=rows, horizon=horizon)
    if out is None:
        raise RuntimeError("no replay window is long enough for this horizon")
    r, t0 = out["rows"], out["t0"]
    steps = t0[:, None] + torch.arange(horizon + 1, device=t0.device)[None]
    real = image_target(batch["image"][r[:, None], steps].flatten(0, 1)).view(len(r), horizon + 1, 3, c.img_size,
                                                                              c.img_size)
    lat = torch.cat([zf[r, t0][:, None], out["teacher"]], dim=1)            # [N, H + 1, D]
    with agent._amp():
        dream = torch.sigmoid(agent.tokenizer.decode(lat.flatten(0, 1)).float())
    dream = dream.view_as(real)
    rmse = (dream - real).square().mean((2, 3, 4)).sqrt()
    return real.cpu(), dream.cpu(), rmse.cpu()


def open_loop_figure(agent, path, rows=4, horizon=15, seed=0):
    """Draws open_loop() as a real / dream grid -> path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    real, dream, rmse = open_loop(agent, rows, horizon, seed)
    n = real.shape[0]
    fig, axes = plt.subplots(2 * n, horizon + 1, figsize=(1.1 * (horizon + 1), 2.3 * n), facecolor="#111")
    for i in range(n):
        for k in range(horizon + 1):
            for j, src in enumerate((real, dream)):
                ax = axes[2 * i + j, k]
                ax.imshow(src[i, k].permute(1, 2, 0).numpy(), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if k == 0:
                    ax.set_ylabel("real" if j == 0 else "dream", color="#ccc")
                if i == 0 and j == 0:
                    ax.set_title(f"t0+{k}", color="#ccc", fontsize=8)
    m = rmse.mean(0).numpy()
    picks = [s for s in (1, 5, horizon) if s <= horizon]
    fig.suptitle(f"{agent.cfg.game} · open loop with the recorded actions\nframe RMSE "
                 + " · ".join(f"{s} steps {m[s]:.3f}" for s in picks) + f" (reconstruction {m[0]:.3f})",
                 color="#eee", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=15)
    a = ap.parse_args()
    ag = load_run(a.run_dir)
    print(open_loop_figure(ag, os.path.join(a.run_dir, "open_loop.png"), a.rows, a.horizon))
