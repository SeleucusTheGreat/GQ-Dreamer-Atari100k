"""Drawing of decoded dreams with matplotlib only (no torch): runs in the training process or in a worker process
(visualize.save_dreams_pdf with an executor), so the PDF is written while the training goes on."""
import gc
import glob
import os

import numpy as np

BG, BLUE, GREEN, RED, GREY = "#0d1117", "#58a6ff", "#3fb950", "#ff7b72", "#8b949e"


def _sign_colour(v):
    return GREEN if v > 0 else (RED if v < 0 else GREY)


def render_dream(d, pdf=None):
    """Draws one dream from its numpy data (visualize.dream_data); adds it to `pdf` or shows it."""
    frames, rewards, actions, alive, adv = d["frames"], d["rewards"], d["actions"], d["alive"], d["adv"]
    names = d.get("action_names")
    horizon = frames.shape[0]
    kw = dict(figsize=(horizon * 2.2, 5.6), facecolor=BG, dpi=110)
    grid = {"height_ratios": [1.0, 1.0]}
    if pdf is not None:
        from matplotlib.figure import Figure
        fig = Figure(**kw)
        axes = fig.subplots(2, horizon, gridspec_kw=grid)
    else:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, horizon, gridspec_kw=grid, **kw)
    if horizon == 1:
        axes = np.expand_dims(axes, axis=1)
    adv_value = d["max_advantage"] if d["max_advantage"] is not None else (
        float(adv.sum()) if adv is not None else None)
    title = f"{d.get('game', '')} · {d['label'] or 'Dream'} dream (decoded latents; frame 0 = posterior)"
    if adv_value is not None:
        title += f"  —  Σ normalised advantage = {adv_value:+.3f}"
    fig.suptitle(title, color=BLUE, fontsize=15, fontweight="bold", y=0.99)

    for i in range(horizon):
        ax_img, ax_info = axes[0, i], axes[1, i]
        ax_img.imshow(frames[i], interpolation="nearest")
        ax_img.axis("off")
        ax_img.set_title(f"t = {i}", color=GREY, fontsize=9)
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        if i < horizon - 1:
            a = int(actions[i])
            name = names[a] if names is not None and a < len(names) else str(a)
            r, al = float(rewards[i]), float(alive[i + 1])
            lines = [(name, BLUE, 12), (f"r: {r:+.2f}", _sign_colour(r), 12),
                     (f"cont {al:.2f}", GREEN if al > 0.5 else RED, 12)]
            if adv is not None:
                lines.append((f"A: {float(adv[i]):+.3f}", _sign_colour(float(adv[i])), 12))
            y = 0.9
            for txt, col, fs in lines:
                ax_info.text(0.5, y, txt, ha="center", va="center", fontsize=fs, fontweight="bold", color=col,
                             fontfamily="monospace", transform=ax_info.transAxes)
                y -= 0.22
        else:
            p = float(alive[i])
            ax_info.text(0.5, 0.72, "END", ha="center", va="center", fontsize=14, fontweight="bold", color=GREY,
                         transform=ax_info.transAxes)
            ax_info.text(0.5, 0.45, f"cont {p:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                         color=GREEN if p > 0.5 else RED, fontfamily="monospace", transform=ax_info.transAxes)
    fig.subplots_adjust(top=0.88, bottom=0.03, hspace=0.1, wspace=0.05)
    if pdf is not None:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
    else:
        import matplotlib.pyplot as plt
        plt.show()
        plt.close(fig)


def render_dreams_pdf(path, dreams, directory, keep=10):
    """Writes the dreams' numpy data to the PDF `path`, keeping the newest `keep` PDFs of `directory` -> path."""
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for d in dreams:
            render_dream(d, pdf=pdf)
    gc.collect()
    existing = sorted(glob.glob(os.path.join(directory, "*.pdf")), key=os.path.getmtime)
    while len(existing) > keep:
        try:
            os.remove(existing.pop(0))
        except OSError:
            pass
    return path
