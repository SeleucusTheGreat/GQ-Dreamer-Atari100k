"""Dark-theme plots: a run's training curve against the published scores of its game, all runs of a game, and the
benchmark comparison figures (aggregate HNS, per-game HNS, performance profile)."""
import csv
import glob
import os

import numpy as np

# --- Dark theme -----------------------------------------------------------------------------------------------------
SURFACE = "#1a1a19"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
BLUE = "#3987e5"             # GQ-Dreamer
ORANGE = "#d95926"
TEAL = "#2a9d8f"
GREEN = "#8ab17d"
YELLOW = "#e2b33c"
PURPLE = "#9d7bd8"
PINK = "#d0679d"
GREY = "#8b8a84"
METHOD_COLOURS = {"DreamerV3": ORANGE, "DreamerV3 (2024)": "#f0a070", "Delta-IRIS": "#e8cf7a", "STORM": TEAL, "IRIS": YELLOW, "DIAMOND": PURPLE, "TWM": PINK,
                  "SimPLe": GREY, "SPR": GREEN, "EfficientZero": "#5aa9a1", "BBF": "#c9a36b", "DER": "#7f7f7f",
                  "CURL": "#a0a0a0", "DrQ": "#b0b0b0", "DrQ(eps)": "#b8b8b8", "Human": INK, "Random": MUTED}
RUN_COLOURS = (BLUE, ORANGE, TEAL, YELLOW, PURPLE, PINK, GREEN)

_STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": MUTED, "text.color": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.labelsize": 10.5, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
}


# --- Data -----------------------------------------------------------------------------------------------------------
def moving_average(values, window):
    """Trailing moving average over the last `window` records (NaNs ignored)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        seg = v[max(0, i - int(window) + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _read(path):
    with open(path, mode="r", newline="") as f:
        return list(csv.DictReader(f))


def _col(rows, name):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, "nan")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def _save_figure(fig, path, dpi=130):
    """Saves and closes the figure through a temporary file (a viewer never sees a half-written PNG)."""
    import matplotlib.pyplot as plt
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.tmp.png")
    try:
        fig.savefig(tmp, dpi=dpi, facecolor=fig.get_facecolor())
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _compact(v, _pos=None):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}".rstrip("0").rstrip(".") + "K"
    return f"{v:g}"


def _style_axes(ax, x_label, y_grid=True):
    import matplotlib.ticker as mticker
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    if y_grid:
        ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    else:
        ax.grid(False, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=4, color=BASELINE, pad=6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_compact))
    ax.set_xlabel(x_label, labelpad=8)


def _header(fig, ax, title, subtitle):
    left = ax.get_position().x0
    fig.text(left, 0.955, title, color=INK, fontsize=17, fontweight="semibold", ha="left", va="top")
    fig.text(left, 0.905, subtitle, color=INK_2, fontsize=10.5, ha="left", va="top")


def _end_marker(ax, x, y, color, caption):
    ax.scatter([x], [y], s=380, color=color, alpha=0.16, lw=0, zorder=5)
    ax.scatter([x], [y], s=70, color=color, edgecolors=SURFACE, linewidths=2.0, zorder=6)
    ax.annotate(f"{y:,.1f}", (x, y), xytext=(-14, 14), textcoords="offset points", ha="right", va="center",
                color=INK, fontsize=14, fontweight="semibold", zorder=7)
    ax.annotate(caption, (x, y), xytext=(-14, -2), textcoords="offset points", ha="right", va="center",
                color=MUTED, fontsize=8.5, zorder=7)


def _reference_lines(ax, refs, x_right=None):
    """Dashed horizontal lines of published scores; call _label_reference_lines once the y limits are final."""
    ax._gq_refs = dict(refs)
    for name, y in refs.items():
        ax.axhline(y, color=METHOD_COLOURS.get(name, MUTED), lw=1.1, ls=(0, (4, 3)), alpha=0.8, zorder=1)


def _label_reference_lines(ax, min_gap=0.035):
    """Labels of the reference lines outside the right edge, spread apart so that close scores stay readable."""
    refs = getattr(ax, "_gq_refs", {})
    if not refs:
        return
    lo, hi = ax.get_ylim()
    items = sorted(((y - lo) / (hi - lo), name, y) for name, y in refs.items())
    pos = []
    for f, _, _ in items:                                             # push up to keep min_gap between labels
        pos.append(max(f, pos[-1] + min_gap) if pos else f)
    over = pos[-1] - 1.0 if pos else 0.0
    if over > 0:                                                      # shift down if the top label left the axes
        pos = [p - over for p in pos]
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - min_gap)
    for (f, name, y), p in zip(items, pos):
        ax.annotate(f"{name} {y:,.0f}", xy=(1.0, f), xycoords="axes fraction", xytext=(1.01, p),
                    textcoords="axes fraction", ha="left", va="center", color=METHOD_COLOURS.get(name, MUTED),
                    fontsize=8.5, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, lw=0.6) if abs(p - f) > 0.005 else None)


# --- Training curves ------------------------------------------------------------------------------------------------
def plot_training(csv_path, out_path, game, window=10, refs=None, budget=100_000):
    """Training return per round (thin) and its moving average over env steps, evaluation returns (dots), and the
    published final scores of the game (dashed: Random, Human, DreamerV3, STORM, IRIS, DIAMOND) -> path or None."""
    if not os.path.exists(csv_path):
        return None
    rows = _read(csv_path)
    if not rows:
        return None
    import matplotlib.pyplot as plt
    x = _col(rows, "envSteps")
    ret = _col(rows, "trainReturn")
    ev, ev_sd = _col(rows, "evalReturn"), _col(rows, "evalReturnStd")
    if refs is None:
        from .baselines import reference_lines
        try:
            refs = reference_lines(game)
        except KeyError:
            refs = {}
    ma = moving_average(ret, window)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7.2))
        fig.subplots_adjust(left=0.065, right=0.88, top=0.84, bottom=0.1)
        x_max = max(float(np.nanmax(x)) if len(x) else 0.0, float(budget))
        _reference_lines(ax, refs, x_max)
        ok = np.isfinite(ret)
        ax.plot(x[ok], ret[ok], color=BLUE, lw=1.0, alpha=0.35, zorder=2)
        ok_ma = np.isfinite(ma)
        ax.plot(x[ok_ma], ma[ok_ma], color=BLUE, lw=2.4, solid_capstyle="round", zorder=3,
                label="training return (moving average)")
        ok_ev = np.isfinite(ev)
        if ok_ev.any():
            ax.errorbar(x[ok_ev], ev[ok_ev], yerr=np.nan_to_num(ev_sd[ok_ev]), fmt="o", color=INK, ms=5, lw=1.0,
                        capsize=3, zorder=4, label="evaluation return (± std)")
        if ok_ma.any():
            i = int(np.flatnonzero(ok_ma)[-1])
            _end_marker(ax, x[i], ma[i], BLUE, f"{window}-round avg")
        ax.set_xlim(0, x_max * 1.01)
        _style_axes(ax, "Environment steps (agent steps; × 4 frames)")
        ax.set_ylabel("Episode return", labelpad=10)
        _label_reference_lines(ax)
        leg = ax.legend(loc="best", frameon=False, fontsize=10)
        for t in leg.get_texts():
            t.set_color(INK_2)
        _header(fig, ax, f"GQ-Dreamer · {game} · Atari 100k",
                "Return of the training episodes finished in each round (thin), moving average (line); dashed: "
                "published scores after 100k steps")
        return _save_figure(fig, out_path)


def plot_game_runs(game_dir, out_path, window=10, budget=100_000):
    """Every run of a game (runs/<Game>/*/metrics.csv: configurations and seeds) on one plot, with the published
    reference lines -> path or None."""
    paths = sorted(glob.glob(os.path.join(game_dir, "*", "metrics.csv")))
    if not paths:
        return None
    import matplotlib.pyplot as plt
    from .baselines import reference_lines
    game = os.path.basename(os.path.normpath(game_dir))
    try:
        refs = reference_lines(game)
    except KeyError:
        refs = {}
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(14, 7.2))
        fig.subplots_adjust(left=0.065, right=0.88, top=0.84, bottom=0.1)
        _reference_lines(ax, refs, budget)
        for k, p in enumerate(paths):
            rows = _read(p)
            x, r = _col(rows, "envSteps"), _col(rows, "trainReturn")
            ma = moving_average(r, window)
            ok = np.isfinite(ma)
            ax.plot(x[ok], ma[ok], color=RUN_COLOURS[k % len(RUN_COLOURS)], lw=2.0,
                    label=os.path.basename(os.path.dirname(p)))
        ax.set_xlim(0, budget * 1.01)
        _style_axes(ax, "Environment steps")
        ax.set_ylabel("Episode return (moving average)", labelpad=10)
        _label_reference_lines(ax)
        leg = ax.legend(loc="best", frameon=False, fontsize=10)
        for t in leg.get_texts():
            t.set_color(INK_2)
        _header(fig, ax, f"GQ-Dreamer · {game} · every run", "Training return per run; dashed: published scores")
        return _save_figure(fig, out_path)


# --- Benchmark figures ----------------------------------------------------------------------------------------------
def plot_aggregates(cmp, out_path):
    """Mean / median / IQM human-normalised score of GQ-Dreamer and the published methods (same games)."""
    import matplotlib.pyplot as plt
    agg, label = cmp["aggregates"], cmp["label"]
    names = sorted(agg, key=lambda n: agg[n]["iqm"])
    keys = (("mean", "Mean HNS"), ("median", "Median HNS"), ("iqm", "IQM HNS"))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(names) + 2.4), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.8, bottom=0.12, wspace=0.08)
        for ax, (k, title) in zip(axes, keys):
            vals = [agg[n][k] for n in names]
            cols = [BLUE if n == label else METHOD_COLOURS.get(n, GREY) for n in names]
            ax.barh(range(len(names)), vals, color=cols, height=0.65)
            ends = list(vals)
            if k in cmp.get("ci", {}) and label in names:
                lo, hi = cmp["ci"][k]
                i = names.index(label)
                ax.plot([lo, hi], [i, i], color=INK, lw=2, solid_capstyle="butt")
                ends[i] = max(vals[i], hi)
            for i, (v, e) in enumerate(zip(vals, ends)):
                ax.text(e, i, f" {v:.2f}", va="center", ha="left", color=INK_2, fontsize=9)
            ax.axvline(1.0, color=MUTED, lw=1, ls=":")
            ax.set_title(title, color=INK, fontsize=12)
            _style_axes(ax, "", y_grid=False)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([f"{n}{'' if agg[n].get('complete', True) else ' *'}" for n in names])
        _header(fig, axes[0], "Atari 100k · aggregate human-normalised score",
                f"{len(cmp['games'])}/26 games (every method on the same games); dotted line = human level; "
                f"* = some games not reported")
        return _save_figure(fig, out_path)


def plot_per_game_hns(cmp, out_path, methods=("DreamerV3", "STORM", "IRIS", "DIAMOND")):
    """Per-game HNS of GQ-Dreamer next to a few published world models (symlog axis)."""
    import matplotlib.pyplot as plt
    from .baselines import human_normalized
    games = cmp["games"]
    methods = [m for m in methods if m in cmp["methods"]]
    series = [(cmp["label"], [r["ours_hns"] for r in cmp["per_game"]], BLUE)]
    for m in methods:
        series.append((m, [human_normalized(r["game"], r[m]) if r[m] is not None else np.nan
                           for r in cmp["per_game"]], METHOD_COLOURS.get(m, GREY)))
    n = len(series)
    width = 0.8 / n
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(games) * (1 + 0.15 * n)), 7.2))
        fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.2)
        for k, (name, vals, col) in enumerate(series):
            ax.bar(np.arange(len(games)) + (k - (n - 1) / 2) * width, vals, width=width, color=col, label=name)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.axhline(0.0, color=BASELINE, lw=1)
        ax.set_yscale("symlog", linthresh=1.0)
        _style_axes(ax, "")
        import matplotlib.ticker as mticker
        ax.xaxis.set_major_formatter(mticker.NullFormatter())
        ax.set_xticks(range(len(games)))
        ax.set_xticklabels(games, rotation=60, ha="right")
        ax.set_ylabel("Human-normalised score (symlog)", labelpad=10)
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=10, ncol=n)
        for t in leg.get_texts():
            t.set_color(INK_2)
        _header(fig, ax, "Atari 100k · per-game human-normalised score", "Dotted line = human level")
        return _save_figure(fig, out_path)


def plot_performance_profile(cmp, out_path, methods=("DreamerV3", "STORM", "IRIS", "DIAMOND", "TWM")):
    """Fraction of games whose HNS exceeds tau (per-game means; rliable's score distribution), for GQ-Dreamer and
    the world models closest to it."""
    import matplotlib.pyplot as plt
    from .baselines import human_normalized, method_scores
    games = cmp["games"]
    taus = np.linspace(0, 2.5, 251)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.subplots_adjust(left=0.08, right=0.8, top=0.84, bottom=0.1)
        curves = [(cmp["label"], np.array([r["ours_hns"] for r in cmp["per_game"]]), BLUE, 3.0)]
        for m in [m for m in methods if m in cmp["methods"]]:
            sc = method_scores(m)
            h = np.array([human_normalized(g, sc[g]) for g in games if g in sc])
            if len(h):
                curves.append((m, h, METHOD_COLOURS.get(m, GREY), 1.6))
        for name, h, col, lw in curves:
            ax.step(taus, [(h > t).mean() for t in taus], where="post", color=col, lw=lw, label=name)
        _style_axes(ax, "Human-normalised score τ")
        ax.set_ylabel("Fraction of games with score > τ", labelpad=10)
        ax.set_ylim(0, 1.02)
        leg = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
        for t in leg.get_texts():
            t.set_color(INK_2)
        _header(fig, ax, "Atari 100k · performance profile", f"{len(games)} games, per-game means")
        return _save_figure(fig, out_path)
