"""Benchmark results of GQ-Dreamer runs against the published Atari 100k results: per-game scores, human-normalised
scores (HNS), mean / median / IQM aggregates (IQM with stratified bootstrap confidence intervals, as rliable), tables
and comparison plots.

A run writes runs/<Game>/<config>_s<seed>/final_eval.json (trainer.Trainer.final_evaluation); report() gathers every
run of a configuration. Aggregates of published methods are computed from their per-game means; when only some games
have been run, every method is compared on the same subset of games."""
import csv
import glob
import json
import os

import numpy as np

from . import baselines
from .atari import ATARI100K_GAMES, canonical_game


# --- Statistics -----------------------------------------------------------------------------------------------------
def iqm(values):
    """Interquartile mean: the mean of the middle 50 % (scipy.stats.trim_mean(x, 0.25), as rliable)."""
    x = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(x)
    if n == 0:
        return float("nan")
    cut = int(0.25 * n)
    return float(x[cut:n - cut].mean())


def aggregate_hns(scores, games=None):
    """{game: score} -> {"mean", "median", "iqm", "superhuman" (games with HNS >= 1), "n_games"} over `games` (default:
    the games present)."""
    games = [g for g in (games or scores.keys()) if g in scores and scores[g] is not None]
    h = np.array([baselines.human_normalized(g, scores[g]) for g in games], dtype=np.float64)
    if not len(h):
        return {"mean": float("nan"), "median": float("nan"), "iqm": float("nan"), "superhuman": 0, "n_games": 0}
    return {"mean": float(h.mean()), "median": float(np.median(h)), "iqm": iqm(h), "superhuman": int((h >= 1).sum()),
            "n_games": len(games)}


def bootstrap_ci(hns_runs, stat, reps=2000, alpha=0.05, seed=0):
    """Stratified bootstrap CI (rliable): hns_runs {game: [HNS of each seed]}; every replicate resamples the seeds of
    each game with replacement; stat(list of per-game arrays) -> float. -> (low, high)."""
    rng = np.random.default_rng(seed)
    games = [g for g, v in hns_runs.items() if len(v)]
    if not games:
        return float("nan"), float("nan")
    arrs = [np.asarray(hns_runs[g], dtype=np.float64) for g in games]
    out = np.empty(reps)
    for b in range(reps):
        out[b] = stat([a[rng.integers(0, len(a), len(a))] for a in arrs])
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def _stat_mean(per_game):
    return float(np.mean([a.mean() for a in per_game]))


def _stat_median(per_game):
    return float(np.median([a.mean() for a in per_game]))


def _stat_iqm_runs(per_game):
    return iqm(np.concatenate(per_game))


# --- Runs -----------------------------------------------------------------------------------------------------------
def run_dir(game, seed=0, config="gq", runs_root="runs"):
    """Directory of one run."""
    return os.path.join(runs_root, canonical_game(game), f"{config}_s{seed}")


def load_runs(runs_root="runs", config="gq", file="final_eval.json"):
    """{game: {seed: result dict}} of every finished run of a configuration (runs/<Game>/<config>_s<seed>/<file>)."""
    out = {}
    for path in glob.glob(os.path.join(runs_root, "*", f"{config}_s*", file)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, ValueError):
            continue
        game = canonical_game(res.get("game") or os.path.basename(os.path.dirname(os.path.dirname(path))))
        seed = int(res.get("seed", os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[-1]))
        out.setdefault(game, {})[seed] = res
    return out


def scores_by_game(runs):
    """{game: [mean evaluation return of each seed]} of load_runs()."""
    return {g: [float(r["mean"]) for _, r in sorted(seeds.items())] for g, seeds in runs.items()}


# --- Comparison -----------------------------------------------------------------------------------------------------
def compare(our_scores, methods=baselines.DEFAULT_COMPARISON, label="GQ-Dreamer", games=None, reps=2000):
    """our_scores {game: [score per seed]} vs the published methods on the same games -> dict with "games",
    "per_game" rows, "aggregates" {method: {mean, median, iqm, superhuman, n_games}}, and our bootstrap CIs."""
    games = [g for g in (games or ATARI100K_GAMES) if g in our_scores and len(our_scores[g])]
    ours_mean = {g: float(np.mean(our_scores[g])) for g in games}
    methods = [m for m in methods if m in baselines.methods()]
    agg = {label: aggregate_hns(ours_mean, games)}
    for m in methods:
        sc = baselines.method_scores(m)
        common = [g for g in games if g in sc]
        agg[m] = aggregate_hns(sc, common)
        agg[m]["complete"] = len(common) == len(games)
    hns_runs = {g: [baselines.human_normalized(g, s) for s in our_scores[g]] for g in games}
    n_seeds = min((len(v) for v in hns_runs.values()), default=0)
    ci = {}
    if games and max(len(v) for v in hns_runs.values()) > 1:
        ci = {"mean": bootstrap_ci(hns_runs, _stat_mean, reps), "median": bootstrap_ci(hns_runs, _stat_median, reps),
              "iqm_runs": bootstrap_ci(hns_runs, _stat_iqm_runs, reps)}
    agg[label]["iqm_runs"] = iqm(np.concatenate([np.asarray(v) for v in hns_runs.values()])) if games else float("nan")
    rows = []
    rnd, hum = baselines.random_scores(), baselines.human_scores()
    for g in games:
        s = np.asarray(our_scores[g], dtype=np.float64)
        row = {"game": g, "random": rnd[g], "human": hum[g], "ours_mean": float(s.mean()),
               "ours_std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"), "seeds": len(s),
               "ours_hns": baselines.human_normalized(g, s.mean())}
        for m in methods:
            row[m] = baselines.method_scores(m).get(g)
        rows.append(row)
    return {"label": label, "games": games, "methods": methods, "per_game": rows, "aggregates": agg, "ci": ci,
            "min_seeds": n_seeds}


def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:,.{d}f}"


def markdown(cmp):
    """Markdown tables (per game and aggregates) of a compare() result."""
    label, methods, games = cmp["label"], cmp["methods"], cmp["games"]
    lines = [f"### Atari 100k: {label} vs published results ({len(games)}/26 games, "
             f"{cmp['min_seeds']}+ seeds per game)", ""]
    head = ["Game", "Random", "Human", f"**{label}**"] + list(methods)
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in cmp["per_game"]:
        ours = _fmt(r["ours_mean"]) + (f" ± {_fmt(r['ours_std'])}" if np.isfinite(r["ours_std"]) else "")
        best = max([r["ours_mean"]] + [r[m] for m in methods if r[m] is not None])
        cells = [r["game"], _fmt(r["random"]), _fmt(r["human"]),
                 f"**{ours}**" if r["ours_mean"] >= best else ours]
        cells += [_fmt(r[m]) for m in methods]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Method | Mean HNS | Median HNS | IQM HNS (per-game means) | # > human | games |",
              "|---|---|---|---|---|---|"]
    ci = cmp["ci"]
    for name, a in cmp["aggregates"].items():
        mark = "" if a.get("complete", True) else " *"
        if name == label and ci:
            mean = f"**{a['mean']:.3f}** [{ci['mean'][0]:.3f}, {ci['mean'][1]:.3f}]"
            med = f"**{a['median']:.3f}** [{ci['median'][0]:.3f}, {ci['median'][1]:.3f}]"
            iq = f"**{a['iqm']:.3f}** (runs: {a['iqm_runs']:.3f} [{ci['iqm_runs'][0]:.3f}, {ci['iqm_runs'][1]:.3f}])"
        elif name == label:
            mean, med, iq = f"**{a['mean']:.3f}**", f"**{a['median']:.3f}**", f"**{a['iqm']:.3f}**"
        else:
            mean, med, iq = f"{a['mean']:.3f}", f"{a['median']:.3f}", f"{a['iqm']:.3f}"
        lines.append(f"| {name}{mark} | {mean} | {med} | {iq} | {a['superhuman']} | {a['n_games']} |")
    if len(games) == len(ATARI100K_GAMES):
        lines += ["", "As printed by the papers (26 games; IQM = rliable IQM over runs):", "",
                  "| Method | Mean HNS | Median HNS | IQM HNS |", "|---|---|---|---|"]
        for name in methods:
            info = baselines.method_info(name)
            lines.append(f"| {name} | {_fmt(info.get('reported_mean_hns'), 3)} | "
                         f"{_fmt(info.get('reported_median_hns'), 3)} | {_fmt(info.get('reported_iqm_hns'), 3)} |")
    lines += ["", "HNS = (score − random) / (human − random). Published aggregates are recomputed from the per-game "
                  "means on the same games; ours use the mean over seeds per game, with 95% stratified-bootstrap CIs "
                  "over seeds and the rliable IQM over all runs when there is more than one seed. "
                  "\\* = the method does not report every game of this subset."]
    return "\n".join(lines)


def write_csv(cmp, path):
    """Per-game rows of a compare() result as CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["game", "random", "human", "ours_mean", "ours_std", "seeds", "ours_hns"] + list(cmp["methods"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(cmp["per_game"])
    return path


def report(runs_root="runs", config="gq", out_dir="results", methods=baselines.DEFAULT_COMPARISON,
           label="GQ-Dreamer", plots=True, verbose=True):
    """Collects every finished run of `config`, compares it with the published results and writes
    <out_dir>/<config>_summary.md, _per_game.csv, _aggregates.png, _per_game_hns.png, _profile.png -> compare()."""
    runs = load_runs(runs_root, config)
    scores = scores_by_game(runs)
    if not scores:
        print(f"[report] no finished run of '{config}' under {runs_root}/ (looked for */{config}_s*/final_eval.json)")
        return None
    cmp = compare(scores, methods, label)
    os.makedirs(out_dir, exist_ok=True)
    md = markdown(cmp)
    with open(os.path.join(out_dir, f"{config}_summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    write_csv(cmp, os.path.join(out_dir, f"{config}_per_game.csv"))
    if plots:
        from .plots import plot_aggregates, plot_per_game_hns, plot_performance_profile
        for fn, name in ((plot_aggregates, "aggregates"), (plot_per_game_hns, "per_game_hns"),
                         (plot_performance_profile, "profile")):
            try:
                fn(cmp, os.path.join(out_dir, f"{config}_{name}.png"))
            except Exception as e:                                      # a plot never blocks the report
                print(f"[!] could not draw {name}: {e}")
    if verbose:
        print(md)
        print(f"\n[report] written to {out_dir}/{config}_*")
    return cmp
