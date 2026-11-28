"""Published Atari 100k results to compare GQ-Dreamer with: the random / human reference scores and the per-game
scores of well-known model-based and model-free agents (DreamerV3, IRIS, STORM, DIAMOND, TWM, SimPLe, EfficientZero,
BBF, SPR, DrQ, CURL, DER, ...).

The numbers live in data/atari100k_baselines.json together with their source (paper table or the authors' score
files), the aggregates each paper reports and notes on how they were obtained."""
import json
import os
from functools import lru_cache

from .atari import ATARI100K_GAMES, canonical_game

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "atari100k_baselines.json")

# methods drawn by default in plots / tables (the world models GQ-Dreamer is closest to, then strong references)
DEFAULT_COMPARISON = ("DreamerV3", "DreamerV3 (2024)", "STORM", "DIAMOND", "IRIS", "Delta-IRIS", "TWM", "SimPLe", "SPR",
                      "BBF", "EfficientZero")
REFERENCE_LINES = ("DreamerV3", "STORM", "IRIS", "DIAMOND")


@lru_cache(maxsize=1)
def load():
    """The baseline data (dict): games, reference {random, human}, methods {name: {scores, paper, ...}}."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def random_scores():
    return dict(load()["reference"]["random"])


def human_scores():
    return dict(load()["reference"]["human"])


def methods(category=None):
    """Names of the published methods (optionally of one category: "world model", "model-free", "lookahead")."""
    m = load()["methods"]
    return [k for k, v in m.items() if category is None or v.get("category") == category]


def method_info(name):
    """Metadata and scores of one method."""
    return load()["methods"][name]


def method_scores(name):
    """{game: published mean score} of one method (games it has no score for are missing)."""
    return {g: float(s) for g, s in load()["methods"][name]["scores"].items() if s is not None}


def human_normalized(game, score):
    """(score - random) / (human - random) with the standard reference scores."""
    game = canonical_game(game)
    r, h = load()["reference"]["random"][game], load()["reference"]["human"][game]
    return (float(score) - r) / (h - r)


def reference_lines(game, names=REFERENCE_LINES):
    """{label: score} of the reference lines of a game's training plot: Random, Human and the given methods."""
    game = canonical_game(game)
    out = {"Random": random_scores().get(game), "Human": human_scores().get(game)}
    for n in names:
        if n in load()["methods"]:
            s = load()["methods"][n]["scores"].get(game)
            if s is not None:
                out[n] = float(s)
    return {k: v for k, v in out.items() if v is not None}


def table(names=DEFAULT_COMPARISON, games=ATARI100K_GAMES):
    """Rows [game, random, human, method scores...] of the published per-game scores."""
    names = [n for n in names if n in load()["methods"]]
    rows = []
    for g in games:
        rows.append([g, random_scores()[g], human_scores()[g]] +
                    [load()["methods"][n]["scores"].get(g) for n in names])
    return ["Game", "Random", "Human"] + names, rows


def print_table(names=DEFAULT_COMPARISON):
    """Prints the published per-game scores and each method's mean / median human-normalised score."""
    from .benchmark import aggregate_hns
    header, rows = table(names)
    w = [max(len(header[0]), 14)] + [max(9, len(h)) for h in header[1:]]
    fmt = lambda v: "–" if v is None else (f"{v:,.1f}" if abs(v) < 1e5 else f"{v:,.0f}")   # noqa: E731
    print("  ".join(h.rjust(w[i]) if i else h.ljust(w[i]) for i, h in enumerate(header)))
    for r in rows:
        print("  ".join((str(v).ljust(w[i]) if i == 0 else fmt(v).rjust(w[i])) for i, v in enumerate(r)))
    for label, key in (("mean HNS", "mean"), ("median HNS", "median"), ("IQM HNS", "iqm")):
        cells = [label.ljust(w[0]), "".rjust(w[1]), "".rjust(w[2])]
        for i, n in enumerate(header[3:], start=3):
            a = aggregate_hns(method_scores(n))
            star = "" if a["n_games"] == len(ATARI100K_GAMES) else "*"
            cells.append(f"{a[key]:.3f}{star}".rjust(w[i]))
        print("  ".join(cells))
    print("(* = aggregate over the games the method reports)")
