"""Command line: python -m gq_dreamer <command> ...

    python -m gq_dreamer train --game Pong --seed 0                  # 100k steps + final evaluation
    python -m gq_dreamer train --game Breakout --config gq_fast --set train_ratio=512 eval_every=20000
    python -m gq_dreamer eval  --game Pong --seed 0 --episodes 100 [--record]
    python -m gq_dreamer suite --games all --seeds 0 1 2            # the 26 games (finished runs are skipped)
    python -m gq_dreamer report [--config gq]                      # tables + plots in results/
    python -m gq_dreamer baselines                                 # the published per-game scores
"""
import argparse
import ast


def _overrides(pairs):
    """['train_ratio=512', 'decoder="transformer"'] -> {'train_ratio': 512, 'decoder': 'transformer'}."""
    out = {}
    for p in pairs or ():
        k, _, v = p.partition("=")
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gq_dreamer", description="GQ-Dreamer on Atari 100k")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "suite"):
        p = sub.add_parser(name)
        if name == "suite":
            p.add_argument("--games", default="all", help="'all' or comma-separated names")
            p.add_argument("--seeds", type=int, nargs="+", default=[0])
            p.add_argument("--no-skip", action="store_true", help="rerun finished runs")
        else:
            p.add_argument("--game", default="Pong")
            p.add_argument("--seed", type=int, default=0)
        p.add_argument("--config", default="gq")
        p.add_argument("--runs-root", default="runs")
        p.add_argument("--device", default=None)
        p.add_argument("--dreams", action="store_true", help="write dream PDFs every round")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE", help="config overrides")
        if name == "eval":
            p.add_argument("--episodes", type=int, default=None)
            p.add_argument("--record", action="store_true")
            p.add_argument("--checkpoint", default=None)
        if name == "train":
            p.add_argument("--no-final-eval", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--config", default="gq")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out", default="results")
    sub.add_parser("baselines")
    a = ap.parse_args(argv)

    if a.cmd == "baselines":
        from .baselines import print_table
        print_table()
        return
    if a.cmd == "report":
        from .benchmark import report
        report(a.runs_root, a.config, a.out)
        return
    from .trainer import run, run_suite
    kw = dict(config=a.config, runs_root=a.runs_root, device=a.device, visualize_dreams=a.dreams,
              verbose=not a.quiet, **_overrides(a.set))
    if a.cmd == "train":
        run(a.game, a.seed, "train", final_eval=not a.no_final_eval, **kw)
    elif a.cmd == "eval":
        run(a.game, a.seed, "eval", eval_episodes=a.episodes, record=a.record, checkpoint=a.checkpoint, **kw)
    else:
        run_suite(a.games, tuple(a.seeds), skip_done=not a.no_skip, **kw)


if __name__ == "__main__":
    main()
