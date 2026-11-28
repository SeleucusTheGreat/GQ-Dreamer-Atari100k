"""GQ-Dreamer (GroundQuery-Dreamer) on the Atari 100k benchmark.

A DreamerV3-style agent whose posterior is a fusion transformer over 64x64 RGB patch tokens read out by learned latent
queries (one 32-class categorical per query), with a transformer state-space model (TSSM) as the sequence model,
trained in imagination with REINFORCE.

    from gq_dreamer import run, run_suite, report
    run("Pong", seed=0)                       # 100k env steps + final evaluation -> runs/Pong/gq_s0/
    run_suite(seeds=(0, 1, 2))                # the 26 games
    report()                                  # comparison with DreamerV3, STORM, IRIS, DIAMOND, ... -> results/
"""
import importlib

# Lazy exports (PEP 562): env worker processes only import gq_dreamer.env, not torch / matplotlib.
_EXPORTS = {"run": ".trainer", "run_suite": ".trainer", "Trainer": ".trainer", "GQDreamer": ".gq_dreamer",
            "create_envs": ".env", "AtariEnv": ".env", "GQDreamerConfig": ".config", "model_config": ".config",
            "MODEL_CONFIGS": ".config", "GQ_CONFIG": ".config", "report": ".benchmark", "compare": ".benchmark",
            "load_runs": ".benchmark", "ATARI100K_GAMES": ".atari", "baselines": ".baselines"}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name], __name__)
        value = mod if name == "baselines" else getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
