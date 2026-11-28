"""Atari 100k benchmark constants: the 26 games, the evaluation protocol and game-name helpers.

The protocol is the one of SimPLe (Kaiser et al. 2019) as used by DER, SPR, IRIS, STORM, DreamerV3 and DIAMOND:
100k agent steps (= 400k frames at frame skip 4) of interaction, no sticky actions, up to 30 no-op starts, the
minimal action set, max-pooling over the last two frames of every action repeat, episodes capped at 108k frames
(27k agent steps, 30 minutes of play), and the score is the mean undiscounted return of evaluation episodes played
after the 100k steps (human-normalised with the standard random / human reference scores)."""
import re

# --- The 26 games (the order used by the papers' tables) -----------------------------------------------------------
ATARI100K_GAMES = (
    "Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
    "CrazyClimber", "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull",
    "KungFuMaster", "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown",
)

# --- Protocol -------------------------------------------------------------------------------------------------------
BUDGET_STEPS = 100_000            # agent steps of interaction (every env step of every training env counts)
FRAMESKIP = 4                     # emulator frames per agent step (action repeat, max-pool of the last two)
MAX_EPISODE_FRAMES = 108_000      # 30 minutes at 60 Hz = 27,000 agent steps
NOOP_MAX = 30                     # 0..30 random no-ops after every reset
STICKY_ACTIONS = 0.0              # repeat_action_probability (Atari 100k: none; Machado et al. 2018 use 0.25)
IMG_SIZE = 64                     # observation side (RGB), as DreamerV3 / IRIS / STORM / DIAMOND
FINAL_EVAL_EPISODES = 100         # evaluation episodes after training (IRIS / rliable convention)

_LOWER = {g.lower(): g for g in ATARI100K_GAMES}


def canonical_game(name):
    """'pong', 'Pong', 'ALE/Pong-v5', 'PongNoFrameskip-v4', 'ms_pacman', 'up_n_down' -> 'Pong' / 'MsPacman' / ...
    Games outside the 26 are returned in CamelCase as given (any ALE game can be trained)."""
    s = str(name).strip()
    s = re.sub(r"^ALE/", "", s)
    s = re.sub(r"(NoFrameskip)?-v\d+$", "", s)
    s = re.sub(r"Deterministic$", "", s)
    key = s.replace("_", "").replace("-", "").replace(" ", "").lower()
    if key in _LOWER:
        return _LOWER[key]
    if "_" in s or s.islower():
        return "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\s-]+", s) if p)
    return s


def env_id(game):
    """Gymnasium id of the ALE v5 environment of a game: 'Pong' -> 'ALE/Pong-v5'."""
    return f"ALE/{canonical_game(game)}-v5"


def resolve_games(games):
    """None / 'all' / '26' -> the 26 games; a comma-separated string or a list -> canonical names (order kept)."""
    if games is None or (isinstance(games, str) and games.lower() in ("all", "26", "atari100k")):
        return list(ATARI100K_GAMES)
    if isinstance(games, str):
        games = [g for g in games.split(",") if g.strip()]
    return [canonical_game(g) for g in games]
