"""CPU smoke tests of the Atari 100k port (tiny model, a few hundred env steps): python -m pytest tests -q"""
import json
import os

import numpy as np
import pytest
import torch

from gq_dreamer import baselines
from gq_dreamer.atari import ATARI100K_GAMES, canonical_game
from gq_dreamer.benchmark import aggregate_hns, compare, iqm
from gq_dreamer.env import AtariEnv
from gq_dreamer.replay import ReplayBuffer


# --- Protocol -------------------------------------------------------------------------------------------------------
def test_game_names():
    assert canonical_game("ALE/MsPacman-v5") == "MsPacman"
    assert canonical_game("PongNoFrameskip-v4") == "Pong"
    assert canonical_game("up_n_down") == "UpNDown"
    assert len(ATARI100K_GAMES) == 26


def test_env_protocol():
    env = AtariEnv("Breakout", seed=0)
    obs, info = env.reset()
    assert obs["image"].shape == (3, 64, 64) and obs["image"].dtype == np.uint8
    assert env.action_space.n == 4 and env.max_episode_steps == 27_000
    frames0 = env.ale.getEpisodeFrameNumber()
    env.step(1)
    assert env.ale.getEpisodeFrameNumber() - frames0 == 4             # frame skip 4
    lives, life_losses, done = info["lives"], 0, False
    while not done:
        _, _, term, trunc, info = env.step(np.random.randint(4))
        life_losses += int(info["life_loss"])
        done = term or trunc
    assert term and life_losses == lives - 1                            # the last life ends the game instead
    env.close()


def test_snapshot_restore():
    env = AtariEnv("Pong", seed=1)
    env.reset()
    for _ in range(30):
        env.step(2)
    snap = env.get_snapshot()
    a = [env.step(k % 6) for k in range(40)]
    env.restore_snapshot(snap)
    b = [env.step(k % 6) for k in range(40)]
    assert all(np.array_equal(x[0]["image"], y[0]["image"]) and x[1] == y[1] for x, y in zip(a, b))
    env.close()


# --- Replay ---------------------------------------------------------------------------------------------------------
def test_replay_windows_stay_in_episode():
    rb = ReplayBuffer(capacity=600, num_streams=2, img_size=8, death_sample_prob=0.3, recent_sample_prob=0.2,
                      reward_sample_prob=0.1, loss_sample_prob=0.1)
    for s in range(2):
        for ep in range(6):
            n = 20 + 13 * ep
            img = np.zeros((n, 3, 8, 8), np.uint8)
            img[:, 0, 0, 0] = np.arange(n) % 256
            c = np.ones(n, np.float32)
            c[-1] = 0.0
            rb.add_rows(s, img, np.zeros(n), (np.arange(n) % 5 == 0).astype(np.float32), c, new_episode=True,
                        end_episode=True)
    for _ in range(20):
        b = rb.sample(8, 16)
        for i in range(8):
            v = int(b["valid"][i].sum())
            x = b["images"][i, :v, 0, 0, 0].long()
            assert torch.all(x[1:] - x[:-1] == 1)
            assert len(set(rb.episode_ids[b["index"][i, :v]].tolist())) == 1


# --- Posterior --------------------------------------------------------------------------------------------------
def test_batch_logit_norm():
    from gq_dreamer.model import BatchLogitNorm
    torch.manual_seed(0)
    bn = BatchLogitNorm(4, 8, scale=3.0)
    x = torch.randn(64, 4, 8) * 0.1 + torch.randn(1, 4, 8) * 5        # nearly frame-independent logits
    y = bn(x.requires_grad_())
    assert torch.allclose(y.mean(0), torch.zeros(4, 8), atol=1e-3)
    assert torch.allclose(y.std(0, unbiased=False), torch.full((4, 8), 3.0), atol=1e-2)
    assert bn.running_mean.abs().sum() > 0                             # running statistics were updated
    with torch.no_grad():                                              # acting: running statistics, per frame
        assert torch.allclose(bn(x[:1]), (x[:1] - bn.running_mean) / (bn.running_var + bn.eps).sqrt() * 3.0)


# --- Baselines ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method,mean,median", [("IRIS", 1.046, 0.289), ("STORM", 1.267, 0.584),
                                                ("TWM", 0.956, 0.505), ("DIAMOND", 1.459, None),
                                                ("BBF", 2.247, 0.917), ("SPR", 0.704, 0.415),
                                                ("DreamerV3 (2024)", 1.25, 0.49)])
def test_published_aggregates_reproduce(method, mean, median):
    a = aggregate_hns(baselines.method_scores(method))
    assert a["n_games"] == 26
    assert abs(a["mean"] - mean) < 0.006
    if median is not None:
        assert abs(a["median"] - median) < 0.006


def test_iqm_matches_scipy_definition():
    x = np.arange(26, dtype=float)
    assert iqm(x) == pytest.approx(x[6:20].mean())


def test_compare_subset():
    ours = {"Pong": [10.0, 12.0], "Breakout": [20.0, 22.0]}
    cmp = compare(ours, reps=100)
    assert cmp["games"] == ["Breakout", "Pong"]
    assert cmp["aggregates"]["DreamerV3"]["n_games"] == 2


# --- A tiny end-to-end run ------------------------------------------------------------------------------------------
def test_tiny_run(tmp_path):
    from gq_dreamer import run
    torch.manual_seed(0)
    tr = run("Breakout", 0, config="gq_debug", runs_root=str(tmp_path), env_steps=300, round_steps=150,
             final_eval_episodes=2, eval_envs=2, verbose=False, checkpoint_every=1, audit=True)
    d = os.path.join(str(tmp_path), "Breakout", "gq_debug_s0")
    res = json.load(open(os.path.join(d, "final_eval.json")))
    assert res["episodes"] == 2 and res["env_steps"] == 300
    assert tr.agent.total_num_updates > 0
    assert os.path.exists(os.path.join(d, "metrics.csv"))
    # resuming loads the final checkpoint and does not train again
    tr2 = run("Breakout", 0, config="gq_debug", runs_root=str(tmp_path), env_steps=300, round_steps=150,
              final_eval=False, verbose=False)
    assert tr2.agent.total_num_steps == 300 and tr2.agent.total_num_updates == tr.agent.total_num_updates
