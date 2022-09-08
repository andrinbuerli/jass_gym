import numpy as np
import pytest

from jass_gym.conv_observation_builder import ConvObservationBuilder
from jass_gym.jass_multi_agent_env import SchieberJassMultiAgentEnv


def test_reset():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    obs = env.reset()

    assert obs is not None


def test_seed():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    env.seed(1)
    obs1 = env.reset()

    env.seed(1)
    obs2 = env.reset()

    assert (list(obs1.values())[0]["observations"] == list(obs2.values())[0]["observations"]).all()


def test_step_without_reset_exception():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    env.seed(1)
    with pytest.raises(ValueError):
        env.step({0: 0})


def test_step_wrong_player_exception():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    env.seed(1)
    env.reset()
    with pytest.raises(ValueError):
        env.step({1: 0})


def test_step():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    env.seed(1)
    env.reset()

    player = env._game.state.player
    obs, reward, done, info = env.step({player: 40})

    next_player = env._game.state.player
    assert obs[next_player] is not None
    assert reward[next_player] == 0
    assert bool(done[next_player]) is False


def test_episode():
    env = SchieberJassMultiAgentEnv(ConvObservationBuilder())
    obs = env.reset()

    done = {"__all__": False}

    i = 0
    data = []
    while not done["__all__"]:
        player = env._game.state.player
        action = np.flatnonzero(obs[player]["action_mask"])[0]
        obs, reward, done, info = env.step({player: action})
        data.append((obs, reward, done, info))
        i += 1

    assert 35 < i < 38

