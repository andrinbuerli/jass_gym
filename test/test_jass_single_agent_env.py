import numpy as np
import pytest

from jass_gym.conv_observation_builder import ConvObservationBuilder
from jass_gym.jass_single_agent_env import SchieberJassSingleAgentEnv


def test_reset():
    env = SchieberJassSingleAgentEnv(ConvObservationBuilder())
    obs = env.reset()

    assert obs is not None


def test_seed():
    env = SchieberJassSingleAgentEnv(ConvObservationBuilder())
    env.seed(1)
    obs1 = env.reset()

    env.seed(1)
    obs2 = env.reset()

    assert (obs1["observations"] == obs2["observations"]).all()


def test_step_without_reset_exception():
    env = SchieberJassSingleAgentEnv(ConvObservationBuilder())
    with pytest.raises(ValueError):
        env.step(0)


def test_step():
    env = SchieberJassSingleAgentEnv(ConvObservationBuilder())
    env.reset()

    obs, reward, done, info = env.step(40)

    assert obs is not None
    assert reward == 0
    assert bool(done) is False


def test_episode():
    env = SchieberJassSingleAgentEnv(ConvObservationBuilder())
    obs = env.reset()

    done = False

    i = 0
    while not done:
        action = np.flatnonzero(obs["action_mask"])[0]
        obs, reward, done, info = env.step(action)
        i += 1

    assert 35 < i < 38

