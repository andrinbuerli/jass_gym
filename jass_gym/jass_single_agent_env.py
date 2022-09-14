import logging

import jasscpp
import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.game.const import ACTION_SET_FULL_SIZE, team

from ray.tune.registry import register_env

from jass_gym.conv_observation_builder import ObservationBuilder, ConvObservationBuilder


class SchieberJassSingleAgentEnv(gym.Env):

    def __init__(self, observation_builder: ObservationBuilder):
        self.observation_builder = observation_builder
        self._logger = logging.getLogger(__name__)

        self.action_space = Discrete(ACTION_SET_FULL_SIZE)
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(ACTION_SET_FULL_SIZE, )),
            "observations": Box(0, 1, shape=observation_builder.shape),
        })

        self._dealing_card_strategy = DealingCardRandomStrategy()

        # the current game that is being played
        self._game = jasscpp.GameSimCpp()  # schieber rule is default
        self._rule = jasscpp.RuleSchieberCpp()  # schieber rule is default

        self.rng = np.random.default_rng()

        self.cum_reward = 0
        self.cum_reward_team = np.zeros(2)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        if self._game.state.hands.sum() < 0:
            raise ValueError("Env needs to be reset initially")

        prev_points = np.copy(self._game.state.points)
        self._game.perform_action_full(action)
        rewards = self._game.state.points - prev_points
        next_player = self._game.state.player
        if next_player > 0:
            next_team = team[next_player]
            reward = rewards[next_team]
        else:
            reward = rewards.max()  # if last card in last trick
        done = self._game.state.nr_played_cards == 36
        obs = self._get_observation(done)

        self.cum_reward += reward
        self.cum_reward_team += rewards

        return obs, reward, done, {
            "cum_reward": self.cum_reward,
            "cum_reward_team": self.cum_reward_team
        }

    def _get_observation(self, done=False):
        obs = self.observation_builder(jasscpp.observation_from_state(self._game.state, -1)) if not done\
            else np.zeros(self.observation_space["observations"].shape)
        mask = self._rule.get_full_valid_actions_from_state(self._game.state) if not done\
            else np.zeros(self.observation_space["action_mask"].shape)
        return {
            "action_mask": mask,
            "observations": obs
        }

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        dealer = self.rng.choice([0, 1, 2, 3])
        self._game.init_from_cards(dealer=dealer, hands=self._dealing_card_strategy.deal_cards())

        obs = self._get_observation()

        return obs

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.rng = np.random.default_rng(seed=seed)
        np.random.seed(seed=seed)


def env_creator(env_config):
    return SchieberJassSingleAgentEnv(observation_builder=ConvObservationBuilder())


register_env("schieber_jass_env", env_creator)
