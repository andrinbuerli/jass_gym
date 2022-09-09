import logging
from typing import Tuple

import jasscpp
import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
from jass.arena.dealing_card_random_strategy import DealingCardRandomStrategy
from jass.game.const import ACTION_SET_FULL_SIZE
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from ray.tune.registry import register_env

from jass_gym.conv_observation_builder import ObservationBuilder, ConvObservationBuilder


class SchieberJassMultiAgentEnv(MultiAgentEnv):

    def __init__(self, observation_builder: ObservationBuilder):
        self.observation_builder = observation_builder
        self._logger = logging.getLogger(__name__)

        self.action_space = Discrete(ACTION_SET_FULL_SIZE)
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(ACTION_SET_FULL_SIZE,)),
            "observations": Box(0, 1, shape=observation_builder.shape),
        })

        self._dealing_card_strategy = DealingCardRandomStrategy()

        # the current game that is being played
        self._game = jasscpp.GameSimCpp()  # schieber rule is default
        self._rule = jasscpp.RuleSchieberCpp()  # schieber rule is default

        self.rng = np.random.default_rng()

        self.prev_points = np.zeros((4, 2))

    def step(self, action_dict: MultiAgentDict) \
            -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns:
            Tuple[dict, dict, dict, dict]: Tuple with 1) new observations for
                each ready agent, 2) reward values for each ready agent. If
                the episode is just started, the value will be None.
                3) Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
                4) Optional info values for each agent id.
        """

        if self._game.state.hands.sum() < 0:
            raise ValueError("Env needs to be reset initially")

        if len(action_dict) != 1 or self._game.state.player not in action_dict:
            raise ValueError(f"Only current_player {self._game.state.player} can select action")

        action = action_dict[self._game.state.player]

        self._game.perform_action_full(action)
        all_done = self._game.state.nr_played_cards == 36

        obs = self._get_observation(all_done)

        next_player = self._game.state.player
        observations = {next_player: obs} \
            if not all_done else {p: obs for p in range(4)}
        rewards = {next_player: self._get_reward(next_player)} \
            if not all_done else {p: self._get_reward(p) for p in range(4)}
        dones = {next_player: False, "__all__": False} \
            if not all_done else {**{p: True for p in range(4)}, "__all__": True}
        infos = {next_player: self._get_infos()} \
            if not all_done else {p: self._get_infos() for p in range(4)}

        return observations, rewards, dones, infos

    def _get_infos(self):
        return {
            "env_max_points": max(self._game.state.points),
            "env_cards_played": self._game.state.nr_played_cards,
            "env_cards_in_trick": self._game.state.nr_cards_in_trick,
            "env_forehand": self._game.state.forehand,
            "env_trump": self._game.state.trump,
            "env_dealer": self._game.state.dealer,
        }

    def _get_reward(self, current_player):
        team = current_player % 2
        rewards = self._game.state.points - self.prev_points[current_player]
        reward = rewards[team]
        self.prev_points[current_player] = np.copy(self._game.state.points)
        return reward

    def _get_observation(self, done=False):
        obs = self.observation_builder(jasscpp.observation_from_state(self._game.state, -1)) if not done \
            else np.zeros(self.observation_space["observations"].shape)
        mask = self._rule.get_full_valid_actions_from_state(self._game.state) if not done \
            else np.zeros(self.observation_space["action_mask"].shape)
        return {
            "action_mask": mask,
            "observations": obs
        }

    def reset(self) -> MultiAgentDict:
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """

        dealer = self.rng.choice([0, 1, 2, 3])
        self._game.init_from_cards(dealer=dealer, hands=self._dealing_card_strategy.deal_cards())

        obs = self._get_observation()

        return {
            self._game.state.player: obs
        }

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
    return SchieberJassMultiAgentEnv(observation_builder=ConvObservationBuilder())


register_env("schieber_jass_multi_agent_env", env_creator)
