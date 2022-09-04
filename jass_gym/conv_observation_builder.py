from abc import abstractmethod

import jasscpp
import numpy as np
from jass.game.const import next_player, team


class ObservationBuilder:

    shape = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ConvObservationBuilder(ObservationBuilder):
    """
    FeaturesSet set for convolutional neural networks with all data in 4x9xK format, with K channels at the end.
    The set is intended for both trump and card play
    """
    FEATURE_LENGTH = 1620             # type: int

    shape = (4, 9, 45)

    # The features are organized as a [4,9,45] matrix, the following constants can be used to access the 45 channels

    CH_CARDS_PLAYER_0  = 0
    CH_CARDS_PLAYER_1  = 1
    CH_CARDS_PLAYER_2  = 2
    CH_CARDS_PLAYER_3  = 3

    CH_CARDS_IN_TRICK  = 4             # 9 Tricks
    CH_CARDS_IN_POSITION = 13          # 4 Positions

    CH_CARDS_TRICK_CURRENT = 17

    CH_HAND           = 18            # 1 Hands
    CH_CARDS_VALID     = 19

    CH_DEALER          = 20
    CH_DECLARE_TRUMP   = 24
    CH_PLAYER          = 28
    CH_TRUMP           = 32            # 6 possibilities
    CH_FOREHAND        = 38            # store forehand as one hot encoded for 3 values with -1, 0, 1
    CH_POINTS_OWN      = 41
    CH_POINTS_OPP      = 42
    CH_TRUMP_VALID     = 43
    CH_PUSH_VALID      = 44

    def __init__(self):
        super().__init__()
        self._rule = jasscpp.RuleSchieberCpp()

    def __call__(self, obs: jasscpp.GameObservationCpp) -> np.ndarray:
        """
        Convert the obs to a feature vector. For convolutional networks, the set will contain the channels
        at the end, so the format will be 36 x K (or 4 x 9 x K)
        Args:
            obs : observation to convert
            rule: rule for calculating the valid cards
        """
        # convert played cards in tricks to several one hot encoded array:
        #  - who played the card (36x4)
        #  - which trick was it played in (36x9)
        #  - which position was it played in the trick (36x4)

        cards_played_by_player = np.zeros([36, 4], dtype=np.float32)
        cards_played_in_trick_number = np.zeros([36, 9], dtype=np.float32)
        cards_played_in_position = np.zeros([36, 4], dtype=np.float32)

        # we go through the one more than the number of tricks to include the current trick
        for trick_id in range(obs.current_trick + 1):
            player = obs.trick_first_player[trick_id]
            for i in range(4):
                card = obs.tricks[trick_id, i]
                if card != -1:
                    cards_played_by_player[card, player] = 1.0
                    cards_played_in_trick_number[card, trick_id] = 1.0
                    cards_played_in_position[card, i] = 1.0
                    player = next_player[player]
        # 612 elements, total 612

        # cards played in the last trick, the information about the position and who played them are already present
        # in the arrays above, so we just mark the cards of the current trick
        current_trick = np.minimum(obs.current_trick, 8)  # could be 9 for last state
        current_trick = obs.tricks[current_trick]
        cards_of_current_trick = np.zeros([36, 1], dtype=np.float32)

        if obs.nr_cards_in_trick > 0:
            cards_of_current_trick[current_trick[0], 0] = 1.0
        if obs.nr_cards_in_trick > 1:
            cards_of_current_trick[current_trick[1], 0] = 1.0
        if obs.nr_cards_in_trick > 2:
            cards_of_current_trick[current_trick[2], 0] = 1.0
        # 36 elements, total 648

        hand = obs.hand.astype(np.float32).reshape(-1, 1)
        # 36 elements, total 684

        # we use 3 planes for the valid actions, one for the cards and one for trump and one for push,
        # however we use the trump layers to the end.
        valid_actions = self._rule.get_valid_cards_from_obs(obs).astype(np.float32)
        valid_cards = np.zeros([36,1], dtype=np.float32)
        valid_cards[:, 0] = valid_actions[0:36]

        # 36 elements, total 720

        # the additional information is added as planes, one-hot encoded
        dealer = np.zeros([36, 4], dtype=np.float32)
        dealer[:, obs.dealer] = 1.0
        # 144 elements, total 864

        # if trump was not declared yet, we use a zero vector
        declare_trump = np.zeros([36, 4], dtype=np.float32)
        if obs.declared_trump_player != -1:
            declare_trump[:, obs.declared_trump_player] = 1.0
        # 144 elements, total 1008

        # player to play
        player = np.zeros([36, 4], dtype=np.float32)
        player[:, obs.player] = 1.0
        # 144 elements, total 1152

        # trump selected
        trump = np.zeros([36, 6], dtype=np.float32)
        if obs.trump != -1:
            trump[:, obs.trump] = 1.0
        # 216 elements, total 1368

        # store forehand as one hot encoded for 3 values with -1, 0, 1 set as the first, second or third entry
        forehand = np.zeros([36, 3], dtype=np.float32)
        forehand[:, obs.forehand + 1] = 1.0
        # 3*36 element, total 1404

        # we omit nr of trick and nr of cards played here

        team_player = team[obs.player]
        points_own = np.full([36, 1], fill_value=obs.points[team_player] / 157.0, dtype=np.float32)
        points_opponent = np.full([36, 1], fill_value=obs.points[1 - team_player] / 157.0, dtype=np.float32)
        # 72 elements, total 1548

        # select trump
        trump_valid = np.zeros([36,1], dtype=np.float32)
        if obs.trump == -1:
            trump_valid.fill(1.0)
        push_valid = np.zeros([36, 1], dtype=np.float32)
        if obs.trump == -1 and obs.forehand == -1:
            push_valid.fill(1.0)

        features = np.concatenate([cards_played_by_player,
                                   cards_played_in_trick_number,
                                   cards_played_in_position,
                                   cards_of_current_trick,
                                   hand,
                                   valid_cards,
                                   dealer,
                                   declare_trump,
                                   player,
                                   trump,
                                   forehand,
                                   points_own,
                                   points_opponent,
                                   trump_valid,
                                   push_valid], axis=1)

        return np.reshape(features, self.shape)
