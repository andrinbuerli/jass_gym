from typing import Dict, List

import gym
import torch
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.utils.typing import ModelConfigDict, TensorType


class ActionMaskingVisionNet(DQNTorchModel, TorchModelV2):

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, *args, **kwargs):
        self.orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(self.orig_space, gym.spaces.Dict) and \
               "action_mask" in self.orig_space.spaces and \
               "observations" in self.orig_space.spaces

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, *args, **kwargs)

        self.internal_model = VisionNetwork(
            self.orig_space["observations"], action_space, num_outputs,
            model_config, name + "_internal")

        self.inf_action_mask = None

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out (Tensor): Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """

        action_scores = self.advantage_module(model_out)
        # compute masked logits
        masked_action_scores = action_scores + self.inf_action_mask

        if self.num_atoms > 1:
            raise ValueError("Distributional Q-learning is not supported with this action masking model!")
        else:
            logits = torch.unsqueeze(torch.ones_like(masked_action_scores), -1)
            return masked_action_scores, logits, logits

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        # Compute the unmasked logits.
        logits, _ = self.internal_model({
            "obs": input_dict["obs"]["observations"]
        })

        # Generate inf mask, if all inf, set mask to zeros to avoid nans in output
        self.inf_action_mask = torch.clamp(torch.log(input_dict["obs"]["action_mask"]), min=float("-inf"))

        if torch.all(torch.eq(self.inf_action_mask, float("-inf"))):
            self.inf_action_mask = torch.zeros_like(self.inf_action_mask)

        return logits, state

    def inference(self, obs: TensorType, actionmask: TensorType):
        """
        Method for export

        :param obs: batched observation
        :param actionmask: batched actionmask
        :return: computed Q values
        """

        # Compute the unmasked logits.
        logits, _ = self.internal_model({
            "obs": obs
        })

        inf_action_mask = torch.clamp(torch.log(actionmask), min=float("-inf"))

        action_scores = self.advantage_module(logits)
        # compute masked logits
        masked_action_scores = action_scores + inf_action_mask
        return masked_action_scores

    def value_function(self):
        return self.internal_model.value_function()

    def export(self, model_path):
        traced = torch.jit.trace_module(
            self.to("cpu"),
            {"inference": (torch.rand(1, *self.orig_space["observations"].shape), torch.rand(1, self.action_space.n))})
        traced.save(model_path)


ModelCatalog.register_custom_model("action_masking_visionnet", ActionMaskingVisionNet)
