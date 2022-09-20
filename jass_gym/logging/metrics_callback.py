from ray.rllib.agents import DefaultCallbacks


class MetricsCallback(DefaultCallbacks):
    """
    Custom metrics can be logged to wandb using this callback.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._processed_agent_ids = []

    def on_episode_end(self, episode, **kwargs):
        """
        This is called when an episode ends or a horizon is reached. Then the stats of the episodes are aggregated
        and added to the custom metrics. The custom metrics are logged to wandb in the wandb logger.
        :param episode: The episode object
        :return: Nothing
        """

        episode_num_agents = 0
        custom_metrics = {}

        # Log agent-wise infos
        for agent, agent_info in episode._agent_to_last_info.items():
            self._processed_agent_ids.append(agent)
            episode_num_agents += 1
            for k, v in agent_info.items():
                if k.startswith('env_'):
                    continue
                if custom_metrics.get(k, None) is None:
                    custom_metrics[k] = v
                else:
                    custom_metrics[k] += v

        # Log env-wise infos
        agent_info = list(episode._agent_to_last_info.values())[0]
        episode_num_agents += 1
        for k, v in agent_info.items():
            if not k.startswith('env_'):
                continue
            if custom_metrics.get(k, None) is None:
                custom_metrics[k] = v
            else:
                custom_metrics[k] += v

        custom_metrics["env_num_agents"] = episode_num_agents

        for k, v in custom_metrics.items():
            episode.custom_metrics[k] = v
