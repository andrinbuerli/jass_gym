import argparse
import collections
import logging
import os
import re
from pathlib import Path

from typing import Callable

import yaml

import ray
import ray.tune
from ray.rllib.agents import Trainer
from ray.tune.registry import get_trainable_cls

# noinspection PyUnresolvedReferences
import jass_gym
from jass_gym.metrics_callback import MetricsCallback
from jass_gym.progress_reporter import OnEpisodeCLIReporter
from jass_gym.wandb_logger import WandbLoggerCallback

from ray.tune.result import (
    TRAINING_ITERATION,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
    EPISODE_REWARD_MEAN,
)


class SchieberJassGymCli(Callable):
    """
    Entrypoint for the sjgym CLI
    """

    CLI_METRICS = collections.OrderedDict(
        {
            TRAINING_ITERATION: "iter",
            TIME_TOTAL_S: "total time (s)",
            TIMESTEPS_TOTAL: "ts",
            EPISODE_REWARD_MEAN: "return_mean",
            "episode_reward_min": "return_min",
            "episode_reward_max": "return_max",
            "episodes_total": "episodes_total",
            "episode_len_mean": "episode_len_mean",
        }
    )

    def __init__(self):
        logging.basicConfig(level=logging.ERROR)
        this_dir, this_filename = os.path.split(__file__)
        package_path = os.path.join(this_dir, "..")
        os.chdir(package_path)

        parser = argparse.ArgumentParser(description="Run a rllib experiment with the Schieber Jass Environment")

        parser.add_argument("--file", help="File with experiment config", type=str)
        parser.add_argument("--local", help="Run ray in local mode", action='store_true')
        parser.add_argument("--latest_checkpoint", help="checkpoint to restore", default=None)
        parser.add_argument("--export", help="export checkpoint", action='store_true')
        parser.add_argument("--log", help="Log data to wandb", action='store_true')
        self.args, self.unknown_args = parser.parse_known_args()

    def __call__(self):
        with open(self.args.file, "r") as f:
            experiment = yaml.safe_load(f)

        if self.args.export:
            agent: Trainer = self.agent_from_experiment(experiment)
            if self.args.latest_checkpoint is not None:
                folder = max([x for x in Path(self.args.latest_checkpoint).iterdir() if x.is_dir()], key=os.path.getmtime)
                checkpoint = folder / re.sub("_0*", "-", folder.name)
                print(f"restoring checkpoint at {checkpoint}")
                agent.restore(str(checkpoint))
            agent.get_policy().model.export("model.pt")
        else:
            loggers = [
                WandbLoggerCallback(
                    api_key_file=os.path.join(os.path.dirname(__file__), "../.wandbkey"),
                )
            ] if self.args.log else []

            list(experiment.values())[0]["config"]["callbacks"] = MetricsCallback

            reporter = OnEpisodeCLIReporter(metric_columns=self.CLI_METRICS)

            ray.init(local_mode=self.args.local)
            ray.tune.run_experiments(experiment, verbose=True, callbacks=loggers, progress_reporter=reporter)
            ray.shutdown()

    @staticmethod
    def agent_from_experiment(experiment_config) -> Trainer:
        experiment_config = experiment_config[list(experiment_config.keys())[0]]
        trainer_cls = get_trainable_cls(experiment_config["run"])
        agent: Trainer = trainer_cls(config=experiment_config["config"], env=experiment_config["env"])
        return agent


def main():
    cli = SchieberJassGymCli()
    cli()


if __name__ == "__main__":
    main()
