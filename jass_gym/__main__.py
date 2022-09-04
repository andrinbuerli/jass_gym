import argparse
import collections
import logging
import os

from typing import Callable

import jass_gym
import yaml

import ray
import ray.tune

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
        parser.add_argument("--log", help="Log data to wandb", action='store_true')

        self.args, self.unknown_args = parser.parse_known_args()

    def __call__(self):
        with open(self.args.file, "r") as f:
            experiment = yaml.safe_load(f)

        loggers = [
            WandbLoggerCallback(
                project="jass-gym",
                entity="andrinburli",
                api_key_file=os.path.join(os.path.dirname(__file__), "../.wandbkey"),
            )
        ] if self.args.log else []

        reporter = OnEpisodeCLIReporter(metric_columns=self.CLI_METRICS)

        ray.init(local_mode=self.args.local)
        ray.tune.run_experiments(experiment, verbose=True, callbacks=loggers, progress_reporter=reporter)
        ray.shutdown()


def main():
    cli = SchieberJassGymCli()
    cli()


if __name__ == "__main__":
    main()
