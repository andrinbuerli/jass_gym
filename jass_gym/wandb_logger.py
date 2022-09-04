import multiprocessing
import numbers
from datetime import datetime
from typing import Dict

import wandb
from ray.tune.logger import LoggerCallback
from ray.tune.trial import Trial
from ray.tune.utils import flatten_dict

ignored_keys = ["callbacks", "policies", "policy_mapping_fn"]


class WandbLoggerCallback(LoggerCallback):
    """
    Log ray trial results and custom metrics to weights and biases.
    """

    def __init__(self, entity, project, group=None, api_key_file=None):
        self.group = group
        self.run = None
        self.metrics_queue_dict = {}
        self.project = project
        self.entity = entity
        with open(api_key_file, "r") as f:
            api_key = f.read().strip()
        wandb.login(key=api_key, relogin=True)

    def log_trial_start(self, trial: "Trial"):
        """Handle logging when a trial starts.
        :param: trial (Trial): Trial object.
        """
        if trial.trial_id not in self.metrics_queue_dict:
            print("=" * 50)
            print("Setting up new w&b logger")
            print("Experiment tag:", trial.experiment_tag)
            print("Experiment id:", trial.trial_id)
            print("=" * 50)

            queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=self._logger_process,
                args=(
                    trial.trainable_name,
                    queue,
                    trial.config,
                    self.project,
                    self.entity,
                ),
            )
            p.start()
            self.metrics_queue_dict[trial.experiment_tag] = queue

    def log_trial_restore(self, trial: "Trial"):
        """
        Handle logging when a trial restores.

        :param: trial (Trial): Trial object.
        """
        pass

    def log_trial_save(self, trial: "Trial"):
        """Handle logging when a trial saves a checkpoint.

        :param: trial (Trial): Trial object.
        """
        pass

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        """Handle logging when a trial reports a result.

        :param: trial (Trial): Trial object.
        :param: result (dict): Result dictionary.
        """
        queue = self.metrics_queue_dict[trial.experiment_tag]
        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]

        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, numbers.Number):
                continue
            metrics[key] = value

        queue.put(metrics)

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        """Handle logging when a trial ends.

        :param: trial (Trial): Trial object.
        :param: failed (bool): True if the Trial finished gracefully, False if
                it failed (e.g. when it raised an exception).
        """
        self._close()

    def _close(self):
        # kills logger processes
        for queue in self.metrics_queue_dict.values():
            metrics = {"KILL": True}
            queue.put(metrics)
        wandb.join()

    @staticmethod
    def _logger_process(run: str, queue: multiprocessing.Queue, config: dict, project: str, entity: str):
        """
        Each logger has to run in a separate process
        :param queue: the queue object containing the log values to be written
        :param config: the configuration info for the run
        :return:
        """
        run_name = "_".join([run, config["env"]]) + "_" + datetime.now().strftime("%m-%d-%Y%_H-%M-%S")
        run = wandb.init(
            entity=entity, reinit=True, name=run_name, project=project, **config.get("env_config", {}).get("wandb", {})
        )

        if config:
            for k in config.keys():
                if k not in ignored_keys:
                    if wandb.config.get(k) is None:
                        try:
                            wandb.config[k] = config[k]
                        except:
                            pass

            if "yaml_config" in config["env_config"]:
                yaml_config = config["env_config"]["yaml_config"]
                print("Saving full experiment config:", yaml_config)
                try:
                    wandb.save(yaml_config)
                except Exception as e:
                    print(e)

        while True:
            metrics = queue.get()

            if "KILL" in metrics:
                break

            run.log(metrics)
