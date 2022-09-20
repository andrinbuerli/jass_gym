from ray.tune import CLIReporter
from typing import List
import numpy as np
from ray.tune.trial import Trial


class OnEpisodeCLIReporter(CLIReporter):
    """
    CLI Reporter for better console logs with ray. Only report progress when there was an interation done.
    """

    def __init__(self, **kwargs):
        self.prev_iterations = None
        super().__init__(**kwargs)

    def should_report(self, trials: List[Trial], done: bool = False):
        """
        Only report progress when there is new trial results.

        :param trials: The ray trial objects
        :param done: Whether the trial is done.
        :return: Bool indicating if we should report.
        """
        new_iterations = [t.last_result.get("training_iteration", -1) for t in trials]

        if self.prev_iterations is None:
            self.prev_iterations = new_iterations
            return False

        res = not (np.array(new_iterations) == np.array(self.prev_iterations)).all()
        self.prev_iterations = new_iterations
        return res
