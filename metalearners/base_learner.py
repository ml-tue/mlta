import warnings
from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from metadatabase import MetaDataBase


class BaseLearner:
    def __init__(self):
        self._top_configurations: List[Pipeline] = []  # ranked on expected performance
        self._configuration_scores: List[float] = []  # scores to rank configurations by

    def offline_phase(self, mdbase: MetaDataBase, **kwargs) -> None:
        """Performs offline computation for the `online_phase()` using the specified metadatabase.
        Gathered information can be stored in the object itself, to later be accessed in `online_phase()`
        May also take in additional kwargs to avoid recomputation, such as datasets meta-features.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        """
        raise NotImplementedError("Must be implemented by child class.")

    def online_phase(self, X: pd.DataFrame, y: pd.DataFrame, max_time: int, metric: str, n_jobs: int, total_n_configs: int) -> None:
        """Execute the meta-learning strategy, with the previous `offline_phase` knowledge,
        on the specified dataset (`X, `y``) within the specified time limit (`max_time`) in seconds.
        Should at least store the following: best `n_configs` solutions (sklearn.pipeline.Pipline) in self._top_configurations
        Note: avoid passing a dataset (`X`,`y`) which is also in `offline_phase()`'s metadatabase `mdbase`

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used during pipeline training and possible characterization and similarity methods.
        y: pd.Series,
            Targets that are used during pipeline training and possible characterization and similarity methods.
        max_time: int,
            The amount of time the online phase is allowed to take. Additionally, when evaluating the method,
                the evaluation method such as LOOCV should take care of time keeping as well.
                This parameter is provided because we allow  meta-learners altering their behavior accordingly.
        metric: str,
            metrics/scoring on which configurations are assessed
        n_jobs: int,
            the `n_jobs` the online phase can use in its computations, especially important for meta-learners
            that evaluate models because they could evaluate a lot more or less depending on this value.
        total_n_configs: integer,
            specifies the number of configurations that should be stored in self._top_configurations
            (ordered high-to-low by estimated performance)
        """
        raise NotImplementedError("Must be implemented by child class.")

    def clear_configurations(self):
        """clears the meta-learners memory concerning configurations and their scores"""
        self._top_configurations = []
        self._configuration_scores = []

    def get_number_of_configurations(self) -> int:
        "Returns an integer indicating the number of stored solutions, 0 if none are stored."
        return len(self._top_configurations)

    def get_top_configurations(self, n: Optional[int]) -> List[Pipeline]:
        """Get the top solutions stored curing the `online_phase`

        Arguments
        ---------
        n: integer,
            Optional: if default (`None`), then all top_configurations are returned,
            else the amount of solutions that should be returned.
        """
        if len(self._top_configurations) == 0:
            warnings.warn("Meta-Learner has no configurations to return")
        if n is None:
            return self._top_configurations
        return self._top_configurations[:n]

    def add_configuration(self, configuration: Pipeline, score: float, higher_is_better: bool = True) -> None:
        """Adds configuration `configuration` to self._top_configurations in approriate place, according to `higher_is_better`

        configuration: Pipeline:
            The pipeline configuration to add
        score: float,
            the score the `configuration` is estimated to have, to rank it by using `higher_is_better` in `self.top_configurations`
        higher_is_better: boolean
            True:  a higher `score` is better, False: a lower score is better
        """
        # simply add if first entry
        if len(self._configuration_scores) == 0 and len(self._top_configurations) == 0:
            self._top_configurations = [configuration]
            self._configuration_scores = [score]
            return

        added = False
        for i, config_score in enumerate(self._configuration_scores):
            if higher_is_better:
                if score > config_score:
                    self._top_configurations.insert(i, configuration)
                    self._configuration_scores.insert(i, score)
                    added = True
                    break
            else:  # lower is better
                if score < config_score:
                    self._top_configurations.insert(i, configuration)
                    self._configuration_scores.insert(i, score)
                    added = True
                    break
        if not added:  # add at end if necessary
            new_index = self.get_number_of_configurations() + 1
            self._top_configurations.insert(new_index, configuration)
            self._configuration_scores.insert(new_index, score)
