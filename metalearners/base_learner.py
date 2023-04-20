from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from metadatabase import MetaDataBase


class BaseLearner:
    def __init__(self):
        self._top_configurations: Optional[List[Pipeline]] = None  # if applicable: high-to-low ranked on expected performance

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

    def online_phase(self, X: pd.DataFrame, y: pd.Series, max_time: int, metric: str, n_jobs: int) -> None:
        """Execute the meta-learning strategy, with the previous `offline_phase` knowledge,
        on the specified dataset (`new_task`) within the specified time limit (`max_time`) in seconds.
        Should at least store the following: best solutions (sklearn.pipeline.Pipline) in self._top_configurations

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
        """
        raise NotImplementedError("Must be implemented by child class.")

    def get_top_configurations(self) -> List[Pipeline]:
        """Get the top solutions stored curing the `online_phase`"""
        if self._top_configurations is None:
            raise Warning("Meta-Learner has no configurations to return")
        return self._top_configurations
