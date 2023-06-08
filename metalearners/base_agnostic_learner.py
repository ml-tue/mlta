import pandas as pd

from metalearners.base_learner import BaseLearner


class BaseAgnosticLearner(BaseLearner):
    def __init__(self):
        super().__init__()

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
