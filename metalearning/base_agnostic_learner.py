from metadatabase import MetaDataBase
from metalearning.base_learner import BaseLearner


class BaseCharacterizationLearner:
    def __init__(self):
        super.__init__()

    def online_phase(self, max_time: int, metric: str, n_jobs: int) -> None:
        """Execute the meta-learning strategy, with the previous `offline_phase` knowledge,
        but without specification of dataset, within the specified time limit (`max_time`) in seconds.
        Should at least store the following: best solutions (sklearn.pipeline.Pipline) in self._top_configurations

        Arguments
        ---------
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
