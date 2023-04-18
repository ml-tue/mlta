import pandas as pd
from sklearn.pipeline import Pipeline
from metadatabase import MetaDataBase
from typing import List

class BaseLearner:
    def __init__(self):
        self._top_solutions: List[Pipeline] = None

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

    def online_phase(self, 
                     X: pd.DataFrame,
                     y: pd.Series, 
                     max_time: int,
                     metric: str,
                     n_jobs: int) -> None:
        """Execute the meta-learning strategy, with the previous `offline_phase` knowledge, 
        on the specified dataset (`new_task`) within the specified time limit (`max_time`) in seconds.
        Should at least store the following: best solution(sklearn.pipeline.Pipline) in self._top_solution

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used in the meta-feature computation and pipeline training
        y: pd.Series,
            Targets that are used in the meta-feature computation and pipeline training
        max_time: int,
            The amount of time the online phase is allowed to take. Additionally, when evaluating the method,
                the evaluation method such as LOOCV will take care of time keeping as well.
                This specific metalearning strategy does not use the available time in its strategy.
        metric: str,
            metrics/ or scoring by which to select the top evaluations from the most similar task
        n_jobs: int,
            the `n_jobs` to use in `sklearn.model_selection.cross_val_score()`
        """
        raise NotImplementedError("Must be implemented by child class.")

    def get_top_solutions(self) -> Pipeline:
        """Get the top solutions stored curing the `online_phase` """
        return self._top_solutions