import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from metadatabase import MetaDataBase
from metalearning.base_learner import BaseLearner
from task_similarity import get_wistuba_metafeatures
from utilities import TimeoutException, time_limit


# baseline meta-learning strategy simply getting the top 10 solutions using Wistuba meta-features,
# evaluating each, returning which is best.
class WistubaTop10Strategy(BaseLearner):
    def __init__(self):
        self._top_solution: Pipeline = None
        self._best_score: float = -100000  # negative value because all metrics are actually scores (e.g. higher=better)

    def offline_phase(self, mdbase: MetaDataBase, **kwargs) -> None:
        """Performs offline meta-feature computation for the `online_phase()` using the specified metadatabase.
        Takes in additional kwargs to avoid recomputation for dataset meta-features.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.

        **kwargs
            if keyword dataset_characterization present, then meta-features are extracted therefrom
            dataset_characterization: List[Tuple[int, List[int | float], List[str]]]
                A list of tuples, where each tuple represents a dataset characterization.
                The first element in the tuple refers to the dataset_id in mdbase,
                The second element is the purely numeric vector representing the dataset,
                The last element is a list of equal size to the second element, specifying its names.
        """
        self._mdbase = mdbase

        if "dataset_characterization" in kwargs:
            meta_features = kwargs["dataset_characterization"]
            for i, characterization in enumerate(meta_features):
                if characterization[0] not in mdbase.list_datasets(by="id"):
                    del meta_features[i]

            self._meta_features = meta_features
        else:
            # compute and store meta-features
            meta_features = []
            for dataset in os.listdir(mdbase._datasets_dir):
                dataset_id = int(dataset.split(".")[0])
                if dataset_id in mdbase.list_datasets(by="id"):  # avoid computing meta-features on new task/dataset
                    dataset_path = os.path.join(mdbase._datasets_dir, dataset)
                    feature_values, feature_names = get_wistuba_metafeatures(dataset_path)
                    meta_features.append((dataset_id, feature_values, feature_names))

            self._meta_features = meta_features

    def online_phase(self, X: pd.DataFrame, y: pd.Series, max_time: int = 120, metric: str = "neg_log_loss", n_jobs: int = 1, verbosity: int = 1) -> None:
        """Execute the meta-learning strategy, with the previous `offline_phase` knowledge,
        on the specified dataset (`new_task`) within the specified time limit (`max_time`) in seconds.
        Strategy: get the top 10 solutions using Wistuba meta-features, evaluating each, returning which is best.
        New task should not be the entire dataset, even for meta-features.

        Stores the best solution (sklearn.pipeline.Pipeline) in self._top_solution

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
        verbosity: int,
            if default (1), then shows information on when it timed out with `max_time`

        """
        try:
            with time_limit(max_time):
                # compute meta-features for a part of the new task's dataset as given by X, y arrays
                feature_values, _ = get_wistuba_metafeatures(X.to_numpy(), y.to_numpy())

                # find most similar dataset
                max_similarity = 0
                most_similar_dataset_id = -1
                for meta_features in self._meta_features:
                    similarity = 1 - spatial.distance.cosine(feature_values, meta_features[1])
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_dataset_id = meta_features[0]

                top_solutions_id = list(self._mdbase.get_df(datasets=[int(most_similar_dataset_id)], top_solutions=(10, metric))["pipeline_id"])
                for id in top_solutions_id:
                    pipe = self._mdbase.get_sklearn_pipeline(id, X, y, True)

                    # we should expect that not all pipelines may work,
                    # for instance some feature selectors may remove all features
                    # therefore try fitting it, if it does not work, then set score very low
                    try:
                        score = np.mean(cross_val_score(pipe, X, y, scoring=metric, n_jobs=n_jobs))
                        if score > self._best_score:
                            self._best_score = score
                            self._top_solution = pipe
                    except ValueError as e:
                        if verbosity == 1:
                            print("pipeline with id {} failed to fit, do not consider it".format(id))

        except TimeoutException as e:
            if verbosity == 1:
                print("online_phase timed out with {} seconds".format(max_time))
