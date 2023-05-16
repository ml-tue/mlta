import os
from typing import List
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from dataset_similarity import BaseSimilarityMeasure, CharacterizationSimilarity
from metadatabase import MetaDataBase
from metalearners import BaseSimilarityLearner
from utilities import TimeoutException, time_limit

# Meta-learning strategy simply getting the top solutions using a similaritymeasure implementing
# BaseSimilarityMeasure, for instance by CharacterizationSimilarity with WistubaMetaFeatures. Tries to evaluate
# the top similar dataset's characterizations and orders them by their evaluation results on the new dataset.


class TopSimilarityLearner(BaseSimilarityLearner):
    def __init__(self, similarity_measure: BaseSimilarityMeasure):
        """Similarity_measure: class inherenting from BaseSimilarityMeasure,
        specifies the similarity measure that is used in similarity computations.
        """
        super().__init__()
        self._similarity_measure = similarity_measure

    def offline_phase(self, mdbase: MetaDataBase, **kwargs) -> None:
        """Typically similarity-measures employing characterizations need to perform computation in the `offline_phase`
        Hence this method could compute mdbase's characterizations for `online_phase()`.
        Takes in additional kwarg(`dataset_characterizations`) to avoid recomputation for dataset characterization.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        **kwargs
            if keyword dataset_characterizations present, then meta-features are extracted therefrom
            dataset_characterizations: List[Tuple[int, List[int | float]]]
                A list of tuples, where each tuple represents a dataset characterization.
                The first element in the tuple refers to the dataset_id in mdbase,
                The second element is the purely numeric vector representing the dataset,
        """
        self._mdbase = mdbase
        if isinstance(self._similarity_measure, CharacterizationSimilarity):
            if "dataset_characterizations" in kwargs:
                characterizations = kwargs["dataset_characterizations"]
                for i, characterization in enumerate(characterizations):
                    if characterization[0] not in mdbase.list_datasets(by="id"):  # only characterizations for mdbase datasets
                        del characterizations[i]
                self._characterizations = characterizations
            else:  # compute and store characterizations for all datasets in mdbase
                characterizations = []
                for dataset in os.listdir(mdbase._datasets_dir):
                    dataset_id = int(dataset.split(".")[0])
                    df_X, df_y = mdbase.get_dataset(dataset_id=dataset_id, type="dataframe")
                    characterization = self._similarity_measure._characterization_method.compute(df_X, df_y)  # type: ignore
                    characterizations.append((dataset_id, characterization))
                self._characterizations = characterizations
        else:  # not characterization-based
            if "dataset_characterizations" in kwargs:
                raise ValueError("Cannot use characterization with a similarity measure not using characterizations")

    def online_phase(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        max_time: int = 120,
        evaluate_recommendations: bool = False,
        metric: str = "neg_log_loss",
        n_jobs: int = 1,
        total_n_configs: int = 25,
        n_datasets: int = 5,
        verbosity: int = 1,
    ) -> None:
        """Execute the meta-learner "TopSimilarity" strategy, possibly with the previous `offline_phase` knowledge.
        The learner simply gets the most similar datasets to the newly specified dataset (`X`, `y`),
        to directly transfer its best-performing configurations to the new task. It evaluates them, one-by-one
        on the new dataset, and stores them if they succesively on the new data, ordered by their performance (high-to-low)
        in self._top_configurations. It does this within the specified time limit (`max_time`) in seconds.

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
        evaluate_recommendations: boolean,
            whether or not the pipeline is evaluated on (X,y) before including it (in `self._top_configurations`).
            If the pipeline are evaluated, then they are ordered by their rank in `self._top_configurations`.
        metric: str,
            metrics/ or scoring by which to select the top evaluations from the most similar task
        n_jobs: int,
            the `n_jobs` to use in `sklearn.model_selection.cross_val_score()`
        total_n_configs: integer,
            specifies the number of configurations that should be stored in `self._top_configurations`
            (ordered high-to-low by estimated performance on `metric`). The learner stops adding and evaluating
            configurations to `self._top_configurations` if it has length `total_n_configs` or `max_time` has passed.
        n_datasets: integer,
            the number of (most similar) datasets that should be considered, the learner takes a uniform
            number of configs per dataset, in other words, the number of configs per dataset is `total_n_configs`/ `n_datasets`
        verbosity: int,
            if default (`1`), then shows information on when it timed out with `max_time`

        """
        try:
            with time_limit(max_time):
                # find most similar datasets
                similar_entries: List = []
                if isinstance(self._similarity_measure, CharacterizationSimilarity):
                    # compute characterization for the new task's dataset
                    new_characterization = self._similarity_measure._characterization_method.compute(X, y)
                    # order entries in mdbase by similarity
                    for dataset_entry in self._characterizations:
                        similarity = self._similarity_measure.compute_from_characterizations(new_characterization[0], dataset_entry[1])  # type: ignore
                        similar_entries.append((dataset_entry[0], similarity))
                    similar_entries = sorted(similar_entries, key=lambda tup: tup[1], reverse=True)
                else:  # non-characterization based
                    for dataset_id in self._mdbase.list_datasets(by="id"):
                        df_X, df_y = self._mdbase.get_dataset(dataset_id, type="dataframe")
                        similarity = self._similarity_measure.compute(X, y, df_X, df_y)  # type: ignore
                        similar_entries.append((dataset_id, similarity))
                most_similar_dataset_ids = [entry[0] for entry in similar_entries[:n_datasets]]

                # evaluate most similar datasets top configurations
                configs_per_dataset = int(total_n_configs / n_datasets)
                pipeline_offset = 0  # number of pipelines to offset per dataset, used in case some cannot be evaluated
                while self.get_number_of_configurations() < total_n_configs:
                    for dataset_id in most_similar_dataset_ids:
                        top_solution_ids = list(self._mdbase.get_df(datasets=[int(dataset_id)], top_solutions=(configs_per_dataset + pipeline_offset, metric))["pipeline_id"])
                        top_solution_ids = top_solution_ids[pipeline_offset : pipeline_offset + configs_per_dataset]  # select approriate pipelines
                        for pipe_id in top_solution_ids:
                            if self.get_number_of_configurations() >= total_n_configs:
                                continue
                            pipe = self._mdbase.get_sklearn_pipeline(pipe_id, X, y, True)
                            # Must expect not all pipelines may work, e.g. feature selectors may remove all features
                            # therefore try fitting pipe, if it does not work do not consider it, fill it in with while loop later
                            if evaluate_recommendations:
                                try:
                                    score = float(np.mean(cross_val_score(pipe, X, y, cv=n_jobs, scoring=metric, n_jobs=n_jobs)))
                                    self.add_configuration(pipe, score, higher_is_better=True)
                                except ValueError as e:
                                    if verbosity == 1:
                                        print("pipeline with id {} failed to fit, do not consider it".format(pipe_id))
                            else:
                                score = None  # did not evaluate
                                self.add_configuration(pipe, score, higher_is_better=True)

        except TimeoutException as e:
            if verbosity == 1:
                print("online_phase timed out with {} seconds".format(max_time))
