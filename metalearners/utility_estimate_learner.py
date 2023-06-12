import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from dataset_similarity import BaseSimilarityMeasure, CharacterizationSimilarity
from metadatabase import MetaDataBase
from metalearners import BaseSimilarityLearner
from utilities import TimeoutException, time_limit

# Meta-learning strategy simply employing direct configuration transfer on the configurations it deems to have most utility,
# It computes the utility by including the score of the configuration on a dataset and the similarty thereto
# It uses a similaritymeasure implementing BaseSimilarityMeasure, for instance by CharacterizationSimilarity with WistubaMetaFeatures.


class UtilityEstimateLearner(BaseSimilarityLearner):
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
        higher_is_better: bool = True,
        n_jobs: int = 1,
        total_n_configs: int = 25,
        verbosity: int = 1,
    ) -> None:
        """Execute the meta-learner "UtilityEstimate", possibly with the previous `offline_phase` knowledge.
        It simply employs direct configuration transfer on the configurations it deems to have most utility,
        It computes the utility by including the score of the configuration on a dataset and the similarty thereto.

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
            metrics/ or scoring by which for which the utility should be computed.
        higher_is_better: boolean,
            whether or not a higher value of `metric `is better, needed to compute utility.
        n_jobs: int,
            the `n_jobs` to use in `sklearn.model_selection.cross_val_score()`
        total_n_configs: integer,
            specifies the number of configurations that should be stored in `self._top_configurations`
            (ordered high-to-low by estimated performance on `metric`). The learner stops adding and evaluating
            configurations to `self._top_configurations` if it has length `total_n_configs` or `max_time` has passed.
        verbosity: int,
            if default (`1`), then shows information on when it timed out with `max_time`
        """

        try:
            with time_limit(max_time):
                # find most similar datasets
                dataset_to_similarity = {}
                if isinstance(self._similarity_measure, CharacterizationSimilarity):
                    # compute characterization for the new task's dataset
                    new_characterization = self._similarity_measure._characterization_method.compute(X, y)
                    for dataset_entry in self._characterizations:
                        similarity = self._similarity_measure.compute_from_characterizations(new_characterization[0], dataset_entry[1])  # type: ignore
                        dataset_to_similarity[dataset_entry[0]] = similarity
                else:  # non-characterization based
                    for dataset_id in self._mdbase.list_datasets(by="id"):
                        df_X, df_y = self._mdbase.get_dataset(dataset_id, type="dataframe")
                        similarity = self._similarity_measure.compute(X, y, df_X, df_y)  # type: ignore
                        dataset_to_similarity[dataset_id] = similarity

                # create utility scores for the pipelines, aggregate multiple pipeline evaluations on different datasets.
                # but should also take into account that the scores should actually be normalized distance to minimum, per dataset.
                pipeline_to_utilityscore_list = {}  # initialize utility score storage (per pipeline)
                for pipe_id in self._mdbase.list_pipelines(by="id"):
                    pipeline_to_utilityscore_list[pipe_id] = []

                for did, similarity in zip(list(dataset_to_similarity.keys()), list(dataset_to_similarity.values())):
                    did_df = self._mdbase.get_df(datasets=[did], metrics=[metric])
                    max_score = max(did_df["score"])
                    min_score = min(did_df["score"])

                    for entry in did_df.iterrows():
                        row_values = entry[1]
                        pipe_id = int(row_values[1])
                        raw_score = float(row_values[2])
                        score = abs(max_score - raw_score) / abs(min_score - max_score)  # normalized distance to highest result for this dataset
                        if higher_is_better:
                            utility_score = similarity * (1 - score)  # if higher metric is better, then a lower score is better (cuz min distance to highest score)
                        else:
                            utility_score = similarity * score
                        pipeline_to_utilityscore_list[pipe_id].extend([utility_score])

                # rank pipes by highest average utility scores
                pipeline_to_utilityscore = {}
                for d_key, u_list in zip(pipeline_to_utilityscore_list.keys(), pipeline_to_utilityscore_list.values()):
                    utility_score = 0  # 0 is good initialization, because score is non-negative
                    if len(u_list) != 0:
                        utility_score = sum(u_list) / len(u_list)
                    pipeline_to_utilityscore[d_key] = utility_score
                ranked_pipelines = sorted(pipeline_to_utilityscore, key=lambda x: pipeline_to_utilityscore[x], reverse=True)[:total_n_configs]

                for pipe_id in ranked_pipelines:
                    pipe = self._mdbase.get_sklearn_pipeline(pipe_id, X, y, True, True)
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
