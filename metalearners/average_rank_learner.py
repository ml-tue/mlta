from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from metadatabase import MetaDataBase
from metalearners.base_agnostic_learner import BaseAgnosticLearner
from utilities import TimeoutException, time_limit


class AverageRankLearner(BaseAgnosticLearner):
    def __init__(self, verbosity: int = 1):
        """Initialize the dataset-agnostic average rank meta-learner. The method computes the average rank
        for all pipelines in the metadataset, based on their normalized regret, in the offline phase, and ranks approaches accordingly.
        In the online phase the best-ranked pipelines, with at least a specified amount of evaluations can be recommended.
        It deos not tailor the online phase to the dataset at hand, making it a dataset-agnostic method.

        Arguments
        ---------
        verbosity: int,
            Set to 1 or larger to get feedback. If smaller no progress is shown.
        """
        super().__init__()
        self._verbosity = verbosity

    def offline_phase(self, mdbase: MetaDataBase, metric: str = "neg_log_loss", higher_is_better: bool = True) -> None:
        """Performs offline computation (e.g. average rank computation) using the specified metadatabase.
        Stores average normalized regret per pipeline, on on how many evaluations this is based.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        metric: string,
            the metric by which the ranking is created from the metadataset.
        higher_is_better: boolean,
            whether or not a higher value of `metric `is better, needed to compute utility.
        """
        self._mdbase = mdbase

        # first process all of the scores such that they are normalized regret between 0 and 1, with 0 being best.
        pipelines_to_ranking_scores = {}
        for pipe_id in mdbase.list_pipelines(by="id"):
            pipelines_to_ranking_scores[pipe_id] = []
        for did in mdbase.list_datasets(by="id"):
            did_df = mdbase.get_df(datasets=[did], metrics=[metric])
            max_score = max(did_df["score"])
            min_score = min(did_df["score"])
            for entry in did_df.iterrows():
                row_values = entry[1]
                pipe_id = int(row_values[1])
                raw_score = float(row_values[2])
                ranking_score = abs(max_score - raw_score) / abs(min_score - max_score)  # normalized distance to highest result for this dataset
                if higher_is_better:
                    ranking_score = 1 - ranking_score  # if higher metric is better, then a lower ranking_score is better (cuz min distance to highest score)
                pipelines_to_ranking_scores[pipe_id].extend([ranking_score])

        self._pipeline_avgrank = {}
        self._pipeline_avgrank_counts = {}
        for pipe_key, ranking_list in zip(pipelines_to_ranking_scores.keys(), pipelines_to_ranking_scores.values()):
            ranking_score = 0  # 0 is good initialization, because score is non-negative
            if len(ranking_list) != 0:
                ranking_score = sum(ranking_list) / len(ranking_list)
            self._pipeline_avgrank[pipe_key] = ranking_score
            self._pipeline_avgrank_counts[pipe_key] = len(ranking_list)

    def online_phase(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        max_time: int,
        evaluate_recommendations: bool = False,
        metric: str = "neg_log_loss",
        n_jobs: int = 1,
        total_n_configs: int = 25,
        min_evals: int = 10,
    ) -> None:
        """Execute the meta-learning strategy, i.e. recommend the pipelines from the average ranking. But at most `total_n_configs`.
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
        evaluate_recommendations: boolean,
            whether or not the pipeline is evaluated on (X,y) before including it (in `self._top_configurations`).
            If the pipeline are evaluated, then they are ordered by their rank in `self._top_configurations`.
        metric: str,
            metrics/scoring on which configurations are assessed
        n_jobs: int,
            the `n_jobs` the online phase can use in its computations, especially important for meta-learners
            that evaluate models because they could evaluate a lot more or less depending on this value.
        total_n_configs: integer,
            specifies the number of configurations that should be stored in self._top_configurations
            (ordered high-to-low by estimated performance)
        min_evals: integer,
            the minimal number of evaluations a pipe should have before its average regret is considered for the recommendation of pipelines.
        """

        # compute which pipes should be recommended
        pipes_with_min_evals = [pipe for pipe in self._pipeline_avgrank_counts if self._pipeline_avgrank_counts[pipe] >= min_evals]
        pipe_rankingscores_min_evals = {pipe: ranking_score for pipe, ranking_score in self._pipeline_avgrank.items() if pipe in pipes_with_min_evals}
        pipe_ranking = sorted(pipe_rankingscores_min_evals, key=lambda x: pipe_rankingscores_min_evals[x], reverse=True)
        pipes_to_recommend = pipe_ranking[:total_n_configs]

        # store recommendations
        try:
            with time_limit(max_time):
                for pipe_id in pipes_to_recommend:
                    pipe = self._mdbase.get_sklearn_pipeline(pipe_id, X, y, True, True)
                    # Must expect not all pipelines may work, e.g. feature selectors may remove all features
                    # therefore try fitting pipe, if it does not work do not consider it, fill it in with while loop later
                    if evaluate_recommendations:
                        try:
                            score = float(np.mean(cross_val_score(pipe, X, y, cv=n_jobs, scoring=metric, n_jobs=n_jobs)))
                            self.add_configuration(pipe, score, higher_is_better=True)
                        except ValueError as e:
                            if self._verbosity >= 1:
                                print("pipeline with id {} failed to fit, do not consider it".format(pipe_id))
                    else:
                        score = None  # did not evaluate
                        self.add_configuration(pipe, score, higher_is_better=True)

        except TimeoutException as e:
            if self._verbosity >= 1:
                print("online_phase timed out with {} seconds".format(max_time))
