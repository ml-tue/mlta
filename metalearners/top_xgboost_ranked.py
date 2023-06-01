import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRanker

from configuration_characterization import BaseConfigurationCharacterization
from dataset_characterization import BaseDatasetCharacterization
from metadatabase import MetaDataBase
from metalearners import BaseCharacterizationLearner
from utilities import TimeoutException, time_limit


class TopXGBoostRanked(BaseCharacterizationLearner):
    """
    Meta-learning strategy following the approach in "RankML: a Meta Learning-Based Approach for Pre-Ranking Machine Learning Pipelines"
          by Laadan 2019, https://arxiv.org/abs/1911.00108
    It requires both a dataset and configuration characterization to use an XGBoost ranker to rank pipelines in the metadataset.
    In the learning to Rank framework the query is the dataset for which we want to find an appropriate pipeline.
    Unlike the previously mentioned work we do allow to choose the amount of candidate models, not just the entire metadataset.
    """

    def __init__(self, dataset_characterization_method: BaseDatasetCharacterization, configuration_characterization_method: BaseConfigurationCharacterization):
        """Initialize the Learner
        Arguments
        ---------
        dataset_characterization_method: class inherenting from BaseDatasetCharacterization,
            specifies the similarity measure that is used in similarity computations.
        """
        super().__init__()
        self._dataset_char_method = dataset_characterization_method
        self._config_char_method = configuration_characterization_method
        self._dataset_characterizations: Optional[List[Tuple[int, List[int | float]]]] = None
        self._config_characterizations: Optional[List[Tuple[int, List[int | float | str]]]] = None

    def offline_phase(
        self,
        mdbase: MetaDataBase,
        dataset_characterizations_name: str,
        configuration_characterizations_name: str,
        n_models: Optional[int] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.1,
        max_depth: int = 8,
        **kwargs,
    ) -> None:
        """This is the place to train the XGBoost Ranker model based on the information in the metadatabase.
            It uses the characterizations of the datasets and pipelines in the metadatabase: store them first!

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        dataset_characterizations_name: str,
            Dataset characterizations are extracted from `mdbase` using the name, so store them over there prior to usage.
        configuration_characterizations_name: Optional str,
            then configuration characterizations are extracted  from `mdbase` using that name.
            Otherwise the specified characteriation method `configuration_characterization_method` is called.
        n_models: Optional integer,
            If specified at most this amount of models are used per dataset in training the ranking models.
            Otherwise when default (`None`) all models are used which may be a lot and in turn causing longer runtime.
        n_estimators: integer,
            number of estimator used in XGBRanker, default set to value in RankML paper.
        learning_rate: float,
            The learning rate used in the XGBRanker, default set to value in RankML paper.
        max_depth: integer,
            max depth of estimators used in XGBRanker, default set to value in RankML paper.
        **kwargs are be passed to  xgboost.XGBRanker instance.
        """
        self._mdbase = mdbase
        self._dataset_characterizations_name = dataset_characterizations_name
        self._configuration_characterizations_name = configuration_characterizations_name

        if n_models is None:
            n_models = len(self._mdbase.get_df().index)  # n_models at most length of evaluations
        top_df = None
        for i, did in enumerate(mdbase.list_datasets(by="id")):
            did_top = mdbase.get_df(datasets=[did], top_solutions=(n_models, "neg_log_loss"))[["dataset_id", "pipeline_id", "score"]]
            # add a rank and the query id per dataset
            did_top["rank"] = [i + 1 for i in reversed(range(0, min(n_models, len(did_top.index))))]
            did_top["qid"] = did_top["dataset_id"]
            if i == 0:
                top_df = did_top
            else:
                top_df = pd.concat([top_df, did_top], ignore_index=True)

        # to create X: need to concatenate the dataset characterization with the pipeline characterization
        self._X = []
        self._y = []
        self._qid = []

        pipe_ids = []  # store to query for all at once
        dataset_ids = []  # store to query for all at once
        for i in range(0, len(top_df.index)):
            row = top_df.loc[i].tolist()
            dataset_ids.append(int(row[0]))
            pipe_ids.append(int(row[1]))
            self._y.append(int(row[3]))
            self._qid.append(int(row[4]))

        # get the corresponding dataset and pipeline characterizations
        dataset_characterizations = mdbase.get_dataset_characterizations(self._dataset_characterizations_name, dataset_ids)
        config_characterizations = mdbase.get_configuration_characterizations(self._configuration_characterizations_name, pipe_ids)

        # # get the datatypes for X, because config_char could be str too which xgboost must get as category
        self._dtypes = {}
        indx_counter = 0
        for i in range(0, len(dataset_characterizations[0][1])):
            self._dtypes[indx_counter] = "float"
            indx_counter += 1
        for val in enumerate(config_characterizations[0][1]):
            if isinstance(val[1], str):
                self._dtypes[indx_counter] = "category"
            else:
                self._dtypes[indx_counter] = "float"
            indx_counter += 1

        # finalize the input for fitting the model
        for d_char, p_char in zip(dataset_characterizations, config_characterizations):
            self._X.append((*d_char[1], *p_char[1]))
        self._X = pd.DataFrame(self._X).astype(self._dtypes)

        self._ranker = XGBRanker(objective="rank:pairwise", n_estimators=150, learning_rate=0.1, max_depth=8, tree_method="approx", enable_categorical=True, kwargs=kwargs)
        self._ranker.fit(X=self._X, y=self._y, qid=self._qid)

    def online_phase(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        max_time: int = 120,
        evaluate_recommendations: bool = False,
        metric: str = "neg_log_loss",
        n_jobs: int = 1,
        total_n_configs: int = 25,
        max_n_models: Optional[int] = None,
        verbosity: int = 1,
    ) -> None:
        """Use the offline phase learned XGBoost ranker together with a dataset characterization to recommend pipelines for dataset (`X`, `y`).
        The pipelines in the metadatabase that are ranked highest for the new task are recommended.
        If multiple pipelines are ranked equally than the pipelines with the highest performance on the previous task is chosen.
        The pipelines are stored ordered by their performance (high-to-low) if the recommendations are specified to be evaluated.

        Arguments
        ---------
        X: pd.DataFrame,
            Dataset's features that are used to create a dataset characterization for the model input.
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
            This score is used to to break ties (using the mdbase score) between equally ranked models.
        n_jobs: int,
            the `n_jobs` to use to predict the new in `sklearn.model_selection.cross_val_score()` if evaluate_recommendations is true
        total_n_configs: integer,
            specifies the number of configurations that should be stored in `self._top_configurations`
            (ordered high-to-low by estimated performance on `metric` if evaluate_recommendation is true).
            The learner stops adding and evaluating configurations to `self._top_configurations` if it has length `total_n_configs` or `max_time` has passed.
        verbosity: int,
            if default (`1`), then shows information on when it timed out with `max_time`
        max_n_models: Optional integer
            The maximal number of models per dataset that are used as candidate models (e.g. those that are ranked.)
            If not specified (`None`), then all pipelines in the metadatabase are used in creating the ranking.
            Make sure the max_n_models*n_datasets > total_n_configs, otherwise there are too little pipelines to recommend.
        """
        if not hasattr(self, "_ranker"):
            raise Warning("Did not find a trained ranker model. Make sure offline phase is ran before the online phase")

        try:
            with time_limit(max_time):
                dataset_characterization = self._dataset_char_method.compute(X, y)[0]
                # get the pipes to use as candidate models in the ranking
                if max_n_models is None:
                    pipe_ids = self._mdbase.list_pipelines(by="id")
                else:  # only at most max_n_models pipelines per dataset
                    pipe_ids = []
                    for did in self._mdbase.list_datasets(by="id"):
                        pipe_ids.extend(self._mdbase.get_df([did], top_solutions=(max_n_models, metric))["pipeline_id"].to_list())
                    # only have to rank each pipe_id once
                    pipe_ids = list(set(pipe_ids))

                # get the pipe characterizations and rank them
                pipe_chars = self._mdbase.get_configuration_characterizations(self._configuration_characterizations_name, pipe_ids)
                X_to_rank_ = []
                for pipe_char in pipe_chars:
                    X_to_rank_.append((*dataset_characterization, *pipe_char[1]))
                X_to_rank_ = pd.DataFrame(X_to_rank_).astype(self._dtypes)
                ranking_scores = self._ranker.predict(X_to_rank_)
                sorted_ranking_scores = sorted(ranking_scores, reverse=True)
                ranks = [sorted_ranking_scores.index(x) for x in ranking_scores]
                new_ranks = [-1] * len(ranks)  # stores the ranks but counting from 0 in steps of 1, e.g.: [0, 1, 2] instead of [0, 2, 2]
                for new_rank, rank in enumerate(set(ranks)):
                    # set all matching indices in new_ranks to the new_rank value
                    for i, val in enumerate(ranks):
                        if val == rank:
                            new_ranks[i] = new_rank

                cur_rank = 0
                max_rank = max(new_ranks)
                pipe_ids_to_recommend = []

                # get which pipes to add
                while len(pipe_ids_to_recommend) < total_n_configs and cur_rank <= max_rank:
                    pipe_ids_with_cur_rank = list(set([pipe_ids[i] for i, val in enumerate(new_ranks) if val is cur_rank]))  # avoid recommending the same pipe more than once
                    n_pipes_to_add = total_n_configs - len(pipe_ids_to_recommend)

                    if len(pipe_ids_with_cur_rank) <= n_pipes_to_add:  # all can be added
                        pipe_ids_to_recommend.extend(pipe_ids_with_cur_rank)
                    else:  # tie needs to be broken, select the ones with lowest average deviation to optimal mdbase score by `metric`
                        # first calculate the scores for the pipes
                        pipe_scores = []
                        df_all = self._mdbase.get_df()
                        for pipe_id in pipe_ids_with_cur_rank:
                            pipe_df = df_all[df_all["pipeline_id"] == pipe_id]
                            pipe_scores = []
                            total_metric_diff = 0
                            for i in range(0, len(pipe_df.index)):
                                did = int(pipe_df.iloc[i]["dataset_id"])
                                pipe_score = float(pipe_df.iloc[i]["score"])
                                df_all_best_score = df_all[df_all["metric"] == metric]
                                df_all_best_score.sort_values(by="score", ascending=False, inplace=True)  # all possible metrics are scores (greater is always better)
                                best_score_did = float(df_all_best_score[:1]["score"])
                                total_metric_diff += abs(pipe_score - best_score_did)
                            pipe_scores.append(float(total_metric_diff / len(pipe_df.index)))
                        # select the ones with the minimal scores
                        inds_added = []
                        for _ in range(0, n_pipes_to_add):  # for the amount of pipes needed
                            # find the minimal score diff, such that it has not been added yet
                            cur_min_diff = 1_000_000_000
                            cur_min_indx = -1
                            for indx, score in enumerate(pipe_scores):
                                if indx not in inds_added:
                                    if score < cur_min_diff:
                                        cur_min_diff = score
                                        cur_min_indx = indx
                            inds_added.append(cur_min_indx)
                            pipe_ids_to_recommend.append(pipe_ids_with_cur_rank[cur_min_indx])
                    cur_rank += 1

                for pipe_id in pipe_ids_to_recommend:
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
