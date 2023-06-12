import datetime
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from metadatabase import MetaDataBase
from metalearners.base_agnostic_learner import BaseAgnosticLearner
from metalearners.base_learner import BaseLearner
from utilities import TimeoutException, get_fixed_preprocessed_data, sklearn_pipe_to_individual_str, time_limit


class PortfolioBuilding(BaseAgnosticLearner):
    def __init__(self, verbosity: int = 1):
        """Initialize the greedy porfolio building meta-learning approach introduced in:
            "Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning" by Feurer et al. (2020)
                See: https://arxiv.org/pdf/2007.04074.pdf

        Arguments
        ---------
        verbosity: int,
            Set to 1 or larger to get feedback on result matrix computation and portfolio building progress.
            If smaller no progress is shown.
        """
        super().__init__()
        self._verbosity = verbosity

    def offline_phase(self, mdbase: MetaDataBase, portfolio_size: int = 25, results_matrix: Optional[List[Tuple[int, List[int | float], List[str]]]] = None) -> None:
        """Performs offline computation (e.g. portfolio construction) using the specified metadatabase.
        Stores the portfolio in as the top configurations in their respective (best-performing) order

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        portfolio_size: integer,
            The size of the portfolio that should be constructed and stored in the meta-learner.
        results_matrix: Optional[List[Tuple[int, List[int | float], List[str]]]]
            Shaped to be consistent with dataset_characterizations such that mdbase functionality can be used for it.
            Each tuple in the list stores a tuple of the dataset_id, the pipe performances on it as a list, and the pipeline ids as a list of strings.
            It thus requires the column/featurenames in using the dataset characterization functionality in mdbase.
            The portfolio is built on this results matrix, it functions without results for all dataset in the mdbase.
        """
        self._mdbase = mdbase
        if results_matrix is None:
            self._results_matrix = self.compute_results_matrix()
        else:
            self._results_matrix = results_matrix

        results_pipe_ids = [int(pipe_id) for pipe_id in self._results_matrix[0][2]]
        dataset_ids = []  # only consider those in the results_matrix to build the portfolio.
        did_to_performances = {}
        for result_entry in self._results_matrix:
            did_to_performances[result_entry[0]] = result_entry[1]
            dataset_ids.append(result_entry[0])

        if portfolio_size > len(results_pipe_ids):
            raise ValueError(f"Cannot build a portfolio of portfolio size: {portfolio_size}, it is larger than the number of candidate pipelines: {len(results_pipe_ids)}.")

        self._portfolio_ids = set()  # stores pipe_ids that are included in portfolio
        dataset_to_best_score = {}  # stores the normalized average distance to mimimum, so no longer a real score; smaller is better now
        for did in dataset_ids:
            dataset_to_best_score[did] = 1

        candidate_pipe_ids = deepcopy(results_pipe_ids)
        while len(self._portfolio_ids) < portfolio_size:
            cur_generalization_err = sum(dataset_to_best_score.values())
            best_pipe = candidate_pipe_ids[0]  # stores info for finding pipe to add, init to first pipe in case no pipe improves the generalization err
            best_generalization_err = cur_generalization_err
            best_dataset_to_best_score = dataset_to_best_score
            for pipe_id in candidate_pipe_ids:  # for each pipeline see how much it changes the current portfolio's generalization error
                pipe_dataset_to_best_score = deepcopy(dataset_to_best_score)
                for did in dataset_ids:  # check for each dataset whether its score can be improved by current pipe
                    pipe_score = did_to_performances[did][results_pipe_ids.index(pipe_id)]
                    pipe_dataset_to_best_score[did] = min(pipe_dataset_to_best_score[did], pipe_score)
                pipe_generalization_err = sum(pipe_dataset_to_best_score.values())
                if pipe_generalization_err < cur_generalization_err:
                    best_pipe = pipe_id
                    best_generalization_err = pipe_generalization_err
                    best_dataset_to_best_score = pipe_dataset_to_best_score

            # enlarge portfolio / bookkeeping next selection
            self._portfolio_ids.add(best_pipe)
            candidate_pipe_ids.remove(best_pipe)
            dataset_to_best_score = best_dataset_to_best_score

            if self._verbosity >= 1:
                print(f"reached a portfolio size of {len(self._portfolio_ids)}")

    def online_phase(
        self, X: pd.DataFrame, y: pd.DataFrame, max_time: int, evaluate_recommendations: bool = False, metric: str = "neg_log_loss", n_jobs: int = 1, total_n_configs: int = 25
    ) -> None:
        """Execute the meta-learning strategy, i.e. recommend the pipelines from the portfolio. But at most `total_n_configs`.
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
        """

        try:
            with time_limit(max_time):
                for pipe_id in self._portfolio_ids:
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

    def compute_results_matrix(self, n_pipes: int = 1, metric: str = "neg_log_loss", n_folds: int = 3, eval_max_time: Optional[int] = 300, dataset_ids: Optional[List[int]] = None):
        """Computes the results matrix, recording the performance for the best `n_pipes` per dataset w.r.t `metric` on all other datasets in the mdbase.
        Be aware running this takes a long time, see `eval_max_time` parameter for more details.

        Arguments
        ---------
        n_pipes: integer,
            The amount of pipes to consider per dataset in the mdbase. Affects the resulting dimensionality and time complexity with a linear factor.
        metric: string,
            which metric wihtin the mdbase to select the best pipeline on, default is negative logistic loss.
        n_folds: integer,
            the amount of folds used in the cross-validation to score the pipeline on a dataset.
            Automatically assigns the same number of jobs to it. Thus assumes a sufficiently powerful machine.
        eval_max_time: integer or None,
            Optional argument to specify the maximum time (in seconds) 1 cross-validation procedure can take (for 1 dataset and pipeline).
            Thus with a default of 600 seconds, creating a portfolio of size 25 could using OpenML18CC (n=72) could still take
                72^2*10 minutes = 36 days, in other words, be mindful of running this function.
        dataset_ids: optional list of ints,
            optional list of integers representing dataset ids as in the mdbase, if not specified results are computed for all datasets in mdbase.
            This selection does not include the selection of the candidate pipelines, only which results are generated (e.g. to do batch runs.)

        Returns
        -------
        results_matrix: List[Tuple[int, List[int | float], List[str]]],
            The output is shaped to be consistent with dataset_characterizations such that mdbase functionality can be used for it.
            Each tuple in the list stores a tuple of the dataset_id, the pipe performances on it as a list, and the pipeline ids as a list of strings.
        """
        # get the pipeline id's, discard duplicate pipelines (datasets may have the same high-perfoming configurations)
        pipe_ids = []
        if dataset_ids is None:
            dataset_ids = self._mdbase.list_datasets(by="id")
        for did in self._mdbase.list_datasets(by="id"):
            top_pipe_ids = self._mdbase.get_df(datasets=[did], top_solutions=(n_pipes, metric))["pipeline_id"].to_list()
            pipe_ids.extend(top_pipe_ids)
        pipe_ids = set(pipe_ids)  # only consider each pipeline once

        results_matrix = []
        # fit the pipes on the mdbase datasets
        for did in dataset_ids:
            X, y = self._mdbase.get_dataset(did)
            raw_pipe_performances = [None] * len(pipe_ids)  # set to None values in case fits fail
            for i, pipe_id in enumerate(pipe_ids):
                pipe = self._mdbase.get_sklearn_pipeline(pipeline_id=pipe_id, X=X, y=y, is_classification=True, include_prepro=True)

                # skip evaluations with pipelines having knn and polynomial features if data is too large
                exclude_knn = False
                exclude_polynomial_features = False
                X_ = get_fixed_preprocessed_data(X)
                if X_.shape[0] * X_.shape[1] > 6_000_000:
                    exclude_knn = True
                if X_.shape[1] > 50:
                    exclude_polynomial_features = True
                ind_str = sklearn_pipe_to_individual_str(pipe)
                if exclude_knn and "KNeighborsClassifier" in ind_str:
                    if self._verbosity >= 1:
                        print(f"Skipped pipeline with id {pipe_id}, because it contains KNN and data is too large")
                    continue  # can continue because default raw_pipe_performances set to None
                if exclude_polynomial_features and "PolynomialFeatures" in ind_str:
                    if self._verbosity >= 1:
                        print(f"Skipped pipeline with id {pipe_id}, because it contains PolynomialFeatures and data is too large")
                    continue
                if self._verbosity >= 1:
                    print(f"Start fitting pipeline with id:{pipe_id} at: {datetime.datetime.now()}")
                try:
                    try:  #  nested try for time_limit if necessary
                        with time_limit(eval_max_time):
                            # limit cross_val_score to only 1 job, otherwise memory issues may occur and the timer does not work properly
                            score = float(np.mean(cross_val_score(pipe, X, y, cv=n_folds, scoring=metric, n_jobs=1, pre_dispatch="1*n_jobs")))
                            raw_pipe_performances[i] = score
                    except TimeoutException as e:
                        if self._verbosity >= 1:
                            print("Evaluating configuration timed out with {} seconds".format(eval_max_time))
                except ValueError as e:
                    if self._verbosity >= 1:
                        print("pipeline with id {} failed to fit, do not consider it".format(pipe_id))
                except MemoryError as me:
                    if self._verbosity >= 1:
                        print("The configuration could not be fitted due to a memory error. Hence added as `None` to results for dataset {}".format(did))

            # process results to get average distance to the minimum (w.r.t dataset), have the failed pipes (None's) set to 1
            max_score = max([val for val in raw_pipe_performances if val is not None])
            min_score = min([val for val in raw_pipe_performances if val is not None])
            pipe_performances = []
            for raw_perf in raw_pipe_performances:
                if raw_perf is None:
                    pipe_performances.append(1)
                else:
                    avg_dist_to_min = abs(max_score - raw_perf) / abs(min_score - max_score)  # the naming is confusing, because its not to minimum but the best score
                    pipe_performances.append(avg_dist_to_min)
            results_matrix.append((did, (pipe_performances, [str(pipe_id) for pipe_id in pipe_ids])))
            if self._verbosity >= 1:
                print(f"Done with dataset with id:{did} at: {datetime.datetime.now()}")
                print("results_matrix: ", results_matrix)

        return results_matrix
