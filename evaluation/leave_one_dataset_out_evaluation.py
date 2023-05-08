from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluation import BaseEvaluation
from metadatabase import MetaDataBase
from metalearners import BaseLearner


class LeaveOneDatasetOutEvaluation(BaseEvaluation):
    def __init__(
        self,
        validation_strategy: str = "holdout",
        test_size: Optional[float] = None,
        n_configs: int = 25,
        max_time: int = 300,
        metric: str = "neg_log_loss",
        n_jobs: int = 1,
        verbosity: int = 1,
    ) -> None:
        super().__init__(n_configs, max_time, metric, n_jobs, verbosity)
        """Initialize the LeaveOneDatasetOutEvaluation procedure, specifying its main static options
        
        Arguments
        ---------
        validation_strategy: str,
            Validation strategy used for each dataset, default is holdout with 80/20 train-test split.
        test_size: optional float,
            Only specifiy if `validation_strategy` is holdout: its portion of each dataset to be used as test set.
            If `validation_strategy` specified but not `test_size`, then default is 0.2
        n_configs: integer,
            the number of solutions that should be evaluated for the metalearner per dataset.
        max_time: int,
            time in seconds allowed for the metalearner's `online_phase()`
        metric: str,
            specify the metric to perform the LeaveOneDatasetOutEvaluation with, default is "neg_log_loss" 
        n_jobs: int,
            the number of threads the `metalearner` is allowed to use. Default is 1.
        verbosity: int,
            if set to 1, then prints out dataset_id when done with dataset
        """
        self._validation_strategy = validation_strategy
        self._test_size = test_size
        self._n_configs = n_configs

    def evaluate(
        self,
        mdbase: MetaDataBase,
        metalearner: BaseLearner,
        dataset_characterizations: Optional[List[Tuple[int, List[int | float]]]],
        config_characterizations: Optional[List[Tuple[int, List[int | float]]]],
    ) -> List[Tuple[int, List[float | None]]]:
        """Evaluates the metalearner using this evaluation method, potentially using pre-computed dataset and configuration characterizations.
        Should store the evaluation results in self._evaluation_results to avoid losing any results""

        Arguments
        ---------
        mdbase: MetaDataBase,
            the metadatabase to perform the LeaveOneDatasetOutEvaluation
        metalearner: Subclass of BaseStrategy,
            the meta-learning strategy that should be evaluated on the mdbase, should inherent from BaseStrategy
        dataset_characterizations: Optional[List[Tuple[int, List[int | float]]]]
            Pre-computed dataset characterizations, to avoid expensive re-computation per metalearner.
            A list of tuples, where each tuple represents a dataset characterization.
                The first element in the tuple refers to the dataset_id in mdbase,
                The second element is the purely numeric vector representing the dataset
        config_characterizations: Optional[List[Tuple[int, List[int | float | str]]]],
            Pre-computed configuration characterizations, to avoid expensive re-computation per metalearner.
            A list of tuples, where each tuple represents a configuration characterization.
                The first element in the tuple refers to the pipeline_id in `mdbase`,
                The second element is the vector representing the configuration (pipeline).

        Returns
        -------
        evaluation_results: List[Tuple[int, List[float | None]]],
            The LeaveOneDatasetOutEvaluation results for each dataset in `mdbase`, each tuple in the list refers to evaluation results for one dataset.
                The first entry refers to the dataset_id as in the `mdbase`.
                The second entry is a list of floats or None, with the evaluation_results according to `metric` for all `n_configs` configurations.
                    Each element is a config score, with `None` for a configuration which could not be evaluated properly on the test set.

        """
        evaluation_results = []

        dataset_ids = mdbase.list_datasets(by="id")
        for did in dataset_ids:
            df_X, df_y = mdbase.get_dataset(did, type="dataframe")  # get data for online_phase
            datasets_to_keep = mdbase.list_datasets(by="id")
            datasets_to_keep.remove(did)
            mdbase.partial_datasets_view(datasets_to_keep)  # remove current dataset from mdbase

            if self._validation_strategy == "holdout":
                if dataset_characterizations != None:
                    # remove characterization for left out dataset
                    selected_characterizations = []
                    for characterization in dataset_characterizations:
                        if characterization[0] != did:
                            selected_characterizations.append(characterization)
                    if config_characterizations != None:
                        metalearner.offline_phase(mdbase=mdbase, dataset_characterizations=selected_characterizations, config_characterizations=config_characterizations)
                    else:  # just dataset characterizations
                        metalearner.offline_phase(mdbase=mdbase, dataset_characterizations=selected_characterizations)
                else:  # compute characterizations in offline_phase for each dataset
                    metalearner.offline_phase(mdbase=mdbase)

                # perform online phase
                if self._test_size is None:
                    self._test_size = 0.2
                X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=self._test_size)
                metalearner.online_phase(X_train, y_train, max_time=self._max_time, metric=self._metric, n_jobs=self._n_jobs, total_n_configs=self._n_configs)

                if metalearner.get_number_of_configurations() == 0:  # nothing to add
                    evaluation_results.append((did, None))
                else:
                    # evaluate each of the configurations on the held out test set
                    dataset_eval_results = []
                    for configuration in metalearner.get_top_configurations(n=self._n_configs):
                        if configuration is None:
                            dataset_eval_results.append(None)
                        else:
                            try:
                                configuration.fit(X_train, y_train)
                            except ValueError as e:
                                if self._verbosity == 1:
                                    print("The configuration could not be fitted, hence added as `None` to results of dataset {}".format(did))
                                    dataset_eval_results.append(None)
                                    continue
                            performance: float = -np.inf  # stores performance according to metric
                            # TODO implement more options
                            if self._metric == "neg_log_loss":
                                y_pred = configuration.predict_proba(X_test)
                                labels = np.unique(LabelEncoder().fit_transform(df_y))
                                performance = float(-1 * log_loss(y_true=LabelEncoder().fit_transform(y_test), y_pred=y_pred, labels=labels))
                            dataset_eval_results.append(performance)
                    evaluation_results.append((did, dataset_eval_results))

            # TODO: implement k-fold
            # if k-fold cv:
            # get relevant dataset_characterizations

            # for each fold
            # perform the offline phase on the subset of the mdbase

            # perform the online phase on the subset of the mdbase within max_time

            # get the top solution

            # avg out the scores

            if self._verbosity == 1:
                print("Done with dataset with id: {}".format(did))
            metalearner.clear_configurations()  # reset the metalearner's memory for next dataset
            mdbase.restore_view()

        self._evaluation_results = evaluation_results
        return evaluation_results
