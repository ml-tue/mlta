from datetime import datetime
from inspect import getfullargspec
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from evaluation import BaseEvaluation
from metadatabase import MetaDataBase
from metalearners import BaseLearner
from utilities import TimeoutException, time_limit


class LeaveOneDatasetOutEvaluation(BaseEvaluation):
    def __init__(
        self,
        validation_strategy: str = "holdout",
        test_size: Optional[float] = None,
        n_configs: int = 25,
        max_time: int = 300,
        metric: str = "neg_log_loss",
        n_jobs: int = 1,
        verbosity: int = 2,
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
            If set to 2, then also prints out when done with the online and offline phase
        """
        self._validation_strategy = validation_strategy
        self._test_size = test_size
        self._n_configs = n_configs

    def evaluate(
        self,
        mdbase: MetaDataBase,
        metalearner: BaseLearner,
        dataset_characterizations: Optional[List[Tuple[int, List[int | float]]]] = None,
        config_characterizations: Optional[List[Tuple[int, List[int | float]]]] = None,
        dataset_ids: Optional[List[int]] = None,
        max_time: Optional[int] = None,
        **metalearner_kwargs,
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
        dataset_ids: list of integers,
            Optional specification of dataset_ids to evaluate. Still all datasets can be used from mdbase in the online phase.
            But only the specified datasets ids are evaluated and returned.
            Intended use is for creating batches or when a run was interrupted midway and must restart but without re-doing evaluations.
        max_time: integer,
            Optional argument to specify the maximum time (in seconds) 1 configuration evaluation can take.
            Thus with a default of 600 seconds, a total of 25 configurations may still take at most 250 minutes per dataset.
        metalearner_kwargs:
            If a keyword in metalearner_kwargs is present in the offline phase it is passed there, and similar for the online phase.

        Returns
        -------
        evaluation_results: List[Tuple[int, List[float | None]]],
            The LeaveOneDatasetOutEvaluation results for each dataset in `mdbase`, each tuple in the list refers to evaluation results for one dataset.
                The first entry refers to the dataset_id as in the `mdbase`.
                The second entry is a list of floats or None, with the evaluation_results according to `metric` for all `n_configs` configurations.
                    Each element is a config score, with `None` for a configuration which could not be evaluated properly on the test set.

        """
        evaluation_results = []
        # get (possible) metalearner kwargs for offline and online phase
        online_phase_kwargs = {}
        offline_phase_kwargs = {}
        if metalearner_kwargs is not None:
            online_args = getfullargspec(metalearner.online_phase).args
            offline_args = getfullargspec(metalearner.offline_phase).args
            for param in metalearner_kwargs:
                if param in online_args:
                    online_phase_kwargs[param] = metalearner_kwargs[param]
                if param in offline_args:
                    offline_phase_kwargs[param] = metalearner_kwargs[param]

        if dataset_ids is None:
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
                        metalearner.offline_phase(
                            mdbase=mdbase, dataset_characterizations=selected_characterizations, config_characterizations=config_characterizations, **offline_phase_kwargs
                        )
                    else:  # just dataset characterizations
                        metalearner.offline_phase(mdbase=mdbase, dataset_characterizations=selected_characterizations, **offline_phase_kwargs)
                else:  # likely compute characterizations in offline_phase for each dataset
                    metalearner.offline_phase(mdbase=mdbase, **offline_phase_kwargs)
                if self._verbosity >= 2:
                    print("Done with offline phase of dataset with id: {} at {}".format(did, str(datetime.now())))

                # perform online phase
                if self._test_size is None:
                    self._test_size = 0.2
                X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=self._test_size)
                metalearner.online_phase(
                    X_train, y_train, max_time=self._max_time, metric=self._metric, n_jobs=self._n_jobs, total_n_configs=self._n_configs, **online_phase_kwargs
                )
                if self._verbosity >= 2:
                    print("Done with online phase of dataset with id: {} at {}".format(did, str(datetime.now())))

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
                                if time_limit is not None:
                                    try:  #  nested try for time_limit if necessary
                                        with time_limit(max_time):
                                            configuration.fit(X_train, y_train)  # error MemoryError, should catch that too
                                    except TimeoutException as e:
                                        if self._verbosity >= 1:
                                            print("Evaluating configuration timed out with {} seconds".format(max_time))
                                        continue
                                else:
                                    configuration.fit(X_train, y_train)
                            except ValueError as e:
                                if self._verbosity >= 1:
                                    print("The configuration could not be fitted, incompatible pipeline. Hence added as `None` to results of dataset {}".format(did))
                                dataset_eval_results.append(None)
                                continue
                            except MemoryError as m:
                                if self._verbosity >= 1:
                                    print("The configuration could not be fitted due to a memory error. Hence added as `None` to results of dataset {}".format(did))
                                dataset_eval_results.append(None)
                                continue
                            except LinAlgError as le:
                                if self._verbosity >= 1:
                                    print("The configuration could not be fitted due to LinAlgError. Hence added as `None` to results of dataset {}".format(did))
                                dataset_eval_results.append(None)
                                continue
                            performance: float = -np.inf  # stores performance according to metric
                            # TODO implement more options
                            if self._metric == "neg_log_loss":
                                try:
                                    y_pred = configuration.predict_proba(X_test)
                                except AttributeError as e:  # can be the case when not training a classifier to predict probabilities
                                    if self._verbosity >= 1:
                                        print("Could not predict probability due to attribute error in pipeline")
                                except ValueError as ve:
                                    if self._verbosity >= 1:
                                        print("Could not predict probability due to ValuerError when applying the pipeline to the new test data{}".format(did))
                                    dataset_eval_results.append(None)
                                    continue
                                labels = np.unique(LabelEncoder().fit_transform(df_y))
                                performance = float(-1 * log_loss(y_true=LabelEncoder().fit_transform(y_test), y_pred=y_pred, labels=labels))
                            dataset_eval_results.append(performance)
                    evaluation_results.append((did, dataset_eval_results))

            if self._verbosity >= 1:
                print("Done with dataset with id: {} at {}".format(did, str(datetime.now())))
            metalearner.clear_configurations()  # reset the metalearner's memory for next dataset
            mdbase.restore_view()

        self._evaluation_results = evaluation_results
        return evaluation_results
