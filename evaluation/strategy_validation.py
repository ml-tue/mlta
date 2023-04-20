import os
from typing import List, Optional, Tuple

import numpy as np
from gama.data_loading import X_y_from_file
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from metadatabase import MetaDataBase
from metalearners import BaseLearner


def strategy_loocv(
    mdbase: MetaDataBase,
    metalearning_strategy: BaseLearner,
    max_time: int,
    dataset_characterization: Optional[List[Tuple[int, List[int | float], List[str]]]],
    metric: str = "neg_log_loss",
    validation_strategy: str = "holdout",
    n_jobs: int = 1,
    verbosity: int = 1,
) -> List[Tuple[int, float]]:
    """Performs leave-one-out-cross-validation (LOOCV) on the metadatabase to estimate the performance of the meta-learning strategy,
        allowing it `max_time` seconds and evaluating on the `metric`, while performing the `validation_strategy` per dataset.

    Arguments
    ---------
    mdbase: MetaDataBase,
        the metadatabase to perform LOOCV on
    metalearning_strategy: Subclass of BaseStrategy,
        the meta-learning strategy that should be evaluated on the mdbase, should inherent from BaseStrategy
    max_time: int,
        time in seconds allowed for the meta-learning strategy online phase
    metric: str,
        specify the metric to perform the LOOCV with
        Default is neg_log_loss, TODO implement other options
    validation_strategy: str,
        Validation strategy used within LOOCV on a single dataset.
        Default is holdout with 80/20 train-test split.
         TODO implement other options
    dataset_characterization: List[Tuple[int, List[int | float], List[str]]]
        A list of tuples, where each tuple represents a dataset characterization.
        The first element in the tuple refers to the dataset_id in mdbase,
        The second element is the purely numeric vector representing the dataset,
        The last element is a list of equal size to the second element, specifying its names.
    n_jobs: int,
        the number of threads the `metalearning_strategy` is allowed to use. Default is 1.
    verbosity: int,
        if set to 1, then shows dataset_id when done with dataset

    Returns
    -------
    List of (dataset_id, performance) tuples for each dataset using LOOCV with the specified metric
    If meta-learning strategy does not provide a solution within `max_time`, then performance is None"""

    # stores the to-be-created output
    dataset_evaluations = []

    dataset_ids = mdbase.list_datasets(by="id")
    for did in dataset_ids:  # leave one dataset out
        if validation_strategy == "holdout":
            if dataset_characterization != None:  # perform online phase with suitable characterizations
                selected_characterizations = []
                for characterization in dataset_characterization:
                    if characterization[0] != did:
                        selected_characterizations.append(characterization)
                metalearning_strategy.offline_phase(mdbase=mdbase, dataset_characterization=selected_characterizations)
            else:
                metalearning_strategy.offline_phase(mdbase=mdbase)

            # perform online phase
            X, y = X_y_from_file(os.path.join(str(mdbase._datasets_dir), "{}.arff".format(did)))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            metalearning_strategy.online_phase(X_train, y_train, max_time=max_time, metric="neg_log_loss", n_jobs=n_jobs)

            # if possible, evaluate top solution
            # TODO: decide how to evaluate when there are multiple solutions
            top_solution = metalearning_strategy.get_top_configurations()[0]
            if top_solution is None:
                dataset_evaluations.append((did, None))
            else:
                top_solution.fit(X_train, y_train)
                performance: float = -np.inf  # stores performance according to metric
                if metric == "neg_log_loss":
                    y_pred = top_solution.predict_proba(X_test)
                    labels = np.unique(LabelEncoder().fit_transform(y))
                    performance = float(-1 * log_loss(y_true=LabelEncoder().fit_transform(y_test), y_pred=y_pred, labels=labels))
                dataset_evaluations.append((did, performance))

        # if k-fold cv:
        # get relevant dataset_characterizations

        # for each fold
        # perform the offline phase on the subset of the mdbase

        # perform the online phase on the subset of the mdbase within max_time

        # get the top solution

        # avg out the scores

        if verbosity == 1:
            print("Done with dataset with id: {}".format(did))
        # reset the meta-learning strategy top solution and best score for next dataset
        metalearning_strategy._top_configurations = None

    return dataset_evaluations
