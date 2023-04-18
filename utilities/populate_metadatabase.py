from datetime import datetime
from typing import List

from gama.search_methods import AsyncEA, AsynchronousSuccessiveHalving, RandomSearch
from openml.datasets import get_dataset

from metadatabase import MetaDataBase


def populate_mdbase_gama(
    batch: List[int],
    mdb_path: str,
    time_per_optimization: int = 3600,
    n_jobs: int = 8,
    score: str = "neg_log_loss",
    verbosity: int = 1,
    search_methods: List[str] = ["AsynchronousSuccessiveHalving", "RandomSearch", "AsyncEA"],
):
    """populates the metadatabase at mdb_path given a list of OpenML datasetIDs. Assumes datasets to already be loaded.
    Assumes the datasets to not yet be present in the metdatabase, stores logs by openml-datasetname

    Arguments
    ---------
    batch: List(int),
        list of openmldataset ids to populate the metadatabase with using gama runs
    mdb_path: str,
        path to the metadatabase to populate
    time_per_optimization: int,
        time in seconds how long each gama search method is allowed to take, without data loading etc.
    n_jobs: int,
        the amount of threads to use during the GAMA run, see metadatabase.add_gama_run() for more info
    score: str,
        the metric/scoring method to use during evaluation, see metadatabase.add_gama_run() docstring for options.
    verbosity: int,
        if set to 1 outputs progress updates
    search_methods: List(int).
        specify which search methods should be ran, by default runs all three GAMA methods per dataset:
            AsynchronousSuccessiveHalving, RandomSearch and AsyncEA.
    """
    for dataset_id in batch:
        # Doing all datasets within the same gama object likely yielded a timeouterror in the framework,
        # thus close and open metadatabase after each dataset run. Also helps to limit the memory usage.
        openml_dataset = get_dataset(dataset_id=dataset_id)
        arff_data_path = openml_dataset.data_file

        if "AsynchronousSuccessiveHalving" in search_methods:
            search_method_name = "AsynchronousSuccessiveHalving"
            mdbase = MetaDataBase(mdb_path)
            logs_name = openml_dataset.name + "_{}".format(search_method_name)
            mdbase.add_gama_run(
                dataset_path=arff_data_path,
                dataset_name=openml_dataset.name,
                scoring=score,
                logs_name=logs_name,
                max_total_time=time_per_optimization,
                search=AsynchronousSuccessiveHalving(),
                n_jobs=n_jobs,
            )
            del mdbase
            if verbosity == 1:
                print("dataset {} done with {} at {}".format(openml_dataset.name, search_method_name, datetime.now().strftime("%H:%M:%S")))

        if "RandomSearch" in search_methods:
            search_method_name = "RandomSearch"
            mdbase = MetaDataBase(mdb_path)
            logs_name = openml_dataset.name + "_{}".format(search_method_name)
            mdbase.add_gama_run(
                dataset_path=arff_data_path,
                dataset_name=openml_dataset.name,
                scoring=score,
                logs_name=logs_name,
                max_total_time=time_per_optimization,
                search=RandomSearch(),
                n_jobs=n_jobs,
            )
            del mdbase
            if verbosity == 1:
                print("dataset {} done with {} at {}".format(openml_dataset.name, search_method_name, datetime.now().strftime("%H:%M:%S")))

        if "AsyncEA" in search_methods:
            search_method_name = "AsyncEA"
            mdbase = MetaDataBase(mdb_path)
            logs_name = openml_dataset.name + "_{}".format(search_method_name)
            mdbase.add_gama_run(
                dataset_path=arff_data_path,
                dataset_name=openml_dataset.name,
                scoring=score,
                logs_name=logs_name,
                max_total_time=time_per_optimization,
                search=AsyncEA(),
                n_jobs=n_jobs,
            )
            del mdbase
            if verbosity == 1:
                print("dataset {} done with {} at {}".format(openml_dataset.name, search_method_name, datetime.now().strftime("%H:%M:%S")))

        if verbosity == 1:
            print("dataset {} done at {}".format(openml_dataset.name, datetime.now().strftime("%H:%M:%S")))
