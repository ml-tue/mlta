import importlib
import os
import shutil
import warnings
from typing import List, Optional, Tuple

import arff
import numpy as np
import pandas as pd
from gama import GamaClassifier
from gama.genetic_programming.components.individual import Individual
from gama.postprocessing import EnsemblePostProcessing
from gama.search_methods.async_ea import AsyncEA
from gama.search_methods.base_search import BaseSearch
from gama.utilities.export import format_import, individual_to_python
from gama.utilities.preprocessing import basic_encoding, basic_pipeline_extension
from sklearn.pipeline import Pipeline

from metadatabase.metadatalookuptable import MetaDataLookupTable
from utilities import hash_pipe_id_to_dir


# the metadatabase consist of datasets that were used to train on, the solutions on these datasets, and the associated scpre, and the corresponding metric
class MetaDataBase:
    def __init__(self, path: str = ""):
        """Initializes the MetaDataBase. `path` is optional, if set then the metadatabase is loaded from path
        path should contain 6 dirs: 'pipelines', 'datasets', 'lookup_tables', 'logs', 'dataset_characterizations', 'configuration_characterizations'
        Additionally it should contain 'metadatabase.csv'"""

        self._mdbase_path = None
        self._pipelines_dir = None
        self._datasets_dir = None
        self._tables_dir = None
        self._logs_dir = None
        self._dataset_char_dir = None
        self._config_char_dir = None
        self._md_table: MetaDataLookupTable  # lookuptable for pipelines and datasets
        self._mdbase: pd.DataFrame  # actually stores the solutions with their scores on datasets

        if os.path.isdir(os.path.join(path, "tmp")):
            warnings.warn("The metadatabase has a partial view w.r.t datasets. Run `restore_view()` for a full view.")

        if path != "":
            self._mdbase_path = os.path.join(path, "metadatabase.csv")
            self._pipelines_dir = os.path.join(path, "pipelines")
            self._datasets_dir = os.path.join(path, "datasets")
            self._tables_dir = os.path.join(path, "lookup_tables")
            self._logs_dir = os.path.join(path, "logs")
            self._dataset_char_dir = os.path.join(path, "dataset_characterizations")
            self._config_char_dir = os.path.join(path, "config_characterizations")

            # check if the specified files exist
            if not os.path.exists(self._mdbase_path):
                raise ValueError("metadatabase.csv not found at `path`", path)
            if not os.path.exists(self._pipelines_dir):
                raise ValueError("pipelines dir not found at `path`", path)
            if not os.path.exists(self._datasets_dir):
                raise ValueError("datasets dir not found at `path`", path)
            if not os.path.exists(self._tables_dir):
                raise ValueError("cannot find lookup_tables dir at `path`", path)
            if not os.path.exists(self._dataset_char_dir):
                raise ValueError("cannot find dataset_characterizations dir at `path`", path)
            if not os.path.exists(self._config_char_dir):
                raise ValueError("cannot find config_characterizations dir at `path`", path)

            self._md_table = MetaDataLookupTable(self._tables_dir)
            self._mdbase = pd.read_csv(self._mdbase_path, dtype={"dataset_id": "int64", "pipeline_id": "int64"})

    def create_empty_metadatabase(self, path: str = "", name: str = "metadatabase_name"):
        "Creates an empty metadatabase at path, only applicable if MetaDataBase was not initialized with exisiting metadatabase"
        # create paths for files
        self._mdbase_path = os.path.join(path, name, "metadatabase.csv")
        self._pipelines_dir = os.path.join(path, name, "pipelines")
        self._datasets_dir = os.path.join(path, name, "datasets")
        self._tables_dir = os.path.join(path, name, "lookup_tables")
        self._logs_dir = os.path.join(path, name, "logs")
        self._dataset_char_dir = os.path.join(path, "dataset_characterizations")
        self._config_char_dir = os.path.join(path, "config_characterizations")

        os.mkdir(os.path.join(path, name))
        os.mkdir(self._pipelines_dir)
        for dir in range(0, 1000):  # create 1000 subdirs in pipelines dir
            os.mkdir(os.path.join(self._pipelines_dir, str(dir)))
        os.mkdir(self._datasets_dir)
        os.mkdir(self._tables_dir)
        os.mkdir(self._logs_dir)
        os.mkdir(self._dataset_char_dir)
        os.mkdir(self._config_char_dir)

        # create empty metadata csv files
        df_pipeline_table = pd.DataFrame(data={"pipeline": list(), "pipeline_id": list()})
        df_pipeline_table.astype(dtype={"pipeline_id": "int64"}, copy=False)
        df_pipeline_table.to_csv(os.path.join(self._tables_dir, "lookup_table_pipelines.csv"), index=False)

        df_datasets_table = pd.DataFrame(data={"dataset": list(), "dataset_id": list()})
        df_datasets_table.astype(dtype={"dataset_id": "int64"}, copy=False)
        df_datasets_table.to_csv(os.path.join(self._tables_dir, "lookup_table_datasets.csv"), index=False)

        df_mdbase = pd.DataFrame(data={"dataset_id": list(), "pipeline_id": list(), "score": list(), "metric": list()})
        df_mdbase.astype(dtype={"dataset_id": "int64", "pipeline_id": "int64"}, copy=False)
        df_mdbase.to_csv(self._mdbase_path, index=False)

        # initialize the lookup tables and metadatabase
        self._mdbase = df_mdbase
        self._md_table = MetaDataLookupTable(self._tables_dir)

    def add_gama_run(
        self,
        dataset_path: str,
        dataset_name: str,
        max_models_to_keep: int = 1000000,
        logs_name: str = "",
        scoring: str = "neg_log_loss",
        regularize_length: bool = True,
        max_pipeline_length: Optional[int] = None,
        random_state: Optional[int] = None,
        max_total_time: int = 3600,
        search: BaseSearch = AsyncEA(),
        n_jobs: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
    ) -> None:
        """Adds the results of a GAMA run to the metadatabase: the associated csv files are updated.
        Additionally, the logs of the gama run are stored in the metadatabase 'logs' directory.
        Stores the results through a GAMA call using EnsemblePostProcessing(), without fitting the ensemble.
        Therefore the logs will contain an error, specifying that the ensemble could not be fitted.
        But this specific function does not throw any error when doing so, hence its safe.

        Arguments
        ----------
        dataset_path: str,
            Path to the dataset, which should be in ARFF format.
        dataset_name: str,
            Name of the dataset, used to identify the dataset in the metadatabase.
            Thus it is important to not use the same datasets with different names.
            To be consistent, use the openml dataset names if available.
        logs_name: str,
            Specifies the name of the logs that will be stored in the metadataset.
            Name must not already be present in the logs.
            If set to None, generate a unique name ("gama_HEXCODE").
        max_models_to_keep: int,
            The amount of models that are stored from the GAMA run and added to the metadatabase.
            Default set to a very large value, essentially keeping all models.
        scoring: str
            Specifies the metric to optimize towards, supports those metrics supported in GAMA.
            Note that for all options a greater value is better (hence named scoring).
            Options for classification:
                accuracy, roc_auc, average_precision, neg_log_loss,
                precision_macro, precision_micro, precision_samples, precision_weighted,
                recall_macro, recall_micro, recall_samples, recall_weighted,
                f1_macro, f1_micro, f1_samples, f1_weighted
            Options for regression:
                explained_variance, r2, neg_mean_absolute_error, neg_mean_squared_log_error,
                neg_median_absolute_error, neg_mean_squared_error
        regularize_length: bool (default=True)
            If True, add pipeline length as an optimization metric.
            Short pipelines should then be preferred over long ones.
        max_pipeline_length: int, optional (default=None)
            If set, limit the maximum number of steps in any evaluated pipeline.
            Encoding and imputation are excluded.
        random_state:  int, optional (default=None)
            Seed for the random number generators used in the process.
            However, with `n_jobs > 1`,
            there will be randomization introduced by multi-processing.
            For reproducible results, set this and use `n_jobs=1`.
        max_total_time: positive int (default=3600)
            Time in seconds that can be used for gama's `fit_from_file` call.
        search: BaseSearch (default=AsyncEA())
            Search method to use to find good pipelines. Should be instantiated.
        n_jobs: int, optional (default=None)
            The amount of parallel processes that may be created to speed up `fit`.
            Accepted values are positive integers, -1 or None.
            If -1 is specified, multiprocessing.cpu_count() processes are created.
            If None is specified, multiprocessing.cpu_count() / 2 processes are created.
        max_memory_mb: int, optional (default=None)
            Sets the total amount of memory GAMA is allowed to use (in megabytes).
            If not set, GAMA will use as much as it needs.
            GAMA is not guaranteed to respect this limit at all times,
            but it should never violate it for too long.
        """
        if self._mdbase_path == None or self._pipelines_dir == None or self._datasets_dir == None or self._tables_dir == None:
            raise ValueError("No csv files associated to the metadatabase. \nFirst create a metadatabase. \nAborted GAMA run.")
        if os.path.exists(os.path.join(str(self._logs_dir), logs_name)):
            raise ValueError("This log file name `logs_name` already exists. \nAborted GAMA run.")

        # instantiate GAMA, start training, keep track of models in `automl` object.
        automl = GamaClassifier(
            scoring=scoring,
            regularize_length=regularize_length,
            max_total_time=max_total_time,
            max_pipeline_length=max_pipeline_length,
            store="logs",
            random_state=random_state,
            n_jobs=n_jobs,
            max_memory_mb=max_memory_mb,
            output_directory=os.path.join(str(self._logs_dir), logs_name),
            search=search,
            post_processing=EnsemblePostProcessing(time_fraction=0, max_models=max_models_to_keep),  # dont fit ensemble
            verbosity=0,
        )
        automl.fit_from_file(dataset_path)
        evaluations = automl._evaluation_library.n_best(n=max_models_to_keep, with_pipelines=True)

        dataset_id = None
        # if needed, store and copy the dataset
        if not self._md_table.dataset_exists(dataset_name):
            self._md_table.add_dataset(dataset_name)
            dataset_id = self._md_table.get_dataset_id(dataset_name)
            shutil.copyfile(dataset_path, os.path.join(self._datasets_dir, "{}.arff".format(dataset_id)))
        else:  # still need to get dataset id
            dataset_id = self._md_table.get_dataset_id(dataset_name)

        # write each evaluated pipeline to scripts to store them in pipelines dir.
        # for efficiency sake store all results in lists first before writing to a dataframe
        datasets, pipelines, scores, metrics, logs_names = [], [], [], [], []
        for eval in evaluations:
            pipe_name = eval.individual.pipeline_str()
            pipe_id = None
            if not self._md_table.pipe_exists(pipe_name):  # only store previously unseen pipeline as code
                pipe_script = individual_to_python(eval.individual)  # dont store prepend steps, those are dataset specific
                self._md_table.add_pipeline(pipe_name)
                pipe_id = int(self._md_table.get_pipeline_id(pipe_name))
                pipe_script_name = "pipeline_{}.py".format(pipe_id)

                # write new pipeline to pipeline_{id}.py file in corresponding subdir in pipelines dir
                dir_id = str(hash_pipe_id_to_dir(pipe_id))
                pipe_path = os.path.join(self._pipelines_dir, dir_id, pipe_script_name)
                with open(pipe_path, "w+") as fh:
                    fh.write(pipe_script)
                    fh.close()

            if pipe_id is None:
                pipe_id = int(self._md_table.get_pipeline_id(pipe_name))
            # store the evaluation results, not yet in the metadatabase
            datasets.append(dataset_id)
            pipelines.append(pipe_id)
            scores.append(eval.score[0])
            metrics.append(scoring)
            logs_names.append(logs_name)

        # store results in metadatabase
        new_records = {"dataset_id": datasets, "pipeline_id": pipelines, "score": scores, "metric": metrics, "logs_name": logs_names}
        new_records_df = pd.DataFrame.from_dict(new_records)
        self._mdbase = pd.concat([self._mdbase, new_records_df])

        # write information to csvs (lookup tables and metadatabase)
        self._md_table.update_tables()
        self._mdbase.to_csv(self._mdbase_path, index=False)

    def add_dataset_evaluations(self, dataset_path: str, dataset_name: str, logs_name: str, metric: str, pipelines: List[str], scores: List[float]):
        """Add metadata evaluations pertaining a single dataset and metric to the metadatabase.
            Currently there is not support for adding multiple datasets or on different metrics in a single call.

        Arguments
        ----------
        dataset_path: str,
            Path to the dataset, which should be in ARFF format.
        dataset_name: str,
            Name of the dataset, used to identify the dataset in the metadatabase.
            Thus it is important to not use the same datasets with different names.
            To be consistent, use the openml dataset names if available.
        logs_name: str,
            Specifies the name of the logs that will be stored in the metadataset.
            Name must not already be present in the logs.
        metric: str
            Specifies the scoring metric the evaluations are evaluated on, supports options where greater values are better (e.g. a score)
            Options for classification:
                accuracy, roc_auc, average_precision, neg_log_loss,
                precision_macro, precision_micro, precision_samples, precision_weighted,
                recall_macro, recall_micro, recall_samples, recall_weighted,
                f1_macro, f1_micro, f1_samples, f1_weighted
            Options for regression:
                explained_variance, r2, neg_mean_absolute_error, neg_mean_squared_log_error,
                neg_median_absolute_error, neg_mean_squared_error
        pipelines: List of strings
            Strings representations of sklearn pipelines adhering to gama individual string representation
                Could use `utilities.sklearn_pipe_to_individual_str()` for the conversion from sklearn pipelines.
        scores: List of floats
            The scores that accompany the `pipelines`, such that matching indices in `pipeline` and `scores` belong together.
        """
        if self._mdbase_path == None or self._pipelines_dir == None or self._datasets_dir == None or self._tables_dir == None:
            raise ValueError("No csv files associated to the metadatabase. \nFirst create a metadatabase. \nAborted GAMA run.")
        if os.path.exists(os.path.join(str(self._logs_dir), logs_name)):
            raise ValueError("This log file name `logs_name` already exists. \nAborted GAMA run.")
        if len(pipelines) != len(scores):
            raise ValueError("Dimensionality of `pipelines` and `scores` does not match, it should have same length.")

        dataset_id = None
        # if needed, store and copy the dataset
        if not self._md_table.dataset_exists(dataset_name):
            self._md_table.add_dataset(dataset_name)
            dataset_id = self._md_table.get_dataset_id(dataset_name)
            shutil.copyfile(dataset_path, os.path.join(self._datasets_dir, "{}.arff".format(dataset_id)))
        else:  # still need to get dataset id
            dataset_id = self._md_table.get_dataset_id(dataset_name)

        # format evaluations for mdbase storage
        n_evals = len(pipelines)
        mdbase_datasets = [dataset_id] * n_evals
        mdbase_metrics = [metric] * n_evals
        mdbase_logs_names = [logs_name] * n_evals
        mdbase_scores = scores
        mdbase_pipeline_ids = []
        for pipe_str in pipelines:
            if not self._md_table.pipe_exists(pipe_str):
                self._md_table.add_pipeline(pipe_str)
            pipe_id = int(self._md_table.get_pipeline_id(pipe_str))
            mdbase_pipeline_ids.append(pipe_id)

            # write new pipeline to pipeline_{id}.py file in corresponding subdir in pipelines dir
            gama = GamaClassifier()
            individual = Individual.from_string(pipe_str, gama._pset, gama._operator_set._safe_compile)
            gama.cleanup(which="all")
            pipe_script = individual_to_python(individual)  # dont store prepend steps, those are dataset specific # TODO check whether this works correctly
            pipe_script_name = "pipeline_{}.py".format(pipe_id)
            dir_id = str(hash_pipe_id_to_dir(pipe_id))
            pipe_path = os.path.join(self._pipelines_dir, dir_id, pipe_script_name)
            with open(pipe_path, "w+") as fh:
                fh.write(pipe_script)
                fh.close()

        # store results in metadatabase
        new_records = {"dataset_id": mdbase_datasets, "pipeline_id": mdbase_pipeline_ids, "score": mdbase_scores, "metric": mdbase_metrics, "logs_name": mdbase_logs_names}
        new_records_df = pd.DataFrame.from_dict(new_records)
        self._mdbase = pd.concat([self._mdbase, new_records_df])

        # write information to csvs (lookup tables and metadatabase)
        self._md_table.update_tables()
        self._mdbase.to_csv(self._mdbase_path, index=False)

    def list_datasets(self, by: str = "both") -> list:
        """Returns a list of datasets in the metadatabase

        Arguments
        ---------
        by: str,
            Can be  "name" to list the dataset names,
            "id" to list the dataset ids.
            "both", to get a list of (name, id) tuples.
        """
        return self._md_table.list_datasets(by)

    def get_dataset(self, dataset_id: int, type: str = "dataframe") -> str | Tuple[List, List] | Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns dataset with `dataset_id` from metadatabase, assuming the last column is the target.

        Arguments
        ---------
        type: str,
            How the dataset is returned. Assumes the last column of the arff is the target. Options are:
            Default option is "dataframe", which returns a tuple of pd.Dataframes for features and outcome.
            Another option is "arrays", which returns the dataset as a tuple: (X, y),
            Another option is "arff_path", which points to the arff file on the system.
        """
        dataset_path = os.path.join(str(self._datasets_dir), "{}.arff".format(dataset_id))
        if not os.path.exists(dataset_path):
            raise ValueError("Dataset with id: {} not present in the metadatabase".format(dataset_id))

        if type == "dataframe":
            with open(dataset_path, "r") as arff_file:
                arff_dict = arff.load(arff_file)

            attribute_names, _ = zip(*arff_dict["attributes"])
            data = pd.DataFrame(arff_dict["data"], columns=attribute_names)
            for attribute_name, dtype in arff_dict["attributes"]:
                if isinstance(dtype, list):
                    data[attribute_name] = data[attribute_name].astype("category")

            return data.iloc[:, :-1], data.iloc[:, -1]  # type: ignore
        elif type == "arff_path":
            return dataset_path
        else:  # type == "arrays":
            data = arff.load(open(dataset_path, "r"))["data"]
            # select the last value to be the target
            X = [i[:-1] for i in data]
            y = [i[-1] for i in data]
            return X, y

    def list_pipelines(self, by: str = "both") -> list:
        """Returns a list of datasets in the metadatabase

        Arguments
        ---------
        by: str,
            Can be "name" to list the pipeline names,
            "id" to list the pipeline ids,
            "both" to get a list of (name, id) tuples
        """
        return self._md_table.list_pipelines(by)

    def get_sklearn_pipeline(
        self,
        pipeline_id: int,
        X: np.ndarray | pd.DataFrame | None = None,
        y: np.ndarray | pd.DataFrame | pd.Series | None = None,
        is_classification: bool = True,
        include_prepro: bool = False,
    ) -> Pipeline:
        """Returns a sklearn.pipeline.Pipeline from the metadatabase representing `pipeline_id`
        Additionally, it loads the imports that are necessary for executing the pipeline.
        You can specify if you want the preprocessing steps too, if they are desired X and y should be given.
        """
        if include_prepro:
            if X is None or y is None:
                raise ValueError("X and Y should be specified if include_prepo is True")

        # prepare loading the pipeline without preprocessing
        dir_id = str(hash_pipe_id_to_dir(pipeline_id))
        pipeline_path = os.path.join(str(self._pipelines_dir), dir_id, "pipeline_{}.py".format(pipeline_id))
        if not os.path.exists(pipeline_path):
            raise ValueError("Pipeline with id: {} is not present in the metadatabase".format(pipeline_id))
        module_name = "pipefile_{}".format(id)
        spec = importlib.util.spec_from_file_location(module_name, pipeline_path)  # type: ignore
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)

        # get the pipeline and load imports
        if include_prepro:
            pipe_wo_prepro = module.pipeline
            prepro_pipe = Pipeline(self._get_preprocessing_steps(X, y, is_classification))
            pipeline = Pipeline(prepro_pipe.steps + pipe_wo_prepro.steps)
            self._load_pipeline_imports(pipeline)
            return pipeline
        else:
            pipe_wo_prepro = module.pipeline
            self._load_pipeline_imports(pipe_wo_prepro)
            return pipe_wo_prepro

    def _load_pipeline_imports(self, pipeline: Pipeline) -> None:
        # import certain modules to enable using the pipeline
        imports = ["from numpy import nan", "from sklearn.pipeline import Pipeline"]
        imports += [format_import(step) for _, step in pipeline.steps]
        [exec(imp) for imp in imports]

    def _get_preprocessing_steps(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series, is_classification: bool) -> list:
        """Returns the preprocessing steps (encoding and imputation) for the data"""
        X_, basic_encoding_pipeline = basic_encoding(X, is_classification)
        fixed_pipeline_extension = basic_pipeline_extension(X_, is_classification)

        return basic_encoding_pipeline.steps + fixed_pipeline_extension

    def get_df(
        self,
        datasets: Optional[List[str] | List[int]] = None,
        pipelines: Optional[List[str] | List[int]] = None,
        logs_name: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        top_solutions: Optional[Tuple[int, str]] = None,
        avg_equivalent: bool = False,
    ) -> pd.DataFrame:
        """Returns a copy of the metadatabase records as a pandas.Dataframe object.

        Arguments
        ---------
        datasets: list(str), list(int), None,
            If list of strings, then selects records by their associated dataset name as present in the metadatabase.
            If list of integers, then selects records by their associated dataset id as present in the metadatabase.
            If default(`None`), then no selection.
        pipelines: list(str), list(int), None,
            If list of strings, then selects records by their associated pipeline name as present in the metadatabase.
            If list of integers, then selects records by their associated pipeline id as present in the metadatabase.
            If default(`None`), then no selection.
        logs_name: list(str),
            specify the runs by their `logs_name` to select
        metrics: list(str),
            Selects records by their associated metric as string.
            See `add_gama_run()` documentation for 'scoring' parameter for supported metric options.
            If default(`None`), then no selection.
        avg_equivalent: bool,
            GAMA may have multiple evaluations using the same metric of equivalent pipelines.
            If True, the equivalent pipelines are averaged out (for the same metric).
            If False, each result is kept seperatel.
        top_solutions: tuple(int, str),
            Select only the top solutions from the metadataset by some score.
            Tuple's first element: maximum number of solutions.
            Tuple's second element: the metric for which to rank the solutions.
                See `add_gama_run()` documentation for 'scoring' parameter for supported metric options..
                Cannot conflict with `metrics`; can only query for top solutions with a metric in `metrics`.
            If default(`None`), then no selection.

        Returns
        -------
        A copy of the metadatabase as pandas.DataFrame with the selected records according to the Arguments.
        If all Arguments are default, then the full metadataset is returned.
        """
        mdbase = self._mdbase.copy(deep=True)

        # select datasets
        if isinstance(datasets, list) and all(isinstance(dataset, str) for dataset in datasets):
            dataset_ids = [self._md_table.get_dataset_id(str(dataset)) for dataset in datasets]
            mdbase = mdbase[mdbase["dataset_id"].isin(dataset_ids)]
        if isinstance(datasets, list) and all(isinstance(dataset, int) for dataset in datasets):
            mdbase = mdbase[mdbase["dataset_id"].isin(datasets)]

        # select pipelines
        if isinstance(pipelines, list) and all(isinstance(pipe, str) for pipe in pipelines):
            pipeline_ids = [self._md_table.get_pipeline_id(str(pipe)) for pipe in pipelines]
            mdbase = mdbase[mdbase["pipeline_id"].isin(pipeline_ids)]
        if isinstance(pipelines, list) and all(isinstance(pipe, int) for pipe in pipelines):
            mdbase = mdbase[mdbase["pipeline_id"].isin(pipelines)]

        # select metrics
        if metrics != None:
            mdbase = mdbase[mdbase["metric"].isin(metrics)]

        if logs_name != None:
            mdbase = mdbase[mdbase["logs_name"].isin(logs_name)]

        # reset index to enable .iloc on dataframe in averaging
        mdbase.reset_index(inplace=True)
        mdbase.drop("index", axis=1, inplace=True)
        # average out equivalent runs
        if avg_equivalent:
            mdbase.drop_duplicates(inplace=True, ignore_index=True)
            ind_to_update = mdbase[mdbase.duplicated(subset=["dataset_id", "pipeline_id", "metric"], keep="last")].index.to_list()
            duplicates_to_remove = []
            for ind in ind_to_update:
                # find which records have equivalent runs to the current one
                record = mdbase.iloc[ind]
                duplicates = mdbase[
                    (mdbase["dataset_id"] == record["dataset_id"]) & (mdbase["pipeline_id"] == record["pipeline_id"]) & (mdbase["metric"] == record["metric"])
                ].index.to_list()
                # update score
                new_score = mdbase.iloc[duplicates]["score"].mean()
                mdbase.at[ind, "score"] = new_score
                # store which duplicates to remove
                duplicates.remove(ind)
                for duplicate_to_remove in duplicates:
                    duplicates_to_remove.append(duplicate_to_remove)

            # remove duplicates
            mdbase.drop(index=duplicates_to_remove, inplace=True)

        # select only top solutions
        if top_solutions != None:
            n_solutions = top_solutions[0]
            metric = top_solutions[1]
            mdbase = mdbase[mdbase["metric"] == metric]
            mdbase.sort_values(by="score", ascending=False, inplace=True)  # all possible metrics are scores (greater is always better)
            mdbase = mdbase[:n_solutions]

        # reset index for cleanliness
        mdbase.reset_index(inplace=True)
        mdbase.drop("index", axis=1, inplace=True)

        return mdbase

    def partial_datasets_view(self, datasets: List[int]) -> None:
        """Facilitates a temporary partial view on the metadatabase with respect to the datasets.
        Alters the metadatabase files, but does not delete any information. To restore the full view run `restore_view()`

        Arguments
        ---------
        datasets: List of integers,
            ids of datasets that should be temporarily removed from the metadatabase views.
        """
        self._set_partial_view_paths()
        if os.path.isdir(self._tmp_dir):
            raise Warning("Cannot create a partial view: already in a partial view. To create a new partial view: \n close current view by `restore_view()`")

        os.mkdir(self._tmp_dir)
        os.mkdir(self._tmp_datasets_dir)

        # copy over the datasets we do not want, and remove them accordingly
        unwanted_dataset_ids = [id for id in self.list_datasets(by="id") if id not in datasets]
        for id in unwanted_dataset_ids:
            dataset_file = "{}.arff".format(id)
            src_path = os.path.join(str(self._datasets_dir), dataset_file)
            dst_path = os.path.join(self._tmp_datasets_dir, dataset_file)
            shutil.copyfile(src_path, dst_path)
            os.remove(src_path)

        # copy over the lookup_tables/lookup_table_datasets.csv to dir metadatabase/temp/lookup_tables
        src_lookup_table = os.path.join(str(self._tables_dir), "lookup_table_datasets.csv")
        shutil.copyfile(src_lookup_table, self._tmp_datasets_table)
        # create new lookup table without the unwanted datasets and update in memory lookuptable
        df_lookup_table = pd.read_csv(self._tmp_datasets_table, index_col=0)
        df_lookup_table = df_lookup_table[df_lookup_table["dataset_id"].isin(datasets)]
        df_lookup_table.to_csv(src_lookup_table)

        # copy over metadatabase.csv to dir metadatabase/temp/
        src_mdbase_table = os.path.join(self._root_dir, "metadatabase.csv")
        shutil.copyfile(src_mdbase_table, self._tmp_mdbase_table)
        # overwrite the src_mdbase_table with file without unwanted datasets
        self._mdbase = self.get_df(datasets=datasets)
        self._mdbase.to_csv(src_mdbase_table, index=False)

        # pipelines update
        src_pipelines_table = os.path.join(str(self._tables_dir), "lookup_table_pipelines.csv")
        shutil.copyfile(src_pipelines_table, self._tmp_pipelines_table)
        remaining_pipes = list(dict.fromkeys(list(self._mdbase["pipeline_id"])))
        df_pipes = pd.read_csv(self._tmp_pipelines_table, index_col=0)
        df_pipes = df_pipes[df_pipes["pipeline_id"].isin(remaining_pipes)]
        df_pipes.to_csv(src_pipelines_table)
        self._md_table = MetaDataLookupTable(path=str(self._tables_dir))  # update for both datasets and pipelines table

    def restore_view(self) -> None:
        """Restores the view on the metadatabase view created with `partial_datasets_view()`."""
        # need to ensure all paths are set, because it enables usage of function when a MetaDataBase object was deleted
        self._set_partial_view_paths()

        if not os.path.isdir(self._tmp_dir):
            raise Warning("There is nothing to restore with `restore_view()`, no tmp directory found in metadatabase dir.")

        # restore datasets
        for dataset_file in os.listdir(self._tmp_datasets_dir):
            src_file = os.path.join(self._tmp_datasets_dir, dataset_file)
            dst_file = os.path.join(str(self._datasets_dir), dataset_file)
            shutil.copyfile(src_file, dst_file)

        # restore datset lookup table
        dst_path = os.path.join(str(self._tables_dir), "lookup_table_datasets.csv")
        os.remove(dst_path)  # remove such that we can replace it with old file in tmp dir
        shutil.copyfile(self._tmp_datasets_table, dst_path)
        # restore pipeline lookup table
        dst_path = os.path.join(str(self._tables_dir), "lookup_table_pipelines.csv")
        os.remove(dst_path)
        shutil.copyfile(self._tmp_pipelines_table, dst_path)
        self._md_table = MetaDataLookupTable(path=str(self._tables_dir))

        # restore mdbase_table
        src_path = os.path.join(self._tmp_dir, "metadatabase.csv")
        dst_path = os.path.join(self._root_dir, "metadatabase.csv")
        self._mdbase = pd.read_csv(src_path)
        os.remove(dst_path)  # remove the adapted `metadatabase.csv`
        shutil.copyfile(src_path, dst_path)

        shutil.rmtree(self._tmp_dir)

    def _set_partial_view_paths(self) -> None:
        """Method setting paths that are used in `partial_datasets_view()` and `restore_view()`"""
        self._root_dir = os.path.split(str(self._datasets_dir))[0]
        self._tmp_dir = os.path.join(self._root_dir, "tmp")
        self._tmp_datasets_dir = os.path.join(self._tmp_dir, "datasets")
        self._tmp_datasets_table = os.path.join(self._tmp_dir, "lookup_table_datasets.csv")
        self._tmp_mdbase_table = os.path.join(self._tmp_dir, "metadatabase.csv")
        self._tmp_pipelines_table = os.path.join(self._tmp_dir, "lookup_table_pipelines.csv")

    def add_dataset_characterizations(self, characterizations: List[Tuple[int, List[int | float], List[str]]], name: str) -> None:
        """Add dataset `characterizations` to the mdbase, referring to them using `name`.
        The user should note that this method can also add dataset characterizations to already existing characterizations,
        but it is not possible to overwrite them. Hence, a ValueError is thrown if `characterizations` contains dataset ids for which
        characterization with `name` already exist.

        Arguments
        ---------
        characterizations: List[Tuple[int, List[int | float], List[str]]]
            A list of tuples, where each tuple represents a dataset characterization.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is the purely numeric vector representing the dataset,
            The last element contains the names of the dimensions/features, which could be number-like if they have no meaning.
        name: string,
            the name by which the characterizations should be referenced and stored.
        """
        # check if datasets are present in mdbase
        new_dataset_ids = [int(entry[0]) for entry in characterizations]
        mdbase_dataset_ids = self.list_datasets(by="id")
        non_existing_datasets = [id for id in new_dataset_ids if id not in mdbase_dataset_ids]
        if len(non_existing_datasets) != 0:
            raise ValueError("Cannot add characterizations, because some datasets do not exist (ids: {})".format(non_existing_datasets))

        # handling existing dataset ids
        df_chars = None
        file_name = "{}.csv".format(name)
        char_file = os.path.join(self._dataset_char_dir, file_name)
        if file_name in os.listdir(self._dataset_char_dir):  # extend this characterization method with new datasets
            existing_dataset_ids = [int(id) for id in list(pd.read_csv(char_file)["dataset_id"])]
            overlap_ids = [id for id in new_dataset_ids if id in existing_dataset_ids]
            print("overlap_ids: ", overlap_ids)
            if len(overlap_ids) != 0:
                raise ValueError("characterizations cannot be added, because there is overlap between already existing datasets with ids: ".format(overlap_ids))
            df_chars = pd.read_csv(char_file)

        # put new characterizations into dataframe format, extend df_chars which could be empty or composed of existing characterizations
        for i, entry in enumerate(characterizations):
            names = entry[1][1]
            values = entry[1][0]
            if i == 0:
                df_chars = pd.DataFrame(columns=["dataset_id"] + names)
            if names != list(df_chars.columns)[1:]:
                raise ValueError("not all characterization entries have the same names or dimensionality thereof, names should be equal")
            df_chars.loc[len(df_chars.index)] = [int(entry[0])] + values
        df_chars.to_csv(char_file, index=False)

    def list_dataset_characterizations(self, characterization_names: Optional[List[str]] = None, dataset_ids: Optional[List[int]] = None) -> List[Tuple[str, List[int]]]:
        """Shows which characterizations are stored for which datasets, can filter on characterization_name and dataset_ids.

        Arguments
        ---------
        characterization_names: Optional[List[str]],
            optional list of characterization names (as stored using `name` in `add_dataset_characterizations`) to filter on.
        dataset_ids: Optional[List[int]],
            optional list of dataset_ids to filter on.
        """
        stored_characterizations = []
        for f in os.listdir(self._dataset_char_dir):
            char_name = f.split(".")[0]
            if characterization_names is not None:  # filter on characterization names
                if char_name not in characterization_names:
                    continue
            char_path = os.path.join(self._dataset_char_dir, f)
            stored_ids = [int(id) for id in list(pd.read_csv(char_path)["dataset_id"])]
            if dataset_ids is not None:  # filter on dataset ids
                stored_ids = [id for id in stored_ids if id in dataset_ids]
            if len(stored_ids) == 0:  # filtered all ids out
                continue
            stored_characterizations.append((char_name, stored_ids))
        return stored_characterizations

    def get_dataset_characterizations(self, characterization_name: str = None, dataset_ids: Optional[List[int]] = None) -> List[Tuple[str, List[int]]]:
        """Returns stored characterizations, can filter on characterization_name and dataset_ids.

        Arguments
        ---------
        characterization_names:str,
            Name of characterization method for which to get the characterizations (as stored using `name` in `add_dataset_characterizations`).
        dataset_ids: Optional[List[int]],
            optional list of dataset_ids to filter on.

        Returns
        -------
        metadatabase_characterizations: List[Tuple[int, List[int | float]]],
            A list of tuples, where each tuple represents a dataset characterization.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is the purely numeric vector representing the dataset,
        """
        # check whether characterization filter is valid
        available_characterizations = []
        for f in os.listdir(self._dataset_char_dir):
            available_characterizations.append(f.split(".")[0])
        if characterization_name not in available_characterizations:
            raise ValueError("The selected characterization `{}` is not stored in the mdbase. Stored options are: {}".format(characterization_name, available_characterizations))

        # get the characterizations file
        file_path = os.path.join(self._dataset_char_dir, characterization_name + ".csv")
        char_df = pd.read_csv(file_path)

        # check whether dataset filter is valid:
        if dataset_ids is not None:
            non_existing_datasets = [id for id in dataset_ids if id not in self.list_datasets(by="id")]
            if len(non_existing_datasets) != 0:
                raise ValueError("Some dataset ids to select on are not stored in the mdbase, namely: {}".format(non_existing_datasets))
        else:
            dataset_ids = [int(val) for val in char_df["dataset_id"].to_list()]

        # get characterizations in proper format
        characterizations = []
        for id in dataset_ids:
            if id not in [int(val) for val in list(char_df["dataset_id"].values)]:  # dataset id to select not in characterization
                raise ValueError("Cannot include dataset with id: {} for characterization: {}. It does not exist in the metadatabase.".format(id, characterization_name))
            char_values = list(float(val) for val in char_df[char_df["dataset_id"] == id].values[0])[1:]
            characterizations.append((id, char_values))

        return characterizations

    # TODO add functionality for pipeline characterizations

    # TODO when both pipeline and dataset characterizations are incorporated in the mdbase, then should also do them in partial view
