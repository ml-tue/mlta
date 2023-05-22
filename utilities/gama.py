import time
from copy import deepcopy
from math import ceil
from typing import List, Optional

import pandas as pd
from gama import GamaClassifier
from gama.configuration.classification import clf_config
from gama.postprocessing.ensemble import EnsemblePostProcessing
from gama.search_methods.async_ea import AsyncEA
from gama.search_methods.base_search import BaseSearch
from gama.utilities.preprocessing import basic_encoding, basic_pipeline_extension
from sklearn._config import set_config
from sklearn.pipeline import Pipeline


def sklearn_pipe_to_individual_str(
    pipe: Pipeline, search_space: dict = clf_config, remove_steps: List[str] = ["ord-enc", "oh-enc", "imputation"], trim_to_search_space: bool = True
) -> str:
    """Converts a sklearn.pipeline.Pipeline to a Gama Individual string format. Does not support pre-processors in the pipe's str representation.

    Arguments
    ---------
    pipe: sklearn.pipeline.Pipeline,
        the pipeline to be converted to a gama individual string format
    search_space: dict,
        the gama search space that is consistent with the individual strings to be created.
        By default this can simply be the search space defined by GAMA itself.
    remove_steps: list of strings,
        step names in `pipe` that should be removed, this should include pre-processing steps, because this function does not support converting those
    trim_to_search_space: boolean,
        If True not all hyperparameters should be displayed for a primitive/algorithm, only those in the `search_space`
        if False, then all hyperparameters are displayed, also those that are not included in the `search_space` and thus have default values.
    """
    # remove steps
    new_pipe = deepcopy(pipe)
    if remove_steps is not None:
        for name in remove_steps:
            if name not in new_pipe.named_steps:
                raise ValueError("Cannot remove step with name: {} because it is not in the pipeline".format(name))
            index_to_remove = list(new_pipe.named_steps.keys()).index(name)
            new_pipe.steps.pop(index_to_remove)

    set_config(print_changed_only=False)  # print out all parameters

    # not all global parameters should get a class prefix
    global_params = [k for k in search_space.keys() if isinstance(k, str)]
    primitives = [key for key in list(search_space.keys()) if not isinstance(key, str)]
    primitive_names = [p.__name__ for p in primitives]
    primitive_params = {}
    for i, name in enumerate(primitive_names):
        primitive_params[name] = list(search_space[primitives[i]].keys())

    # get the primitive names that do need a prefix for their global params
    prefix_primitives = []
    for prim in primitives:
        for glob_par in global_params:
            if glob_par in list(search_space[prim].keys()):
                if search_space[prim][glob_par] != []:
                    if prefix_primitives.count(prim) == 0:
                        prefix_primitives.append(prim.__name__)
    ind_str = ""
    for i, step in enumerate(new_pipe.steps):
        add_front = ""
        add_back = ""
        s = str(step[1]).replace("\n", "")
        tmp = s.split("(")
        class_name = tmp[0]
        if i == 0:
            tmp.insert(1, "(data, ")
            for elem in tmp:
                add_back += elem
        else:
            add_front = tmp[0] + "("
            add_back = ", " + tmp[1]

        # add class prefix for hyperparams
        hp_strs = add_back.split(", ")
        add_back = ""
        for i, hp_s in enumerate(hp_strs):
            if i == 0:  # first entry has no hp's, so dont adapt it
                add_back += hp_s
                continue
            hp = hp_s.split("=")[0]
            if hp_s == ")":  # prevent adapting entries without hp's like 'StandardScaler()'
                add_back = add_back + hp_s
            else:
                if hp.strip() not in global_params:
                    hp_s = ", " + class_name + "." + hp_s.strip()
                    add_back += hp_s
                else:
                    if class_name in prefix_primitives:
                        hp_s = ", " + class_name + "." + hp_s.strip()
                    else:
                        hp_s = ", " + hp_s.strip()
                    add_back += hp_s
        ind_str = add_front + ind_str + add_back

    if trim_to_search_space:  # if specified remove all hyperparameters that are not in the search space
        tmp_strs = ind_str.split(", ")

        ind_str = ""  # reset output to nothing
        bracket_added_classes = []  # avoid adding closing brackets more than once for the same class
        for s in tmp_strs:
            if "function f_classif" in s:
                if "SelectPercentile" in s:
                    s = "SelectPercentile.score_func=f_classif)"
                if "SelectFwe" in s:
                    s = "SelectFwe.score_func=f_classif)"
            for global_param in global_params:
                if global_param == s.split("=")[0]:
                    ind_str = ind_str + ", " + s
            if "data" in s:  # always add first substr
                ind_str = ind_str + s
            else:  # check if a search space param is in substring
                bracket_added = False
                for p in primitive_names:
                    if p in s:  # p is the class_name in substr
                        if len(primitive_params[p]) == 0 and p not in bracket_added_classes:  # no primitive param so close bracket
                            ind_str = ind_str + ")"
                            bracket_added = True
                            bracket_added_classes.append(p)
                        for i, param in enumerate(primitive_params[p]):
                            # print("param: ", param)
                            # print("s: ", s)
                            if param == s.split("=")[0].split(".")[1]:  # only select param part, because names can partially overlap
                                ind_str = ind_str + ", " + s
                            else:
                                # if the substr contained a bracket need to close it
                                if ")" in s and not bracket_added and i + 1 == len(primitive_params[p]):
                                    # print("add brackets")
                                    ind_str = ind_str + ")"
                                    bracket_added = True

    return ind_str


def create_warm_starters(ind_strs: List[str], n_ind: int) -> List[str]:
    """Create a list of `n_ind` gama individual string representations that can be used in warm_starting.

    Arguments
    ---------
    ind_strs: list of strings,
        gama individual strings that should be incorporated in the warm_starters (i.e. those recommended by a metalearner)
    n_ind: integer,
        The number of warm_starters that should be returned. Should be smaller than the amount of individuals in `ind_strs`,
        the remaining number of individuals are created randomly.
    """
    if len(ind_strs) >= n_ind:
        raise ValueError("The amount of specified individual strings is too large for the amount of individuals")

    gama = GamaClassifier(store="nothing", search_space=clf_config)
    random_starters = [gama._operator_set.individual().pipeline_str() for _ in range(0, n_ind - len(ind_strs))]
    gama.cleanup(which="all")
    return ind_strs + random_starters


def warm_started_gama(
    metalearner,  # should be of type BaseLearner #TODO fix this and actually enforce type, circular import
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_warm_start_configs: int = 25,
    online_phase_max_time: int = 300,
    logs_path: Optional[str] = None,
    scoring: str = "neg_log_loss",
    regularize_length: bool = True,
    max_pipeline_length: Optional[int] = None,
    random_state: Optional[int] = None,
    max_total_time: int = 3600,
    search: BaseSearch = AsyncEA(),
    n_jobs: Optional[int] = None,
    max_memory_mb: Optional[int] = None,
) -> GamaClassifier:
    """Runs GAMA on data (`X`, `y`) and returns its ran object, but warm-started with the `metalearner` using (`X`, `y`).

    Arguments
    ---------
    metalearner: BaseLearner,
        The metalearner whose online phase is ran to warm-start the GAMA run, the object should be initialized properly,
        and the offline_phase should have been run.
    X: pd.DataFrame,
        Features that are used during pipeline training and possible characterization and similarity methods.
    y: pd.Series,
        Targets that are used during pipeline training and possible characterization and similarity methods.
    n_warm_start_configs: integer,
        Specifies the number of configurations that should be create by the metalearner to warm-start the GAMA run.
    logs_path: string or None,
        Option to specify the path where the logs of the run should be stored, if not specified no logs are stored.
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
        Time in seconds that can be used for both the meta-learner's online phase and the subsequent GAMA run.
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

    if online_phase_max_time >= max_total_time:
        raise ValueError("`online_phase_max_time` should be lower than the `max_total_time`")

    # get the warm-starters from the metalearner, record how much time it took and substract it from total time
    start = time.perf_counter()
    metalearner.online_phase(X, y, max_time=online_phase_max_time, metric=scoring, n_jobs=n_jobs, total_n_configs=n_warm_start_configs)
    end = time.perf_counter()
    time_spent = ceil(end - start)
    gama_time = max_total_time - time_spent

    exclude_knn = False
    exclude_polynomial_features = False
    X_ = get_fixed_preprocessed_data(X)
    if X_.shape[0] * X_.shape[1] > 6_000_000:
        exclude_knn = True
    if X_.shape[1] > 50:
        exclude_polynomial_features = True
    ind_strs = []
    for pipe in metalearner.get_top_configurations():
        ind_str = sklearn_pipe_to_individual_str(pipe)
        if exclude_knn and "KNeighborsClassifier" in ind_str:
            continue
        if exclude_polynomial_features and "PolynomialFeatures" in ind_str:
            continue
        ind_strs.append(ind_str)

    # also need to exclude relevant components from randomly generated warmstarters
    if exclude_knn or exclude_polynomial_features:
        warm_starters_to_select = create_warm_starters(ind_strs, n_ind=100)
        final_warm_starters = []
        n = 0
        while len(final_warm_starters) < 50:
            warm_starter = warm_starters_to_select[n]
            if exclude_knn or exclude_polynomial_features:
                if "KNeighborsClassifier" not in warm_starter and "PolynomialFeatures" not in warm_starter:
                    final_warm_starters.append(warm_starter)
            n += 1
    else:
        final_warm_starters = create_warm_starters(ind_strs, n_ind=50)

    if logs_path is None:
        # instantiate GAMA, start training, keep track of models in `automl` object.
        automl = GamaClassifier(
            search_space=clf_config,
            scoring=scoring,
            regularize_length=regularize_length,
            max_total_time=gama_time,
            max_pipeline_length=max_pipeline_length,
            store="nothing",
            random_state=random_state,
            n_jobs=n_jobs,
            max_memory_mb=max_memory_mb,
            search=search,
            post_processing=EnsemblePostProcessing(time_fraction=0, max_models=100),  # dont fit ensemble
            verbosity=0,
        )
    else:
        automl = GamaClassifier(
            search_space=clf_config,
            scoring=scoring,
            regularize_length=regularize_length,
            max_total_time=gama_time,
            max_pipeline_length=max_pipeline_length,
            store="logs",
            random_state=random_state,
            n_jobs=n_jobs,
            max_memory_mb=max_memory_mb,
            output_directory=logs_path,
            search=search,
            post_processing=EnsemblePostProcessing(time_fraction=0, max_models=100),  # dont fit ensemble
            verbosity=0,
        )
    automl.fit(X, y, warm_start=final_warm_starters)

    return automl


def get_fixed_preprocessed_data(X: pd.DataFrame):
    X_, basic_encoding_pipeline = basic_encoding(X, True)
    fixed_pipeline_extension = basic_pipeline_extension(X_, True)
    pipe = Pipeline(basic_encoding_pipeline.steps + fixed_pipeline_extension)
    return pipe.fit_transform(X)


def get_fixed_preprocessing_steps(X: pd.DataFrame):
    X_, basic_encoding_pipeline = basic_encoding(X, True)
    fixed_pipeline_extension = basic_pipeline_extension(X_, True)
    return basic_encoding_pipeline.steps + fixed_pipeline_extension
