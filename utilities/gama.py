from copy import deepcopy
from typing import List

from sklearn._config import set_config
from sklearn.pipeline import Pipeline

from gama import GamaClassifier
from gama.configuration.classification import clf_config


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

    # print("ind_str: ", ind_str)

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
