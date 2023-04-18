import arff
from pymfe.mfe import MFE
from typing import List, Tuple
import math
from utilities import isNaN
from collections.abc import Iterable

def get_wistuba_metafeatures_from_arff(arff_file: str) -> Tuple[List, List]:
    """Get the meta-features as defined in the paper by Wistuba et al. (2016):
            'Two-stage transfer surrogate model for automatic hyperparameter optimization'
        Assumes the last attribute of the arff file to be the target

    Arguments
    ---------
    arff_file: str
        path to the arff file to compute the metafeatures for


    Returns
    -------
    Tuple of lists, the first being the metafeature values, the second being the names
    """
    data = arff.load(open(arff_file, "r"))["data"]
    # select the last value to be the target
    X = [i[:-1] for i in data]
    y = [i[-1] for i in data]

    wistuba_features = ["nr_class",
                    "nr_inst",
                    # log_nr_inst, # needs to be added through tranformation
                    "nr_attr",
                    # log_nr_attr, # needs to be added through tranformation
                    "cat_to_num",
                    "nr_cat",
                    "nr_num",
                    "num_to_cat",
                    "attr_to_inst",
                    # "log_attr_to_inst", # needs to be added through tranformation
                    "inst_to_attr",
                    # "log_inst_to_attr", # needs to be added through tranformation
                    "class_ent",
                    "freq_class", # is computed for min, max, mean, std in summary
                    "kurtosis", # is computed for min, max, mean, std in summary
                    "skewness" # is computed for min, max, mean, std in summary
    ]
    wistuba_summary = ["min", "max", "mean", "sd"]
    wistuba_mfe = MFE(groups="all", features=wistuba_features, summary=wistuba_summary, suppress_warnings=True)
    wistuba_mfe.fit(X, y, transform_num=None, transform_cat=None, suppress_warnings=True)
    ft_untransformed = wistuba_mfe.extract(suppress_warnings=True)
    feature_names = ft_untransformed[0]
    feature_values = ft_untransformed[1]
    
    # get the log transforms
    features_to_add = ["log_nr_inst", "log_nr_attr", "log_attr_to_inst", "log_inst_to_attr"]
    for feature in features_to_add:
        index = feature_names.index(feature.split("log_")[1]) # index of feature to take log of
        feature_names.insert(index + 1, feature)
        feature_values.insert(index + 1, math.log(feature_values[index]))
    
    # set NaNs to 0, likely only happens to class/num ratios, so 0 is a sensible value.
    for i, feature_value in enumerate(feature_values):
        if isNaN(feature_value):
            feature_values[i] = 0

    # features_to_remove = ["nr_inst", "inst_to_attr"]
    # for i, feature in enumerate(feature_names):
    #     if feature in features_to_remove:
    #         del feature_values[i]
    #         del feature_names[i]
    
    return feature_values, feature_names

# overloaded function, to also work with arrays format instead of arff file
def get_wistuba_metafeatures(X: Iterable, y) -> Tuple[List, List]:
    """Get the meta-features as defined in the paper by Wistuba et al. (2016):
            'Two-stage transfer surrogate model for automatic hyperparameter optimization'

    Arguments
    ---------
    X: Iterable,
        Features that are used in the meta-feature computation and pipeline training
    y: Iterable,
        Targets that are used in the meta-feature computation and pipeline training

    Returns
    -------
    Tuple of lists, the first being the metafeature values, the second being the names
    """
    wistuba_features = ["nr_class",
                    "nr_inst",
                    # log_nr_inst, # needs to be added through tranformation
                    "nr_attr",
                    # log_nr_attr, # needs to be added through tranformation
                    "cat_to_num",
                    "nr_cat",
                    "nr_num",
                    "num_to_cat",
                    "attr_to_inst",
                    # "log_attr_to_inst", # needs to be added through tranformation
                    "inst_to_attr",
                    # "log_inst_to_attr", # needs to be added through tranformation
                    "class_ent",
                    "freq_class", # is computed for min, max, mean, std in summary
                    "kurtosis", # is computed for min, max, mean, std in summary
                    "skewness" # is computed for min, max, mean, std in summary
    ]
    wistuba_summary = ["min", "max", "mean", "sd"]
    wistuba_mfe = MFE(groups="all", features=wistuba_features, summary=wistuba_summary, suppress_warnings=True)
    wistuba_mfe.fit(X, y, transform_num=None, transform_cat=None, suppress_warnings=True)
    ft_untransformed = wistuba_mfe.extract(suppress_warnings=True)
    feature_names = ft_untransformed[0]
    feature_values = ft_untransformed[1]
    
    # get the log transforms
    features_to_add = ["log_nr_inst", "log_nr_attr", "log_attr_to_inst", "log_inst_to_attr"]
    for feature in features_to_add:
        index = feature_names.index(feature.split("log_")[1]) # index of feature to take log of
        feature_names.insert(index + 1, feature)
        feature_values.insert(index + 1, math.log(feature_values[index]))
    
    # set NaNs to 0, likely only happens to class/num ratios, so 0 is a sensible value.
    for i, feature_value in enumerate(feature_values):
        if isNaN(feature_value):
            feature_values[i] = 0
    
    return feature_values, feature_names