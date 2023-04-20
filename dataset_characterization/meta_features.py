import math
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from pymfe.mfe import MFE

from dataset_characterization.base_dataset_characterization import BaseDatasetCharacterization
from metadatabase.metadatabase import MetaDataBase
from utilities import isNaN


class WistubaMetaFeatures(BaseDatasetCharacterization):
    def __init__(self):
        super().__init__()

    def compute(self, X: pd.DataFrame, y: pd.DataFrame) -> List[int | float]:
        """Computes and returns a characterization using Wistuba Meta-features for the dataset given by X and y,
        specifically: Get the meta-features as defined in the paper by Wistuba et al. (2016):
            'Two-stage transfer surrogate model for automatic hyperparameter optimization'
        For meta-feature computation the package pyMFE is used.

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used during meta-feature characterization of the dataset, passed to MFE. Must be specified.
        y: pd.Series or None,
            Targets used during meta-feature computation.
        Returns
        -------
        characterization: List[int | float],
            A list of numerical values characterizating the dataset (given by `X` and `y`)
        """

        wistuba_features = [
            "nr_class",
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
            "freq_class",  # is computed for min, max, mean, std in summary
            "kurtosis",  # is computed for min, max, mean, std in summary
            "skewness",  # is computed for min, max, mean, std in summary
        ]

        wistuba_summary = ["min", "max", "mean", "sd"]
        wistuba_mfe = MFE(groups="all", features=wistuba_features, summary=wistuba_summary, suppress_warnings=True)
        wistuba_mfe.fit(X.to_numpy(), y.to_numpy(), transform_num=None, transform_cat=None, suppress_warnings=True)
        ft_untransformed = wistuba_mfe.extract(suppress_warnings=True)
        feature_names = ft_untransformed[0]
        feature_values = ft_untransformed[1]

        # get the log transforms
        features_to_add = ["log_nr_inst", "log_nr_attr", "log_attr_to_inst", "log_inst_to_attr"]
        for feature in features_to_add:
            index = feature_names.index(feature.split("log_")[1])  # index of feature to take log of
            feature_names.insert(index + 1, feature)
            feature_values.insert(index + 1, math.log(feature_values[index]))

        # set NaNs to 0, likely only happens to class/num ratios, so 0 is a sensible value.
        for i, feature_value in enumerate(feature_values):
            if isNaN(feature_value):
                feature_values[i] = 0

        return feature_values

    def compute_mdbase_characterizations(self, mdbase: MetaDataBase, ids: Optional[List[int]] = None, verbosity: int = 1) -> List[Tuple[int, List[int | float]]]:
        """Computes and returns the Wistuba Meta-features for all datasets in the specified metadatabase.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        ids: List of integers,
            specifies which datasets (as given by their `mdbase` ids) should be characterized,
            default is `None`, meaning that all datasets in `mdbase` will be characterized
        verbosity: integer,
            if 1 or greater, then a message is printed for each completed dataset

        Returns
        -------
        dataset_characterizations: List[Tuple[int, List[int | float]]],
            A list of tuples, where each tuple represents a dataset characterization.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is the purely numeric vector representing the dataset,
        """
        dataset_characterizations = []
        if ids is None:
            ids = mdbase.list_datasets("id")
        for dataset_id in ids:
            df_X, df_y = mdbase.get_dataset(dataset_id, type="dataframe")
            meta_features = self.compute(df_X, df_y)  # type: ignore
            dataset_characterizations.append((dataset_id, meta_features))
            if verbosity >= 1:
                print("Done characterizing dataset with id: {} at {}".format(dataset_id, str(datetime.now())))
        return dataset_characterizations
