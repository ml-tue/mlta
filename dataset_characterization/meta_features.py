import math
from typing import List, Tuple

import pandas as pd
from numpy import mean, percentile, std
from pymfe.mfe import MFE
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from dataset_characterization.base_dataset_characterization import BaseDatasetCharacterization
from utilities import get_fixed_preprocessed_data, isNaN


def _try_fit_mfe(mfe: MFE(), X: pd.DataFrame, y: pd.DataFrame) -> None:
    """Helper function to encapsulate fitting the MFE object for meta-feature computation on data (`X`, `y`)."""
    try:  # TODO: cleanup try except clauses in a function
        mfe.fit(X.to_numpy(), y.to_numpy(), transform_num=None, transform_cat="gray", suppress_warnings=True)
    except ValueError as e:  # data transformation may fail, because of preprocessing issues
        processed_X = get_fixed_preprocessed_data(X)
        mfe.fit(processed_X, y.to_numpy(), transform_num=None, transform_cat=None, suppress_warnings=True)
    except RecursionError as r:  # gray may fail because of a bug with Recursion, in that case try to preprocess data.
        processed_X = get_fixed_preprocessed_data(X)
        mfe.fit(processed_X, y.to_numpy(), transform_num=None, transform_cat=None, suppress_warnings=True)


class WistubaMetaFeatures(BaseDatasetCharacterization):
    def __init__(self):
        super().__init__()

    def compute(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[List[int | float], List[str]]:
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
        characterization: Tuple[List[int | float], List[str]],
            A tuple consisting of lists of feature values and feature names respectively
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
        _try_fit_mfe(wistuba_mfe, X, y)
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

        return feature_values, feature_names


class FeurerMetaFeatures(BaseDatasetCharacterization):
    def __init__(self):
        super().__init__()

    def compute(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[List[int | float], List[str]]:
        """Computes and returns a characterization using Feurer Meta-features for the dataset given by X and y,
        specifically: Get the meta-features as defined in the paper by Feurer et al. (2014):
            'Using Meta-Learning to Initialize Bayesian Optimization of Hyperparameters'
        For meta-feature computation the package pyMFE is used.

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used during meta-feature characterization of the dataset, passed to MFE. Must be specified.
        y: pd.Series or None,
            Targets used during meta-feature computation.
        Returns
        -------
        characterization: Tuple[List[int | float], List[str]],
            A tuple consisting of lists of feature values and feature names respectively
        """
        # first compute pyMFE meta-features
        feurer_features_sum = [
            "nr_inst",  # nr of patterns in paper
            # log_nr_inst, # needs to be added through tranformation
            "nr_class",
            "nr_attr",  # nr of features in paper
            # log_nr_attr, # needs to be added through tranformation
            "cat_to_num",
            "nr_num",
            "nr_cat",
            "num_to_cat",  # ratio numerical to categorical
            "cat_to_num",  # ratio categorical to numerical
            "attr_to_inst",  # also the data dimensionality
            # "log_attr_to_inst", # needs to be added through tranformation
            "inst_to_attr",  # inverse of the data dimensionality
            # "log_inst_to_attr", # needs to be added through tranformation
            "freq_class",  # is computed for min, max, mean, std in summary
            "class_ent",
            "kurtosis",  # is computed for min, max, mean, std in summary
            "skewness",  # is computed for min, max, mean, std in summary
        ]
        # run pyMFE for features that do want summarization functions if applicable
        feurer_summary = ["min", "max", "mean", "sd"]
        feurer_mfe_sum = MFE(groups="all", features=feurer_features_sum, summary=feurer_summary, suppress_warnings=True)
        _try_fit_mfe(feurer_mfe_sum, X, y)
        ft_untransformed = feurer_mfe_sum.extract(suppress_warnings=True)
        feature_names = ft_untransformed[0]
        feature_values = ft_untransformed[1]

        # run pyMFE a second time for the meta-features without summarization functions. Use "mean" summarization to mimic no summary.
        feurer_features_nosum = [
            "one_nn",  # One Nearest Neighbor
            "linear_discr",  # Linear Discriminant Analysis
            "naive_bayes",  # Naive Bayes
            "best_node",  # Decision Node Learner
            "random_node",  # Random Node Learner
        ]
        feurer_mfe_nosum = MFE(groups="all", features=feurer_features_nosum, summary=("mean"), suppress_warnings=True)
        _try_fit_mfe(feurer_mfe_nosum, X, y)
        ft_nosum = feurer_mfe_nosum.extract(suppress_warnings=True)
        feature_names = feature_names + ft_nosum[0]
        feature_values = feature_values + ft_nosum[1]

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

        ### next blocks of code: add the meta-features not faciliated by pyMFE
        # nr of patterns with missing values, % of patterns with missing values
        miss_data_rows = X[X.isnull().any(axis=1)]
        nr_rows_with_missval = miss_data_rows.shape[0]
        miss_rows_perc = float(nr_rows_with_missval / X.shape[0])
        feature_values.append(nr_rows_with_missval)
        feature_names.append("nr_rows_with_missval")
        feature_values.append(miss_rows_perc)
        feature_names.append("miss_rows_perc")

        # nr of features with missing values, % of features with missing values
        miss_data_cols = X.loc[:, X.isnull().any()]
        nr_cols_with_missval = miss_data_cols.shape[1]
        miss_cols_perc = float(nr_cols_with_missval / X.shape[1])
        feature_values.append(nr_cols_with_missval)
        feature_names.append("nr_cols_with_missval")
        feature_values.append(miss_cols_perc)
        feature_names.append("miss_cols_perc")

        # nr of missing values, % of missing values
        n_miss_values = 0
        for col in X.columns:
            n_miss_values += sum(X[col].isnull())
        perc_miss_values = float(n_miss_values / int(X.shape[0] * X.shape[1]))
        feature_values.append(n_miss_values)
        feature_names.append("n_miss_values")
        feature_values.append(perc_miss_values)
        feature_names.append("perc_miss_values")

        # categorical values per feature in dataset, and take min, max, mean, std and total of that
        n_distinct_categories = []
        for col in X.columns:
            if isinstance(X[col].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
                n_distinct_categories.append(X[col].unique().size)
        names = ["categorical_values.min", "categorical_values.max", "categorical_values.mean", "categorical_values.std", "categorical_values.total"]
        values = []
        if len(n_distinct_categories) == 0:  # no categorical atrributes/values, so default values
            values = [0, 0, 0, 0, 0]
        else:
            values = [
                min(n_distinct_categories),
                max(n_distinct_categories),
                sum(n_distinct_categories) / len(n_distinct_categories),
                std(n_distinct_categories),
                sum(n_distinct_categories),
            ]
        for value, name in zip(values, names):
            feature_values.append(value)
            feature_names.append(name)

        # perform PCA and compute various statistics of the principal components (95%, skewness and kurtosis of 1st pc)
        processed_X = get_fixed_preprocessed_data(X)
        first_pc = PCA().fit_transform(processed_X)[0]
        names = ["pca.95%", "pca.skewness", "pca.kurtosis"]
        values = [percentile(first_pc, 0.95), skew(first_pc), kurtosis(first_pc)]
        for value, name in zip(values, names):
            feature_values.append(value)
            feature_names.append(name)

        # train a decision tree and record performance
        dt_score = mean(cross_validate(DecisionTreeClassifier(), processed_X, y, scoring="f1_macro")["test_score"])
        feature_values.append(dt_score)
        feature_names.append("Decision tree")

        return feature_values, feature_names  # TODO adapt this after testing
