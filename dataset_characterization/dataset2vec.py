"""
This file provides an (adopted) implementation of dataset2vec from the Github Repo:
    https://github.com/hadijomaa/dataset2vec
It's code (in utilties package) has been adapted to fit the toolbox structure and to allow for easy use of its pre-trained models.
"""

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from dataset_characterization import BaseDatasetCharacterization
from utilities import get_fixed_preprocessed_data
from utilities.dataset2vec import Batch, Dataset, Model, TestSampling, get_configuration, get_weights_path


class PreTrainedDataset2Vec(BaseDatasetCharacterization):
    def __init__(self, split_n: int = 0, n_batches=10):
        """
        split_n: integer,
            Which of the pre-trained models from dataset2vec to use. Can choose among 0, 1, 2, 3 or 4
        n_batches
            The number of batches (size 32 each) that are taken from the dataset to compute the characterization.
            It affects the characterization values; batches are averaged out. But also the running time, almost linearly.
            Default in dataset2vec source code is 10, and therefore this `n_batches` defaults to 10.
        """
        super().__init__()
        self._split_n = split_n
        self._n_batches = n_batches

    def compute(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[List[int | float], List[str]]:
        """Computes and returns a characterization using the `split_n`'th pre-trained model from Dataset2Vec for the dataset given by (`X`, `y`),

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used during dataset2vec's characterization of the dataset.
            Though Dataset2Vec assumes cleaned data, the data passed can be unclean. Data preprocessing (imputation/encoding) is performed in this function.
        y: pd.Series or None,
            Targets used during dataset2vecs characterization of the dataset.
            Though Dataset2Vec assumes cleaned data, the data passed can be unclean. Data preprocessing (imputation/encoding) is performed in this function.

        Returns
        -------
        characterization: Tuple[List[int | float], List[str]],
            A tuple consisting of lists of feature values and feature names respectively.
            The feature names refer to dimension of the latent NN space's dimensions,
            but they also includes the batch number and pre-computed split that is used (which are constant) for reproducability.
        """

        if self._split_n not in [0, 1, 2, 3, 4]:
            raise ValueError(f"The number of splits `n_splits` given is {self._split_n}, but it must be 0, 1, 2, 3, or 4")

        # set similar seeds as Dataset2Vec paper for reproducible meta-features
        tf.random.set_seed(0)
        np.random.seed(42)
        random.seed(3718)

        # intialize dataset2vec model
        configuration = get_configuration()  # pre-trained model's configuration
        model = Model(configuration)
        d2v_model = model.dataset2vecmodel(trainable=False)  # use pre-trained model, so do not train model
        batch = Batch(configuration["batch_size"])
        d2v_model.load_weights(get_weights_path(self._split_n), by_name=False, skip_mismatch=False)

        # dataset loading. Dataset2vec seems to use preprocessed data, so do that first.
        y_processed = np.asarray(LabelEncoder().fit_transform(y))
        X_processed = np.asarray(get_fixed_preprocessed_data(X))
        dataset = Dataset(X_processed, y_processed)
        testsampler = TestSampling(dataset=dataset)

        # compute meta-features
        metafeatures = pd.DataFrame(data=None)
        datasetmf = []
        for _ in range(self._n_batches):  # the number of batches that should be sampled. Determines output size.
            batch = testsampler.sample_from_one_dataset(batch)
            batch.collect()
            datasetmf.append(d2v_model(batch.input)["metafeatures"].numpy())
        metafeatures = list(np.vstack(datasetmf).mean(axis=0)[None][0])

        # create the meta-feature names based on the split, number of batches and which latent dimensions it is.
        feature_names = [f"split_{self._split_n}_batch_{self._n_batches}_dim{dim}" for dim in range(0, len(metafeatures))]

        return metafeatures, feature_names
