from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from configuration_characterization import BaseConfigurationCharacterization
from dataset_characterization import BaseDatasetCharacterization
from metadatabase import MetaDataBase
from metalearning import BaseLearner


class BaseCharacterizationLearner(BaseLearner):
    def __init__(self):
        super.__init__()
        self._dataset_characterizations: List[Tuple[int, List[int | float]]] = None
        self._config_characterizations: List[Tuple[int, List[int | float | str]]] = None

    def offline_phase(
        self,
        mdbase: MetaDataBase,
        dataset_characterization: Union[List[Tuple[int, List[int | float]]], BaseDatasetCharacterization] = None,
        config_characterization: Union[List[Tuple[int, List[int | float | str]]], BaseConfigurationCharacterization] = None,
    ) -> None:
        """Performs offline computation before the `online_phase()` using the specified metadatabase.
        After this method at least:
            the dataset characterizations should be stored in _dataset_characterizations
            the configuration characterizations should be stored in _config_characterizations
        Gathered information can be stored in the object itself, to later be accessed in `online_phase()`

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        dataset_characterization: List[Tuple[int, List[int | float]]] or BaseDatasetCharacterization,
            If type is BaseDatasetCharacterization, then the specified Characterization measure is computed.
            Otherwise, use given pre-computed dataset characterizations instead of computing them. Should be:
                A list of tuples, where each tuple represents a dataset characterization.
                The first element in the tuple refers to the dataset_id in `mdbase`,
                The second element is the purely numeric vector representing the dataset.
        config_characterization: List[Tuple[int, List[int | float | str]] or BaseConfigurationCharacterization,
            If type is BaseConfigurationCharacterization, then specified characterization method is computed on `mdbase`
            Otherwise, use given pre-computed config characterizations instead of computing them. Should be:
                A list of tuples, where each tuple represents a configuraiton characterization.
                The first element in the tuple refers to the pipeline_id in `mdbase`,
                The second element is the vector representing the configuration (pipeline).
        """
        raise NotImplementedError("Must be implemented by child class.")
