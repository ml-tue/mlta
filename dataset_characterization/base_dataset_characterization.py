from typing import List, Optional, Tuple

import pandas as pd

from metadatabase import MetaDataBase


class BaseDatasetCharacterization:
    def __init__(self):
        pass

    def compute(self, X: pd.DataFrame, y: Optional[pd.DataFrame]) -> List[int | float]:
        """Computes and returns a characterization for the dataset given by X and y.

        Arguments
        ---------
        X: pd.DataFrame,
            Features that are used during characterization of the dataset. Must be specified.
        y: pd.Series or None,
            Targets that could be used during dataset characterization.
            Optional argument because there can be characterization methods only using features not outcomes.

        Returns
        -------
        characterization: List[int | float],
            A list of numerical values characterizating the dataset (given by `X` and `y`)
        """
        raise NotImplementedError("Method `compute` must be implemented by child class.")

    def compute_mdbase_characterizations(self, mdbase: MetaDataBase) -> List[Tuple[int, List[int | float]]]:
        """Computes and returns the characterizations for all datasets in the specified metadatabase.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.

        Returns
        -------
        dataset_characterizations: List[Tuple[int, List[int | float]]],
            A list of tuples, where each tuple represents a dataset characterization.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is the purely numeric vector representing the dataset,
        """
        raise NotImplementedError("Method `compute_mdbase_characterizations` must be implemented by child class.")
