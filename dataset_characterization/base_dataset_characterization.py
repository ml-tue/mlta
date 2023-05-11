from datetime import datetime
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

    def compute_mdbase_characterizations(self, mdbase: MetaDataBase, ids: Optional[List[int]] = None, verbosity: int = 1) -> List[Tuple[int, List[int | float]]]:
        """Computes and returns the dataset characterization for all datasets in the specified metadatabase.

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
