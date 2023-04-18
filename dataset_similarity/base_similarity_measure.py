from typing import List, Optional, Tuple

import pandas as pd

from metadatabase import MetaDataBase


class BaseSimilarityMeasure:
    def __init__(self):
        pass

    def compute(X1: pd.DataFrame, y1: Optional[pd.Series], X2: pd.DataFrame, y2: Optional[pd.Series]) -> float:
        """Computes and returns a pair-wise similarity-value representing the (dis)similarity between the datasets.

        Arguments
        ---------
        X1: pd.DataFrame,
            The 1st datasets' features used during similarity computation of the dataset. Must be specified.
        y1: pd.Series or None,
            The 1st dataset's targets that could be used during similarity computation.
            Optional argument because there can be similarity measures only using features not outcomes.
        X2: pd.DataFrame,
            The 2nd datasets' features used during similarity computation of the dataset. Must be specified.
        y2: pd.Series or None,
            The 2nd dataset's targets that could be used during similarity computation.
            Optional argument because there can be similarity measures only using features not outcomes.

        Returns
        -------
        similarity_value: float,
            A numerical value representing the (dis)similarity between the datasets given by (X1, y1) and (X2, y2)
        """
        return NotImplementedError("Method `compute` must be implemented by child class.")

    def get_datasets(X: pd.DataFrame, y: Optional[pd.Series], mdbase: MetaDataBase, n: int, by: str = "similarity") -> List[Tuple[int, List[int | float]]]:
        """Returns mdbase's dataset_ids by their (dis)similarity to specified dataset.

        Arguments
        ---------
        X: pd.DataFrame,
            The dataset's features used during similarity computation of the dataset. Must be specified.
        y: pd.Series or None,
            The dataset's targets that could be used during similarity computation.
            Optional argument because there can be similarity measures only using features not outcomes.
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        n: int,
            the number of datasets that should be returned
        by: str,
            By which criterion the datasets should be selected, two options:
                "similarity", then the `n` datasets most similar to (X,y) are returned.
                "dissimilarity", then the `n` datasets most dissimilar to (X,y) are returned.

        Returns
        -------
        similar_entries: List[Tuple[int, List[int | float]]],
            A list of tuples, where each tuple entry represents one dataset.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is (dis)similarity-value between (X,y) and dataset with dataset_id.
        """
        return NotImplementedError("Method `get_datasets` must be implemented by child class.")
