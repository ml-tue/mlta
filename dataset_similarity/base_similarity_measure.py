from typing import List, Optional, Tuple

import pandas as pd

from metadatabase import MetaDataBase


class BaseSimilarityMeasure:
    def __init__(self):
        pass

    def compute(self, X1: pd.DataFrame, y1: Optional[pd.DataFrame], X2: pd.DataFrame, y2: Optional[pd.Series]) -> float | NotImplementedError:
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

    def get_datasets(self, X: pd.DataFrame, y: Optional[pd.DataFrame], mdbase: MetaDataBase, n: int, by: str = "similarity") -> List[Tuple[int, float]]:
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
        similar_entries: List[Tuple[int, float]],
            A list of tuples, where each tuple entry represents one dataset.
            The first element in the tuple refers to the dataset_id in `mdbase`,
            The second element is (dis)similarity-value between (X,y) and dataset with dataset_id.
        """

        similar_entries: List = []

        # compute similarity between current dataset (`X`, `y`) and all other datasets in `mdbase`
        for dataset_id in mdbase.list_datasets(by="id"):
            df_X, df_y = mdbase.get_dataset(dataset_id, type="dataframe")
            similarity = self.compute(df_X, df_y, X, y)  # type: ignore
            similar_entries.append((dataset_id, similarity))

        # sort and select the `n` datasets by criteria `by`
        if by == "similarity":
            similar_entries = sorted(similar_entries, key=lambda tup: tup[1], reverse=True)
        else:  # "dissimilarity"
            similar_entries = sorted(similar_entries, key=lambda tup: tup[1], reverse=False)

        return similar_entries[:n]
