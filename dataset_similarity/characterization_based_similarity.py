from typing import List, Optional

import pandas as pd
from scipy import spatial

from dataset_characterization import BaseDatasetCharacterization
from dataset_similarity import BaseSimilarityMeasure


class CharacterizationSimilarity(BaseSimilarityMeasure):
    def __init__(self, characterization_method: BaseDatasetCharacterization, compare_characterizations_by: str = "cosine_similarity"):
        """
        Arguments
        ---------
        characterization_method: class inherenting from BaseDatasetCharacterization,
            characterization method to characterize the dataset.
        compare_characterizations_by: string,
            specifies the option by which the (dis)similarity between characterizations should be computed, options are:
                "cosine_similarity" or "euclidian_distance"
        """
        super().__init__()
        self._characterization_method = characterization_method
        self._compare_characterizations_by = compare_characterizations_by

    def compute(self, X1: pd.DataFrame, y1: Optional[pd.DataFrame], X2: pd.DataFrame, y2: Optional[pd.DataFrame]) -> float:
        """Computes and returns a pair-wise similarity-value representing the (dis)similarity between the datasets
        By using the specified Characterization Method.

        Arguments
        ---------
        X1: pd.DataFrame,
            The 1st datasets' features used during similarity computation of the dataset. Must be specified.
        y1: pd.DataFrame or None,
            The 1st dataset's targets that could be used during similarity computation.
            Optional argument because there can be similarity measures only using features not outcomes.
        X2: pd.DataFrame,
            The 2nd datasets' features used during similarity computation of the dataset. Must be specified.
        y2: pd.DataFrame or None,
            The 2nd dataset's targets that could be used during similarity computation.
            Optional argument because there can be similarity measures only using features not outcomes.


        Returns
        -------
        similarity_value: float,
            A numerical value representing the (dis)similarity between the datasets given by (X1, y1) and (X2, y2)
        """
        charac_1 = self._characterization_method.compute(X1, y1)
        charac_2 = self._characterization_method.compute(X2, y2)

        return self.compute_from_characterizations(charac_1, charac_2)  # type: ignore

    def compute_from_characterizations(self, characterization_1: List[int | float], characterization_2: List[int | float]) -> float:
        """
        Computes the similarity between two characterizations by the method specified with argument `by`.
        Assumes characterizations to be vectors of similar dimensions.

        Arguments
        ---------
        characterization_1: List[int | float],
            characterizes the first dataset
        characterization_2: List[int | float],
            characterizes the second dataset
        """
        if any(isinstance(i, List) for i in characterization_1) or any(isinstance(i, List) for i in characterization_2):
            raise ValueError("Characterization(s) is/are not single-dimensional, they should be vectors")
        if len(characterization_1) != len(characterization_2):
            raise ValueError("Characterizations should be of similar length, they are different now.")

        if self._compare_characterizations_by == "cosine_similarity":
            return 1 - spatial.distance.cosine(characterization_1, characterization_2)
        else:  # euclidian_distance
            return spatial.distance.euclidean(characterization_1, characterization_2)
