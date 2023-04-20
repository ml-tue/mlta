from typing import List, Optional, Tuple

from metadatabase import MetaDataBase
from metalearners.base_learner import BaseLearner


class BaseCharacterizationLearner(BaseLearner):
    def __init__(self):
        self._dataset_characterizations: Optional[List[Tuple[int, List[int | float]]]] = None
        self._config_characterizations: Optional[List[Tuple[int, List[int | float | str]]]] = None

    def offline_phase(
        self,
        mdbase: MetaDataBase,
        dataset_characterization: Optional[List[Tuple[int, List[int | float]]]],
        config_characterization: Optional[List[Tuple[int, List[int | float | str]]]],
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
        dataset_characterization: List[Tuple[int, List[int | float]]] or None,
            If type is None, then the Characterizations should be computed within this function.
            Otherwise, use given pre-computed dataset characterizations instead of computing them. Should be:
                A list of tuples, where each tuple represents a dataset characterization.
                The first element in the tuple refers to the dataset_id in `mdbase`,
                The second element is the purely numeric vector representing the dataset.
        config_characterization: List[Tuple[int, List[int | float | str]] or None,
            If type is None, then specified configuration characterization method should be computed on `mdbase`
            Otherwise, use given pre-computed config characterizations instead of computing them. Should be:
                A list of tuples, where each tuple represents a configuraiton characterization.
                The first element in the tuple refers to the pipeline_id in `mdbase`,
                The second element is the vector representing the configuration (pipeline).
        """
        raise NotImplementedError("Must be implemented by child class.")
