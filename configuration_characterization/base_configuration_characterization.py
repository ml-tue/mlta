from typing import List, Optional, Tuple

from sklearn.pipeline import Pipeline

from metadatabase import MetaDataBase


class BaseConfigurationCharacterization:
    def __init__(self):
        pass

    def compute(pipe: Pipeline) -> List[int | float | str]:
        """Computes and returns a characterization for a configuration for the specified Pipeline.

        Arguments
        ---------
        pipe: sklearn.pipeline.Pipeline,
            The configuration (in this framework a sklearn pipeline object) to be characterized.

        Returns
        -------
        config_characterization: List[int | float | str],
            A list consisting of numerical values or possibly also str characterizating the pipeline,
            str is supported because a characterization of a configuration could contain qualitative information.
        """
        return NotImplementedError("Method `compute` must be implemented by child class.")

    def compute_mdbase_characterizations(mdbase: MetaDataBase) -> List[Tuple[int, List[int | float | str]]]:
        """Computes and returns the characterizations for all configuration (pipes) in the specified metadatabase.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.

        Returns
        -------
        config_characterizations: List[Tuple[int, List[int | float | str]]],
            A list of tuples, where each tuple represents a configuration characterization.
            The first element in the tuple refers to the pipeline_id in `mdbase`,
            The second element is the vector representing the configuration (pipeline),
        """
        return NotImplementedError("Method `compute_mdbase_characterizations` must be implemented by child class.")
