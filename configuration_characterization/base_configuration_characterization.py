from typing import List, Tuple

from sklearn.pipeline import Pipeline

from metadatabase import MetaDataBase


class BaseConfigurationCharacterization:
    def __init__(self):
        pass

    def compute(self, pipe: Pipeline) -> List[int | float | str]:
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
        raise NotImplementedError("Method `compute` must be implemented by child class.")

    def compute_mdbase_characterizations(self, mdbase: MetaDataBase, verbosity: int = 1) -> List[Tuple[int, List[int | float | str]]]:
        """Computes and returns the characterizations for all configuration (pipes) in the specified metadatabase.

        Arguments
        ---------
        mdbase: MetaDataBase,
            metadatabase of prior experiences, as created in metadatabase.MetaDataBase class.
        verbosity: int,
            if set to 1 then an update is given every 1000 pipes

        Returns
        -------
        config_characterizations: List[Tuple[int, List[int | float | str]]],
            A list of tuples, where each tuple represents a configuration characterization.
            The first element in the tuple refers to the pipeline_id in `mdbase`,
            The second element is the vector representing the configuration (pipeline),
        """
        config_characterizations = []
        for pipe_id in mdbase.list_pipelines(by="id"):
            pipe = mdbase.get_sklearn_pipeline(pipeline_id=pipe_id, is_classification=True)
            config_characterizations.append((pipe_id, self.compute(pipe)))
            del pipe
        return config_characterizations
