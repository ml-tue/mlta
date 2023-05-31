from hashlib import md5
from typing import List

from sklearn.pipeline import Pipeline

from configuration_characterization.base_configuration_characterization import BaseConfigurationCharacterization
from metadatabase import MetaDataBase


class RankMLPipelineRepresentation(BaseConfigurationCharacterization):
    def __init__(self, mdbase: MetaDataBase):
        """Initializes the RankMLPipelineRepresentation

        Arguments
        ---------
        mdbase: MetaDataBase,
            The metadatabase that stores all pipelines to be characterized. This is necessary because the method needs the
                maximal length of any pipeline in the metadatabase.
        """
        super().__init__()
        self._mdbase = mdbase
        self._max_pipe_length: int = 0

    def compute(self, pipe: Pipeline) -> List[str]:
        """Computes and returns a pipeline characterization for a configuration for the specified Pipeline based on the paper:
            "RankML: a Meta Learning-Based Approach for Pre-Ranking Machine Learning Pipelines" by Laadan 2019, https://arxiv.org/abs/1911.00108

        The paper however is not entirely clear in its method, hence filling in the freedom we chose:
            - To let the hash represent the entire primitive (including hyperparameters)
            - Use the hash on the byte representation of the string representation of the sklearn pipeline.
            - Use UTF-8 encoding to transform string representation into byte representation for hash algorithm.
            - Use the MD5 hashing algorithm.
            - Use the empty string to create the 'blank' hash.

        Moreover, note the configuration characterization does not support preprocessing operators, thus pipe steps with the names:
            "ord-enc", "oh-enc" and "imputation" are removed by default.
        Lastly, it is currently only implemented for linear pipelines (so without a parallel phase), as with any of the GAMA pipelines.
        To create the

        Arguments
        ---------
        pipe: sklearn.pipeline.Pipeline,
            The configuration (in the MLTA framework a sklearn pipeline) to be characterized.

        Returns
        -------
        rankml_config_characterization: List[str],
            A list consisting 32 length hashes (strings) representing each of the primitives in the pipe
        """
        # convert pipe to pipe without preprocessing steps if needed
        pipe_without_prepro_steps = []
        for step in pipe.steps:
            if step[0] != "ord-enc" and step[0] != "oh-enc" and step[0] != "imputation":
                pipe_without_prepro_steps.append(step)
        pipe_without_prepro = Pipeline([(named_step[0], pipe[named_step[0]]) for named_step in pipe_without_prepro_steps])

        if self._max_pipe_length == 0:  # need to get the max pipeline length in the mdbase, check avoids re-computation
            for pipe_id in self._mdbase.list_pipelines(by="id"):
                pipe = self._mdbase.get_sklearn_pipeline(pipe_id, is_classification=True, include_prepro=False)
                self._max_pipe_length = max(len(pipe), self._max_pipe_length)

        blank_hash = md5("".encode("UTF-8")).hexdigest()
        pipe_characterization = [blank_hash] * self._max_pipe_length
        indx = 0
        for step in reversed(pipe_without_prepro.steps):  # sequence pipeline reversed order. So reversed traversal through list but normal index counting.
            primitive_hash = md5(str(step[1]).encode("UTF-8")).hexdigest()
            pipe_characterization[indx] = primitive_hash
            indx += 1

        return pipe_characterization
