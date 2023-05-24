from typing import List

from sklearn.pipeline import Pipeline

from configuration_characterization.base_configuration_characterization import BaseConfigurationCharacterization


class AlgorithmsPropositionalization(BaseConfigurationCharacterization):
    def __init__(self, search_space_config: dict):
        """Intialize the AP pipeline configuration characterization method with the search space it operates on.

        Arguments
        ---------
        search_space_config: dictionary,
            Should follow similar style as in search_space construction in GAMA (e.g. gama.configuration.classification)
            The dictionary should have classes as keys, representing the algorithms to use in the propositionalization.
            The value should be a dictionary with possible parameter values for the classes arguments.
        """
        self._search_space = search_space_config
        super().__init__()

    def compute(self, pipe: Pipeline) -> List[int | float | str]:
        """Computes and returns a configuration characterization for a configuration for the specified Pipeline based on
        algorithms propositionalization. It is a vector of length: #algorithms + #algorithms^2.
        The first #algorithms are 1 if the corresponding algorithm in the search space is present in `pipe`, else 0.
        The other #algorithms^2 entries encode with 1 whether two algorithms direclty follow each other, else the entry is 0.
        Note the configuration characterization does not support preprocessing operators, thus pipe steps with the names:
            "ord-enc", "oh-enc" and "imputation" are removed by default.

        Example
        -------
        A pipe with steps: PolynomialFeatures, StandardScaler, ExtraTreesClassifier has values 1 for the entries corresponding to:
        ExtraTreesClassifier, PolynomialFeatures, StandardScaler, PolynomialFeatures-->StandardScaler and StandardScaler-->ExtraTreesClassifier
        All other values are 0. Given a search space of 25 algorithms, it thus produces a configucation characterization
        composed of a 650 (25 + 25^2) dimensional vector with 5 times a 1 and 645 times a 0.

        Arguments
        ---------
        pipe: sklearn.pipeline.Pipeline,
            The configuration (in the MLTA framework a sklearn pipeline) to be characterized.

        Returns
        -------
        ap_config_characterization: List[int | float | str],
            A list consisting of numerical values or possibly also str characterizating the pipeline,
            str is supported because a characterization of a configuration could contain qualitative information.
        """
        # convert pipe to pipe without preprocessing steps if
        pipe_without_prepro_steps = []
        for step in pipe.steps:
            if step[0] != "ord-enc" and step[0] != "oh-enc" and step[0] != "imputation":
                pipe_without_prepro_steps.append(step)
        pipe_without_prepro = Pipeline([(named_step[0], pipe[named_step[0]]) for named_step in pipe_without_prepro_steps])

        # get the search space algorithms
        algorithms = [entry for entry in list(self._search_space.keys()) if not isinstance(entry, str)]
        algorithm_names = [a.__name__ for a in algorithms]

        # create the one-hot names, vector with 0's and a dictonary mapping names to indices.
        # also include the algorithms themselves therein, to be able to characterize single-algorithm pipelines.
        one_hot_names = [a.__name__ for a in algorithms]
        pipe_config = [0] * int(len(algorithm_names) + int(len(algorithm_names) ** 2))
        for a1 in algorithm_names:
            for a2 in algorithm_names:
                one_hot_names.append(f"{a1}-->{a2}")
        name_to_index = {}
        for i, name in enumerate(one_hot_names):
            name_to_index[name] = i
        pipe_algorithm_names = [step[1].__class__.__name__ for step in pipe_without_prepro.steps]

        # create the configuration characterization
        for i in range(0, len(pipe_algorithm_names)):
            step_name = pipe_algorithm_names[i]
            pipe_config[name_to_index[step_name]] = 1
            if i != len(pipe_algorithm_names) - 1:  # can only take N - 1 2 algoriothms
                alg_comb_name = f"{ pipe_algorithm_names[i]}-->{ pipe_algorithm_names[i+1]}"
                pipe_config[name_to_index[alg_comb_name]] = 1

        return pipe_config
