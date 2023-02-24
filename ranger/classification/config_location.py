from typing import Dict, List
import yaml


class ClassificationLocationConfig:
    '''A class to hold the configuration for classification experiments.

    Attributes
    ----------
    display_names : List[str]
        the display names for each of the experiments

    eval_metrics : List[Dict[str, List[str]]]
        the available metrics from the evaluation for each of the experiments

    Methods
    -------
    __init__(filename: str)
        Load the configuration from the provided file
    '''

    display_names: Dict[str, str]
    eval_metrics: Dict[str, List[str]]

    def __init__(self, filename: str):

        with open(filename, 'r', encoding="utf8") as stream:
            yaml_config = yaml.safe_load(stream)

        self.display_names = yaml_config['display_names']
        self.eval_metrics = yaml_config['eval_metrics']