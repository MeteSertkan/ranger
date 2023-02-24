from typing import Dict
import yaml


class RetrievalLocationConfig:
    '''A class to hold the configuration for retrieval experiments.

    Attributes
    ----------
    display_names : List[str]
        the display names for each of the experiments

    qrels : List[str]
        the qrels files for each of the experiments

    available_rankings : List[Dict[str, str]]
        the available rankings for each of the experiments

    Methods
    -------
    __init__(filename: str)
        Load the configuration from the provided file
    '''

    display_names: Dict[str, str]
    qrels: Dict[str, str]
    available_rankings: Dict[str, Dict[str, str]]

    def __init__(self, filename: str):

        with open(filename, 'r', encoding="utf8") as stream:
            yaml_config = yaml.safe_load(stream)

        self.display_names = yaml_config['display_names']
        self.qrels = yaml_config['qrels']
        self.available_rankings = yaml_config['available_rankings']
