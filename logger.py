"""Module bundling all logging related classes and function for LargeRasterOR model training.

Provides a DefaultModelLogger class that prints logs and copies file artifacts to a provided directory.
"""

from typing import Protocol
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import os
import shutil
from typing import Dict, Any
from Experiment import Experiment

class LoggerProtocol(Protocol):
    def log(self, message: str) -> None:
        pass

class DefaultModelLogger(Logger):
    '''
    Provides an interface to log experiment parameters and artifacts to a given directory.

    Attributes:
    -----------
    log_save_dir : str
        The directory where all the logs and artifacts will be saved.

    Methods:
    --------
    experiment:
        Property method to get the underlying Experiment instance.

    name:
        Property method to get the name of the logger.

    version:
        Property method to get the version of the logger.

    log_hyperparams(params: Dict[str, Any]):
        Log hyperparameters to the experiment.

    log_metrics(metrics: Dict[str, Any], step: int):
        Log metrics to the experiment.

    save():
        Save the experiment configuration and metrics to a JSON file.

    finalize(status: str):
        Finalize the logging, saving the experiment and logging the final status.

    log(message: str):
        Log a custom message to the experiment.

    default_artifact_logging(filepath: str, name: str, type: str, description: str):
        Log an artifact by copying it to a designated artifact directory and log the action.

    Example:
    --------
    >>> logger = DefaultModelLogger("logs/")
    >>> logger.log_hyperparams({"learning_rate": 0.01})
    >>> logger.log_metrics({"accuracy": 0.9}, step=1)
    >>> logger.save()
    >>> logger.finalize("success")
    '''
    def __init__(self, log_save_dir: str):
        super().__init__()
        self.log_save_dir = log_save_dir
        self._experiment = Experiment()

    @property
    @rank_zero_only
    def experiment(self):
        return self._experiment

    @property
    def name(self):
        return "DefaultModelLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]):
        self._experiment.log_config(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        self._experiment.log_metrics(metrics)

    @rank_zero_only
    def save(self):
        experiment_path = os.path.join(self.log_save_dir, 'experiment.json')
        self._experiment.save(experiment_path)
        
    @rank_zero_only
    def finalize(self, status: str):
        self._experiment.log_message(f"Training ended with status: {status}")
        self.save()

    def log(self, message: str):
        self._experiment.log_message(message)

    @rank_zero_only
    def default_artifact_logging(self, filepath: str, name: str, type: str, description: str):
        artifact_dir = os.path.join(self.log_save_dir, 'artifacts')
        os.makedirs(artifact_dir, exist_ok=True)
        shutil.copy(filepath, os.path.join(artifact_dir, name))
        self._experiment.log_message(f'Artifact file {name} copied to {artifact_dir}')

def get_logging_key(hyperparams: dict) -> str:
    '''
    Creates a descriptive name for a given model given its hyperparameters.

    Parameters:
    - hyperparams (dict): Dictionary of hyperpameters that will be used to generate a name.

    Returns:
    - str: Descriptive model name made from the provided hyperparameters.

    '''
    logging_key_parts = []
    
    key_to_abbreviation = {
        'patch_size': 'patch',
        'n_epochs': 'epochs',
        'sample_size': 'sample'
    }
    
    # Iterate through the abbreviation-key mappings
    for key, abbreviation in key_to_abbreviation.items():
        # If the key exists in hyperparams, append its abbreviation and value to logging_key_parts
        if key in hyperparams:
            logging_key_parts.append(f"{abbreviation}_{hyperparams[key]}")
    
    # Handle any additional keys that were not anticipated
    for key in hyperparams.keys():
        if key not in key_to_abbreviation:
            logging_key_parts.append(f"{key}_{hyperparams[key]}")
    
    # Join all parts to form the final logging key
    logging_key = '_'.join(logging_key_parts)
    
    return logging_key