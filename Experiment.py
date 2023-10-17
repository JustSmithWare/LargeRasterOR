from typing import Dict, Any
import json

class Experiment:
    '''
    Manages and logs experimental settings, metrics, logs, and exceptions for machine learning experiments.
    
    Attributes:
    -----------
    config : dict
        Stores the configuration settings for the experiment.
        
    metrics : dict
        Stores evaluation metrics for the experiment.
        
    logs : list
        Stores log messages related to the experiment.
        
    exceptions : list
        Stores exceptions that were encountered during the experiment.
        
    Methods:
    --------
    log_config(config: Dict[str, Any]):
        Updates the configuration settings for the experiment.
        
    log_metrics(metrics: Dict[str, Any]):
        Updates the evaluation metrics for the experiment.
        
    log_message(message: str):
        Appends a textual message to the experiment's logs.
        
    log_exception(exception: str):
        Appends an exception message to the experiment's exception list.
        
    save(path: str):
        Saves the experiment data (config, metrics, logs, exceptions) to a JSON file at the specified path.
        
    Example:
    --------
    >>> exp = Experiment()
    >>> exp.log_config({"learning_rate": 0.01})
    >>> exp.log_metrics({"accuracy": 0.9})
    >>> exp.log_message("Training started")
    >>> exp.log_exception("NullPointerError")
    >>> exp.save("experiment.json")
    '''
    def __init__(self):
        self.config = {}
        self.metrics = {}
        self.logs = []
        self.exceptions = []

    def log_config(self, config: Dict[str, Any]):
        self.config.update(config)

    def log_metrics(self, metrics: Dict[str, Any]):
        self.metrics.update(metrics)

    def log_message(self, message: str):
        self.logs.append(message)

    def log_exception(self, exception: str):
        self.exceptions.append(exception)

    def save(self, path: str):
        data = {
            'config': self.config,
            'metrics': self.metrics,
            'logs': self.logs,
            'exceptions': self.exceptions
        }
        with open(path, 'w') as f:
            json.dump(data, f)