import mlflow
from pathlib import Path
from typing import Dict, Optional, Any

class MLFlowTracker:
    """
    Wrapper class for ML flow
    """

    def __init__(self, tracking_uri: str = "file:./mlruns", experiment_name: str = "nodule_detection"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = None

    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Starts an mlflow run
        """

        self.run = mlflow.start_run(run_name = run_name)

        if tags: mlflow.set_tags(tags)

        return self.run
    
    def log_parameters(self, parameters: Dict[str, Any]):
        """
        Log config parameters - > will be flattened  so mlflow can use
        """

        flattend_parameters = self.flatten_dictionary(parameters)
        
        for key, value in flattend_parameters.items():
            mlflow.log_param(key, str(value)[:250]) # upper char limit on mlflow pararms


    def flatten_dictionary(self, dictionary: dict, parent_key: str ='', separator: str ='.') -> dict:
        """
        Flatten nested dictionary so mlflow can log stuff
        """

        items = []
        for key, value in dictionary.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dictionary):
                items.extend(self._flatten_dictionary(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        
        return dictionary
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log a batch of metrics, with an optonial step
        """

        for key, value in metrics:
            mlflow.log_metric(key, value, step)

    
    def log_artifacts(self, path: Path):
        """
        Used to log files as run artifacts
        """

        mlflow.log_artifact(str(path))

    def end_run(self):
        if self.run is not None:
            mlflow.end_run()
            self.run = None