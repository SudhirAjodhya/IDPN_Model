from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import dacite

@dataclass
class AnatomyConfig:
    lung_y: List[float]
    lung_x_left: List[float]
    lung_x_right: List[float]
    mediastinum_y: List[float]
    mediastinum_x: List[float]

@dataclass
class DataConfig:
    dataset: str
    input_size: int
    target_size: int
    pixel_spacing_mm: float
    anatomy: AnatomyConfig
    root: Optional[str] = None
    cross_val: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None

@dataclass
class ModelConfig:
    variant: str
    head: str
    backbone: str
    fpn_out: int

@dataclass
class OptimizerConfig:
    type: str
    learning_rate: float
    backbone_lr_factor: float
    weight_decay: float

@dataclass
class SchedulerConfig:
    type: str
    eta_min: float

@dataclass
class EarlyStoppingConfig:
    patience: int
    monitor: str

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    num_workers: int
    seed: int
    test_seed: int
    patience: int
    grad_accum: int
    mixed_precision: bool
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: Optional[EarlyStoppingConfig] = None

@dataclass
class LossConfig:
    type: str
    alpha: float
    beta: float
    tv_weight: float
    aux_mask_weight: float
    aux_density_high: float
    aux_density_low: float

@dataclass
class EvaluationConfig:
    score_threshold: float
    dist_threshold_mm: float
    heatmap_sigma_min: float
    froc: bool

@dataclass
class LoggingConfig:
    mlflow_uri: str
    mlflow_experiment: str

@dataclass
class ExperimentConfig:
    name: str
    direction: str
    method_tag: str
    notes: str

@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    _base_: Optional[str] = None


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge dict `override` into `base`."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result

def load_config(path: Path) -> Config:
    """
    Load a YAML config file into a typed Config dataclass, handling _base_ inheritance.
    """
    with open(path) as file:
        raw = yaml.safe_load(file)

    if "_base_" in raw:
        base_path = path.parent / raw["_base_"]
        with open(base_path) as file:
            base_raw = yaml.safe_load(file)
        
        # remove _base_ key so it's not strictly required in the dataclass
        del raw["_base_"]
        raw = merge_dicts(base_raw, raw)

    return dacite.from_dict(data_class=Config, data=raw)