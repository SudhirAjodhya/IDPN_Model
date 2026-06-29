from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
import dacite

@dataclass
class DataConfig:
    dataset: str
    root: str
    input_size: int
    output_size: int 
    cross_val: Dict[str, Any]
    augmentation: Dict[str, Any]


@dataclass
class OptimizerConfig:
    type: str
    learning_rate: float
    weight_decay: float


@dataclass
class EarlyStoppingConfig:
    patience: int
    monitor: str


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    scheduler: Dict[str, Any]
    early_stopping: EarlyStoppingConfig
    mixed_precision: bool


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
    model: Dict[str, Any]
    loss: Dict[str, Any]
    training: TrainingConfig
    evaluation: Dict[str, Any]
    logging: Dict[str, Any]


def load_config(path: Path) -> Config:
    """
    Load a YAML config file into a typed Config dataclass
    """

    with open(path) as file:
        raw = yaml.safe_load(file)

    return dacite.from_dict(data_class=Config, data=raw)