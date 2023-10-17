'''
Module bundling the dataclasses used for model configuration.
'''
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    epochs: int = 5
    csv_file: str = ''
    root_dir: str = ''
    lr: float = 0.001

@dataclass
class ValidationConfig:
    csv_file: str = ''
    root_dir: str = ''
    pr_compute_interval: int = 1
    

@dataclass
class TestConfig:
    csv_file: str = ''
    root_dir: str = ''
    iou_threshold: float = 0.4
    score_threshold: float = 0.1

@dataclass
class ModelConfig:
    batch_size: int = 1
    workers: int = 4
    accelerator: str = 'cpu'
    devices: int = 1
    train: TrainConfig = field(default_factory=TrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    test: TestConfig = field(default_factory=TestConfig)