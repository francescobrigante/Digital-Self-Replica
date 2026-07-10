# ==============================================================================
# Typed access to config.yaml.
# Usage: cfg = load_config()  →  cfg.lora.r, cfg.data.max_length, ...
# ==============================================================================

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    base_model_id: str


@dataclass
class DataConfig:
    chat_dir: str
    csv_dir: str
    dataset_json: str
    datasets_dir: str
    assistant_name: str
    max_context_messages: int
    time_gap_minutes: int
    max_length: int
    test_size: float
    seed: int

    @property
    def time_gap_seconds(self) -> int:
        return self.time_gap_minutes * 60


@dataclass
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: list = field(default_factory=list)


@dataclass
class TrainingSettings:
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    eval_accumulation_steps: int
    warmup_steps: int
    learning_rate: float
    lr_scheduler_type: str
    weight_decay: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    max_grad_norm: float
    early_stopping_patience: int
    early_stopping_threshold: float
    wandb_project: str


@dataclass
class GenerationSettings:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    lora: LoraSettings
    training: TrainingSettings
    generation: GenerationSettings


def load_config(path: str | Path = "config.yaml") -> Config:
    """Loads config.yaml into a typed Config object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraSettings(**raw["lora"]),
        training=TrainingSettings(**raw["training"]),
        generation=GenerationSettings(**raw["generation"]),
    )
