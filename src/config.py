"""Central, CPU-safe configuration for the experiment.

Change ``dataset_name`` and add a parser in data/prepare_dataset.py when moving
from Devign to another corpus such as BigVul or DiverseVul.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "devign"
    dataset_path: str | None = None  # Optional local file/dataset override.
    model_name: str = "google/gemma-2-2b-it"
    label_mode: str = "binary"  # "binary" or "cwe" (when source supplies CWEs)
    max_seq_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    models_dir: Path = ROOT / "models"
    results_dir: Path = ROOT / "results"
    metrics_dir: Path = ROOT / "results" / "metrics"
    plots_dir: Path = ROOT / "results" / "plots"

    @property
    def model_predictions_path(self) -> Path:
        return self.results_dir / "model_predictions.jsonl"
    @property
    def static_predictions_path(self) -> Path:
        return self.results_dir / "static_predictions.jsonl"
    @property
    def codebert_predictions_path(self) -> Path:
        return self.results_dir / "codebert_predictions.jsonl"

CFG = ExperimentConfig()
