import argparse
import contextlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from tasks.build_graphs import get_cache_filename


class ConfigError(Exception):
    pass


class NestedConfig:
    """Enables dot-notation access for nested dictionaries."""

    def __init__(self, data: dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, NestedConfig(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, NestedConfig) else value
        return result


@dataclass
class Config:
    """Merges command line arguments and YAML configuration files."""

    config_data: dict[str, Any] = field(default_factory=dict)
    dataset: str = ""
    seed: int = 42
    cpu: bool = False
    cache_dir: str = "./cache/"
    outputs_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    force_restart: bool = False
    test: bool = False
    checkpoint: str | None = None
    config: str | None = None
    data_dir: str = "./data/DARPA"
    device: str | None = None
    model: str = "theseus"
    dataset_info: NestedConfig | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Builds Config from parsed CLI arguments, defaults, and external YAMLs."""
        config = cls()
        config.model = "theseus"

        # Map CLI arguments to config attributes
        for key, value in vars(args).items():
            config_key = key.replace("-", "_")
            setattr(config, config_key, value)

        # Auto-detect device unless forced to CPU
        if getattr(config, "cpu", False):
            config.device = "cpu"
        elif config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load external configurations and clean cache if needed
        config._setup_dataset_config()
        config._load_yaml_configs()
        config._cleanup_existing_files()

        return config

    def _setup_dataset_config(self) -> None:
        """Loads dataset parameters from `configs/datasets/*.yml`."""
        configs_dir = Path(__file__).parent.parent / "configs" / "datasets"
        dataset_configs: dict[str, dict[str, Any]] = {}

        for yaml_file in configs_dir.glob("*.yml"):
            with open(yaml_file) as f:
                file_configs = yaml.safe_load(f)
                if not file_configs:
                    continue
                if not isinstance(file_configs, dict):
                    raise ConfigError(
                        f"Expected a mapping in dataset config {yaml_file}, got {type(file_configs).__name__}"
                    )
                dataset_configs.update(cast(dict[str, dict[str, Any]], file_configs))

        if self.dataset not in dataset_configs:
            raise KeyError(
                f"Dataset '{self.dataset}' not found. Available: {list(dataset_configs.keys())}"
            )

        self.dataset_info = NestedConfig(dataset_configs[str(self.dataset)])

    def _load_yaml_configs(self) -> None:
        """Loads YAML config. Priority: Custom Path > Tuned Config > Default Model Config."""
        custom_config = getattr(self, "config", None)

        if custom_config:
            config_path = Path(str(custom_config))
            if not config_path.exists():
                raise ConfigError(f"Custom configuration file not found: {config_path}")
            print(f"Loading custom configuration from: {config_path}")
            with open(config_path) as file:
                loaded = yaml.safe_load(file) or {}
                if not isinstance(loaded, dict):
                    raise ConfigError(
                        f"Custom configuration must be a mapping, got {type(loaded).__name__}: {config_path}"
                    )
                self.config_data = cast(dict[str, Any], loaded)
        else:
            model_config_path = (
                f"./configs/tuned/{self.model}_{self.dataset.lower()}.yml"
            )

            if os.path.exists(model_config_path):
                print(f"Loading configuration from: {model_config_path}")
                with open(model_config_path) as file:
                    loaded = yaml.safe_load(file) or {}
                    if not isinstance(loaded, dict):
                        raise ConfigError(
                            f"Model configuration must be a mapping, got {type(loaded).__name__}: {model_config_path}"
                        )
                    self.config_data = cast(dict[str, Any], loaded)
            elif os.path.exists(f"./configs/models/{self.model}.yml"):
                print("Model-specific config not found. Loading default configuration.")
                with open(f"./configs/models/{self.model}.yml") as file:
                    loaded = yaml.safe_load(file) or {}
                    if not isinstance(loaded, dict):
                        raise ConfigError(
                            f"Model configuration must be a mapping, got {type(loaded).__name__}: {self.model}"
                        )
                    self.config_data = cast(dict[str, Any], loaded)
            else:
                raise ConfigError(
                    f"Model configuration file not found: {model_config_path}"
                )

        self._create_nested_attributes()

    def _create_nested_attributes(self) -> None:
        """Converts the raw config dictionary into object attributes."""
        if not self.config_data:
            return

        for key, value in self.config_data.items():
            if isinstance(value, dict):
                setattr(self, key, NestedConfig(value))
            else:
                setattr(self, key, value)

    def override_from_args(self, args: argparse.Namespace) -> None:
        """Updates configuration with non-null CLI arguments, excluding internal flags."""
        for param, value in vars(args).items():
            if param in [
                "model",
                "dataset",
                "config",
                "force_restart",
                "wandb",
                "project",
            ]:
                continue

            if value is not None:
                setattr(self, param, value)
                if param in self.config_data:
                    self.config_data[param] = value
                print(f"Overriding config.{param} = {value}")

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the current configuration state."""
        result: dict[str, Any] = {}
        result.update(self.config_data)

        for key, value in self.__dict__.items():
            if not key.startswith("_") and key not in ["config_data", "dataset_info"]:
                result[key] = (
                    value.to_dict() if isinstance(value, NestedConfig) else value
                )

        return result

    def __repr__(self) -> str:
        model = getattr(self, "model", "unknown")
        dataset = getattr(self, "dataset", "unknown")
        seed = getattr(self, "seed", "unknown")
        return f"Config(model='{model}', dataset='{dataset}', seed={seed})"

    def _cleanup_existing_files(self):
        """Clears seed-specific cache if `force_restart` is active."""
        if not getattr(self, "force_restart", False):
            return

        print("Force restart enabled - cleaning up existing files...")

        seed = getattr(self, "seed", 42)
        seed_cache_dir = os.path.join(
            self.cache_dir, self.dataset.lower(), f"seed_{seed}"
        )
        if os.path.exists(seed_cache_dir):
            print(f"Removing seed-specific cache directory: {seed_cache_dir}")
            shutil.rmtree(seed_cache_dir)

        cache_filename = get_cache_filename(self.dataset, self)
        graph_cache_file = os.path.join(self.cache_dir, cache_filename)
        if os.path.exists(graph_cache_file):
            print(f"Removing graph cache: {graph_cache_file}")
            with contextlib.suppress(FileNotFoundError):
                os.remove(graph_cache_file)

        # Remove checkpoint files
        checkpoint_path = getattr(self, "checkpoint", None)
        if checkpoint_path:
            if os.path.dirname(checkpoint_path):
                checkpoint_file = checkpoint_path
            else:
                checkpoint_file = os.path.join(
                    self.checkpoint_dir, "theseus", checkpoint_path
                )
            if os.path.exists(checkpoint_file):
                print(f"Removing checkpoint: {checkpoint_file}")
                with contextlib.suppress(FileNotFoundError):
                    os.remove(checkpoint_file)

        print("Force restart cleanup completed.")
