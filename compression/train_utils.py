#!/usr/bin/env python3
#
# train_utils.py  Andrew Belles  April 10th, 2026
#
# Shared config, device, and seed helpers.
#

import copy
import random
from pathlib import Path

import torch
import yaml


def merge_config(defaults: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(defaults)

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_file():
        return config_path

    example_path = config_path.with_name(f"{config_path.stem}.example{config_path.suffix}")
    if example_path.is_file():
        return example_path

    raise FileNotFoundError(f"missing config: {config_path}")


def load_config(config_path: Path, defaults: dict) -> dict:
    resolved_path = resolve_config_path(config_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {resolved_path}")

    return merge_config(defaults, loaded)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
