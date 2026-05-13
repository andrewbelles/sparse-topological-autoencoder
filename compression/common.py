#!/usr/bin/env python3
#
# Shared compression utilities for ABT manifold experiments.
#

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SPLITS = ("training", "validation", "test")
METADATA_COLUMNS = {
    "method",
    "family",
    "dataset",
    "source",
    "split",
    "ratio_percent",
    "m_dim",
    "input_dim",
    "seed",
    "track_id",
    "genre_top",
}


def report(message: str) -> None:
    print(message, flush=True)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("embedding_"))
    if not columns:
        raise ValueError("frame has no embedding columns")
    return columns


def load_anchor_splits(anchor_dir: Path, source: str, dataset: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None
    for split in SPLITS:
        path = anchor_dir / f"{source}_{dataset}_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing anchor parquet: {path}")
        frame = pd.read_parquet(path).copy()
        columns = embedding_columns(frame)
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"embedding columns differ across anchor splits at {path}")
        frame["split"] = split
        frames[split] = frame
    return frames


def metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[column for column in frame.columns if not column.startswith("embedding_")]].copy()


def feature_matrix(frame: pd.DataFrame, m_dim: int | None = None) -> np.ndarray:
    columns = embedding_columns(frame)
    if m_dim is not None:
        columns = columns[: int(m_dim)]
    return frame[columns].to_numpy(dtype=np.float32, copy=True)


def ratio_to_dim(input_dim: int, ratio_percent: int, convenient_dims: list[int] | None = None) -> int:
    ratio = int(ratio_percent)
    if convenient_dims:
        target = max(1, int(round(input_dim * ratio / 100.0)))
        eligible = [value for value in convenient_dims if value <= input_dim]
        return min(eligible, key=lambda value: (abs(value - target), value))
    return max(1, min(input_dim, int(round(input_dim * ratio / 100.0))))


def dim_to_ratio(input_dim: int, dim: int) -> int:
    return int(round(100.0 * int(dim) / int(input_dim)))


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def gaussian_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / math.sqrt(output_dim), size=(input_dim, output_dim)).astype("float32")


def fast_walsh_hadamard(matrix: np.ndarray) -> np.ndarray:
    output = matrix.astype("float32", copy=True)
    n_features = output.shape[1]
    if n_features <= 0 or n_features & (n_features - 1):
        raise ValueError(f"SRHT requires power-of-two input dimension, got {n_features}")
    step = 1
    while step < n_features:
        reshaped = output.reshape(output.shape[0], -1, step * 2)
        left = reshaped[:, :, :step].copy()
        right = reshaped[:, :, step : step * 2].copy()
        reshaped[:, :, :step] = left + right
        reshaped[:, :, step : step * 2] = left - right
        step *= 2
    output /= math.sqrt(n_features)
    return output


def srht_project(matrix: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    input_dim = matrix.shape[1]
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_dim)
    indices = np.sort(rng.choice(input_dim, size=output_dim, replace=False))
    transformed = fast_walsh_hadamard(matrix * signs.reshape(1, -1))
    return (transformed[:, indices] * math.sqrt(input_dim / output_dim)).astype("float32", copy=False)


def fit_pca(training_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = training_matrix.mean(axis=0, keepdims=True).astype("float32")
    centered = training_matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt.astype("float32", copy=False)


def pca_project(matrix: np.ndarray, mean: np.ndarray, components: np.ndarray, output_dim: int) -> np.ndarray:
    return ((matrix - mean) @ components[:output_dim].T).astype("float32", copy=False)


def pad_features(features: np.ndarray, max_dim: int) -> pd.DataFrame:
    if features.shape[1] > max_dim:
        raise ValueError(f"features dim {features.shape[1]} exceeds max_dim={max_dim}")
    output = np.full((features.shape[0], max_dim), np.nan, dtype=np.float32)
    output[:, : features.shape[1]] = features
    return pd.DataFrame(output, columns=[f"embedding_{index:04d}" for index in range(max_dim)])


def build_output_frame(
    split_frame: pd.DataFrame,
    features: np.ndarray,
    *,
    method: str,
    family: str,
    dataset: str,
    source: str,
    split: str,
    ratio_percent: int,
    input_dim: int,
    seed: int,
    max_dim: int,
) -> pd.DataFrame:
    metadata = metadata_frame(split_frame)
    metadata["method"] = method
    metadata["family"] = family
    metadata["dataset"] = dataset
    metadata["source"] = source
    metadata["split"] = split
    metadata["ratio_percent"] = int(ratio_percent)
    metadata["m_dim"] = int(features.shape[1])
    metadata["input_dim"] = int(input_dim)
    metadata["seed"] = int(seed)
    return pd.concat([metadata.reset_index(drop=True), pad_features(features, max_dim)], axis=1)


def write_method_parquet(rows: list[pd.DataFrame], output_path: Path) -> Path:
    if not rows:
        raise ValueError("no rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(output_path, index=False)
    return output_path


def method_parquet_path(output_dir: Path, method: str, source: str, dataset: str) -> Path:
    return output_dir / f"{method}_{source}_{dataset}.parquet"


def read_method_parquets(data_dir: Path, methods: list[str] | None = None) -> list[Path]:
    paths = sorted(data_dir.glob("*.parquet"))
    if methods:
        wanted = set(methods)
        paths = [path for path in paths if path.stem.split("_anchor_", 1)[0] in wanted or path.stem in wanted]
    return paths


def active_feature_columns(frame: pd.DataFrame, m_dim: int) -> list[str]:
    columns = embedding_columns(frame)
    if int(m_dim) > len(columns):
        raise ValueError(f"requested m_dim={m_dim} but only {len(columns)} embedding columns exist")
    return columns[: int(m_dim)]
