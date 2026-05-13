#!/usr/bin/env python3
#
# Non-CS manifold baselines over the selected ABT anchor manifold.
#

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fma-compression-umap-numba")

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from compression.common import (
    SPLITS,
    build_output_frame,
    feature_matrix,
    load_anchor_splits,
    method_parquet_path,
    report,
    write_method_parquet,
)
from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "manifold.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
DEFAULT_OUTPUT_DIR = Path("compression/data")
METHODS = ("umap", "laplacian_eigenmaps")
DEFAULT_CONFIG = {
    "seed": 0,
    "source": "anchor",
    "dataset": "fma_small_mel",
    "methods": ["umap", "laplacian_eigenmaps"],
    "dims": [2, 4, 8, 16, 32, 64],
    "seeds": [0, 1, 2],
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean",
    "laplacian_weight": "heat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manifold compression baselines over the ABT anchor manifold.")
    parser.add_argument("-a", "--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    invalid = sorted(set(str(method) for method in config["methods"]) - set(METHODS))
    if invalid:
        raise ValueError(f"unsupported manifold methods: {', '.join(invalid)}")
    if not config.get("dims"):
        raise ValueError("dims must be non-empty")
    if int(config["n_neighbors"]) <= 1:
        raise ValueError("n_neighbors must be greater than 1")


def ratio_for_dim(input_dim: int, dim: int) -> int:
    return int(round(100.0 * dim / input_dim))


def run_umap(train: np.ndarray, splits: dict[str, np.ndarray], dim: int, seed: int, config: dict) -> dict[str, np.ndarray]:
    from umap import UMAP

    reducer = UMAP(
        n_components=dim,
        n_neighbors=int(config["n_neighbors"]),
        min_dist=float(config["min_dist"]),
        metric=str(config["metric"]),
        random_state=seed,
    )
    outputs = {"training": reducer.fit_transform(train).astype("float32", copy=False)}
    for split in ("validation", "test"):
        outputs[split] = reducer.transform(splits[split]).astype("float32", copy=False)
    return outputs


def laplacian_embedding(train: np.ndarray, dim: int, config: dict) -> tuple[np.ndarray, NearestNeighbors, float]:
    n_neighbors = int(config["n_neighbors"])
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=str(config["metric"]))
    neighbors.fit(train)
    distances, indices = neighbors.kneighbors(train)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    sigma = float(np.median(distances[distances > 0])) if np.any(distances > 0) else 1.0

    weights = np.zeros((train.shape[0], train.shape[0]), dtype=np.float32)
    for row in range(train.shape[0]):
        if str(config["laplacian_weight"]) == "binary":
            values = np.ones(n_neighbors, dtype=np.float32)
        else:
            values = np.exp(-(distances[row] ** 2) / max(sigma**2, 1e-8)).astype("float32")
        weights[row, indices[row]] = values
    weights = np.maximum(weights, weights.T)

    laplacian = csgraph.laplacian(weights, normed=True)
    k = min(dim + 1, train.shape[0] - 1)
    eigenvalues, eigenvectors = eigsh(laplacian, k=k, which="SM")
    order = np.argsort(eigenvalues)
    embedding = eigenvectors[:, order[1 : dim + 1]].astype("float32", copy=False)
    return embedding, neighbors, sigma


def interpolate_laplacian(points: np.ndarray, train_embedding: np.ndarray, neighbors: NearestNeighbors, sigma: float) -> np.ndarray:
    distances, indices = neighbors.kneighbors(points)
    weights = np.exp(-(distances**2) / max(sigma**2, 1e-8)).astype("float32")
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return np.einsum("nk,nkd->nd", weights, train_embedding[indices]).astype("float32", copy=False)


def run_laplacian(train: np.ndarray, splits: dict[str, np.ndarray], dim: int, config: dict) -> dict[str, np.ndarray]:
    train_embedding, neighbors, sigma = laplacian_embedding(train, dim, config)
    return {
        "training": train_embedding,
        "validation": interpolate_laplacian(splits["validation"], train_embedding, neighbors, sigma),
        "test": interpolate_laplacian(splits["test"], train_embedding, neighbors, sigma),
    }


def run(config: dict, anchor_dir: Path = DEFAULT_ANCHOR_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    validate_config(config)
    source = str(config["source"])
    dataset = str(config["dataset"])
    methods = [str(method) for method in config["methods"]]
    dims = [int(value) for value in config["dims"]]
    seeds = [int(value) for value in config["seeds"]]
    base_seed = int(config["seed"])

    split_frames = load_anchor_splits(anchor_dir, source, dataset)
    split_features = {split: feature_matrix(frame) for split, frame in split_frames.items()}
    input_dim = int(split_features["training"].shape[1])
    max_dim = max(dims)
    written: list[Path] = []

    report(f"compression.manifold methods={','.join(methods)} dims={','.join(str(d) for d in dims)} source={source}")
    for method in methods:
        rows = []
        for dim in dims:
            for seed in seeds:
                method_seed = base_seed + seed * 100_000 + dim
                if method == "umap":
                    outputs = run_umap(split_features["training"], split_features, dim, method_seed, config)
                elif method == "laplacian_eigenmaps":
                    outputs = run_laplacian(split_features["training"], split_features, dim, config)
                else:
                    raise ValueError(f"unsupported method: {method}")
                for split in SPLITS:
                    rows.append(
                        build_output_frame(
                            split_frames[split],
                            outputs[split],
                            method=method,
                            family="manifold",
                            dataset=dataset,
                            source=source,
                            split=split,
                            ratio_percent=ratio_for_dim(input_dim, dim),
                            input_dim=input_dim,
                            seed=seed,
                            max_dim=max_dim,
                        )
                    )
        output_path = method_parquet_path(output_dir, method, source, dataset)
        written.append(write_method_parquet(rows, output_path))
        report(f"wrote method={method} rows={sum(len(row) for row in rows)} path={output_path}")
    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    run(config, args.anchor_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
