#!/usr/bin/env python3
#
# Topology-aware CS projection learners over the ABT anchor manifold.
#

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression.common import (
    SPLITS,
    build_output_frame,
    dim_to_ratio,
    fast_walsh_hadamard,
    feature_matrix,
    gaussian_projection,
    load_anchor_splits,
    method_parquet_path,
    ratio_to_dim,
    report,
    write_method_parquet,
)
from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "topo_cs.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
DEFAULT_OUTPUT_DIR = Path("compression/data")
METHODS = ("learned_gaussian", "learned_srht_mask")
DEFAULT_CONFIG = {
    "seed": 0,
    "source": "anchor",
    "dataset": "fma_small_mel",
    "methods": ["learned_gaussian", "learned_srht_mask"],
    "ratios": [10, 20, 40, 60],
    "convenient_dims": [16, 32, 64, 96, 128, 192, 256],
    "seeds": [0],
    "device": "cuda",
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 1.0e-3,
    "balanced_batch": True,
    "orth_lambda": 0.05,
    "budget_lambda": 0.1,
    "binary_lambda": 0.001,
    "row_normalize": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train topology-aware CS projections over the ABT anchor manifold.")
    parser.add_argument("-a", "--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def validate_config(config: dict) -> None:
    invalid = sorted(set(str(method) for method in config["methods"]) - set(METHODS))
    if invalid:
        raise ValueError(f"unsupported topo-CS methods: {', '.join(invalid)}")
    if int(config["batch_size"]) < 4:
        raise ValueError("batch_size must be at least 4")
    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be positive")


def output_dims(input_dim: int, config: dict) -> dict[int, int]:
    explicit_dims = [int(value) for value in config.get("dims", [])]
    if explicit_dims:
        return {dim_to_ratio(input_dim, dim): min(input_dim, max(1, dim)) for dim in explicit_dims}
    convenient_dims = [int(value) for value in config.get("convenient_dims", [])]
    return {int(ratio): ratio_to_dim(input_dim, int(ratio), convenient_dims) for ratio in config["ratios"]}


def normalize_pairwise(points: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(points, points)
    upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    positive = upper[upper > 0]
    median = positive.median() if positive.numel() else torch.tensor(1.0, device=points.device)
    return distances / median.clamp_min(1e-8)


def topology_proxy_loss(original: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
    original_dist = normalize_pairwise(original)
    projected_dist = normalize_pairwise(projected)
    return F.mse_loss(projected_dist, original_dist)


def make_batches(labels: np.ndarray, batch_size: int, balanced: bool, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if not balanced:
        indices = rng.permutation(len(labels))
        return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size) if len(indices[start : start + batch_size]) >= 4]

    classes = sorted(np.unique(labels).tolist())
    per_class = max(1, batch_size // len(classes))
    class_indices = {label: np.where(labels == label)[0] for label in classes}
    batches = []
    steps = max(1, len(labels) // batch_size)
    for _ in range(steps):
        parts = []
        for label in classes:
            pool = class_indices[label]
            parts.append(rng.choice(pool, size=per_class, replace=len(pool) < per_class))
        batch = np.concatenate(parts)
        rng.shuffle(batch)
        if len(batch) >= 4:
            batches.append(batch)
    return batches


def train_learned_gaussian(train: np.ndarray, labels: np.ndarray, m_dim: int, seed: int, config: dict, device: torch.device) -> np.ndarray:
    torch.manual_seed(seed)
    input_dim = train.shape[1]
    init = gaussian_projection(input_dim, m_dim, seed).T
    phi = torch.nn.Parameter(torch.from_numpy(init).to(device))
    optimizer = torch.optim.Adam([phi], lr=float(config["learning_rate"]))
    train_tensor = torch.from_numpy(train).to(device)
    identity = torch.eye(m_dim, device=device)

    for epoch in range(int(config["epochs"])):
        for batch_indices in make_batches(labels, int(config["batch_size"]), bool(config["balanced_batch"]), seed + epoch):
            batch = train_tensor[batch_indices]
            projected = batch @ phi.t()
            topo = topology_proxy_loss(batch, projected)
            orth = F.mse_loss(phi @ phi.t(), identity)
            loss = topo + float(config["orth_lambda"]) * orth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if bool(config["row_normalize"]):
                with torch.no_grad():
                    phi.div_(phi.norm(dim=1, keepdim=True).clamp_min(1e-8))

    return phi.detach().cpu().numpy().astype("float32", copy=False)


def train_srht_mask(train: np.ndarray, labels: np.ndarray, m_dim: int, seed: int, config: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    input_dim = train.shape[1]
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_dim)
    transformed = fast_walsh_hadamard(train * signs.reshape(1, -1))
    transformed_tensor = torch.from_numpy(transformed).to(device)
    original_tensor = torch.from_numpy(train).to(device)
    labels_array = labels
    logits = torch.nn.Parameter(torch.zeros(input_dim, device=device))
    optimizer = torch.optim.Adam([logits], lr=float(config["learning_rate"]))

    for epoch in range(int(config["epochs"])):
        for batch_indices in make_batches(labels_array, int(config["batch_size"]), bool(config["balanced_batch"]), seed + epoch):
            original = original_tensor[batch_indices]
            transformed_batch = transformed_tensor[batch_indices]
            weights = torch.sigmoid(logits)
            soft_projected = transformed_batch * weights.unsqueeze(0)
            topo = topology_proxy_loss(original, soft_projected)
            budget = (weights.sum() - float(m_dim)).pow(2) / float(input_dim * input_dim)
            binary = torch.mean(weights * (1.0 - weights))
            loss = topo + float(config["budget_lambda"]) * budget + float(config["binary_lambda"]) * binary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    weights = torch.sigmoid(logits).detach().cpu().numpy()
    indices = np.sort(np.argsort(weights)[-m_dim:])
    return signs.astype("float32", copy=False), indices.astype(int, copy=False)


def apply_learned_gaussian(matrix: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return (matrix @ phi.T).astype("float32", copy=False)


def apply_srht_mask(matrix: np.ndarray, signs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    transformed = fast_walsh_hadamard(matrix * signs.reshape(1, -1))
    return (transformed[:, indices] * math.sqrt(matrix.shape[1] / len(indices))).astype("float32", copy=False)


def label_array(frame) -> np.ndarray:
    labels = {label: index for index, label in enumerate(sorted(frame["genre_top"].unique()))}
    return frame["genre_top"].map(labels).to_numpy(dtype=np.int64)


def run(config: dict, anchor_dir: Path = DEFAULT_ANCHOR_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    validate_config(config)
    source = str(config["source"])
    dataset = str(config["dataset"])
    methods = [str(method) for method in config["methods"]]
    seeds = [int(value) for value in config["seeds"]]
    device = resolve_device(str(config["device"]))

    split_frames = load_anchor_splits(anchor_dir, source, dataset)
    split_features = {split: feature_matrix(frame) for split, frame in split_frames.items()}
    labels = label_array(split_frames["training"])
    input_dim = int(split_features["training"].shape[1])
    dims_by_ratio = output_dims(input_dim, config)
    max_dim = max(dims_by_ratio.values())
    written: list[Path] = []

    report(f"compression.topo_cs methods={','.join(methods)} ratios={','.join(str(r) for r in dims_by_ratio)} device={device}")
    for method in methods:
        rows = []
        for ratio, m_dim in dims_by_ratio.items():
            for seed in seeds:
                train_seed = int(config["seed"]) + seed * 100_000 + ratio
                if method == "learned_gaussian":
                    phi = train_learned_gaussian(split_features["training"], labels, m_dim, train_seed, config, device)
                    outputs = {split: apply_learned_gaussian(matrix, phi) for split, matrix in split_features.items()}
                elif method == "learned_srht_mask":
                    signs, indices = train_srht_mask(split_features["training"], labels, m_dim, train_seed, config, device)
                    outputs = {split: apply_srht_mask(matrix, signs, indices) for split, matrix in split_features.items()}
                else:
                    raise ValueError(f"unsupported method: {method}")

                for split in SPLITS:
                    rows.append(
                        build_output_frame(
                            split_frames[split],
                            outputs[split],
                            method=method,
                            family="topo_cs",
                            dataset=dataset,
                            source=source,
                            split=split,
                            ratio_percent=ratio,
                            input_dim=input_dim,
                            seed=seed,
                            max_dim=max_dim,
                        )
                    )
                report(f"trained method={method} ratio={ratio} m_dim={m_dim} seed={seed}")
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
