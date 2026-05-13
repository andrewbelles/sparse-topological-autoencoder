#!/usr/bin/env python3
#
# PH-feature transfer probes for ABT, SAE, and TopoSAE representations.
#

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-evaluation-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from compression.common import SPLITS, active_feature_columns, embedding_columns
from compression.train_utils import load_config
from evaluation.linear import build_estimator, compute_metrics, optimize_hyperparameters
from evaluation.topology import betti_curve, compute_diagrams, finite_diagram
from evaluation.visualizations import clean_label


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "transfer.yaml"
DEFAULT_DATA_DIR = Path("compression/data")
DEFAULT_CONFIG = {
    "seed": 7,
    "dataset": "fma_small_mel",
    "anchor_dir": "representation/data",
    "compression_dir": "compression/data",
    "source": "anchor",
    "methods": ["anchor", "sae", "topo_sae"],
    "neighbors": [15, 30, 50],
    "homology_dims": [0, 1],
    "features": ["betti", "entropy"],
    "betti_grid_size": 64,
    "filtration_max": 3.0,
    "max_homology_dim": 1,
    "n_perm": 64,
    "classifier": "logistic",
    "optuna": {
        "trials": 20,
        "target_metric": "f1_macro",
    },
    "knn_neighbors": [3, 5, 9, 15, 25],
    "c_min": 1e-4,
    "c_max": 1.0,
    "max_iter": 10000,
    "tol": 1e-3,
    "device": "cuda",
    "torch_epochs": 100,
    "torch_lr": 0.05,
    "torch_batch_size": 2048,
    "include_code_dims": [],
    "exclude_code_dims": [],
    "include_target_active": [],
    "exclude_target_active": [],
    "cache_features": True,
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PH-feature probes over ABT/SAE representations.")
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def optional_int_set(values: object) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, int):
        return {int(values)}
    if isinstance(values, list | tuple):
        return {int(value) for value in values}
    return {int(values)}


def passes_grid_filters(code_dim: int | None, target_active: int | None, config: dict) -> bool:
    include_k = optional_int_set(config.get("include_code_dims", []))
    exclude_k = optional_int_set(config.get("exclude_code_dims", []))
    include_s = optional_int_set(config.get("include_target_active", []))
    exclude_s = optional_int_set(config.get("exclude_target_active", []))
    if code_dim is not None:
        if include_k and code_dim not in include_k:
            return False
        if code_dim in exclude_k:
            return False
    if target_active is not None:
        if include_s and target_active not in include_s:
            return False
        if target_active in exclude_s:
            return False
    return True


def cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def load_anchor_run(config: dict) -> dict[str, object]:
    anchor_dir = Path(str(config["anchor_dir"])).expanduser()
    source = str(config["source"])
    dataset = str(config["dataset"])
    frames: dict[str, pd.DataFrame] = {}
    columns: list[str] | None = None
    for split in SPLITS:
        path = anchor_dir / f"{source}_{dataset}_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing anchor parquet: {path}")
        frame = pd.read_parquet(path).copy()
        split_columns = embedding_columns(frame)
        columns = split_columns if columns is None else columns
        frames[split] = frame
    return {
        "run_name": f"{source}_ph",
        "method": source,
        "family": "baseline",
        "code_dim": len(columns or []),
        "target_active": None,
        "seed": 0,
        "frames": frames,
        "columns": columns or [],
    }


def load_sparse_runs(data_dir: Path, config: dict) -> list[dict[str, object]]:
    wanted = {str(method) for method in config["methods"] if str(method) != "anchor"}
    if not wanted:
        return []
    runs: list[dict[str, object]] = []
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        required = {"method", "split", "code_dim", "target_active", "seed", "track_id", "genre_top", "m_dim"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[frame["method"].astype(str).isin(wanted)].copy()
        if frame.empty:
            continue
        for keys, group in frame.groupby(["method", "code_dim", "target_active", "seed"], dropna=False):
            method, code_dim, target_active, seed = keys
            code_dim = int(code_dim)
            target_active = int(target_active)
            seed = int(seed)
            if not passes_grid_filters(code_dim, target_active, config):
                continue
            if set(group["split"].unique()) < set(SPLITS):
                continue
            columns = active_feature_columns(group, code_dim)
            frames = {split: group[group["split"] == split].copy() for split in SPLITS}
            runs.append(
                {
                    "run_name": f"{method}_k{code_dim:04d}_s{target_active:03d}_seed{seed:02d}_ph",
                    "method": str(method),
                    "family": str(group["family"].iloc[0]) if "family" in group.columns else "sparse_dictionary",
                    "code_dim": code_dim,
                    "target_active": target_active,
                    "seed": seed,
                    "frames": frames,
                    "columns": columns,
                }
            )
    return runs


def run_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=np.float32, copy=True)


def persistence_entropy(diagram: np.ndarray) -> float:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return 0.0
    persistence = np.maximum(finite[:, 1] - finite[:, 0], 0.0)
    total = float(np.sum(persistence))
    if total <= 0.0:
        return 0.0
    probabilities = persistence / total
    return float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))


def diagram_summaries(diagram: np.ndarray) -> list[float]:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    persistence = np.maximum(finite[:, 1] - finite[:, 0], 0.0)
    return [
        persistence_entropy(finite),
        float(np.sum(persistence)),
        float(np.max(persistence)) if len(persistence) else 0.0,
        float(len(finite)),
    ]


def ph_vector(points: np.ndarray, config: dict) -> np.ndarray:
    diagrams = compute_diagrams(points, config)
    grid = np.linspace(0.0, float(config["filtration_max"]), int(config["betti_grid_size"]))
    values: list[np.ndarray] = []
    for homology_dim in [int(dim) for dim in config["homology_dims"]]:
        diagram = diagrams[homology_dim] if homology_dim < len(diagrams) else np.empty((0, 2), dtype=np.float64)
        if "betti" in config["features"]:
            values.append(betti_curve(diagram, grid).astype(np.float32, copy=False))
        if "entropy" in config["features"]:
            values.append(np.asarray(diagram_summaries(diagram), dtype=np.float32))
    return np.concatenate(values).astype(np.float32, copy=False)


def compute_split_features(matrix: np.ndarray, k: int, config: dict) -> np.ndarray:
    n_neighbors = min(int(k), matrix.shape[0])
    if n_neighbors < 3:
        raise ValueError(f"k={k} leaves fewer than 3 neighbors for split with n={matrix.shape[0]}")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors.fit(matrix)
    indices = neighbors.kneighbors(matrix, return_distance=False)
    rows: list[np.ndarray] = []
    for index, local_indices in enumerate(indices):
        rows.append(ph_vector(matrix[local_indices], config))
        if (index + 1) % 500 == 0:
            log(f"ph_features n={index + 1}/{len(indices)} k={k}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def feature_cache_path(run: dict[str, object], split: str, k: int, config: dict, cache_dir: Path) -> Path:
    payload = {
        "run_name": run["run_name"],
        "split": split,
        "k": int(k),
        "features": config["features"],
        "homology_dims": config["homology_dims"],
        "betti_grid_size": int(config["betti_grid_size"]),
        "filtration_max": float(config["filtration_max"]),
        "max_homology_dim": int(config["max_homology_dim"]),
        "n_perm": int(config["n_perm"]),
    }
    return cache_dir / f"{run['run_name']}_{split}_k{k}_{cache_key(payload)}.npz"


def load_or_compute_features(run: dict[str, object], split: str, k: int, config: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = run["frames"][split]
    columns = run["columns"]
    path = feature_cache_path(run, split, k, config, cache_dir)
    if bool(config.get("cache_features", True)) and path.is_file():
        cached = np.load(path, allow_pickle=True)
        return cached["features"], cached["labels"], cached["track_ids"]

    matrix = run_matrix(frame, columns)
    features = compute_split_features(matrix, k, config)
    labels = frame["genre_top"].astype(str).to_numpy()
    track_ids = frame["track_id"].to_numpy()
    if bool(config.get("cache_features", True)):
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features, labels=labels, track_ids=track_ids)
    return features, labels, track_ids


def encode_labels(labels_by_split: dict[str, np.ndarray]) -> tuple[LabelEncoder, dict[str, np.ndarray]]:
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_by_split[split] for split in SPLITS]))
    return encoder, {split: encoder.transform(labels_by_split[split]) for split in SPLITS}


def evaluate_transfer_run(run: dict[str, object], k: int, config: dict, data_root: Path) -> dict[str, object]:
    cache_dir = data_root / "transfer_features"
    features: dict[str, np.ndarray] = {}
    labels_text: dict[str, np.ndarray] = {}
    for split in SPLITS:
        split_features, split_labels, _ = load_or_compute_features(run, split, k, config, cache_dir)
        features[split] = split_features
        labels_text[split] = split_labels

    label_encoder, labels = encode_labels(labels_text)
    classifier = str(config["classifier"])
    target_metric = str(config["optuna"]["target_metric"])
    best_params, best_val_score = optimize_hyperparameters(
        classifier,
        config,
        features["training"],
        labels["training"],
        features["validation"],
        labels["validation"],
    )
    estimator = build_estimator(
        classifier,
        float(best_params["C"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        config=config,
    )
    estimator.fit(features["training"], labels["training"])
    n_classes = len(label_encoder.classes_)
    val_metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes=n_classes)
    test_metrics = compute_metrics(estimator, features["test"], labels["test"], n_classes=n_classes)
    record = {
        "run_name": str(run["run_name"]),
        "method": str(run["method"]),
        "family": str(run["family"]),
        "code_dim": run["code_dim"],
        "target_active": run["target_active"],
        "seed": int(run["seed"]),
        "k": int(k),
        "feature_dim": int(features["training"].shape[1]),
        "classifier": classifier,
        "target_metric": target_metric,
        "best_c": float(best_params["C"]),
        f"validation_{target_metric}": float(best_val_score),
        "validation_accuracy": val_metrics["accuracy"],
        "validation_f1_macro": val_metrics["f1_macro"],
        "validation_pr_auc_macro": val_metrics["pr_auc_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_pr_auc_macro": test_metrics["pr_auc_macro"],
        "n_train": int(len(labels["training"])),
        "n_validation": int(len(labels["validation"])),
        "n_test": int(len(labels["test"])),
    }
    report(
        f"transfer method={record['method']} K={record['code_dim']} s={record['target_active']} "
        f"k={k} f1={record['test_f1_macro']:.3f} pr_auc={record['test_pr_auc_macro']:.3f}"
    )
    return record


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "code_dim",
        "target_active",
        "k",
        "feature_dim",
        "validation_f1_macro",
        "test_f1_macro",
        "test_pr_auc_macro",
    ]
    compact = summary[columns].rename(
        columns={
            "code_dim": "K",
            "target_active": "s",
            "validation_f1_macro": "val_f1",
            "test_f1_macro": "f1",
            "test_pr_auc_macro": "pr_auc",
        }
    )
    return compact.sort_values(["method", "K", "s", "k"], na_position="first")


def feature_metadata(summary: pd.DataFrame, config: dict) -> pd.DataFrame:
    columns = [
        "run_name",
        "method",
        "family",
        "code_dim",
        "target_active",
        "seed",
        "k",
        "feature_dim",
        "n_train",
        "n_validation",
        "n_test",
    ]
    metadata = summary[columns].copy()
    metadata["features"] = ",".join(str(value) for value in config["features"])
    metadata["homology_dims"] = ",".join(str(value) for value in config["homology_dims"])
    metadata["betti_grid_size"] = int(config["betti_grid_size"])
    return metadata


def attach_sparse_topology(summary: pd.DataFrame) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "sparse_dictionary_topology_summary.csv"
    if not path.is_file():
        return summary
    topology = pd.read_csv(path)
    required = {"method", "code_dim", "target_active", "seed", "betti_Z_A_h0", "betti_Z_A_h1", "betti_Z_Zhat_h0", "betti_Z_Zhat_h1"}
    if topology.empty or not required.issubset(topology.columns):
        return summary
    grouped = (
        topology.groupby(["method", "code_dim", "target_active", "seed"], as_index=False)
        .agg(
            betti_Z_A_h0=("betti_Z_A_h0", "mean"),
            betti_Z_A_h1=("betti_Z_A_h1", "mean"),
            betti_Z_Zhat_h0=("betti_Z_Zhat_h0", "mean"),
            betti_Z_Zhat_h1=("betti_Z_Zhat_h1", "mean"),
        )
    )
    grouped["betti_Z_A"] = (grouped["betti_Z_A_h0"] + grouped["betti_Z_A_h1"]) / 2.0
    grouped["betti_Z_Zhat"] = (grouped["betti_Z_Zhat_h0"] + grouped["betti_Z_Zhat_h1"]) / 2.0
    merged = summary.copy()
    for column in ["code_dim", "target_active", "seed"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce")
    return merged.merge(grouped, on=["method", "code_dim", "target_active", "seed"], how="left")


def save_transfer_plots(summary: pd.DataFrame, image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def set_tight_score_axis(ax: plt.Axes, values: pd.Series) -> None:
        finite = pd.to_numeric(values, errors="coerce").dropna()
        if finite.empty:
            ax.set_ylim(0.0, 1.0)
            return
        lower = float(finite.min())
        upper = float(finite.max())
        span = max(upper - lower, 0.02)
        padding = max(0.015, span * 0.2)
        ax.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))

    for metric in ["test_f1_macro", "test_pr_auc_macro"]:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        plot_frame = summary.copy()
        plot_frame["label"] = plot_frame.apply(
            lambda row: str(row["method"])
            if pd.isna(row["target_active"])
            else f"{row['method']} K={int(row['code_dim'])} s={int(row['target_active'])}",
            axis=1,
        )
        sns.lineplot(data=plot_frame, x="k", y=metric, hue="method", marker="o", ax=ax)
        ax.set_xlabel("kNN Neighborhood Size")
        ax.set_ylabel(clean_label(metric.replace("test_", "")))
        set_tight_score_axis(ax, plot_frame[metric])
        ax.set_title(f"PH Probe {clean_label(metric.replace('test_', ''))} by Representation")
        if ax.get_legend() is not None:
            ax.legend(title="Method")
        fig.savefig(image_root / f"transfer_{metric}_by_k.png", dpi=200)
        plt.close(fig)

    sparse = summary[summary["target_active"].notna()].copy()
    if not sparse.empty:
        for metric in ["test_f1_macro", "test_pr_auc_macro"]:
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            sns.lineplot(
                data=sparse,
                x="target_active",
                y=metric,
                hue="method",
                style="code_dim",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Target Active Coefficients")
            ax.set_ylabel(clean_label(metric.replace("test_", "")))
            set_tight_score_axis(ax, sparse[metric])
            ax.set_title(f"PH Probe {clean_label(metric.replace('test_', ''))} by Sparse Grid")
            if ax.get_legend() is not None:
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
            fig.savefig(image_root / f"transfer_sparse_{metric}_by_s.png", dpi=200)
            plt.close(fig)

    if {"betti_Z_A", "betti_Z_Zhat"}.issubset(summary.columns):
        sparse = summary[summary["target_active"].notna()].copy()
        for topology_metric in ["betti_Z_A", "betti_Z_Zhat"]:
            for performance_metric in ["test_f1_macro", "test_pr_auc_macro"]:
                plot_frame = sparse.dropna(subset=[topology_metric, performance_metric])
                if plot_frame.empty:
                    continue
                fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
                sns.scatterplot(
                    data=plot_frame,
                    x=topology_metric,
                    y=performance_metric,
                    hue="method",
                    style="code_dim",
                    size="target_active",
                    sizes=(45, 150),
                    alpha=0.85,
                    ax=ax,
                )
                ax.set_xlabel(clean_label(topology_metric))
                ax.set_ylabel(clean_label(performance_metric.replace("test_", "")))
                ax.set_title(f"PH Probe {clean_label(performance_metric.replace('test_', ''))} vs {clean_label(topology_metric)}")
                set_tight_score_axis(ax, plot_frame[performance_metric])
                ax.grid(True, alpha=0.25)
                if ax.get_legend() is not None:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
                fig.savefig(image_root / f"transfer_{topology_metric}_vs_{performance_metric}.png", dpi=200)
                plt.close(fig)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data_dir = args.data_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    report(f"START module=evaluation.transfer data_dir={data_dir} config={args.config}")
    runs: list[dict[str, object]] = []
    if "anchor" in {str(method) for method in config["methods"]}:
        runs.append(load_anchor_run(config))
    runs.extend(load_sparse_runs(data_dir, config))
    if not runs:
        raise FileNotFoundError("no transfer runs found")

    records: list[dict[str, object]] = []
    for run in runs:
        for k in [int(value) for value in config["neighbors"]]:
            records.append(evaluate_transfer_run(run, k, config, data_root))

    summary = attach_sparse_topology(pd.DataFrame.from_records(records))
    compact = compact_summary(summary)
    metadata = feature_metadata(summary, config)
    summary_path = data_root / "transfer_summary.csv"
    compact_path = data_root / "transfer_compact_summary.csv"
    metadata_path = data_root / "transfer_features_metadata.csv"
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    save_transfer_plots(summary, image_root)
    log("transfer_summary")
    log(compact.round(4).to_string(index=False))
    report(f"DONE module=evaluation.transfer runs={len(summary)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
