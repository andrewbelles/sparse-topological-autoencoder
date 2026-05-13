#!/usr/bin/env python3
#
# Hybrid coordinate + local-PH probes for sparse ABT dictionary representations.
#

from __future__ import annotations

import argparse
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
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from compression.common import SPLITS
from compression.train_utils import load_config
from evaluation.transfer import load_anchor_run, load_or_compute_features, load_sparse_runs, run_matrix
from evaluation.visualizations import anchor_baseline, clean_label, set_tight_score_axis


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "hybrid.yaml"
DEFAULT_DATA_DIR = Path("compression/data")
DEFAULT_CONFIG = {
    "seed": 7,
    "dataset": "fma_small_mel",
    "anchor_dir": "representation/data",
    "source": "anchor",
    "methods": ["sae", "topo_sae"],
    "k": 30,
    "feature_modes": ["coord", "ph", "coord_ph"],
    "homology_dims": [0, 1],
    "features": ["betti", "entropy"],
    "betti_grid_size": 64,
    "filtration_max": 3.0,
    "max_homology_dim": 1,
    "n_perm": 64,
    "cache_features": True,
    "optuna": {
        "trials": 20,
        "target_metric": "f1_macro",
    },
    "c_min": 1.0e-4,
    "c_max": 10.0,
    "l1_ratio_min": 0.05,
    "l1_ratio_max": 0.95,
    "max_iter": 5000,
    "tol": 1.0e-3,
    "device": "cuda",
    "torch_epochs": 200,
    "torch_lr": 0.05,
    "torch_batch_size": 2048,
    "weight_threshold": 1.0e-8,
    "include_code_dims": [],
    "exclude_code_dims": [],
    "include_target_active": [],
    "exclude_target_active": [],
    "output_prefix": "hybrid",
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coordinate + local-PH hybrid probes.")
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def load_runs(data_dir: Path, config: dict) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    methods = {str(method) for method in config["methods"]}
    if "anchor" in methods:
        runs.append(load_anchor_run(config))
    runs.extend(load_sparse_runs(data_dir, config))
    if not runs:
        raise FileNotFoundError(f"no hybrid runs found in {data_dir}")
    return runs


def encode_labels(labels_by_split: dict[str, np.ndarray]) -> tuple[LabelEncoder, dict[str, np.ndarray]]:
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_by_split[split] for split in SPLITS]))
    return encoder, {split: encoder.transform(labels_by_split[split]) for split in SPLITS}


def align_ph_features(
    frame: pd.DataFrame,
    ph_features: np.ndarray,
    ph_labels: np.ndarray,
    ph_track_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_track_ids = frame["track_id"].to_numpy()
    if np.array_equal(expected_track_ids, ph_track_ids):
        return ph_features, ph_labels

    positions = {track_id: index for index, track_id in enumerate(ph_track_ids)}
    try:
        order = np.asarray([positions[track_id] for track_id in expected_track_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"PH cache is missing track_id={error.args[0]}") from error
    return ph_features[order], ph_labels[order]


def load_feature_blocks(
    run: dict[str, object],
    k: int,
    config: dict,
    cache_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    coord: dict[str, np.ndarray] = {}
    ph: dict[str, np.ndarray] = {}
    labels_text: dict[str, np.ndarray] = {}

    for split in SPLITS:
        frame = run["frames"][split]
        coord[split] = run_matrix(frame, run["columns"])
        split_ph, split_labels, split_track_ids = load_or_compute_features(run, split, k, config, cache_dir)
        split_ph, split_labels = align_ph_features(frame, split_ph, split_labels, split_track_ids)
        ph[split] = split_ph
        labels_text[split] = split_labels

    return coord, ph, labels_text


def standardize_blocks(
    coord: dict[str, np.ndarray],
    ph: dict[str, np.ndarray],
    mode: str,
) -> tuple[dict[str, np.ndarray], int, int]:
    coord_dim = int(coord["training"].shape[1])
    ph_dim = int(ph["training"].shape[1])
    if mode == "coord":
        scaler = StandardScaler().fit(coord["training"])
        return {split: scaler.transform(coord[split]).astype(np.float32, copy=False) for split in SPLITS}, coord_dim, 0
    if mode == "ph":
        scaler = StandardScaler().fit(ph["training"])
        return {split: scaler.transform(ph[split]).astype(np.float32, copy=False) for split in SPLITS}, 0, ph_dim
    if mode != "coord_ph":
        raise ValueError(f"unsupported feature mode: {mode}")

    coord_scaler = StandardScaler().fit(coord["training"])
    ph_scaler = StandardScaler().fit(ph["training"])
    features = {
        split: np.concatenate(
            [
                coord_scaler.transform(coord[split]).astype(np.float32, copy=False),
                ph_scaler.transform(ph[split]).astype(np.float32, copy=False),
            ],
            axis=1,
        )
        for split in SPLITS
    }
    return features, coord_dim, ph_dim


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


class TorchElasticNetProbe:
    def __init__(self, c_value: float, l1_ratio: float, config: dict):
        self.c_value = float(c_value)
        self.l1_ratio = float(l1_ratio)
        self.config = config
        self.device = resolve_device(str(config["device"]))
        self.classes_: np.ndarray | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        torch.manual_seed(int(self.config["seed"]))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.config["seed"]))

        self.classes_ = np.unique(labels)
        class_to_index = {int(label): index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)
        n_samples, n_features = features.shape
        n_classes = int(len(self.classes_))

        x_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(encoded, dtype=torch.long, device=self.device)
        weight = torch.empty((n_classes, n_features), dtype=torch.float32, device=self.device)
        torch.nn.init.xavier_uniform_(weight)
        bias = torch.zeros(n_classes, dtype=torch.float32, device=self.device)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        optimizer = torch.optim.Adam([weight, bias], lr=float(self.config["torch_lr"]))
        batch_size = max(1, int(self.config["torch_batch_size"]))
        epochs = max(1, int(self.config["torch_epochs"]))
        regularization = 1.0 / max(self.c_value * n_samples, 1.0e-12)

        for _ in range(epochs):
            permutation = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                logits = F.linear(x_tensor[indices], weight, bias)
                l2 = weight.pow(2).sum()
                penalty = regularization * 0.5 * (1.0 - self.l1_ratio) * l2
                loss = F.cross_entropy(logits, y_tensor[indices]) + penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if self.l1_ratio > 0.0:
                    with torch.no_grad():
                        shrinkage = float(self.config["torch_lr"]) * regularization * self.l1_ratio
                        weight.copy_(torch.sign(weight) * torch.clamp(weight.abs() - shrinkage, min=0.0))

        self.weight = weight.detach()
        self.bias = bias.detach()
        return self

    @torch.no_grad()
    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.weight is None or self.bias is None:
            raise RuntimeError("TorchElasticNetProbe must be fit before inference")
        batch_size = max(1, int(self.config["torch_batch_size"]))
        batches: list[np.ndarray] = []
        for start in range(0, features.shape[0], batch_size):
            x_batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=self.device)
            logits = F.linear(x_batch, self.weight, self.bias)
            batches.append(logits.cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = self.decision_function(features)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TorchElasticNetProbe must be fit before prediction")
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]

    @property
    def coef_(self) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("TorchElasticNetProbe must be fit before reading weights")
        return self.weight.detach().cpu().numpy()


def build_elastic_net_probe(c_value: float, l1_ratio: float, config: dict) -> TorchElasticNetProbe:
    return TorchElasticNetProbe(c_value, l1_ratio, config)


def decision_scores(estimator: TorchElasticNetProbe, features: np.ndarray, n_classes: int) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        scores = estimator.decision_function(features)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)

    class_ids = np.asarray(estimator.classes_, dtype=int)
    full_scores = np.full((scores.shape[0], n_classes), float(np.min(scores) - 1.0), dtype=np.float64)
    full_scores[:, class_ids] = scores
    return full_scores


def compute_metrics(estimator: TorchElasticNetProbe, features: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    predictions = estimator.predict(features)
    scores = decision_scores(estimator, features, n_classes)
    binarized = label_binarize(labels, classes=np.arange(n_classes))
    present = np.any(binarized == 1, axis=0)
    pr_auc = 0.0
    if np.any(present):
        pr_auc = float(average_precision_score(binarized[:, present], scores[:, present], average="macro"))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "pr_auc_macro": pr_auc,
    }


def optimize_probe(
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict[str, float], float]:
    trials = int(config["optuna"]["trials"])
    target_metric = str(config["optuna"]["target_metric"])
    n_classes = len(np.unique(np.concatenate([labels["training"], labels["validation"]])))
    sampler = optuna.samplers.TPESampler(seed=int(config["seed"]))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        c_value = trial.suggest_float("C", float(config["c_min"]), float(config["c_max"]), log=True)
        l1_ratio = trial.suggest_float("l1_ratio", float(config["l1_ratio_min"]), float(config["l1_ratio_max"]))
        estimator = build_elastic_net_probe(c_value, l1_ratio, config)
        estimator.fit(features["training"], labels["training"])
        metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes)
        return float(metrics[target_metric])

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return {"C": float(study.best_params["C"]), "l1_ratio": float(study.best_params["l1_ratio"])}, float(study.best_value)


def feature_importance(estimator: TorchElasticNetProbe, coord_dim: int, ph_dim: int, threshold: float) -> dict[str, float]:
    if coord_dim <= 0 or ph_dim <= 0:
        return {
            "coord_weight_l1": np.nan,
            "ph_weight_l1": np.nan,
            "ph_weight_l1_share": np.nan,
            "coord_weight_l2": np.nan,
            "ph_weight_l2": np.nan,
            "ph_weight_l2_share": np.nan,
            "coord_active_features": np.nan,
            "ph_active_features": np.nan,
            "ph_active_share": np.nan,
        }

    weights = estimator.coef_
    coord_weights = weights[:, :coord_dim]
    ph_weights = weights[:, coord_dim : coord_dim + ph_dim]
    coord_l1 = float(np.sum(np.abs(coord_weights)))
    ph_l1 = float(np.sum(np.abs(ph_weights)))
    coord_l2 = float(np.sqrt(np.sum(coord_weights**2)))
    ph_l2 = float(np.sqrt(np.sum(ph_weights**2)))
    coord_active = int(np.count_nonzero(np.any(np.abs(coord_weights) > threshold, axis=0)))
    ph_active = int(np.count_nonzero(np.any(np.abs(ph_weights) > threshold, axis=0)))
    return {
        "coord_weight_l1": coord_l1,
        "ph_weight_l1": ph_l1,
        "ph_weight_l1_share": ph_l1 / max(coord_l1 + ph_l1, 1.0e-12),
        "coord_weight_l2": coord_l2,
        "ph_weight_l2": ph_l2,
        "ph_weight_l2_share": ph_l2 / max(coord_l2 + ph_l2, 1.0e-12),
        "coord_active_features": coord_active,
        "ph_active_features": ph_active,
        "ph_active_share": ph_active / max(ph_dim, 1),
    }


def evaluate_mode(
    run: dict[str, object],
    mode: str,
    k: int,
    coord: dict[str, np.ndarray],
    ph: dict[str, np.ndarray],
    labels_text: dict[str, np.ndarray],
    config: dict,
) -> dict[str, object]:
    features, coord_dim, ph_dim = standardize_blocks(coord, ph, mode)
    _, labels = encode_labels(labels_text)
    best_params, best_val = optimize_probe(features, labels, config)
    estimator = build_elastic_net_probe(best_params["C"], best_params["l1_ratio"], config)
    estimator.fit(features["training"], labels["training"])

    n_classes = len(np.unique(np.concatenate([labels[split] for split in SPLITS])))
    val_metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes)
    test_metrics = compute_metrics(estimator, features["test"], labels["test"], n_classes)
    importance = feature_importance(estimator, coord_dim, ph_dim, float(config["weight_threshold"]))
    record = {
        "run_name": str(run["run_name"]),
        "method": str(run["method"]),
        "family": str(run["family"]),
        "code_dim": run["code_dim"],
        "target_active": run["target_active"],
        "seed": int(run["seed"]),
        "mode": mode,
        "k": int(k),
        "coord_dim": int(coord_dim),
        "ph_dim": int(ph_dim),
        "feature_dim": int(features["training"].shape[1]),
        "target_metric": str(config["optuna"]["target_metric"]),
        "best_c": float(best_params["C"]),
        "best_l1_ratio": float(best_params["l1_ratio"]),
        f"validation_{config['optuna']['target_metric']}": float(best_val),
        "validation_accuracy": val_metrics["accuracy"],
        "validation_f1_macro": val_metrics["f1_macro"],
        "validation_pr_auc_macro": val_metrics["pr_auc_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_pr_auc_macro": test_metrics["pr_auc_macro"],
        "n_train": int(len(labels["training"])),
        "n_validation": int(len(labels["validation"])),
        "n_test": int(len(labels["test"])),
        **importance,
    }
    report(
        f"hybrid method={record['method']} K={record['code_dim']} s={record['target_active']} "
        f"mode={mode} k={k} f1={record['test_f1_macro']:.3f} pr_auc={record['test_pr_auc_macro']:.3f} "
        f"ph_l1_share={record['ph_weight_l1_share'] if np.isfinite(record['ph_weight_l1_share']) else np.nan:.3f}"
    )
    return record


def compact_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "code_dim",
        "target_active",
        "mode",
        "k",
        "validation_f1_macro",
        "test_f1_macro",
        "test_pr_auc_macro",
        "best_c",
        "best_l1_ratio",
        "ph_weight_l1_share",
        "ph_weight_l2_share",
        "ph_active_features",
        "ph_active_share",
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
    return compact.sort_values(["method", "K", "s", "mode"], na_position="first")


def aggregate_for_plot(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["method", "code_dim", "target_active", "mode"], dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    return grouped


def add_anchor_line(ax: plt.Axes, metric: str) -> float | None:
    value = anchor_baseline(metric)
    if value is None:
        return None
    ax.axhline(value, color="black", linestyle="--", linewidth=1.0, label="Anchor upper bound")
    return value


def save_metric_plot(summary: pd.DataFrame, metric: str, image_root: Path, prefix: str) -> None:
    sparse = summary[summary["target_active"].notna()].copy()
    if sparse.empty:
        return
    image_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    code_dims = sorted(int(value) for value in sparse["code_dim"].dropna().unique())
    fig, axes = plt.subplots(1, len(code_dims), figsize=(6 * len(code_dims), 5.5), sharey=True, constrained_layout=True)
    if len(code_dims) == 1:
        axes = [axes]
    palette = {"sae": "#4c78a8", "topo_sae": "#f58518", "anchor": "#54a24b"}
    markers = {"coord": "o", "ph": "s", "coord_ph": "^"}
    linestyles = {"coord": "-", "ph": ":", "coord_ph": "--"}
    all_scores: list[float] = []
    anchor_value = anchor_baseline(metric)
    if anchor_value is not None:
        all_scores.append(anchor_value)

    for ax, code_dim in zip(axes, code_dims, strict=True):
        panel = aggregate_for_plot(sparse[sparse["code_dim"] == code_dim], metric)
        if panel.empty:
            continue
        all_scores.extend(panel["mean"].dropna().astype(float).tolist())
        for (method, mode), group in panel.groupby(["method", "mode"], dropna=False):
            group = group.sort_values("target_active")
            color = palette.get(str(method), None)
            ax.errorbar(
                group["target_active"].astype(float),
                group["mean"].astype(float),
                yerr=group["se"].astype(float),
                marker=markers.get(str(mode), "o"),
                linestyle=linestyles.get(str(mode), "-"),
                linewidth=1.6,
                elinewidth=0.9,
                capsize=3,
                color=color,
                label=f"{clean_label(str(method))} {clean_label(str(mode))}",
            )
        line_value = add_anchor_line(ax, metric)
        if line_value is not None:
            all_scores.append(line_value)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Target Active Coefficients")
        ax.set_title(f"K={code_dim}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(clean_label(metric.replace("test_", "")))
    set_tight_score_axis(axes[0], pd.Series(all_scores), extra_values=[])
    for ax in axes[1:]:
        ax.set_ylim(axes[0].get_ylim())
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[-1].legend(unique.values(), unique.keys(), title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle(f"Hybrid Probe {clean_label(metric.replace('test_', ''))}", fontsize=15)
    fig.savefig(image_root / f"{prefix}_{metric}_by_s.png", dpi=200)
    plt.close(fig)


def save_importance_plot(summary: pd.DataFrame, metric: str, image_root: Path, prefix: str) -> None:
    hybrid = summary[(summary["mode"] == "coord_ph") & summary["target_active"].notna()].copy()
    if hybrid.empty:
        return
    image_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    code_dims = sorted(int(value) for value in hybrid["code_dim"].dropna().unique())
    fig, axes = plt.subplots(1, len(code_dims), figsize=(6 * len(code_dims), 5.0), sharey=True, constrained_layout=True)
    if len(code_dims) == 1:
        axes = [axes]
    palette = {"sae": "#4c78a8", "topo_sae": "#f58518"}
    values: list[float] = []
    for ax, code_dim in zip(axes, code_dims, strict=True):
        panel = aggregate_for_plot(hybrid[hybrid["code_dim"] == code_dim], metric)
        values.extend(panel["mean"].dropna().astype(float).tolist())
        for method, group in panel.groupby("method", dropna=False):
            group = group.sort_values("target_active")
            ax.errorbar(
                group["target_active"].astype(float),
                group["mean"].astype(float),
                yerr=group["se"].astype(float),
                marker="o",
                linewidth=1.6,
                elinewidth=0.9,
                capsize=3,
                color=palette.get(str(method), None),
                label=clean_label(str(method)),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Target Active Coefficients")
        ax.set_title(f"K={code_dim}")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(clean_label(metric))
    finite = pd.Series(values).dropna()
    if not finite.empty:
        lower = max(0.0, float(finite.min()) - 0.05)
        upper = min(1.0, float(finite.max()) + 0.05)
        axes[0].set_ylim(lower, upper)
        for ax in axes[1:]:
            ax.set_ylim(axes[0].get_ylim())
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[-1].legend(unique.values(), unique.keys(), title="Method", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle(f"Hybrid Probe {clean_label(metric)}", fontsize=15)
    fig.savefig(image_root / f"{prefix}_{metric}_by_s.png", dpi=200)
    plt.close(fig)


def save_plots(summary: pd.DataFrame, image_root: Path, prefix: str) -> None:
    save_metric_plot(summary, "test_f1_macro", image_root, prefix)
    save_metric_plot(summary, "test_pr_auc_macro", image_root, prefix)
    save_importance_plot(summary, "ph_weight_l1_share", image_root, prefix)
    save_importance_plot(summary, "ph_active_share", image_root, prefix)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data_dir = args.data_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    prefix = str(config["output_prefix"])
    k = int(config["k"])
    device = resolve_device(str(config["device"]))

    report(f"START module=evaluation.hybrid data_dir={data_dir} k={k} device={device} config={args.config}")
    runs = load_runs(data_dir, config)
    cache_dir = data_root / "transfer_features"
    records: list[dict[str, object]] = []
    for run in runs:
        coord, ph, labels_text = load_feature_blocks(run, k, config, cache_dir)
        for mode in [str(value) for value in config["feature_modes"]]:
            records.append(evaluate_mode(run, mode, k, coord, ph, labels_text, config))

    summary = pd.DataFrame.from_records(records)
    compact = compact_summary(summary)
    summary_path = data_root / f"{prefix}_summary.csv"
    compact_path = data_root / f"{prefix}_compact_summary.csv"
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    save_plots(summary, image_root, prefix)

    log("hybrid_summary")
    log(compact.round(4).to_string(index=False))
    report(f"DONE module=evaluation.hybrid rows={len(summary)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
