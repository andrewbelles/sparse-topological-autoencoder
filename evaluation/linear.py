#!/usr/bin/env python3
#
# linear.py  Andrew Belles  April 13th, 2026
#
# Linear probe evaluation over compression parquet embeddings.
#

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from evaluation.visualizations import (
    save_confusion_matrix_plot,
    save_dual_metric_method_plot,
    save_ratio_metric_plot,
    save_subset_accuracy_plot,
    save_topology_performance_scatter,
)
from evaluation.filters import FILTER_DEFAULTS, passes_run_filters
from compression.common import SPLITS, active_feature_columns, embedding_columns
from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "linear.yaml"
DEFAULT_CONFIG = {
    "seed": 7,
    "classifier": "logistic",
    "classifiers": [],
    "optuna": {
        "trials": 20,
        "target_metric": "pr_auc_macro",
    },
    "knn_neighbors": [3, 5, 9, 15, 25],
    "subset_fractions": [0.01, 0.03, 0.1, 0.3, 0.5, 1.0],
    "c_min": 1e-4,
    "c_max": 1.0,
    "max_iter": 10000,
    "tol": 1e-3,
    "device": "cuda",
    "torch_epochs": 200,
    "torch_lr": 0.05,
    "torch_batch_size": 2048,
    "weight_threshold": 1e-8,
    "save_confusion_matrices": False,
    "run_subset_curves": False,
    "anchor_dir": "representation/data",
    "source": "anchor",
    "dataset": "fma_small_mel",
    "include_anchor": False,
    **FILTER_DEFAULTS,
}


def log(message: str) -> None:
    print(message, flush=True)


def report(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compression embeddings with sparse linear probes.")
    parser.add_argument(
        "-p",
        "--parquet",
        type=Path,
        default=None,
        help="Path to one split parquet, typically the training parquet under compression/data/.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "compression" / "data",
        help="Directory of compression parquet split outputs. Used when --parquet is omitted.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def discover_split_groups(data_root: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}

    for path in sorted(data_root.glob("*.parquet")):
        stem = path.stem
        for split in SPLITS:
            suffix = f"_{split}"
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                groups.setdefault(prefix, {})[split] = path
                break

    return {
        prefix: paths
        for prefix, paths in groups.items()
        if all(split in paths for split in SPLITS)
    }


def parse_run_metadata(run_name: str) -> dict[str, object]:
    projected = re.match(r"^(?P<method>.+)_r(?P<ratio>\d+)_(?P<dataset>.+)$", run_name)
    if projected:
        return {
            "method": projected.group("method"),
            "ratio_percent": int(projected.group("ratio")),
            "dataset": projected.group("dataset"),
        }

    representation = re.match(r"^(?P<method>.+)_(?P<dataset>fma_.+)$", run_name)
    if representation:
        return {
            "method": representation.group("method"),
            "ratio_percent": None,
            "dataset": representation.group("dataset"),
        }

    return {"method": run_name.split("_", 1)[0], "ratio_percent": None, "dataset": run_name}


def filter_split_groups(groups: dict[str, dict[str, Path]], config: dict) -> dict[str, dict[str, Path]]:
    filtered: dict[str, dict[str, Path]] = {}
    for run_name, split_paths in groups.items():
        metadata = parse_run_metadata(run_name)
        if passes_run_filters(
            run_name,
            method=str(metadata["method"]),
            ratio=metadata["ratio_percent"] if metadata["ratio_percent"] is None else int(metadata["ratio_percent"]),
            config=config,
        ):
            filtered[run_name] = split_paths
    return filtered


def has_explicit_run_filters(config: dict) -> bool:
    return any(config.get(key) for key in FILTER_DEFAULTS)


def anchor_scoped_config_for_compression(data_root: Path, config: dict) -> dict:
    if has_explicit_run_filters(config):
        return config

    if data_root.name == "data" and data_root.parent.name == "compression":
        scoped = dict(config)
        scoped["include_methods"] = ["*_anchor"]
        return scoped

    return config


def is_default_anchor_scope_active(data_root: Path, config: dict) -> bool:
    return (
        data_root.name == "data"
        and data_root.parent.name == "compression"
        and not has_explicit_run_filters(config)
    )


def infer_split_paths(parquet_path: Path) -> tuple[str, dict[str, Path]]:
    stem = parquet_path.stem
    matched_split = None
    for split in SPLITS:
        suffix = f"_{split}"
        if stem.endswith(suffix):
            matched_split = split
            prefix = stem[: -len(suffix)]
            break

    if matched_split is None:
        raise ValueError(f"expected parquet path ending in one of {SPLITS}: {parquet_path}")

    split_paths = {split: parquet_path.with_name(f"{prefix}_{split}.parquet") for split in SPLITS}
    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} parquet: {path}")

    return prefix, split_paths


def discover_method_parquets(data_root: Path, config: dict) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for path in sorted(data_root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        required = {"method", "split", "ratio_percent", "m_dim", "seed", "track_id", "genre_top"}
        if not required.issubset(frame.columns):
            continue
        group_columns = ["method", "ratio_percent", "seed"]
        if {"code_dim", "target_active"}.issubset(frame.columns):
            group_columns.extend(["code_dim", "target_active"])
        for keys, group in frame.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_values = dict(zip(group_columns, keys, strict=True))
            method = str(key_values["method"])
            ratio = key_values["ratio_percent"]
            seed = key_values["seed"]
            ratio_value = None if pd.isna(ratio) else int(ratio)
            run_name = f"{method}_r{ratio_value:03d}_s{int(seed):02d}" if ratio_value is not None else method
            if "code_dim" in key_values:
                run_name = (
                    f"{method}_k{int(key_values['code_dim']):04d}_"
                    f"a{int(key_values['target_active']):03d}_r{ratio_value:03d}_s{int(seed):02d}"
                )
            if not passes_run_filters(run_name, method, ratio_value, config):
                continue
            if set(group["split"].unique()) >= set(SPLITS):
                groups[run_name] = group.copy()
    return groups


def anchor_group(config: dict) -> dict[str, pd.DataFrame]:
    anchor_dir = Path(str(config["anchor_dir"])).expanduser()
    source = str(config["source"])
    dataset = str(config["dataset"])
    frames = []
    for split in SPLITS:
        path = anchor_dir / f"{source}_{dataset}_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing anchor parquet: {path}")
        frame = pd.read_parquet(path).copy()
        frame["method"] = source
        frame["family"] = "baseline"
        frame["dataset"] = dataset
        frame["source"] = source
        frame["split"] = split
        frame["ratio_percent"] = 100
        frame["m_dim"] = len(embedding_columns(frame))
        frame["input_dim"] = frame["m_dim"]
        frame["seed"] = 0
        frames.append(frame)
    return {f"{source}_r100_s00": pd.concat(frames, ignore_index=True)}


def split_frames_from_group(group: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[str]]:
    m_dim = int(group["m_dim"].iloc[0])
    columns = active_feature_columns(group, m_dim)
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_frame = group[group["split"] == split].copy()
        if split_frame.empty:
            raise ValueError(f"group is missing split={split}")
        frames[split] = split_frame[[*(column for column in split_frame.columns if not column.startswith("embedding_")), *columns]]
    return frames, columns


def get_embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("embedding_"))
    if not columns:
        raise ValueError("parquet file does not contain embedding columns")
    return columns


def load_split_frames(split_paths: dict[str, Path]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames = {split: pd.read_parquet(path).copy() for split, path in split_paths.items()}
    embedding_columns = get_embedding_columns(frames["training"])

    for split, frame in frames.items():
        missing_columns = [column for column in embedding_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{split} parquet is missing embedding columns: {missing_columns[:5]}")

    return frames, embedding_columns


def encode_labels(frames: dict[str, pd.DataFrame]) -> tuple[LabelEncoder, dict[str, np.ndarray]]:
    encoder = LabelEncoder()
    all_labels = pd.concat([frame["genre_top"] for frame in frames.values()], ignore_index=True)
    encoder.fit(all_labels)
    encoded = {
        split: encoder.transform(frame["genre_top"])
        for split, frame in frames.items()
    }
    return encoder, encoded


def build_features(
    frames: dict[str, pd.DataFrame],
    embedding_columns: list[str],
) -> dict[str, np.ndarray]:
    return {
        split: frame[embedding_columns].to_numpy(dtype=np.float32, copy=True)
        for split, frame in frames.items()
    }


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class TorchLogisticProbe:
    def __init__(
        self,
        c_value: float,
        device: torch.device,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        seed: int,
    ):
        self.c_value = float(c_value)
        self.device = device
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.classes_ = np.unique(labels)
        n_classes = int(len(self.classes_))
        n_features = int(features.shape[1])

        class_to_index = {int(label): index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([class_to_index[int(label)] for label in labels], dtype=np.int64)

        mean_np = features.mean(axis=0, keepdims=True).astype("float32")
        std_np = np.maximum(features.std(axis=0, keepdims=True), 1e-6).astype("float32")
        self.mean = torch.from_numpy(mean_np).to(self.device)
        self.std = torch.from_numpy(std_np).to(self.device)

        generator = torch.Generator().manual_seed(self.seed)
        weight = torch.empty((n_classes, n_features), device=self.device)
        torch.nn.init.xavier_uniform_(weight)
        bias = torch.zeros(n_classes, device=self.device)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        optimizer = torch.optim.Adam([weight, bias], lr=self.learning_rate)
        labels_tensor = torch.from_numpy(encoded)
        n_samples = int(features.shape[0])
        l2_scale = 0.5 / max(self.c_value * n_samples, 1e-8)

        for _ in range(self.epochs):
            permutation = torch.randperm(n_samples, generator=generator)
            for start in range(0, n_samples, self.batch_size):
                indices = permutation[start : start + self.batch_size]
                batch_x = torch.from_numpy(features[indices.numpy()]).to(self.device)
                batch_y = labels_tensor[indices].to(self.device)
                batch_x = (batch_x - self.mean) / self.std

                logits = F.linear(batch_x, weight, bias)
                loss = F.cross_entropy(logits, batch_y) + l2_scale * weight.pow(2).sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self.weight = weight.detach()
        self.bias = bias.detach()
        return self

    @torch.no_grad()
    def decision_function(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None or self.weight is None or self.bias is None:
            raise RuntimeError("TorchLogisticProbe must be fit before inference")

        batches: list[np.ndarray] = []
        for start in range(0, features.shape[0], self.batch_size):
            batch_x = torch.from_numpy(features[start : start + self.batch_size]).to(self.device)
            batch_x = (batch_x - self.mean) / self.std
            logits = F.linear(batch_x, self.weight, self.bias)
            batches.append(logits.cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("TorchLogisticProbe must be fit before prediction")
        logits = self.decision_function(features)
        return self.classes_[np.argmax(logits, axis=1)]

    def weight_matrix(self) -> np.ndarray:
        if self.weight is None:
            raise RuntimeError("TorchLogisticProbe must be fit before reading weights")
        return self.weight.detach().cpu().numpy()


def build_estimator(classifier: str, c_value: float, max_iter: int, tol: float, config: dict):
    if classifier == "logistic":
        return TorchLogisticProbe(
            c_value=c_value,
            device=resolve_device(str(config["device"])),
            epochs=int(config["torch_epochs"]),
            learning_rate=float(config["torch_lr"]),
            batch_size=int(config["torch_batch_size"]),
            seed=int(config["seed"]),
        )

    if classifier == "knn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=max(1, int(round(c_value))))),
            ]
        )

    if classifier == "sparse_logistic":
        base = LogisticRegression(
            penalty="l1",
            C=c_value,
            solver="liblinear",
            max_iter=max_iter,
            tol=tol,
        )
    elif classifier == "svm":
        base = LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual="auto",
            C=c_value,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        raise ValueError(f"unsupported classifier: {classifier}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(base)),
        ]
    )


def sample_training_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction <= 1:
        raise ValueError(f"subset fraction must be in (0, 1], got {fraction}")

    if fraction >= 1.0:
        return x_train, y_train

    n_classes = len(np.unique(y_train))
    subset_size = max(n_classes, int(round(len(y_train) * fraction)))
    subset_size = min(subset_size, len(y_train))

    indices = np.arange(len(y_train))
    _, class_counts = np.unique(y_train, return_counts=True)
    if np.min(class_counts) < 2:
        generator = np.random.default_rng(seed)
        required_indices: list[int] = []
        for label in np.unique(y_train):
            class_indices = indices[y_train == label]
            required_indices.append(int(generator.choice(class_indices, size=1, replace=False)[0]))

        remaining_pool = np.array([index for index in indices if index not in set(required_indices)], dtype=int)
        remaining_needed = max(0, subset_size - len(required_indices))
        if remaining_needed > 0 and len(remaining_pool) > 0:
            sampled_extra = generator.choice(remaining_pool, size=remaining_needed, replace=False)
            subset_indices = np.sort(np.concatenate([np.asarray(required_indices, dtype=int), sampled_extra]))
        else:
            subset_indices = np.sort(np.asarray(required_indices, dtype=int))
    else:
        subset_indices, _ = train_test_split(
            indices,
            train_size=subset_size,
            stratify=y_train,
            random_state=seed,
        )
    return x_train[subset_indices], y_train[subset_indices]


def compute_margin(estimator, features: np.ndarray, n_classes: int) -> float:
    scores = get_decision_scores(estimator, features, n_classes=n_classes)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    return float(np.mean(margins))


def get_decision_scores(estimator, features: np.ndarray, n_classes: int) -> np.ndarray:
    if isinstance(estimator, TorchLogisticProbe):
        scores = estimator.decision_function(features)
        if scores.shape[1] != n_classes:
            full_scores = np.full((scores.shape[0], n_classes), float(np.min(scores) - 1.0), dtype=np.float64)
            class_ids = np.asarray(estimator.classes_, dtype=int)
            full_scores[:, class_ids] = scores
            return full_scores
        return scores

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        scores = estimator.decision_function(features)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)

    classifier = estimator.named_steps["classifier"]
    class_ids = np.asarray(classifier.classes_, dtype=int)
    full_scores = np.full((scores.shape[0], n_classes), float(np.min(scores) - 1.0), dtype=np.float64)
    if scores.shape[1] == len(class_ids):
        full_scores[:, class_ids] = scores
    elif len(class_ids) == 1 and scores.shape[1] == 2:
        full_scores[:, class_ids[0]] = scores[:, 1]
    else:
        raise ValueError(
            f"could not align decision scores: scores shape={scores.shape}, class_ids shape={class_ids.shape}"
        )
    return full_scores


def get_weight_matrix(estimator) -> np.ndarray:
    if isinstance(estimator, TorchLogisticProbe):
        return estimator.weight_matrix()

    classifier = estimator.named_steps["classifier"]
    if isinstance(classifier, KNeighborsClassifier):
        return np.zeros((1, int(classifier.n_features_in_)), dtype=np.float32)
    return np.stack([sub_estimator.coef_.reshape(-1) for sub_estimator in classifier.estimators_], axis=0)


def compute_effective_dimensions(estimator, threshold: float) -> tuple[int, float]:
    weights = get_weight_matrix(estimator)
    active_dims = np.any(np.abs(weights) > threshold, axis=0)
    count = int(np.count_nonzero(active_dims))
    ratio = float(count / active_dims.size)
    return count, ratio


def normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.clip(row_sums, a_min=1.0, a_max=None)


def compute_pr_auc_macro(scores: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    binarized = label_binarize(labels, classes=np.arange(n_classes))
    present_mask = np.any(binarized == 1, axis=0)
    if not np.any(present_mask):
        return 0.0
    return float(average_precision_score(binarized[:, present_mask], scores[:, present_mask], average="macro"))


def compute_metrics(
    estimator,
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    predictions = estimator.predict(features)
    scores = get_decision_scores(estimator, features, n_classes=n_classes)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "pr_auc_macro": compute_pr_auc_macro(scores, labels, n_classes=n_classes),
    }


def optimize_hyperparameters(
    classifier: str,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, float], float]:
    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    trials = int(config["optuna"]["trials"])
    seed = int(config["seed"])
    target_metric = str(config["optuna"]["target_metric"])
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        if classifier == "knn":
            choices = [int(value) for value in config.get("knn_neighbors", [3, 5, 9, 15, 25])]
            c_value = float(trial.suggest_categorical("n_neighbors", choices))
        else:
            c_value = trial.suggest_float("C", float(config["c_min"]), float(config["c_max"]), log=True)
        estimator = build_estimator(classifier, c_value, max_iter=max_iter, tol=tol, config=config)
        estimator.fit(x_train, y_train)
        metrics = compute_metrics(estimator, x_val, y_val, n_classes=n_classes)
        return float(metrics[target_metric])

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    if classifier == "knn":
        return {"C": float(study.best_params["n_neighbors"]), "n_neighbors": int(study.best_params["n_neighbors"])}, float(study.best_value)
    return study.best_params, float(study.best_value)


def format_model_name(run_name: str) -> str:
    return run_name.replace("_", " ").strip().title()


def run_descriptor(first: pd.Series) -> str:
    parts = [
        f"method={first['method']}",
        f"ratio={int(first['ratio_percent'])}",
        f"seed={int(first['seed'])}",
    ]
    if "code_dim" in first.index and "target_active" in first.index:
        parts.insert(1, f"K={int(first['code_dim'])}")
        parts.insert(2, f"s={int(first['target_active'])}")
    return " ".join(parts)


def evaluate_subset_curve(
    classifier: str,
    config: dict,
    best_c: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> pd.DataFrame:
    max_iter = int(config["max_iter"])
    tol = float(config["tol"])
    threshold = float(config["weight_threshold"])
    subset_fractions = [float(value) for value in config["subset_fractions"]]
    seed = int(config["seed"])
    records: list[dict[str, float]] = []

    for index, fraction in enumerate(subset_fractions):
        subset_x, subset_y = sample_training_subset(x_train, y_train, fraction, seed + index)
        estimator = build_estimator(classifier, best_c, max_iter=max_iter, tol=tol, config=config)
        estimator.fit(subset_x, subset_y)
        metrics = compute_metrics(estimator, x_test, y_test, n_classes=n_classes)
        margin = compute_margin(estimator, x_test, n_classes=n_classes)
        effective_dims, effective_ratio = compute_effective_dimensions(estimator, threshold)

        record = {
            "subset_fraction": float(fraction),
            "subset_count": int(len(subset_y)),
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "test_pr_auc_macro": metrics["pr_auc_macro"],
            "mean_margin": margin,
            "effective_dimensions": effective_dims,
            "effective_dimension_ratio": effective_ratio,
        }
        records.append(record)
        log(
            f"subset_fraction={fraction:.4f} subset_count={len(subset_y)} "
            f"test_accuracy={metrics['accuracy']:.4f} test_f1_macro={metrics['f1_macro']:.4f} "
            f"test_pr_auc_macro={metrics['pr_auc_macro']:.4f} mean_margin={margin:.4f} effective_dims={effective_dims}"
        )

    return pd.DataFrame.from_records(records)


def evaluate_run(
    run_name: str,
    split_paths: dict[str, Path],
    classifier: str,
    target_metric: str,
    config: dict,
    data_root: Path,
    image_root: Path,
) -> pd.DataFrame:
    frames, embedding_columns = load_split_frames(split_paths)
    features = build_features(frames, embedding_columns)
    label_encoder, labels = encode_labels(frames)

    best_params, best_val_accuracy = optimize_hyperparameters(
        classifier,
        config,
        features["training"],
        labels["training"],
        features["validation"],
        labels["validation"],
    )
    best_c = float(best_params["C"])
    log(
        f"best_hparams classifier={classifier} C={best_c:.6g} "
        f"target_metric={target_metric} validation_{target_metric}={best_val_accuracy:.4f}"
    )

    estimator = build_estimator(
        classifier,
        best_c,
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        config=config,
    )
    estimator.fit(features["training"], labels["training"])

    n_classes = len(label_encoder.classes_)
    val_metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes=n_classes)
    test_metrics = compute_metrics(estimator, features["test"], labels["test"], n_classes=n_classes)
    test_predictions = estimator.predict(features["test"])
    mean_margin = compute_margin(estimator, features["test"], n_classes=n_classes)
    effective_dims, effective_ratio = compute_effective_dimensions(
        estimator,
        threshold=float(config["weight_threshold"]),
    )

    summary_frame = pd.DataFrame.from_records(
        [
            {
                "run_name": run_name,
                **parse_run_metadata(run_name),
                "classifier": classifier,
                "target_metric": target_metric,
                "best_c": best_c,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_f1_macro": val_metrics["f1_macro"],
                "validation_pr_auc_macro": val_metrics["pr_auc_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_pr_auc_macro": test_metrics["pr_auc_macro"],
                "mean_margin": mean_margin,
                "effective_dimensions": effective_dims,
                "effective_dimension_ratio": effective_ratio,
                "n_train": int(len(labels["training"])),
                "n_validation": int(len(labels["validation"])),
                "n_test": int(len(labels["test"])),
            }
        ]
    )

    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    output_prefix = f"{run_name}_{classifier}"
    summary_path = data_root / f"{output_prefix}_summary.csv"
    params_path = data_root / f"{output_prefix}_best_params.json"

    summary_frame.to_csv(summary_path, index=False)
    params_path.write_text(json.dumps({"classifier": classifier, **best_params}, indent=2), encoding="utf-8")

    if bool(config.get("save_confusion_matrices", False)):
        confusion = normalized_confusion(labels["test"], test_predictions, n_classes=n_classes)
        display_labels = [str(label) for label in label_encoder.classes_]
        confusion_path = image_root / f"{output_prefix}_confusion_matrix.png"
        save_confusion_matrix_plot(
            confusion,
            display_labels,
            confusion_path,
            title=f"Confusion Matrix - {format_model_name(output_prefix)}",
        )

    if bool(config.get("run_subset_curves", False)):
        subset_frame = evaluate_subset_curve(
            classifier,
            config,
            best_c,
            features["training"],
            labels["training"],
            features["test"],
            labels["test"],
            n_classes=n_classes,
        )
        subset_path = data_root / f"{output_prefix}_subset_accuracy.csv"
        subset_plot_path = image_root / f"{output_prefix}_subset_accuracy.png"
        subset_frame.to_csv(subset_path, index=False)
        save_subset_accuracy_plot(
            subset_frame,
            subset_plot_path,
            title=f"Subset Accuracy - {format_model_name(output_prefix)}",
        )

    row = summary_frame.iloc[0]
    log(
        f"run={run_name} method={row['method']} ratio={row['ratio_percent']} "
        f"f1={row['test_f1_macro']:.3f} pr_auc={row['test_pr_auc_macro']:.3f}"
    )
    return summary_frame


def evaluate_group(
    run_name: str,
    group: pd.DataFrame,
    classifier: str,
    target_metric: str,
    config: dict,
    data_root: Path,
    image_root: Path,
) -> pd.DataFrame:
    frames, columns = split_frames_from_group(group)
    features = build_features(frames, columns)
    label_encoder, labels = encode_labels(frames)

    best_params, best_val_score = optimize_hyperparameters(
        classifier,
        config,
        features["training"],
        labels["training"],
        features["validation"],
        labels["validation"],
    )
    best_c = float(best_params["C"])
    estimator = build_estimator(
        classifier,
        best_c,
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        config=config,
    )
    estimator.fit(features["training"], labels["training"])

    n_classes = len(label_encoder.classes_)
    val_metrics = compute_metrics(estimator, features["validation"], labels["validation"], n_classes=n_classes)
    test_metrics = compute_metrics(estimator, features["test"], labels["test"], n_classes=n_classes)
    test_predictions = estimator.predict(features["test"])
    mean_margin = compute_margin(estimator, features["test"], n_classes=n_classes)
    effective_dims, effective_ratio = compute_effective_dimensions(
        estimator,
        threshold=float(config["weight_threshold"]),
    )

    first = group.iloc[0]
    extra_metadata = {}
    if {"code_dim", "target_active"}.issubset(group.columns):
        extra_metadata = {
            "code_dim": int(first["code_dim"]),
            "target_active": int(first["target_active"]),
            "actual_active_mean": float(first.get("actual_active_mean", np.nan)),
            "l1_lambda": float(first.get("l1_lambda", np.nan)),
            "topology_weight": float(first.get("topology_weight", np.nan)),
        }
    summary_frame = pd.DataFrame.from_records(
        [
            {
                "run_name": run_name,
                "method": str(first["method"]),
                "family": str(first.get("family", "")),
                "ratio_percent": int(first["ratio_percent"]),
                "m_dim": int(first["m_dim"]),
                "input_dim": int(first["input_dim"]),
                "seed": int(first["seed"]),
                **extra_metadata,
                "dataset": str(first.get("dataset", config["dataset"])),
                "classifier": classifier,
                "target_metric": target_metric,
                "best_c": best_c,
                f"validation_{target_metric}": best_val_score,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_f1_macro": val_metrics["f1_macro"],
                "validation_pr_auc_macro": val_metrics["pr_auc_macro"],
                "test_accuracy": test_metrics["accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_pr_auc_macro": test_metrics["pr_auc_macro"],
                "mean_margin": mean_margin,
                "effective_dimensions": effective_dims,
                "effective_dimension_ratio": effective_ratio,
                "n_train": int(len(labels["training"])),
                "n_validation": int(len(labels["validation"])),
                "n_test": int(len(labels["test"])),
            }
        ]
    )

    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{run_name}_{classifier}"
    summary_frame.to_csv(data_root / f"{output_prefix}_summary.csv", index=False)
    (data_root / f"{output_prefix}_best_params.json").write_text(
        json.dumps({"classifier": classifier, **best_params}, indent=2),
        encoding="utf-8",
    )

    if bool(config.get("save_confusion_matrices", False)):
        confusion = normalized_confusion(labels["test"], test_predictions, n_classes=n_classes)
        save_confusion_matrix_plot(
            confusion,
            [str(label) for label in label_encoder.classes_],
            image_root / f"{output_prefix}_confusion_matrix.png",
            title=f"Confusion Matrix - {format_model_name(output_prefix)}",
        )

    log(f"linear {run_descriptor(first)} f1={test_metrics['f1_macro']:.3f} pr_auc={test_metrics['pr_auc_macro']:.3f}")
    return summary_frame


def attach_topology_summary(summary_frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    topology_path = data_root / "topology_summary.csv"
    if not topology_path.is_file():
        return summary_frame

    topology = pd.read_csv(topology_path)
    if topology.empty:
        return summary_frame
    if not {"method", "ratio_percent", "wasserstein_distance", "critical_pair_distortion"}.issubset(topology.columns):
        return summary_frame

    summary = summary_frame.copy()
    topology = topology.copy()
    summary["ratio_percent"] = pd.to_numeric(summary["ratio_percent"], errors="coerce")
    topology["ratio_percent"] = pd.to_numeric(topology["ratio_percent"], errors="coerce")

    aggregations = {
        "topology_wasserstein": ("wasserstein_distance", "mean"),
        "wasserstein_h0": ("wasserstein_h0", "mean"),
        "wasserstein_h1": ("wasserstein_h1", "mean"),
        "betti_dist_h0": ("betti_dist_h0", "mean"),
        "betti_dist_h1": ("betti_dist_h1", "mean"),
        "critical_pair_distortion": ("critical_pair_distortion", "mean"),
    }
    if {"persistence_image_h0", "persistence_image_h1", "persistence_image_distance"}.issubset(topology.columns):
        aggregations.update(
            {
                "persistence_image_h0": ("persistence_image_h0", "mean"),
                "persistence_image_h1": ("persistence_image_h1", "mean"),
                "persistence_image_distance": ("persistence_image_distance", "mean"),
            }
        )
    grouped = topology.groupby(["method", "ratio_percent", "seed"], dropna=False).agg(**aggregations).reset_index()
    summary["seed"] = pd.to_numeric(summary["seed"], errors="coerce")
    grouped["seed"] = pd.to_numeric(grouped["seed"], errors="coerce")
    return summary.merge(grouped, on=["method", "ratio_percent", "seed"], how="left")


def topology_genre_join(summary_frame: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    topology_path = data_root / "topology_summary.csv"
    if not topology_path.is_file():
        return pd.DataFrame()
    topology = pd.read_csv(topology_path)
    if topology.empty:
        return pd.DataFrame()
    summary = summary_frame.copy()
    for column in ["ratio_percent", "seed"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
        topology[column] = pd.to_numeric(topology[column], errors="coerce")
    return topology.merge(
        summary[
            [
                "method",
                "ratio_percent",
                "seed",
                "test_f1_macro",
                "test_pr_auc_macro",
                "test_accuracy",
                "classifier",
            ]
        ],
        on=["method", "ratio_percent", "seed"],
        how="inner",
    )


def compact_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compact = frame.copy()
    compact["m/N"] = compact["ratio_percent"].apply(lambda value: "base" if pd.isna(value) else f"{int(value)}%")

    columns = {
        "method": "method",
        "m/N": "m/N",
        "validation_pr_auc_macro": "val_pr",
        "test_f1_macro": "f1",
        "test_pr_auc_macro": "pr_auc",
    }
    if "code_dim" in compact.columns and "target_active" in compact.columns:
        columns = {
            "method": "method",
            "code_dim": "K",
            "target_active": "s",
            "m/N": "m/N",
            "validation_pr_auc_macro": "val_pr",
            "test_f1_macro": "f1",
            "test_pr_auc_macro": "pr_auc",
        }
    if "topology_wasserstein" in compact.columns:
        columns["topology_wasserstein"] = "top_wass"
    if "persistence_image_distance" in compact.columns:
        columns["persistence_image_distance"] = "pi_dist"
    if "critical_pair_distortion" in compact.columns:
        columns["critical_pair_distortion"] = "crit_dist"

    result = compact[list(columns)].rename(columns=columns)
    result["_sort_method"] = compact["method"].to_numpy()
    result["_sort_ratio"] = compact["ratio_percent"].fillna(10_000).astype(float).to_numpy()
    if "code_dim" in compact.columns and "target_active" in compact.columns:
        result["_sort_k"] = compact["code_dim"].fillna(0).astype(float).to_numpy()
        result["_sort_s"] = compact["target_active"].fillna(0).astype(float).to_numpy()
        result = result.sort_values(["_sort_method", "_sort_k", "_sort_s"]).drop(
            columns=["_sort_method", "_sort_ratio", "_sort_k", "_sort_s"]
        )
    else:
        result = result.sort_values(["_sort_method", "_sort_ratio"]).drop(columns=["_sort_method", "_sort_ratio"])

    for column in result.columns:
        if column not in {"method", "m/N"}:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(3)
    return result


def sensing_methods(frame: pd.DataFrame) -> list[str]:
    methods = frame.loc[frame["ratio_percent"].notna(), "method"].dropna().unique().tolist()
    return sorted(str(method) for method in methods)


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    classifier = str(config["classifier"]).lower()
    target_metric = str(config["optuna"]["target_metric"])
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"

    groups: dict[str, pd.DataFrame] = {}
    if bool(config.get("include_anchor", False)):
        groups.update(anchor_group(config))
    if args.parquet is not None:
        parquet_path = args.parquet.expanduser().resolve()
        frame = pd.read_parquet(parquet_path)
        required = {"method", "split", "ratio_percent", "m_dim", "seed", "track_id", "genre_top"}
        if not required.issubset(frame.columns):
            raise ValueError(f"parquet does not use the method schema: {parquet_path}")
        group_columns = ["method", "ratio_percent", "seed"]
        if {"code_dim", "target_active"}.issubset(frame.columns):
            group_columns.extend(["code_dim", "target_active"])
        for keys, group in frame.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_values = dict(zip(group_columns, keys, strict=True))
            method = str(key_values["method"])
            ratio_value = int(key_values["ratio_percent"])
            seed_value = int(key_values["seed"])
            run_name = f"{method}_r{ratio_value:03d}_s{seed_value:02d}"
            if "code_dim" in key_values:
                run_name = (
                    f"{method}_k{int(key_values['code_dim']):04d}_"
                    f"a{int(key_values['target_active']):03d}_r{ratio_value:03d}_s{seed_value:02d}"
                )
            if passes_run_filters(run_name, method, ratio_value, config):
                groups[run_name] = group.copy()
        source_description = f"parquet={parquet_path}"
    else:
        compression_data_root = args.data_dir.expanduser().resolve()
        groups.update(discover_method_parquets(compression_data_root, config))
        source_description = f"data_dir={compression_data_root}"
    if not groups:
        raise FileNotFoundError(f"no method parquet groups matched filters for {source_description}")

    device_note = f" device={resolve_device(str(config['device']))}" if classifier == "logistic" else ""
    report(
        f"START module=evaluation.linear {source_description} classifier={classifier}"
        f"{device_note} config={args.config}"
    )

    summary_frames = [
        evaluate_group(run_name, group, classifier, target_metric, config, data_root, image_root)
        for run_name, group in sorted(groups.items())
    ]

    combined = pd.concat(summary_frames, ignore_index=True)
    combined = attach_topology_summary(combined, data_root)
    topology_joined = topology_genre_join(combined, data_root)
    combined_path = data_root / f"linear_{classifier}_summary.csv"
    compact = compact_summary_frame(combined)
    compact_path = data_root / f"linear_{classifier}_compact_summary.csv"
    combined.to_csv(combined_path, index=False)
    compact.to_csv(compact_path, index=False)

    save_ratio_metric_plot(
        combined,
        "test_f1_macro",
        image_root / f"linear_{classifier}_ratio_f1_macro.png",
        title="Linear Probe F1-Macro vs Compression Ratio",
    )
    save_ratio_metric_plot(
        combined,
        "test_pr_auc_macro",
        image_root / f"linear_{classifier}_ratio_pr_auc_macro.png",
        title="Linear Probe PR-AUC vs Compression Ratio",
    )
    save_dual_metric_method_plot(
        combined,
        image_root / f"linear_{classifier}_ratio_f1_pr_auc.png",
        title="Linear Probe Macro-F1 and PR-AUC vs Compression Ratio",
    )
    for method in sensing_methods(combined):
        save_dual_metric_method_plot(
            combined,
            image_root / f"linear_{classifier}_{method}_ratio_f1_pr_auc.png",
            title=f"Linear Probe Metrics vs Compression Ratio - {format_model_name(method)}",
            method_filter=method,
        )
        save_ratio_metric_plot(
            combined,
            "test_f1_macro",
            image_root / f"linear_{classifier}_{method}_ratio_f1_macro.png",
            title=f"Linear Probe F1-Macro vs Compression Ratio - {format_model_name(method)}",
            method_filter=method,
        )
        save_ratio_metric_plot(
            combined,
            "test_pr_auc_macro",
            image_root / f"linear_{classifier}_{method}_ratio_pr_auc_macro.png",
            title=f"Linear Probe PR-AUC vs Compression Ratio - {format_model_name(method)}",
            method_filter=method,
        )

    if not topology_joined.empty:
        topology_metrics = ["persistence_image_h0", "persistence_image_h1", "betti_dist_h0", "betti_dist_h1", "wasserstein_h0", "wasserstein_h1"]
        for topology_metric in topology_metrics:
            if topology_metric not in topology_joined.columns:
                continue
            for performance_metric in ["test_f1_macro", "test_pr_auc_macro"]:
                save_topology_performance_scatter(
                    topology_joined,
                    topology_metric,
                    performance_metric,
                    image_root / f"linear_{classifier}_{topology_metric}_vs_{performance_metric}.png",
                    title=f"{format_model_name(performance_metric)} vs {format_model_name(topology_metric)}",
                )

    log("combined_summary")
    log(compact.to_string(index=False))
    log(f"saved combined_summary={combined_path}")
    log(f"saved compact_summary={compact_path}")

    report(f"DONE module=evaluation.linear classifier={classifier} runs={len(summary_frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
