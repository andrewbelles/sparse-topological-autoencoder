#!/usr/bin/env python3
#
# End-to-end construction and selection of the fixed ABT anchor manifold.
#

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from compression.common import SPLITS, embedding_columns
from compression.train_utils import load_config, merge_config, resolve_device, set_seed
from evaluation.transfer import load_or_compute_features
from representation import barlow, extract


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "representation_manifold.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG = {
    "seed": 17,
    "dataset": "fma_small_mel",
    "anchor_source": "anchor",
    "run_training": True,
    "run_extraction": True,
    "run_evaluation": True,
    "barlow": {},
    "extract": {},
    "selection": {
        "run_pattern": "barlow_d*_a*",
        "primary_metric": "validation_logistic_f1_macro",
        "secondary_metric": "validation_knn_f1_macro",
        "tertiary_metric": "validation_ph_f1_macro",
        "within_best_tolerance": 0.0,
    },
    "baseline": {
        "logistic_c": 1.0,
        "logistic_max_iter": 3000,
        "knn_neighbors": 15,
        "purity_neighbors": 30,
    },
    "ph": {
        "enabled": True,
        "k": 30,
        "features": ["betti", "entropy"],
        "homology_dims": [0, 1],
        "betti_grid_size": 64,
        "filtration_max": 3.0,
        "max_homology_dim": 1,
        "n_perm": 64,
        "cache_features": True,
    },
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, score, and select the ABT anchor manifold.")
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_MEL_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    return parser.parse_args()


def source_embedding_dim(source: str) -> int:
    match = re.search(r"d(?P<dim>\d+)", source)
    return int(match.group("dim")) if match else 10**9


def source_augmentation(source: str) -> str:
    match = re.search(r"_(a\d+)$", source)
    return match.group(1) if match else ""


def checkpoint_sources(checkpoints: list[Path]) -> list[str]:
    sources: list[str] = []
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sources.append(str(payload["source_name"]))
    return sorted(set(sources))


def discover_candidate_sources(representation_dir: Path, dataset: str, pattern: str) -> list[str]:
    suffix = f"_{dataset}_training.parquet"
    sources = []
    for path in sorted(representation_dir.glob(f"*{suffix}")):
        source = path.name[: -len(suffix)]
        if fnmatchcase(source, pattern):
            sources.append(source)
    return sources


def train_candidates(data_dir: Path, checkpoint_dir: Path, config: dict, device: torch.device) -> list[Path]:
    barlow_config = merge_config(barlow.DEFAULT_CONFIG, config["barlow"])
    set_seed(int(barlow_config["seed"]))
    written: list[Path] = []
    for embedding_dim in [int(value) for value in barlow_config["embedding_dims"]]:
        for policy in [str(value) for value in barlow_config["augmentations"]]:
            written.append(barlow.train_one(data_dir, checkpoint_dir, embedding_dim, policy, barlow_config, device))
    manifest_path = checkpoint_dir / f"barlow_{data_dir.name}_checkpoints.json"
    manifest_path.write_text(json.dumps({"dataset": data_dir.name, "checkpoints": [path.as_posix() for path in written]}, indent=2), encoding="utf-8")
    return written


def extract_candidates(data_dir: Path, representation_dir: Path, checkpoints: list[Path], config: dict) -> list[str]:
    extract_config = merge_config(extract.DEFAULT_CONFIG, config["extract"])
    extract_config["normalize_embeddings"] = True
    device = resolve_device(str(extract_config["device"]))
    sources: list[str] = []
    for checkpoint in checkpoints:
        encoder, payload = extract.load_encoder(checkpoint, device)
        sources.append(str(payload["source_name"]))
        for split in [str(value) for value in extract_config["splits"]]:
            output_path = extract.write_split_embeddings(encoder, payload, data_dir, split, extract_config, representation_dir, device)
            log(f"extracted source={payload['source_name']} split={split} path={output_path}")
    return sorted(set(sources))


def read_candidate(representation_dir: Path, source: str, dataset: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    columns: list[str] | None = None
    for split in SPLITS:
        path = representation_dir / f"{source}_{dataset}_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate parquet: {path}")
        frame = pd.read_parquet(path).copy()
        split_columns = embedding_columns(frame)
        columns = split_columns if columns is None else columns
        if split_columns != columns:
            raise ValueError(f"embedding columns differ for source={source} split={split}")
        frames[split] = frame
    return frames, columns or []


def feature_matrices(frames: dict[str, pd.DataFrame], columns: list[str]) -> dict[str, np.ndarray]:
    return {split: frames[split][columns].to_numpy(dtype=np.float32, copy=True) for split in SPLITS}


def encode_labels(frames: dict[str, pd.DataFrame]) -> tuple[LabelEncoder, dict[str, np.ndarray]]:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([frames[split]["genre_top"].astype(str) for split in SPLITS], ignore_index=True))
    return encoder, {split: encoder.transform(frames[split]["genre_top"].astype(str)) for split in SPLITS}


def pr_auc_macro(scores: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    binarized = label_binarize(labels, classes=np.arange(n_classes))
    present = np.any(binarized == 1, axis=0)
    if not np.any(present):
        return 0.0
    return float(average_precision_score(binarized[:, present], scores[:, present], average="macro"))


def evaluate_classifier(estimator, features: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    predictions = estimator.predict(features)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    else:
        scores = estimator.decision_function(features)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro")),
        "pr_auc_macro": pr_auc_macro(scores, labels, n_classes),
    }


def scaled_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(features["training"])
    return {split: scaler.transform(features[split]).astype(np.float32, copy=False) for split in SPLITS}


def coordinate_probe_metrics(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], config: dict) -> dict[str, float]:
    scaled = scaled_features(features)
    n_classes = len(np.unique(np.concatenate([labels[split] for split in SPLITS])))
    logistic = LogisticRegression(
        C=float(config["baseline"]["logistic_c"]),
        max_iter=int(config["baseline"]["logistic_max_iter"]),
        random_state=int(config["seed"]),
    )
    logistic.fit(scaled["training"], labels["training"])
    knn = KNeighborsClassifier(n_neighbors=int(config["baseline"]["knn_neighbors"]))
    knn.fit(scaled["training"], labels["training"])

    metrics: dict[str, float] = {}
    for split in ["validation", "test"]:
        logistic_metrics = evaluate_classifier(logistic, scaled[split], labels[split], n_classes)
        knn_metrics = evaluate_classifier(knn, scaled[split], labels[split], n_classes)
        metrics[f"{split}_logistic_f1_macro"] = logistic_metrics["f1_macro"]
        metrics[f"{split}_logistic_pr_auc_macro"] = logistic_metrics["pr_auc_macro"]
        metrics[f"{split}_logistic_accuracy"] = logistic_metrics["accuracy"]
        metrics[f"{split}_knn_f1_macro"] = knn_metrics["f1_macro"]
        metrics[f"{split}_knn_pr_auc_macro"] = knn_metrics["pr_auc_macro"]
        metrics[f"{split}_knn_accuracy"] = knn_metrics["accuracy"]
    return metrics


def neighborhood_purity(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], config: dict) -> dict[str, float]:
    scaled = scaled_features(features)
    k = int(config["baseline"]["purity_neighbors"])
    neighbors = NearestNeighbors(n_neighbors=min(k, len(scaled["training"])), metric="euclidean")
    neighbors.fit(scaled["training"])
    metrics: dict[str, float] = {}
    for split in ["validation", "test"]:
        indices = neighbors.kneighbors(scaled[split], return_distance=False)
        neighbor_labels = labels["training"][indices]
        metrics[f"{split}_neighbor_purity"] = float(np.mean(neighbor_labels == labels[split].reshape(-1, 1)))
    return metrics


def ph_probe_metrics(
    source: str,
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    labels: dict[str, np.ndarray],
    data_root: Path,
    config: dict,
) -> dict[str, float]:
    if not bool(config["ph"]["enabled"]):
        return {
            "validation_ph_f1_macro": np.nan,
            "validation_ph_pr_auc_macro": np.nan,
            "test_ph_f1_macro": np.nan,
            "test_ph_pr_auc_macro": np.nan,
        }
    ph_config = dict(config["ph"])
    ph_config["seed"] = int(config["seed"])
    run = {
        "run_name": f"{source}_manifold_ph",
        "frames": frames,
        "columns": columns,
    }
    ph_features: dict[str, np.ndarray] = {}
    for split in SPLITS:
        features, _, _ = load_or_compute_features(run, split, int(ph_config["k"]), ph_config, data_root / "manifold_ph_features")
        ph_features[split] = features
    scaled = scaled_features(ph_features)
    n_classes = len(np.unique(np.concatenate([labels[split] for split in SPLITS])))
    logistic = LogisticRegression(
        C=float(config["baseline"]["logistic_c"]),
        max_iter=int(config["baseline"]["logistic_max_iter"]),
        random_state=int(config["seed"]),
    )
    logistic.fit(scaled["training"], labels["training"])
    metrics: dict[str, float] = {}
    for split in ["validation", "test"]:
        split_metrics = evaluate_classifier(logistic, scaled[split], labels[split], n_classes)
        metrics[f"{split}_ph_f1_macro"] = split_metrics["f1_macro"]
        metrics[f"{split}_ph_pr_auc_macro"] = split_metrics["pr_auc_macro"]
        metrics[f"{split}_ph_accuracy"] = split_metrics["accuracy"]
    return metrics


def evaluate_source(source: str, representation_dir: Path, data_root: Path, config: dict) -> dict[str, object]:
    frames, columns = read_candidate(representation_dir, source, str(config["dataset"]))
    features = feature_matrices(frames, columns)
    _, labels = encode_labels(frames)
    record: dict[str, object] = {
        "source": source,
        "embedding_dim": int(len(columns)),
        "augmentation": source_augmentation(source),
    }
    record.update(coordinate_probe_metrics(features, labels, config))
    record.update(neighborhood_purity(features, labels, config))
    record.update(ph_probe_metrics(source, frames, columns, labels, data_root, config))
    return record


def select_source(summary: pd.DataFrame, config: dict) -> pd.Series:
    selection = config["selection"]
    primary = str(selection["primary_metric"])
    secondary = str(selection["secondary_metric"])
    tertiary = str(selection["tertiary_metric"])
    best = float(pd.to_numeric(summary[primary], errors="coerce").max())
    tolerance = float(selection["within_best_tolerance"])
    candidates = summary[pd.to_numeric(summary[primary], errors="coerce") >= best - tolerance].copy()
    candidates["_embedding_dim"] = candidates["source"].map(source_embedding_dim)
    return candidates.sort_values(
        [primary, secondary, tertiary, "_embedding_dim"],
        ascending=[False, False, False, True],
    ).iloc[0]


def materialize_anchor(selected: pd.Series, representation_dir: Path, config: dict, summary: pd.DataFrame) -> list[Path]:
    source = str(selected["source"])
    dataset = str(config["dataset"])
    anchor_source = str(config["anchor_source"])
    written: list[Path] = []
    for split in SPLITS:
        source_path = representation_dir / f"{source}_{dataset}_{split}.parquet"
        anchor_path = representation_dir / f"{anchor_source}_{dataset}_{split}.parquet"
        shutil.copyfile(source_path, anchor_path)
        written.append(anchor_path)

    def json_safe(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if pd.isna(value):
            return None
        return value

    metadata = {
        "anchor_source": anchor_source,
        "selected_source": source,
        "dataset": dataset,
        "selection": dict(config["selection"]),
        "supcon": dict(merge_config(barlow.DEFAULT_CONFIG, config["barlow"])["supcon"]),
        "selected_metrics": {
            key: json_safe(value)
            for key, value in selected.drop(labels=[label for label in ["_embedding_dim"] if label in selected.index]).to_dict().items()
        },
        "summary_path": (representation_dir / "manifold_selection_summary.csv").as_posix(),
        "candidate_count": int(len(summary)),
        "anchor_files": [path.as_posix() for path in written],
    }
    metadata_path = representation_dir / f"{anchor_source}_{dataset}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written.append(metadata_path)
    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    set_seed(int(config["seed"]))
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    representation_dir = args.output_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent.parent / "evaluation" / "data"
    representation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(merge_config(barlow.DEFAULT_CONFIG, config["barlow"])["device"]))
    report(f"START module=representation.manifold data_dir={data_dir} device={device} config={args.config}")

    checkpoint_paths: list[Path] = []
    if bool(config["run_training"]):
        checkpoint_paths = train_candidates(data_dir, checkpoint_dir, config, device)
    else:
        extract_config = merge_config(extract.DEFAULT_CONFIG, config["extract"])
        checkpoint_paths = extract.discover_checkpoints(extract_config)

    sources: list[str] = []
    if bool(config["run_extraction"]):
        sources = extract_candidates(data_dir, representation_dir, checkpoint_paths, config)
    elif checkpoint_paths:
        sources = checkpoint_sources(checkpoint_paths)

    pattern = str(config["selection"]["run_pattern"])
    discovered = discover_candidate_sources(representation_dir, str(config["dataset"]), pattern)
    sources = sorted(set(sources).union(discovered))
    if not sources:
        raise FileNotFoundError(f"no representation candidates matched pattern={pattern}")

    records: list[dict[str, object]] = []
    if bool(config["run_evaluation"]):
        for source in sources:
            record = evaluate_source(source, representation_dir, data_root, config)
            records.append(record)
            report(
                f"candidate={source} val_f1={record['validation_logistic_f1_macro']:.4f} "
                f"val_knn={record['validation_knn_f1_macro']:.4f} val_ph={record['validation_ph_f1_macro']:.4f}"
            )
        summary = pd.DataFrame.from_records(records)
        summary_path = representation_dir / "manifold_selection_summary.csv"
        summary.to_csv(summary_path, index=False)
    else:
        summary_path = representation_dir / "manifold_selection_summary.csv"
        summary = pd.read_csv(summary_path)

    selected = select_source(summary, config)
    written = materialize_anchor(selected, representation_dir, config, summary)
    report(
        f"selected={selected['source']} validation_logistic_f1_macro={selected['validation_logistic_f1_macro']:.4f} "
        f"test_logistic_f1_macro={selected['test_logistic_f1_macro']:.4f}"
    )
    report(f"DONE module=representation.manifold anchor_source={config['anchor_source']} files={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
