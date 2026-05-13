#!/usr/bin/env python3
#
# SAE-specific diagnostics for sparse dictionary codes over the ABT manifold.
#

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-evaluation-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ripser import ripser

from compression.common import SPLITS, active_feature_columns, embedding_columns, load_anchor_splits
from compression.train_utils import load_config, resolve_device
from evaluation.topology import betti_distances, compute_diagrams, diagram_wasserstein, persistence_image_distances
from evaluation.visualizations import clean_label


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sparse_dictionary_eval.yaml"
DEFAULT_DATA_DIR = Path("compression/data")
DEFAULT_CONFIG = {
    "dataset": "fma_small_mel",
    "anchor_dir": "representation/data",
    "reference_source": "anchor",
    "methods": ["sae", "topo_sae"],
    "seed": 7,
    "tracks_per_genre": 150,
    "bootstrap_replicates": 20,
    "max_homology_dim": 1,
    "n_perm": 64,
    "betti_grid_size": 64,
    "persistence_image_resolution": 24,
    "persistence_image_sigma": 0.05,
    "persistence_image_weight_power": 1.0,
    "persistence_image_max_birth": 2.5,
    "persistence_image_max_persistence": 2.5,
    "active_threshold": 1e-3,
    "linear_classifier": "logistic",
    "device": "auto",
    "torch_batch_size": 8192,
    "gpu_pairwise_distances": True,
    "support_plot": {
        "enabled": True,
        "split": "training",
        "normalize_by_genre": True,
    },
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sparse dictionary SAE diagnostics.")
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def read_sparse_parquets(data_dir: Path, config: dict) -> pd.DataFrame:
    methods = {str(value) for value in config["methods"]}
    frames: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if not {"method", "code_dim", "target_active", "split", "track_id", "genre_top"}.issubset(frame.columns):
            continue
        frame = frame[frame["method"].astype(str).isin(methods)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no sparse dictionary parquets matched methods={sorted(methods)} in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def reconstruction_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("recon_"))
    if not columns:
        raise ValueError("sparse dictionary parquet has no recon_* columns")
    return columns


def group_columns(frame: pd.DataFrame) -> list[str]:
    columns = ["method", "code_dim", "target_active", "seed"]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"sparse dictionary frame missing columns: {missing}")
    return columns


@torch.no_grad()
def tensor_reconstruction_stats(
    z: np.ndarray,
    z_hat: np.ndarray,
    codes: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    batch_size = max(1, int(config.get("torch_batch_size", 8192)))
    active_counts: list[torch.Tensor] = []
    recon_sums: list[torch.Tensor] = []
    for start in range(0, z.shape[0], batch_size):
        stop = start + batch_size
        z_batch = torch.as_tensor(z[start:stop], dtype=torch.float32, device=device)
        z_hat_batch = torch.as_tensor(z_hat[start:stop], dtype=torch.float32, device=device)
        code_batch = torch.as_tensor(codes[start:stop], dtype=torch.float32, device=device)
        active_counts.append(torch.count_nonzero(code_batch.abs() > float(config["active_threshold"]), dim=1).float())
        recon_sums.append(torch.sum((z_batch - z_hat_batch) ** 2, dim=1))
    active = torch.cat(active_counts) if active_counts else torch.empty(0, device=device)
    recon = torch.cat(recon_sums) if recon_sums else torch.empty(0, device=device)
    return {
        "avg_active": float(active.mean().detach().cpu()) if active.numel() else 0.0,
        "med_active": float(active.median().detach().cpu()) if active.numel() else 0.0,
        "recon_mse": float(recon.mean().detach().cpu()) if recon.numel() else 0.0,
    }


def reconstruction_and_sparsity_summary(
    frame: pd.DataFrame,
    anchors: dict[str, pd.DataFrame],
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, group in frame.groupby([*group_columns(frame), "split"], dropna=False):
        method, code_dim, target_active, seed, split = keys
        split = str(split)
        anchor = anchors[split]
        anchor_cols = embedding_columns(anchor)
        recon_cols = reconstruction_columns(group)
        active_cols = active_feature_columns(group, int(code_dim))
        anchor_feature_cols = [f"z_{column}" for column in anchor_cols]
        active_feature_cols = [f"a_{column}" for column in active_cols]
        anchor_part = anchor[["track_id", *anchor_cols]].rename(columns=dict(zip(anchor_cols, anchor_feature_cols, strict=True)))
        group_part = group[["track_id", *active_cols, *recon_cols]].rename(columns=dict(zip(active_cols, active_feature_cols, strict=True)))
        merged = anchor_part.merge(
            group_part,
            on="track_id",
            how="inner",
        )
        if merged.empty:
            continue
        z = merged[anchor_feature_cols].to_numpy(dtype=np.float32, copy=True)
        z_hat = merged[recon_cols].to_numpy(dtype=np.float32, copy=True)
        codes = merged[active_feature_cols].to_numpy(dtype=np.float32, copy=True)
        stats = tensor_reconstruction_stats(z, z_hat, codes, config, device)
        records.append(
            {
                "method": str(method),
                "code_dim": int(code_dim),
                "target_active": int(target_active),
                "seed": int(seed),
                "split": split,
                "avg_active": stats["avg_active"],
                "med_active": stats["med_active"],
                "recon_mse": stats["recon_mse"],
                "n": int(len(merged)),
            }
        )
    return pd.DataFrame.from_records(records)


@torch.no_grad()
def pairwise_distances_device(points: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
    distances = torch.cdist(tensor, tensor, p=2)
    upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    positive = upper[upper > 0]
    median = positive.median() if positive.numel() else torch.tensor(1.0, device=device)
    distances = distances / torch.clamp(median, min=1e-8)
    return distances.detach().cpu().numpy().astype("float64", copy=False)


def compute_diagrams_device(points: np.ndarray, config: dict, device: torch.device) -> list[np.ndarray]:
    if device.type != "cuda" or not bool(config.get("gpu_pairwise_distances", True)):
        return compute_diagrams(points, config)
    distances = pairwise_distances_device(points, device)
    n_perm = min(int(config["n_perm"]), points.shape[0])
    kwargs: dict[str, object] = {
        "maxdim": int(config.get("max_homology_dim", 1)),
        "distance_matrix": True,
    }
    if n_perm < points.shape[0]:
        kwargs["n_perm"] = n_perm
    return [np.asarray(diagram, dtype="float64") for diagram in ripser(distances, **kwargs)["dgms"]]


def score_topology_group(
    method: str,
    code_dim: int,
    target_active: int,
    seed: int,
    group: pd.DataFrame,
    anchor: pd.DataFrame,
    config: dict,
    device: torch.device,
) -> list[dict[str, object]]:
    active_cols = active_feature_columns(group, int(code_dim))
    recon_cols = reconstruction_columns(group)
    anchor_cols = embedding_columns(anchor)
    anchor_feature_cols = [f"z_{column}" for column in anchor_cols]
    active_feature_cols = [f"a_{column}" for column in active_cols]
    anchor_part = anchor[["track_id", "genre_top", *anchor_cols]].rename(
        columns=dict(zip(anchor_cols, anchor_feature_cols, strict=True))
    )
    group_part = group[["track_id", *active_cols, *recon_cols]].rename(
        columns=dict(zip(active_cols, active_feature_cols, strict=True))
    )
    merged = anchor_part.merge(
        group_part,
        on="track_id",
        how="inner",
    )
    rng = np.random.default_rng(int(config["seed"]) + int(seed) * 10_000 + int(code_dim) + int(target_active))
    records: list[dict[str, object]] = []

    for genre, genre_frame in sorted(merged.groupby("genre_top")):
        sample_size = min(int(config["tracks_per_genre"]), len(genre_frame))
        if sample_size < 3:
            continue
        for bootstrap in range(int(config["bootstrap_replicates"])):
            indices = rng.choice(len(genre_frame), size=sample_size, replace=len(genre_frame) < sample_size)
            sample = genre_frame.iloc[np.sort(indices)]
            z = sample[anchor_feature_cols].to_numpy(dtype=np.float32, copy=True)
            codes = sample[active_feature_cols].to_numpy(dtype=np.float32, copy=True)
            z_hat = sample[recon_cols].to_numpy(dtype=np.float32, copy=True)
            z_diagrams = compute_diagrams_device(z, config, device)
            code_diagrams = compute_diagrams_device(codes, config, device)
            recon_diagrams = compute_diagrams_device(z_hat, config, device)
            code_betti = betti_distances(z_diagrams, code_diagrams, int(config["betti_grid_size"]))
            recon_betti = betti_distances(z_diagrams, recon_diagrams, int(config["betti_grid_size"]))
            code_wass = diagram_wasserstein(z_diagrams, code_diagrams)
            recon_wass = diagram_wasserstein(z_diagrams, recon_diagrams)
            code_pi = persistence_image_distances(z_diagrams, code_diagrams, config)
            recon_pi = persistence_image_distances(z_diagrams, recon_diagrams, config)
            records.append(
                {
                    "method": method,
                    "code_dim": int(code_dim),
                    "target_active": int(target_active),
                    "seed": int(seed),
                    "genre_top": str(genre),
                    "bootstrap": int(bootstrap),
                    "track_count": int(sample_size),
                    "betti_Z_A_h0": code_betti.get(0, 0.0),
                    "betti_Z_A_h1": code_betti.get(1, 0.0),
                    "betti_Z_Zhat_h0": recon_betti.get(0, 0.0),
                    "betti_Z_Zhat_h1": recon_betti.get(1, 0.0),
                    "wasserstein_Z_A_h0": code_wass.get(0, 0.0),
                    "wasserstein_Z_A_h1": code_wass.get(1, 0.0),
                    "wasserstein_Z_Zhat_h0": recon_wass.get(0, 0.0),
                    "wasserstein_Z_Zhat_h1": recon_wass.get(1, 0.0),
                    "persistence_image_Z_A_h0": code_pi.get(0, 0.0),
                    "persistence_image_Z_A_h1": code_pi.get(1, 0.0),
                    "persistence_image_Z_Zhat_h0": recon_pi.get(0, 0.0),
                    "persistence_image_Z_Zhat_h1": recon_pi.get(1, 0.0),
                }
            )
    return records


def topology_summary(
    frame: pd.DataFrame,
    anchor_training: pd.DataFrame,
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    training = frame[frame["split"] == "training"].copy()
    for keys, group in training.groupby(group_columns(training), dropna=False):
        method, code_dim, target_active, seed = keys
        records.extend(
            score_topology_group(
                str(method),
                int(code_dim),
                int(target_active),
                int(seed),
                group,
                anchor_training,
                config,
                device,
            )
        )
        report(f"topology method={method} K={int(code_dim)} s={int(target_active)} seed={int(seed)}")
    detail = pd.DataFrame.from_records(records)
    if detail.empty:
        return detail, detail
    summary = (
        detail.groupby(["method", "code_dim", "target_active", "seed", "genre_top"], as_index=False)
        .agg(
            betti_Z_A_h0=("betti_Z_A_h0", "mean"),
            betti_Z_A_h1=("betti_Z_A_h1", "mean"),
            betti_Z_Zhat_h0=("betti_Z_Zhat_h0", "mean"),
            betti_Z_Zhat_h1=("betti_Z_Zhat_h1", "mean"),
            wasserstein_Z_A_h0=("wasserstein_Z_A_h0", "mean"),
            wasserstein_Z_A_h1=("wasserstein_Z_A_h1", "mean"),
            wasserstein_Z_Zhat_h0=("wasserstein_Z_Zhat_h0", "mean"),
            wasserstein_Z_Zhat_h1=("wasserstein_Z_Zhat_h1", "mean"),
            persistence_image_Z_A_h0=("persistence_image_Z_A_h0", "mean"),
            persistence_image_Z_A_h1=("persistence_image_Z_A_h1", "mean"),
            persistence_image_Z_Zhat_h0=("persistence_image_Z_Zhat_h0", "mean"),
            persistence_image_Z_Zhat_h1=("persistence_image_Z_Zhat_h1", "mean"),
            bootstrap_replicates=("bootstrap", "count"),
            track_count=("track_count", "mean"),
        )
    )
    return detail, summary


def join_linear_metrics(summary: pd.DataFrame, classifier: str) -> pd.DataFrame:
    linear_path = Path(__file__).resolve().parent / "data" / f"linear_{classifier}_summary.csv"
    if not linear_path.is_file():
        summary["f1"] = np.nan
        summary["pr_auc"] = np.nan
        return summary
    linear = pd.read_csv(linear_path)
    required = {"method", "code_dim", "target_active", "seed", "test_f1_macro", "test_pr_auc_macro"}
    if not required.issubset(linear.columns):
        summary["f1"] = np.nan
        summary["pr_auc"] = np.nan
        return summary
    linear = linear[list(required)].rename(
        columns={"test_f1_macro": "f1", "test_pr_auc_macro": "pr_auc"}
    )
    for column in ["code_dim", "target_active", "seed"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").astype("Int64")
        linear[column] = pd.to_numeric(linear[column], errors="coerce").astype("Int64")
    return summary.merge(linear, on=["method", "code_dim", "target_active", "seed"], how="left")


def build_compact_summary(recon: pd.DataFrame, topology: pd.DataFrame, config: dict) -> pd.DataFrame:
    recon_test = recon[recon["split"] == "test"].copy()
    if topology.empty:
        topo_grouped = pd.DataFrame(columns=["method", "code_dim", "target_active", "seed"])
    else:
        topo_grouped = (
            topology.groupby(["method", "code_dim", "target_active", "seed"], as_index=False)
            .agg(
                betti_Z_A=("betti_Z_A_h0", "mean"),
                betti_Z_A_h1=("betti_Z_A_h1", "mean"),
                betti_Z_Zhat=("betti_Z_Zhat_h0", "mean"),
                betti_Z_Zhat_h1=("betti_Z_Zhat_h1", "mean"),
                pi_Z_A=("persistence_image_Z_A_h0", "mean"),
                pi_Z_A_h1=("persistence_image_Z_A_h1", "mean"),
                pi_Z_Zhat=("persistence_image_Z_Zhat_h0", "mean"),
                pi_Z_Zhat_h1=("persistence_image_Z_Zhat_h1", "mean"),
            )
        )
        topo_grouped["betti_Z_A"] = (topo_grouped["betti_Z_A"] + topo_grouped["betti_Z_A_h1"]) / 2.0
        topo_grouped["betti_Z_Zhat"] = (topo_grouped["betti_Z_Zhat"] + topo_grouped["betti_Z_Zhat_h1"]) / 2.0
        topo_grouped["pi_Z_A"] = (topo_grouped["pi_Z_A"] + topo_grouped["pi_Z_A_h1"]) / 2.0
        topo_grouped["pi_Z_Zhat"] = (topo_grouped["pi_Z_Zhat"] + topo_grouped["pi_Z_Zhat_h1"]) / 2.0
        topo_grouped = topo_grouped.drop(columns=["betti_Z_A_h1", "betti_Z_Zhat_h1", "pi_Z_A_h1", "pi_Z_Zhat_h1"])
    compact = recon_test.merge(topo_grouped, on=["method", "code_dim", "target_active", "seed"], how="left")
    compact = join_linear_metrics(compact, str(config["linear_classifier"]))
    compact = compact.rename(columns={"code_dim": "K", "target_active": "target_s"})
    columns = [
        "method",
        "K",
        "target_s",
        "avg_active",
        "med_active",
        "recon_mse",
        "f1",
        "pr_auc",
        "pi_Z_A",
        "pi_Z_Zhat",
        "betti_Z_A",
        "betti_Z_Zhat",
    ]
    for column in columns:
        if column not in compact.columns:
            compact[column] = np.nan
    compact = compact[columns].sort_values(["method", "K", "target_s"])
    for column in columns:
        if column not in {"method"}:
            compact[column] = pd.to_numeric(compact[column], errors="coerce")
    return compact


def save_metric_plot(frame: pd.DataFrame, metric: str, output_path: Path, title: str, ylabel: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_frame = frame.copy()
    if not plot_frame.empty:
        sns.lineplot(
            data=plot_frame,
            x="target_s",
            y=metric,
            hue="method",
            style="K",
            markers=True,
            dashes=False,
            ax=ax,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Target Active Coefficients")
    ax.set_ylabel(ylabel or clean_label(metric))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_scatter(frame: pd.DataFrame, x_metric: str, y_metric: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    plot_frame = frame.dropna(subset=[x_metric, y_metric]).copy()
    if not plot_frame.empty:
        sns.scatterplot(
            data=plot_frame,
            x=x_metric,
            y=y_metric,
            hue="method",
            style="K",
            size="target_s",
            sizes=(45, 150),
            alpha=0.85,
            ax=ax,
        )
    ax.set_xlabel(clean_label(x_metric))
    ax.set_ylabel(clean_label(y_metric))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_topology_panel(topology: pd.DataFrame, image_root: Path) -> None:
    if topology.empty:
        return
    output_path = image_root / "sparse_dictionary_topology_panel.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plot_rows: list[pd.DataFrame] = []
    metric_prefix = "persistence_image" if "persistence_image_Z_A_h0" in topology.columns else "betti"
    if metric_prefix == "persistence_image":
        specs = [
            ("persistence_image_Z_A", "Z vs Sparse Codes", "Mean Persistence-Image Distance"),
            ("persistence_image_Z_Zhat", "Z vs Reconstruction", "Mean Persistence-Image Distance"),
        ]
    else:
        specs = [
            ("betti_Z_A", "Z vs Sparse Codes", "Mean Betti Distortion"),
            ("betti_Z_Zhat", "Z vs Reconstruction", "Mean Betti Distortion"),
        ]
    y_label = specs[0][2]
    for prefix, label, _ in specs:
        columns = [f"{prefix}_h0", f"{prefix}_h1"]
        if not set(columns).issubset(topology.columns):
            continue
        frame = topology.copy()
        frame["metric"] = label
        frame["value"] = (pd.to_numeric(frame[columns[0]], errors="coerce") + pd.to_numeric(frame[columns[1]], errors="coerce")) / 2.0
        plot_rows.append(frame)
    if not plot_rows:
        return
    plot_frame = pd.concat(plot_rows, ignore_index=True)
    grouped = (
        plot_frame.groupby(["metric", "method", "code_dim", "target_active"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))

    metrics = ["Z vs Sparse Codes", "Z vs Reconstruction"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False, constrained_layout=True)
    palette = {"sae": "#4c78a8", "topo_sae": "#f58518"}
    for ax, metric in zip(axes, metrics, strict=True):
        panel = grouped[grouped["metric"] == metric].copy()
        for (method, code_dim), group in panel.groupby(["method", "code_dim"], dropna=False):
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
                label=f"{clean_label(str(method))} K={int(code_dim)}",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Target Active Coefficients")
        ax.set_ylabel(y_label)
        ax.set_title(metric)
        ax.grid(True, which="both", alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axes[-1].legend(unique.values(), unique.keys(), title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("SAE Topology Diagnostics", fontsize=15)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def support_counts_by_genre(
    group: pd.DataFrame,
    active_cols: list[str],
    active_threshold: float,
    device: torch.device,
) -> list[pd.DataFrame]:
    codes = group[active_cols].to_numpy(dtype=np.float32, copy=True)
    support = torch.as_tensor(codes, dtype=torch.float32, device=device).abs() > active_threshold
    atom_indices = np.arange(len(active_cols), dtype=int)
    rows: list[pd.DataFrame] = []
    group = group.reset_index(drop=True)
    for genre, genre_indices in group.groupby("genre_top").groups.items():
        indices = torch.as_tensor(np.asarray(list(genre_indices), dtype=np.int64), device=device)
        counts = support.index_select(0, indices).sum(dim=0).detach().cpu().numpy().astype(np.float64)
        denominator = max(1, int(indices.numel()))
        rows.append(
            pd.DataFrame(
                {
                    "genre_top": str(genre),
                    "atom": atom_indices,
                    "activation_count": counts,
                    "activation_frequency": counts / denominator,
                    "genre_count": int(denominator),
                }
            )
        )
    return rows


def support_frequency_summary(frame: pd.DataFrame, config: dict, device: torch.device) -> pd.DataFrame:
    support_config = config.get("support_plot", {})
    split = str(support_config.get("split", "training"))
    active_threshold = float(config["active_threshold"])
    rows: list[pd.DataFrame] = []
    split_frame = frame[frame["split"] == split].copy()
    if split_frame.empty:
        return pd.DataFrame()

    for keys, group in split_frame.groupby(group_columns(split_frame), dropna=False):
        method, code_dim, target_active, seed = keys
        active_cols = active_feature_columns(group, int(code_dim))
        for counts in support_counts_by_genre(group, active_cols, active_threshold, device):
            rows.append(
                counts.assign(
                    method=str(method),
                    code_dim=int(code_dim),
                    target_active=int(target_active),
                    seed=int(seed),
                    split=split,
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_support_frequency_plots(summary: pd.DataFrame, image_root: Path, config: dict) -> None:
    support_config = config.get("support_plot", {})
    if not bool(support_config.get("enabled", True)) or summary.empty:
        return
    image_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    value_column = "activation_frequency" if bool(support_config.get("normalize_by_genre", True)) else "activation_count"
    value_label = "Within-Genre Activation Frequency" if value_column == "activation_frequency" else "Activation Count"

    for keys, group in summary.groupby(["method", "code_dim", "target_active", "seed"], dropna=False):
        method, code_dim, target_active, seed = keys
        atoms = list(range(int(code_dim)))
        plot_frame = group.copy()
        plot_frame["atom"] = pd.Categorical(plot_frame["atom"].astype(int), categories=atoms, ordered=True)
        matrix = plot_frame.pivot_table(
            index="genre_top",
            columns="atom",
            values=value_column,
            aggfunc="sum",
            observed=False,
        ).reindex(columns=atoms).fillna(0.0)
        genres = sorted(matrix.index.tolist())
        matrix = matrix.reindex(index=genres)
        values = matrix.to_numpy(dtype=np.float64)
        global_support = values.mean(axis=0)
        total_support = values.sum(axis=0)
        specificity = np.divide(values.max(axis=0), np.maximum(total_support, 1e-12))

        fig_width = max(12.0, min(28.0, 0.035 * int(code_dim)))
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(fig_width, 8.8),
            gridspec_kw={"height_ratios": [1.0, 4.2, 1.0]},
            sharex=True,
            constrained_layout=True,
        )
        tick_step = max(1, int(code_dim) // 16)
        tick_positions = np.arange(0, int(code_dim), tick_step)

        axes[0].bar(np.arange(int(code_dim)), global_support, width=1.0, color="#525252", linewidth=0)
        axes[0].set_ylabel("Global")
        axes[0].set_title(f"Support by Genre - {clean_label(str(method))} K={int(code_dim)} s={int(target_active)}")
        axes[0].grid(True, axis="y", alpha=0.25)

        vmax = float(np.nanmax(values)) if values.size else 1.0
        sns.heatmap(
            matrix,
            ax=axes[1],
            cmap="mako",
            vmin=0.0,
            vmax=max(vmax, 1e-8),
            cbar_kws={"label": value_label},
            xticklabels=False,
            yticklabels=True,
        )
        axes[1].set_ylabel("Genre")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="y", rotation=0)

        axes[2].plot(np.arange(int(code_dim)), specificity, color="#c2410c", linewidth=1.2)
        axes[2].fill_between(np.arange(int(code_dim)), specificity, color="#fed7aa", alpha=0.7)
        axes[2].set_ylabel("Specificity")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].grid(True, axis="y", alpha=0.25)
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels([str(int(position)) for position in tick_positions], rotation=0, fontsize=8)
        axes[2].set_xlim(-0.5, int(code_dim) - 0.5)
        axes[2].set_xlabel("Dictionary Atom Index")
        output_path = (
            image_root
            / f"sparse_dictionary_support_by_genre_{method}_K{int(code_dim)}_s{int(target_active)}_seed{int(seed)}.png"
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def save_plots(compact: pd.DataFrame, topology: pd.DataFrame, image_root: Path) -> None:
    save_metric_plot(compact, "recon_mse", image_root / "sparse_dictionary_reconstruction.png", "SAE Reconstruction Error")
    save_metric_plot(compact, "avg_active", image_root / "sparse_dictionary_avg_active.png", "SAE Average Active Coefficients")
    save_metric_plot(compact, "med_active", image_root / "sparse_dictionary_median_active.png", "SAE Median Active Coefficients")
    save_topology_panel(topology, image_root)
    for topology_metric in ["pi_Z_A", "pi_Z_Zhat", "betti_Z_A", "betti_Z_Zhat", "recon_mse"]:
        if topology_metric not in compact.columns:
            continue
        for performance_metric in ["f1", "pr_auc"]:
            save_scatter(
                compact,
                topology_metric,
                performance_metric,
                image_root / f"sparse_dictionary_{topology_metric}_vs_{performance_metric}.png",
                f"{clean_label(performance_metric)} vs {clean_label(topology_metric)}",
            )


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(str(config.get("device", "auto")))
    data_dir = args.data_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    ph_backend = "ripser_cpu"
    pairwise_backend = str(device) if device.type == "cuda" and bool(config.get("gpu_pairwise_distances", True)) else "cpu"
    report(
        f"START module=evaluation.sparse_dictionary data_dir={data_dir} device={device} "
        f"pairwise={pairwise_backend} ph={ph_backend} config={args.config}"
    )
    frame = read_sparse_parquets(data_dir, config)
    anchors = load_anchor_splits(Path(str(config["anchor_dir"])).expanduser(), str(config["reference_source"]), str(config["dataset"]))
    recon_summary = reconstruction_and_sparsity_summary(frame, anchors, config, device)
    topo_detail, topo_summary = topology_summary(frame, anchors["training"], config, device)
    compact = build_compact_summary(recon_summary, topo_summary, config)
    support_summary = support_frequency_summary(frame, config, device)

    recon_summary.to_csv(data_root / "sparse_dictionary_reconstruction_summary.csv", index=False)
    topo_detail.to_csv(data_root / "sparse_dictionary_topology_bootstrap.csv", index=False)
    topo_summary.to_csv(data_root / "sparse_dictionary_topology_summary.csv", index=False)
    compact.to_csv(data_root / "sparse_dictionary_compact_summary.csv", index=False)
    support_summary.to_csv(data_root / "sparse_dictionary_support_summary.csv", index=False)
    save_plots(compact, topo_summary, image_root)
    save_support_frequency_plots(support_summary, image_root, config)

    log("sparse_dictionary_summary")
    log(compact.round(4).to_string(index=False))
    report(f"DONE module=evaluation.sparse_dictionary rows={len(compact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
