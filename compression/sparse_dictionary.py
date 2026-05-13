#!/usr/bin/env python3
#
# Sparse dictionary autoencoders over the selected ABT anchor manifold.
#

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_topological.nn import VietorisRipsComplex

from compression.common import SPLITS, embedding_columns, load_anchor_splits, metadata_frame, method_parquet_path
from compression.train_utils import load_config, resolve_device, set_seed


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sparse_dictionary.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
DEFAULT_OUTPUT_DIR = Path("compression/data")
DEFAULT_CONFIG = {
    "source": "anchor",
    "dataset": "fma_small_mel",
    "methods": ["sae", "topo_sae"],
    "code_dims": [128, 256, 512, 1024],
    "target_active": [2, 4, 8, 16, 32],
    "seeds": [0],
    "device": "auto",
    "epochs": 300,
    "batch_size": 512,
    "learning_rate": 3e-3,
    "min_learning_rate": 1e-5,
    "weight_decay": 1e-6,
    "active_threshold": 1e-3,
    "topology_weight": 0.005,
    "topology_metric": "persistence_image",
    "topology_target": "raw_anchor",
    "topology_dims": [0, 1],
    "topology_grid_size": 64,
    "topology_temperature": 0.05,
    "topology_batch_size": 96,
    "topology_global_batch_size": 384,
    "topology_eval_batches": 1,
    "topology_scale_mode": "reference_median",
    "topology_scale_penalty_weight": 0.01,
    "persistence_image_resolution": 24,
    "persistence_image_sigma": 0.05,
    "persistence_image_weight_power": 1.0,
    "persistence_image_max_birth": 2.5,
    "persistence_image_max_persistence": 2.5,
    "supcon": {
        "enabled": True,
        "weight": 2.5,
        "temperature": 0.07,
    },
    "early_stopping": {
        "enabled": True,
        "gl_threshold": 2.0,
        "min_epochs": 10,
        "patience": 5,
    },
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sparse dictionary autoencoders over ABT anchor embeddings.")
    parser.add_argument("-a", "--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, code_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, code_dim)
        self.decoder = nn.Linear(code_dim, input_dim, bias=False)
        self.b0 = nn.Parameter(torch.zeros(input_dim))
        self.normalize_decoder_atoms()

    @torch.no_grad()
    def normalize_decoder_atoms(self) -> None:
        self.decoder.weight.div_(torch.clamp(self.decoder.weight.norm(dim=0, keepdim=True), min=1e-8))

    def forward(self, features: torch.Tensor, target_active: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features - self.b0)
        activations = F.relu(hidden)
        active = min(int(target_active), activations.shape[1])
        if active < activations.shape[1]:
            values, indices = torch.topk(activations, k=active, dim=1)
            codes = torch.zeros_like(activations)
            codes.scatter_(1, indices, values)
        else:
            codes = activations
        reconstruction = self.decoder(codes) + self.b0
        return codes, reconstruction


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=np.float32, copy=True)


def fit_standardizer(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=0, keepdims=True).astype("float32")
    std = np.maximum(training.std(axis=0, keepdims=True), 1e-6).astype("float32")
    return mean, std


def standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / std).astype("float32", copy=False)


def inverse_standardize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix * std + mean).astype("float32", copy=False)


def label_arrays(split_frames: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    genres = sorted(
        set(
            pd.concat(
                [frame["genre_top"].astype(str) for frame in split_frames.values()],
                ignore_index=True,
            ).dropna()
        )
    )
    genre_to_index = {genre: index for index, genre in enumerate(genres)}
    return {
        split: frame["genre_top"].astype(str).map(genre_to_index).to_numpy(dtype=np.int64, copy=True)
        for split, frame in split_frames.items()
    }


def distance_matrix_and_median(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points, points, p=2)
    upper = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    positive = upper[upper > 0]
    median = positive.median() if positive.numel() else torch.tensor(1.0, device=points.device)
    return distances, torch.clamp(median, min=1e-6)


def normalized_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    distances, median = distance_matrix_and_median(points)
    return distances / median


def scale_distance_matrices(
    reference: torch.Tensor,
    projected: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_distances, reference_median = distance_matrix_and_median(reference.detach())
    projected_distances, projected_median = distance_matrix_and_median(projected)
    if mode == "reference_median":
        reference_scale = reference_median
        projected_scale = reference_median
    elif mode == "separate_median":
        reference_scale = reference_median
        projected_scale = projected_median
    elif mode == "none":
        reference_scale = torch.tensor(1.0, device=reference.device)
        projected_scale = torch.tensor(1.0, device=projected.device)
    else:
        raise ValueError("topology_scale_mode must be one of: reference_median, separate_median, none")
    return (
        reference_distances / torch.clamp(reference_scale, min=1e-6),
        projected_distances / torch.clamp(projected_scale, min=1e-6),
        reference_median,
        projected_median,
    )


def soft_betti_curve(diagram: torch.Tensor, grid: torch.Tensor, temperature: float) -> torch.Tensor:
    if diagram.numel() == 0:
        return torch.zeros_like(grid)
    finite = diagram[torch.isfinite(diagram).all(dim=1)]
    if finite.numel() == 0:
        return torch.zeros_like(grid)
    births = finite[:, 0:1]
    deaths = finite[:, 1:2]
    values = grid.reshape(1, -1)
    alive = torch.sigmoid((values - births) / temperature) * torch.sigmoid((deaths - values) / temperature)
    return alive.sum(dim=0)


class SoftBettiLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dimensions = [int(value) for value in config["topology_dims"]]
        self.temperature = float(config["topology_temperature"])
        self.grid_size = int(config["topology_grid_size"])
        self.vr = VietorisRipsComplex(dim=max(self.dimensions), keep_infinite_features=False)

    def forward(self, reference: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
        reference_distances = normalized_distance_matrix(reference.detach())
        projected_distances = normalized_distance_matrix(projected)
        reference_info = self.vr(reference_distances, treat_as_distances=True)
        projected_info = self.vr(projected_distances, treat_as_distances=True)

        loss = projected.sum() * 0.0
        for homology_dim in self.dimensions:
            if homology_dim >= len(reference_info) or homology_dim >= len(projected_info):
                continue
            reference_diagram = reference_info[homology_dim].diagram.detach()
            projected_diagram = projected_info[homology_dim].diagram
            if reference_diagram.numel() == 0 and projected_diagram.numel() == 0:
                continue
            max_death = torch.tensor(1.0, device=projected.device)
            for diagram in (reference_diagram, projected_diagram.detach()):
                finite = diagram[torch.isfinite(diagram).all(dim=1)]
                if finite.numel():
                    max_death = torch.maximum(max_death, finite[:, 1].max())
            grid = torch.linspace(0.0, float(max_death.detach().cpu()), self.grid_size, device=projected.device)
            reference_curve = soft_betti_curve(reference_diagram, grid, self.temperature)
            projected_curve = soft_betti_curve(projected_diagram, grid, self.temperature)
            loss = loss + torch.mean((reference_curve - projected_curve) ** 2)
        return loss / max(len(self.dimensions), 1)


class PersistenceImageLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.dimensions = [int(value) for value in config["topology_dims"]]
        self.resolution = int(config["persistence_image_resolution"])
        self.sigma = float(config["persistence_image_sigma"])
        self.weight_power = float(config["persistence_image_weight_power"])
        self.scale_mode = str(config.get("topology_scale_mode", "reference_median"))
        self.scale_penalty_weight = float(config.get("topology_scale_penalty_weight", 0.0))
        self.vr = VietorisRipsComplex(dim=max(self.dimensions), keep_infinite_features=False)

        birth_grid = torch.linspace(0.0, float(config["persistence_image_max_birth"]), self.resolution)
        persistence_grid = torch.linspace(0.0, float(config["persistence_image_max_persistence"]), self.resolution)
        yy, xx = torch.meshgrid(persistence_grid, birth_grid, indexing="ij")
        self.register_buffer("birth_grid", xx.reshape(1, -1))
        self.register_buffer("persistence_grid", yy.reshape(1, -1))

    def persistence_image(self, diagram: torch.Tensor) -> torch.Tensor:
        zero = diagram.sum() * 0.0
        if diagram.numel() == 0:
            return zero + torch.zeros(self.resolution, self.resolution, device=self.birth_grid.device)
        finite = diagram[torch.isfinite(diagram).all(dim=1)]
        if finite.numel() == 0:
            return zero + torch.zeros(self.resolution, self.resolution, device=self.birth_grid.device)

        births = finite[:, 0:1]
        persistence = torch.clamp(finite[:, 1:2] - finite[:, 0:1], min=0.0)
        weights = torch.clamp(persistence, min=0.0).pow(self.weight_power)
        squared = (births - self.birth_grid) ** 2 + (persistence - self.persistence_grid) ** 2
        kernels = torch.exp(-0.5 * squared / max(self.sigma**2, 1e-12))
        image = torch.sum(weights * kernels, dim=0)
        return image.reshape(self.resolution, self.resolution)

    def forward(self, reference: torch.Tensor, projected: torch.Tensor) -> torch.Tensor:
        reference_distances, projected_distances, reference_median, projected_median = scale_distance_matrices(
            reference,
            projected,
            self.scale_mode,
        )
        reference_info = self.vr(reference_distances, treat_as_distances=True)
        projected_info = self.vr(projected_distances, treat_as_distances=True)

        image_loss = projected.sum() * 0.0
        used_dimensions = 0
        for homology_dim in self.dimensions:
            if homology_dim >= len(reference_info) or homology_dim >= len(projected_info):
                continue
            reference_image = self.persistence_image(reference_info[homology_dim].diagram.detach())
            projected_image = self.persistence_image(projected_info[homology_dim].diagram)
            image_loss = image_loss + F.mse_loss(projected_image, reference_image)
            used_dimensions += 1
        image_loss = image_loss / max(used_dimensions, 1)

        scale_loss = (projected_median / torch.clamp(reference_median, min=1e-6) - 1.0) ** 2
        return image_loss + self.scale_penalty_weight * scale_loss


def make_topology_loss(config: dict) -> nn.Module:
    metric = str(config.get("topology_metric", "persistence_image"))
    if metric == "persistence_image":
        return PersistenceImageLoss(config)
    if metric == "soft_betti":
        return SoftBettiLoss(config)
    raise ValueError("topology_metric must be one of: persistence_image, soft_betti")


def select_topology_subset(
    reference: torch.Tensor,
    codes: torch.Tensor,
    labels: torch.Tensor,
    sample_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_size = min(int(sample_size), reference.shape[0])
    if sample_size >= reference.shape[0]:
        return reference, codes

    selected_parts: list[torch.Tensor] = []
    selected_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    unique_labels = torch.unique(labels)
    quota = max(1, sample_size // max(int(unique_labels.numel()), 1))
    for label in unique_labels:
        candidates = torch.nonzero(labels == label, as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        take = min(int(candidates.numel()), quota)
        chosen = candidates[:take]
        selected_parts.append(chosen)
        selected_mask[chosen] = True

    selected = torch.cat(selected_parts) if selected_parts else torch.empty(0, dtype=torch.long, device=labels.device)
    if selected.numel() < sample_size:
        remaining = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        selected = torch.cat([selected, remaining[: sample_size - selected.numel()]])
    selected = selected[:sample_size]
    return reference.index_select(0, selected), codes.index_select(0, selected)


def supervised_code_contrastive_loss(codes: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if codes.size(0) != labels.size(0):
        raise ValueError("codes and labels must have matching batch sizes")
    if codes.size(0) < 2:
        return codes.sum() * 0.0
    features = F.normalize(codes, dim=1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    positive_mask = labels.reshape(-1, 1).eq(labels.reshape(1, -1)) & ~self_mask
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return codes.sum() * 0.0
    row_loss = -(positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1).float()
    return row_loss[valid].mean()


def batch_loss(
    model: SparseAutoencoder,
    batch: torch.Tensor,
    raw_reference: torch.Tensor,
    labels: torch.Tensor,
    *,
    method: str,
    target_active: int,
    config: dict,
    topo_loss: nn.Module | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    codes, reconstruction = model(batch, target_active)
    recon = F.mse_loss(reconstruction, batch)
    active_counts = torch.count_nonzero(torch.abs(codes) > float(config["active_threshold"]), dim=1).float()
    topology = codes.sum() * 0.0
    supcon = codes.sum() * 0.0
    if method == "topo_sae" and topo_loss is not None:
        topo_batch_size = min(int(config.get("topology_global_batch_size", config["topology_batch_size"])), batch.shape[0])
        if str(config.get("topology_target", "raw_anchor")) != "raw_anchor":
            raise ValueError("topology_target must be raw_anchor")
        topology_reference, topology_codes = select_topology_subset(raw_reference, codes, labels, topo_batch_size)
        topology = topo_loss(topology_reference, topology_codes)
    supcon_config = config.get("supcon", {})
    if bool(supcon_config.get("enabled", False)):
        supcon = supervised_code_contrastive_loss(codes, labels, float(supcon_config["temperature"]))
    total = recon + float(config["topology_weight"]) * topology + float(supcon_config.get("weight", 0.0)) * supcon
    metrics = {
        "loss": float(total.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "active": float(active_counts.mean().detach().cpu()),
        "topology": float(topology.detach().cpu()),
        "supcon": float(supcon.detach().cpu()),
    }
    return total, metrics


@torch.no_grad()
def evaluate_loss(
    model: SparseAutoencoder,
    loader: DataLoader,
    *,
    method: str,
    target_active: int,
    config: dict,
    topo_loss: nn.Module | None,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {"loss": [], "recon": [], "active": [], "topology": [], "supcon": []}
    topo_batches = 0
    for batch, raw_reference, labels in loader:
        batch = batch.to(device)
        raw_reference = raw_reference.to(device)
        labels = labels.to(device)
        loss_topo = topo_loss
        if method == "topo_sae":
            topo_batches += 1
            if topo_batches > int(config["topology_eval_batches"]):
                loss_topo = None
        _, metrics = batch_loss(
            model,
            batch,
            raw_reference,
            labels,
            method=method,
            target_active=target_active,
            config=config,
            topo_loss=loss_topo,
        )
        for key, value in metrics.items():
            totals[key].append(value)
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def train_one(
    training: np.ndarray,
    validation: np.ndarray,
    training_raw: np.ndarray,
    validation_raw: np.ndarray,
    training_labels: np.ndarray,
    validation_labels: np.ndarray,
    *,
    method: str,
    code_dim: int,
    target_active: int,
    seed: int,
    config: dict,
    device: torch.device,
) -> tuple[SparseAutoencoder, dict[str, float]]:
    set_seed(seed)
    model = SparseAutoencoder(training.shape[1], code_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["epochs"])),
        eta_min=float(config.get("min_learning_rate", 1e-5)),
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(training), torch.from_numpy(training_raw), torch.from_numpy(training_labels)),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation), torch.from_numpy(validation_raw), torch.from_numpy(validation_labels)),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=False,
    )
    topo_loss = make_topology_loss(config).to(device) if method == "topo_sae" else None
    early = config["early_stopping"]
    early_enabled = bool(early.get("enabled", True))
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    stale = 0
    final_epoch = 0

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_metrics: list[dict[str, float]] = []
        for batch, raw_reference, labels in train_loader:
            batch = batch.to(device)
            raw_reference = raw_reference.to(device)
            labels = labels.to(device)
            loss, metrics = batch_loss(
                model,
                batch,
                raw_reference,
                labels,
                method=method,
                target_active=target_active,
                config=config,
                topo_loss=topo_loss,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder_atoms()
            train_metrics.append(metrics)

        val_metrics = evaluate_loss(
            model,
            val_loader,
            method=method,
            target_active=target_active,
            config=config,
            topo_loss=topo_loss,
            device=device,
        )
        train_loss = float(np.mean([item["loss"] for item in train_metrics]))
        val_loss = float(val_metrics["loss"])
        gl = 0.0 if not math.isfinite(best_val) else 100.0 * (val_loss / max(best_val, 1e-12) - 1.0)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        elif early_enabled and epoch >= int(early["min_epochs"]) and gl > float(early["gl_threshold"]):
            stale += 1
        else:
            stale = 0
        final_epoch = epoch
        current_lr = float(scheduler.get_last_lr()[0])
        log(
            f"method={method} K={code_dim} s={target_active} epoch={epoch} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} gl={gl:.3f} "
            f"lr={current_lr:.6g} active={val_metrics['active']:.2f} "
            f"recon={val_metrics['recon']:.6f} topo={val_metrics['topology']:.6f} "
            f"supcon={val_metrics['supcon']:.6f}"
        )
        if early_enabled and stale >= int(early["patience"]):
            break
        scheduler.step()

    model.load_state_dict(best_state)
    model.normalize_decoder_atoms()
    return model, {"best_validation_loss": float(best_val), "epochs": int(final_epoch)}


@torch.no_grad()
def encode_split(
    model: SparseAutoencoder,
    matrix: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_active: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    codes: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], batch_size):
        batch = torch.from_numpy(matrix[start : start + batch_size]).to(device)
        code, reconstruction = model(batch, target_active)
        codes.append(code.cpu().numpy().astype("float32", copy=False))
        reconstructions.append(reconstruction.cpu().numpy().astype("float32", copy=False))
    code_matrix = np.concatenate(codes, axis=0)
    recon_matrix = inverse_standardize(np.concatenate(reconstructions, axis=0), mean, std)
    return code_matrix, recon_matrix


def pad_codes(codes: np.ndarray, max_dim: int) -> pd.DataFrame:
    output = np.full((codes.shape[0], max_dim), np.nan, dtype=np.float32)
    output[:, : codes.shape[1]] = codes
    return pd.DataFrame(output, columns=[f"embedding_{index:04d}" for index in range(max_dim)])


def reconstruction_frame(reconstructions: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(reconstructions, columns=[f"recon_{index:04d}" for index in range(reconstructions.shape[1])])


def build_sparse_frame(
    split_frame: pd.DataFrame,
    codes: np.ndarray,
    reconstructions: np.ndarray,
    *,
    method: str,
    dataset: str,
    source: str,
    split: str,
    code_dim: int,
    target_active: int,
    topology_weight: float,
    topology_metric: str,
    topology_target: str,
    supcon_enabled: bool,
    supcon_weight: float,
    supcon_temperature: float,
    seed: int,
    input_dim: int,
    max_code_dim: int,
    best_validation_loss: float,
    epochs: int,
    active_threshold: float,
) -> pd.DataFrame:
    metadata = metadata_frame(split_frame)
    active_counts = np.sum(np.abs(codes) > active_threshold, axis=1)
    metadata["method"] = method
    metadata["family"] = "sparse_dictionary"
    metadata["dataset"] = dataset
    metadata["source"] = source
    metadata["split"] = split
    metadata["ratio_percent"] = int(round(100.0 * target_active / code_dim))
    metadata["m_dim"] = int(code_dim)
    metadata["input_dim"] = int(input_dim)
    metadata["seed"] = int(seed)
    metadata["code_dim"] = int(code_dim)
    metadata["target_active"] = int(target_active)
    metadata["actual_active"] = active_counts.astype(np.float32)
    metadata["actual_active_mean"] = float(np.mean(active_counts))
    metadata["l1_lambda"] = 0.0
    metadata["topology_weight"] = float(topology_weight)
    metadata["topology_metric"] = str(topology_metric)
    metadata["topology_target"] = str(topology_target)
    metadata["supcon_enabled"] = bool(supcon_enabled)
    metadata["supcon_weight"] = float(supcon_weight)
    metadata["supcon_temperature"] = float(supcon_temperature)
    metadata["best_validation_loss"] = float(best_validation_loss)
    metadata["epochs"] = int(epochs)
    return pd.concat(
        [
            metadata.reset_index(drop=True),
            pad_codes(codes, max_code_dim),
            reconstruction_frame(reconstructions),
        ],
        axis=1,
    )


def run(config: dict, anchor_dir: Path = DEFAULT_ANCHOR_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    source = str(config["source"])
    dataset = str(config["dataset"])
    methods = [str(value) for value in config["methods"]]
    code_dims = [int(value) for value in config["code_dims"]]
    target_values = [int(value) for value in config["target_active"]]
    seeds = [int(value) for value in config.get("seeds", [0])]
    max_code_dim = max(code_dims)
    device = resolve_device(str(config["device"]))

    split_frames = load_anchor_splits(anchor_dir, source, dataset)
    columns = embedding_columns(split_frames["training"])
    raw_training = feature_matrix(split_frames["training"], columns)
    mean, std = fit_standardizer(raw_training)
    raw = {
        split: feature_matrix(frame, columns)
        for split, frame in split_frames.items()
    }
    standardized = {
        split: standardize(raw[split], mean, std)
        for split, frame in split_frames.items()
    }
    labels = label_arrays(split_frames)
    supcon_config = config.get("supcon", {})
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    report(
        f"START module=compression.sparse_dictionary source={source} dataset={dataset} "
        f"methods={','.join(methods)} device={device} "
        f"supcon={bool(supcon_config.get('enabled', False))}"
    )
    for method in methods:
        rows: list[pd.DataFrame] = []
        for seed in seeds:
            for code_dim in code_dims:
                for target_active in target_values:
                    if target_active > code_dim:
                        continue
                    model, metadata = train_one(
                        standardized["training"],
                        standardized["validation"],
                        raw["training"],
                        raw["validation"],
                        labels["training"],
                        labels["validation"],
                        method=method,
                        code_dim=code_dim,
                        target_active=target_active,
                        seed=seed,
                        config=config,
                        device=device,
                    )
                    for split in SPLITS:
                        codes, reconstructions = encode_split(
                            model,
                            standardized[split],
                            mean,
                            std,
                            device,
                            int(config["batch_size"]),
                            target_active,
                        )
                        rows.append(
                            build_sparse_frame(
                                split_frames[split],
                                codes,
                                reconstructions,
                                method=method,
                                dataset=dataset,
                                source=source,
                                split=split,
                                code_dim=code_dim,
                                target_active=target_active,
                                topology_weight=float(config["topology_weight"]) if method == "topo_sae" else 0.0,
                                topology_metric=str(config.get("topology_metric", "persistence_image")) if method == "topo_sae" else "",
                                topology_target=str(config.get("topology_target", "raw_anchor")) if method == "topo_sae" else "",
                                supcon_enabled=bool(supcon_config.get("enabled", False)),
                                supcon_weight=float(supcon_config.get("weight", 0.0)),
                                supcon_temperature=float(supcon_config.get("temperature", 0.07)),
                                seed=seed,
                                input_dim=len(columns),
                                max_code_dim=max_code_dim,
                                best_validation_loss=float(metadata["best_validation_loss"]),
                                epochs=int(metadata["epochs"]),
                                active_threshold=float(config["active_threshold"]),
                            )
                        )
                    report(
                        f"trained method={method} K={code_dim} s={target_active} seed={seed} "
                        f"best_val={metadata['best_validation_loss']:.6f} epochs={metadata['epochs']}"
                    )
        if rows:
            output_path = method_parquet_path(output_dir, method, source, dataset)
            pd.concat(rows, ignore_index=True).to_parquet(output_path, index=False)
            written.append(output_path)
            report(f"wrote method={method} path={output_path}")
    report(f"DONE module=compression.sparse_dictionary files={len(written)}")
    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    run(config, args.anchor_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
