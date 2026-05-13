#!/usr/bin/env python3
#
# Topology preservation scoring for compressed ABT manifold features.
#

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-evaluation-matplotlib")

import gudhi as gd
import numpy as np
import pandas as pd
from persim import wasserstein
from ripser import ripser

from compression.common import SPLITS, active_feature_columns, embedding_columns
from compression.train_utils import load_config
from evaluation.filters import FILTER_DEFAULTS, passes_run_filters
from evaluation.visualizations import save_topology_metric_panel, save_topology_preservation_plot


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "topology.yaml"
DEFAULT_DATA_DIR = Path("compression/data")
DEFAULT_CONFIG = {
    "seed": 7,
    "dataset": "fma_small_mel",
    "anchor_dir": "representation/data",
    "reference_source": "anchor",
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
    "salient_pairs_per_dim": 16,
    "compute_critical_pairs": True,
    **FILTER_DEFAULTS,
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score topology preservation against the ABT anchor manifold.")
    parser.add_argument("-d", "--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    if int(config["tracks_per_genre"]) < 3:
        raise ValueError("tracks_per_genre must be at least 3")
    if int(config["bootstrap_replicates"]) <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if int(config["max_homology_dim"]) < 0:
        raise ValueError("max_homology_dim must be non-negative")
    if int(config["n_perm"]) <= 1:
        raise ValueError("n_perm must be greater than 1")
    if int(config["betti_grid_size"]) <= 1:
        raise ValueError("betti_grid_size must be greater than 1")
    if int(config["persistence_image_resolution"]) <= 1:
        raise ValueError("persistence_image_resolution must be greater than 1")


def load_anchor_frame(config: dict) -> pd.DataFrame:
    anchor_dir = Path(str(config["anchor_dir"])).expanduser()
    source = str(config["reference_source"])
    dataset = str(config["dataset"])
    path = anchor_dir / f"{source}_{dataset}_training.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing anchor parquet: {path}")
    return pd.read_parquet(path).copy()


def discover_groups(data_dir: Path, config: dict) -> dict[str, pd.DataFrame]:
    groups: dict[str, pd.DataFrame] = {}
    for path in sorted(data_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        required = {"method", "split", "ratio_percent", "m_dim", "seed", "track_id", "genre_top"}
        if not required.issubset(frame.columns):
            continue
        training = frame[frame["split"] == "training"].copy()
        for (method, ratio, seed), group in training.groupby(["method", "ratio_percent", "seed"], dropna=False):
            method = str(method)
            ratio = int(ratio)
            seed = int(seed)
            run_name = f"{method}_r{ratio:03d}_s{seed:02d}"
            if passes_run_filters(run_name, method, ratio, config):
                groups[run_name] = group.copy()
    return groups


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2).astype("float64", copy=False)
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
    return distances / max(median, 1e-8)


def compute_diagrams(points: np.ndarray, config: dict) -> list[np.ndarray]:
    distances = pairwise_distances(points)
    n_perm = min(int(config["n_perm"]), points.shape[0])
    kwargs: dict[str, object] = {
        "maxdim": int(config["max_homology_dim"]),
        "distance_matrix": True,
    }
    if n_perm < points.shape[0]:
        kwargs["n_perm"] = n_perm
    return [np.asarray(diagram, dtype="float64") for diagram in ripser(distances, **kwargs)["dgms"]]


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype="float64")
    return np.asarray(diagram[np.isfinite(diagram).all(axis=1)], dtype="float64")


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    births = finite[:, 0:1]
    deaths = finite[:, 1:2]
    values = grid.reshape(1, -1)
    return np.sum((births <= values) & (values < deaths), axis=0).astype("float64")


def betti_distances(reference: list[np.ndarray], projected: list[np.ndarray], grid_size: int) -> dict[int, float]:
    all_finite = [finite_diagram(diagram) for diagram in [*reference, *projected]]
    non_empty = [diagram for diagram in all_finite if len(diagram) > 0]
    if not non_empty:
        grid = np.linspace(0.0, 1.0, grid_size)
    else:
        stacked = np.concatenate(non_empty, axis=0)
        upper = float(np.max(stacked[:, 1])) if len(stacked) else 1.0
        grid = np.linspace(0.0, max(upper, 1e-6), grid_size)
    distances: dict[int, float] = {}
    for homology_dim in range(min(len(reference), len(projected))):
        left = betti_curve(reference[homology_dim], grid)
        right = betti_curve(projected[homology_dim], grid)
        distances[homology_dim] = float(np.mean((left - right) ** 2))
    return distances


def diagram_wasserstein(reference: list[np.ndarray], projected: list[np.ndarray]) -> dict[int, float]:
    distances: dict[int, float] = {}
    for homology_dim in range(min(len(reference), len(projected))):
        left = finite_diagram(reference[homology_dim])
        right = finite_diagram(projected[homology_dim])
        distances[homology_dim] = 0.0 if len(left) == 0 and len(right) == 0 else float(wasserstein(left, right, matching=False))
    return distances


def persistence_image(diagram: np.ndarray, config: dict) -> np.ndarray:
    finite = finite_diagram(diagram)
    resolution = int(config["persistence_image_resolution"])
    if finite.size == 0:
        return np.zeros((resolution, resolution), dtype=np.float64)

    births = finite[:, 0:1]
    persistence = np.maximum(finite[:, 1:2] - finite[:, 0:1], 0.0)
    birth_grid = np.linspace(0.0, float(config["persistence_image_max_birth"]), resolution)
    persistence_grid = np.linspace(0.0, float(config["persistence_image_max_persistence"]), resolution)
    grid_birth, grid_persistence = np.meshgrid(birth_grid, persistence_grid)
    squared = (births[:, :, None] - grid_birth[None, :, :]) ** 2
    squared = squared + (persistence[:, :, None] - grid_persistence[None, :, :]) ** 2
    sigma = max(float(config["persistence_image_sigma"]), 1e-12)
    weights = np.maximum(persistence, 0.0) ** float(config["persistence_image_weight_power"])
    image = np.sum(weights[:, :, None] * np.exp(-0.5 * squared / sigma**2), axis=0)
    return np.asarray(image, dtype=np.float64)


def persistence_image_distances(reference: list[np.ndarray], projected: list[np.ndarray], config: dict) -> dict[int, float]:
    distances: dict[int, float] = {}
    for homology_dim in range(min(len(reference), len(projected))):
        left = persistence_image(reference[homology_dim], config)
        right = persistence_image(projected[homology_dim], config)
        distances[homology_dim] = float(np.mean((left - right) ** 2))
    return distances


def edge_distance(matrix: np.ndarray, left: int, right: int) -> float:
    return float(np.linalg.norm(matrix[left] - matrix[right]))


def simplex_diameter_edge(matrix: np.ndarray, vertices: list[int]) -> tuple[int, int] | None:
    unique = sorted(set(int(vertex) for vertex in vertices))
    if len(unique) < 2:
        return None
    return max(combinations(unique, 2), key=lambda pair: edge_distance(matrix, pair[0], pair[1]))


def persistence_pairs(points: np.ndarray, config: dict) -> list[tuple[int, float, tuple[int, int]]]:
    max_edge = float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)))
    tree = gd.RipsComplex(points=points, max_edge_length=max_edge).create_simplex_tree(
        max_dimension=int(config["max_homology_dim"]) + 1
    )
    persistence = tree.persistence()
    pairs = tree.persistence_pairs()
    records: list[tuple[int, float, tuple[int, int]]] = []
    for persistence_item, pair in zip(persistence, pairs):
        homology_dim, interval = persistence_item
        if homology_dim > int(config["max_homology_dim"]):
            continue
        birth, death = interval
        if not np.isfinite(death):
            continue
        edge = simplex_diameter_edge(points, list(pair[0]) + list(pair[1]))
        if edge is not None:
            records.append((int(homology_dim), float(death - birth), edge))
    return sorted(records, key=lambda item: item[1], reverse=True)


def critical_pair_distortion(reference_points: np.ndarray, projected_points: np.ndarray, config: dict) -> float:
    if not bool(config.get("compute_critical_pairs", True)):
        return float("nan")
    pairs = persistence_pairs(reference_points, config)
    if not pairs:
        return 0.0
    distortions: list[float] = []
    counts: dict[int, int] = {}
    limit = int(config["salient_pairs_per_dim"])
    for homology_dim, _, (left, right) in pairs:
        if counts.get(homology_dim, 0) >= limit:
            continue
        original = edge_distance(reference_points, left, right)
        projected = edge_distance(projected_points, left, right)
        distortions.append(abs(original - projected) / max(original, 1e-8))
        counts[homology_dim] = counts.get(homology_dim, 0) + 1
    return float(np.mean(distortions)) if distortions else 0.0


def align(reference: pd.DataFrame, projected: pd.DataFrame, m_dim: int) -> pd.DataFrame:
    ref_cols = embedding_columns(reference)
    proj_cols = active_feature_columns(projected, m_dim)
    merged = reference[["track_id", "genre_top", *ref_cols]].merge(
        projected[["track_id", *proj_cols]],
        on="track_id",
        suffixes=("_reference", "_projected"),
    )
    if merged.empty:
        raise ValueError("reference and projected frames have no matching track_id")
    return merged


def score_group(run_name: str, reference: pd.DataFrame, projected: pd.DataFrame, config: dict) -> list[dict[str, object]]:
    first = projected.iloc[0]
    method = str(first["method"])
    ratio = int(first["ratio_percent"])
    seed = int(first["seed"])
    m_dim = int(first["m_dim"])
    merged = align(reference, projected, m_dim)
    ref_cols = sorted(column for column in merged.columns if column.startswith("embedding_") and column.endswith("_reference"))
    proj_cols = sorted(column for column in merged.columns if column.startswith("embedding_") and column.endswith("_projected"))
    rng = np.random.default_rng(int(config["seed"]) + seed * 10_000 + ratio)
    records: list[dict[str, object]] = []

    for genre, genre_frame in sorted(merged.groupby("genre_top")):
        sample_size = min(int(config["tracks_per_genre"]), len(genre_frame))
        if sample_size < 3:
            continue
        for bootstrap in range(int(config["bootstrap_replicates"])):
            indices = rng.choice(len(genre_frame), size=sample_size, replace=len(genre_frame) < sample_size)
            sample = genre_frame.iloc[np.sort(indices)]
            ref_points = sample[ref_cols].to_numpy(dtype=np.float32, copy=True)
            proj_points = sample[proj_cols].to_numpy(dtype=np.float32, copy=True)
            ref_diagrams = compute_diagrams(ref_points, config)
            proj_diagrams = compute_diagrams(proj_points, config)
            betti = betti_distances(ref_diagrams, proj_diagrams, int(config["betti_grid_size"]))
            wass = diagram_wasserstein(ref_diagrams, proj_diagrams)
            image_distances = persistence_image_distances(ref_diagrams, proj_diagrams, config)
            critical = critical_pair_distortion(ref_points, proj_points, config)
            records.append(
                {
                    "run_name": run_name,
                    "method": method,
                    "family": str(first.get("family", "")),
                    "ratio_percent": ratio,
                    "m_dim": m_dim,
                    "seed": seed,
                    "genre_top": genre,
                    "bootstrap": bootstrap,
                    "track_count": sample_size,
                    "betti_dist_h0": betti.get(0, 0.0),
                    "betti_dist_h1": betti.get(1, 0.0),
                    "wasserstein_h0": wass.get(0, 0.0),
                    "wasserstein_h1": wass.get(1, 0.0),
                    "wasserstein_distance": float(np.mean(list(wass.values()))) if wass else 0.0,
                    "persistence_image_h0": image_distances.get(0, 0.0),
                    "persistence_image_h1": image_distances.get(1, 0.0),
                    "persistence_image_distance": float(np.mean(list(image_distances.values()))) if image_distances else 0.0,
                    "critical_pair_distortion": critical,
                }
            )
    return records


def standard_error(values: pd.Series) -> float:
    return float(values.std(ddof=1) / np.sqrt(max(len(values), 1))) if len(values) > 1 else 0.0


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    validate_config(config)
    data_dir = args.data_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"
    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    report(f"START module=evaluation.topology data_dir={data_dir} config={args.config}")
    reference = load_anchor_frame(config)
    groups = discover_groups(data_dir, config)
    if not groups:
        raise FileNotFoundError(f"no method parquet groups matched filters in {data_dir}")

    records: list[dict[str, object]] = []
    for run_name, group in sorted(groups.items()):
        records.extend(score_group(run_name, reference, group, config))
        first = group.iloc[0]
        report(f"topology method={first['method']} ratio={int(first['ratio_percent'])} seed={int(first['seed'])}")

    detail = pd.DataFrame.from_records(records)
    summary = (
        detail.groupby(["method", "family", "ratio_percent", "m_dim", "seed", "genre_top"], as_index=False)
        .agg(
            betti_dist_h0=("betti_dist_h0", "mean"),
            betti_dist_h1=("betti_dist_h1", "mean"),
            wasserstein_h0=("wasserstein_h0", "mean"),
            wasserstein_h1=("wasserstein_h1", "mean"),
            wasserstein_distance=("wasserstein_distance", "mean"),
            persistence_image_h0=("persistence_image_h0", "mean"),
            persistence_image_h1=("persistence_image_h1", "mean"),
            persistence_image_distance=("persistence_image_distance", "mean"),
            critical_pair_distortion=("critical_pair_distortion", "mean"),
            betti_dist_h0_se=("betti_dist_h0", standard_error),
            betti_dist_h1_se=("betti_dist_h1", standard_error),
            wasserstein_h0_se=("wasserstein_h0", standard_error),
            wasserstein_h1_se=("wasserstein_h1", standard_error),
            persistence_image_h0_se=("persistence_image_h0", standard_error),
            persistence_image_h1_se=("persistence_image_h1", standard_error),
            bootstrap_replicates=("bootstrap", "count"),
            track_count=("track_count", "mean"),
        )
    )
    detail_path = data_root / "topology_bootstrap.csv"
    summary_path = data_root / "topology_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    save_topology_preservation_plot(
        summary,
        "wasserstein_distance",
        image_root / "topology_wasserstein_by_genre.png",
        title="Topology Wasserstein by Genre",
    )
    for method in sorted(summary["method"].unique()):
        save_topology_metric_panel(
            summary,
            str(method),
            image_root / f"{method}_topology_metric_panel.png",
        )
    report(f"DONE module=evaluation.topology runs={len(groups)} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
