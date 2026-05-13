#!/usr/bin/env python3
#
# variation.py  Andrew Belles  May 8th, 2026
#
# Bootstrap topology variation within and across genres for the anchor manifold.
#

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-persistence-matplotlib")

import numpy as np
import pandas as pd
import yaml
from persim import wasserstein
from ripser import ripser

from persistence.diagrams import resolve_config_path
from persistence.visualizations import (
    save_between_genre_distance_heatmaps,
    save_between_genre_distance_source_panel,
    save_genre_covariance_heatmaps,
    save_genre_covariance_source_panel,
    save_topology_meaningfulness_plot,
    save_within_genre_variation_plot,
    save_within_genre_variation_source_panel,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "variation.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
SPLITS = ("training", "validation", "test")


@dataclass(frozen=True)
class VariationConfig:
    seed: int = 7
    anchor_source: str = "anchor"
    sources: tuple[str, ...] = ("anchor",)
    dataset: str = "fma_small_mel"
    splits: tuple[str, ...] = ("training",)
    genres: tuple[str, ...] = ()
    bootstrap_replicates: int = 40
    tracks_per_genre: int = 64
    max_homology_dim: int = 1
    n_perm: int = 64
    metric: str = "euclidean"
    standardize_rows: bool = True
    summary_metric: str = "total_persistence"
    between_genre_replicates: int = 12


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure bootstrapped topology variation within and across genres for the anchor manifold."
    )
    parser.add_argument(
        "-a",
        "--anchor-dir",
        type=Path,
        default=DEFAULT_ANCHOR_DIR,
        help=f"Directory containing anchor split parquets. Defaults to {DEFAULT_ANCHOR_DIR}.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def _as_tuple(raw_value: object, default: Iterable[str]) -> tuple[str, ...]:
    if raw_value is None:
        return tuple(default)
    if isinstance(raw_value, str):
        return (raw_value,)
    if isinstance(raw_value, list | tuple):
        return tuple(str(value) for value in raw_value)
    raise ValueError("expected a string or list")


def load_config(config_path: Path) -> VariationConfig:
    resolved_path = resolve_config_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"config must be a mapping: {resolved_path}")

    return VariationConfig(
        seed=int(raw_config.get("seed", 7)),
        anchor_source=str(raw_config.get("anchor_source", "anchor")),
        sources=_as_tuple(raw_config.get("sources"), (str(raw_config.get("anchor_source", "anchor")),)),
        dataset=str(raw_config.get("dataset", "fma_small_mel")),
        splits=_as_tuple(raw_config.get("splits"), ("training",)),
        genres=_as_tuple(raw_config.get("genres"), ()),
        bootstrap_replicates=int(raw_config.get("bootstrap_replicates", 40)),
        tracks_per_genre=int(raw_config.get("tracks_per_genre", 64)),
        max_homology_dim=int(raw_config.get("max_homology_dim", 1)),
        n_perm=int(raw_config.get("n_perm", 64)),
        metric=str(raw_config.get("metric", "euclidean")),
        standardize_rows=bool(raw_config.get("standardize_rows", True)),
        summary_metric=str(raw_config.get("summary_metric", "total_persistence")),
        between_genre_replicates=int(raw_config.get("between_genre_replicates", 12)),
    )


def validate_config(config: VariationConfig) -> None:
    invalid_splits = sorted(set(config.splits) - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"unsupported splits: {', '.join(invalid_splits)}")
    if not config.sources:
        raise ValueError("sources must contain at least one source")
    if config.bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least 2")
    if config.tracks_per_genre < 3:
        raise ValueError("tracks_per_genre must be at least 3")
    if config.max_homology_dim < 0:
        raise ValueError("max_homology_dim must be non-negative")
    if config.n_perm <= 1:
        raise ValueError("n_perm must be greater than 1")
    if config.summary_metric not in {"total_persistence", "mean_lifetime", "max_lifetime", "feature_count"}:
        raise ValueError("summary_metric must be one of: total_persistence, mean_lifetime, max_lifetime, feature_count")
    if config.between_genre_replicates < 0:
        raise ValueError("between_genre_replicates must be non-negative")


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("embedding_"))
    if not columns:
        raise ValueError("anchor parquet has no embedding columns")
    return columns


def load_anchor_frame(anchor_dir: Path, config: VariationConfig, source: str) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    for split in config.splits:
        path = anchor_dir / f"{source}_{config.dataset}_{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing anchor parquet: {path}")
        frame = pd.read_parquet(path).copy()
        columns = embedding_columns(frame)
        if feature_columns is None:
            feature_columns = columns
        elif columns != feature_columns:
            raise ValueError(f"embedding columns differ across splits at {path}")
        frame["split"] = split
        frames.append(frame)

    if feature_columns is None:
        raise ValueError("no splits configured")

    combined = pd.concat(frames, ignore_index=True)
    if "genre_top" not in combined.columns:
        raise ValueError("anchor parquet is missing genre_top")
    if config.genres:
        wanted = {genre.lower() for genre in config.genres}
        combined = combined[combined["genre_top"].str.lower().isin(wanted)].copy()
    if combined.empty:
        raise ValueError("no anchor rows matched configured genres")
    combined["source"] = source
    return combined, feature_columns


def standardize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    mean = point_cloud.mean(axis=1, keepdims=True)
    std = np.maximum(point_cloud.std(axis=1, keepdims=True), 1e-6)
    return ((point_cloud - mean) / std).astype("float32", copy=False)


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype="float64")
    return np.asarray(diagram[np.isfinite(diagram).all(axis=1)], dtype="float64")


def compute_diagrams(point_cloud: np.ndarray, config: VariationConfig) -> list[np.ndarray]:
    n_perm = min(config.n_perm, point_cloud.shape[0])
    kwargs: dict[str, object] = {
        "maxdim": config.max_homology_dim,
        "metric": config.metric,
    }
    if n_perm < point_cloud.shape[0]:
        kwargs["n_perm"] = n_perm
    result = ripser(point_cloud, **kwargs)
    return [finite_diagram(np.asarray(diagram, dtype="float64")) for diagram in result["dgms"]]


def diagram_stats(diagram: np.ndarray) -> dict[str, float]:
    if diagram.size == 0:
        return {
            "feature_count": 0.0,
            "total_persistence": 0.0,
            "mean_lifetime": 0.0,
            "max_lifetime": 0.0,
            "persistence_entropy": 0.0,
        }

    lifetimes = np.maximum(diagram[:, 1] - diagram[:, 0], 0.0)
    total = float(lifetimes.sum())
    if total <= 0.0:
        entropy = 0.0
    else:
        probabilities = lifetimes / total
        entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    return {
        "feature_count": float(len(diagram)),
        "total_persistence": total,
        "mean_lifetime": float(lifetimes.mean()) if len(lifetimes) else 0.0,
        "max_lifetime": float(lifetimes.max()) if len(lifetimes) else 0.0,
        "persistence_entropy": entropy,
    }


def diagram_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(wasserstein(left, right, matching=False))


def bootstrap_diagrams(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: VariationConfig,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], np.ndarray]]:
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, object]] = []
    diagrams_by_key: dict[tuple[str, int, int], np.ndarray] = {}

    for genre, genre_frame in sorted(frame.groupby("genre_top")):
        if len(genre_frame) < 3:
            log(f"skipping genre={genre} rows={len(genre_frame)}")
            continue

        sample_size = min(config.tracks_per_genre, len(genre_frame))
        replace = len(genre_frame) < config.tracks_per_genre
        features = genre_frame[feature_columns].to_numpy(dtype=np.float32, copy=True)

        for bootstrap_index in range(config.bootstrap_replicates):
            sample_indices = rng.choice(len(features), size=sample_size, replace=replace)
            point_cloud = features[sample_indices]
            if config.standardize_rows:
                point_cloud = standardize_point_cloud(point_cloud)

            diagrams = compute_diagrams(point_cloud, config)
            for homology_dim, diagram in enumerate(diagrams):
                stats = diagram_stats(diagram)
                diagrams_by_key[(genre, bootstrap_index, homology_dim)] = diagram
                records.append(
                    {
                        "genre_top": genre,
                        "bootstrap": bootstrap_index,
                        "homology_dim": homology_dim,
                        "track_count": sample_size,
                        **stats,
                    }
                )

        report(f"bootstrapped genre={genre} rows={len(genre_frame)} sample_size={sample_size}")

    if not records:
        raise ValueError("no bootstrap records were computed")
    return pd.DataFrame.from_records(records), diagrams_by_key


def within_genre_variation(diagrams_by_key: dict[tuple[str, int, int], np.ndarray]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    genres = sorted({key[0] for key in diagrams_by_key})
    homology_dims = sorted({key[2] for key in diagrams_by_key})

    for genre in genres:
        for homology_dim in homology_dims:
            bootstrap_ids = sorted(key[1] for key in diagrams_by_key if key[0] == genre and key[2] == homology_dim)
            distances = [
                diagram_distance(
                    diagrams_by_key[(genre, left, homology_dim)],
                    diagrams_by_key[(genre, right, homology_dim)],
                )
                for left, right in combinations(bootstrap_ids, 2)
            ]
            records.append(
                {
                    "genre_top": genre,
                    "homology_dim": homology_dim,
                    "pair_count": len(distances),
                    "mean_pairwise_wasserstein": float(np.mean(distances)) if distances else 0.0,
                    "std_pairwise_wasserstein": float(np.std(distances)) if distances else 0.0,
                    "median_pairwise_wasserstein": float(np.median(distances)) if distances else 0.0,
                }
            )

    return pd.DataFrame.from_records(records)


def cross_genre_covariance(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for homology_dim, subset in summary.groupby("homology_dim"):
        pivot = subset.pivot(index="bootstrap", columns="genre_top", values=metric).sort_index(axis=1)
        covariance = pivot.cov()
        correlation = pivot.corr()
        for genre_a in pivot.columns:
            for genre_b in pivot.columns:
                records.append(
                    {
                        "homology_dim": int(homology_dim),
                        "metric": metric,
                        "genre_a": genre_a,
                        "genre_b": genre_b,
                        "covariance": float(covariance.loc[genre_a, genre_b]),
                        "correlation": float(correlation.loc[genre_a, genre_b]),
                    }
                )
    return pd.DataFrame.from_records(records)


def between_genre_distance(
    diagrams_by_key: dict[tuple[str, int, int], np.ndarray],
    replicate_limit: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    genres = sorted({key[0] for key in diagrams_by_key})
    homology_dims = sorted({key[2] for key in diagrams_by_key})
    bootstrap_ids = sorted({key[1] for key in diagrams_by_key})
    if replicate_limit > 0:
        bootstrap_ids = bootstrap_ids[:replicate_limit]

    for homology_dim in homology_dims:
        for genre_a in genres:
            for genre_b in genres:
                distances: list[float] = []
                for bootstrap_id in bootstrap_ids:
                    left = diagrams_by_key.get((genre_a, bootstrap_id, homology_dim))
                    right = diagrams_by_key.get((genre_b, bootstrap_id, homology_dim))
                    if left is None or right is None:
                        continue
                    distances.append(diagram_distance(left, right))
                records.append(
                    {
                        "homology_dim": homology_dim,
                        "genre_a": genre_a,
                        "genre_b": genre_b,
                        "replicates": len(distances),
                        "mean_wasserstein": float(np.mean(distances)) if distances else 0.0,
                        "std_wasserstein": float(np.std(distances)) if distances else 0.0,
                    }
                )

    return pd.DataFrame.from_records(records)


def topology_meaningfulness_summary(within: pd.DataFrame, between: pd.DataFrame) -> pd.DataFrame:
    within_summary = (
        within.groupby(["source", "homology_dim"], as_index=False)
        .agg(
            mean_within_wasserstein=("mean_pairwise_wasserstein", "mean"),
            median_within_wasserstein=("median_pairwise_wasserstein", "mean"),
            within_genre_count=("genre_top", "nunique"),
        )
    )
    between_off_diagonal = between[between["genre_a"] != between["genre_b"]].copy()
    between_summary = (
        between_off_diagonal.groupby(["source", "homology_dim"], as_index=False)
        .agg(
            mean_between_wasserstein=("mean_wasserstein", "mean"),
            std_between_wasserstein=("mean_wasserstein", "std"),
            between_pair_count=("mean_wasserstein", "count"),
        )
    )
    summary = within_summary.merge(between_summary, on=["source", "homology_dim"], how="left")
    summary["between_within_ratio"] = (
        summary["mean_between_wasserstein"] / summary["mean_within_wasserstein"].clip(lower=1e-8)
    )
    summary["between_minus_within"] = summary["mean_between_wasserstein"] - summary["mean_within_wasserstein"]
    summary["topology_is_separated"] = summary["between_within_ratio"] > 1.0
    return summary


def run_source(
    source: str,
    anchor_dir: Path,
    config: VariationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, feature_columns = load_anchor_frame(anchor_dir, config, source)
    summary, diagrams_by_key = bootstrap_diagrams(frame, feature_columns, config)
    within = within_genre_variation(diagrams_by_key)
    covariance = cross_genre_covariance(summary, config.summary_metric)
    between = between_genre_distance(diagrams_by_key, config.between_genre_replicates)

    for output in (summary, within, covariance, between):
        output.insert(0, "source", source)

    return summary, within, covariance, between


def write_source_outputs(
    source: str,
    summary: pd.DataFrame,
    within: pd.DataFrame,
    covariance: pd.DataFrame,
    between: pd.DataFrame,
    config: VariationConfig,
    data_root: Path,
    image_root: Path,
) -> None:
    summary_path = data_root / f"{source}_bootstrap_topology_summary.csv"
    within_path = data_root / f"{source}_within_genre_topology_variation.csv"
    covariance_path = data_root / f"{source}_cross_genre_topology_covariance.csv"
    between_path = data_root / f"{source}_between_genre_topology_distance.csv"
    summary.to_csv(summary_path, index=False)
    within.to_csv(within_path, index=False)
    covariance.to_csv(covariance_path, index=False)
    between.to_csv(between_path, index=False)

    save_within_genre_variation_plot(within, image_root / f"{source}_within_genre_topology_variation.png")
    save_genre_covariance_heatmaps(
        covariance,
        "correlation",
        image_root / f"{source}_cross_genre_topology_correlation.png",
        title=f"{source.replace('_', ' ').title()} Cross-Genre Topology Correlation "
        f"({config.summary_metric.replace('_', ' ').title()})",
    )
    save_genre_covariance_heatmaps(
        covariance,
        "covariance",
        image_root / f"{source}_cross_genre_topology_covariance.png",
        title=f"{source.replace('_', ' ').title()} Cross-Genre Topology Covariance "
        f"({config.summary_metric.replace('_', ' ').title()})",
    )
    save_between_genre_distance_heatmaps(between, image_root / f"{source}_between_genre_topology_distance.png")

    log(f"summary={summary_path}")
    log(f"within={within_path}")
    log(f"covariance={covariance_path}")
    log(f"between={between_path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    anchor_dir = args.anchor_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"
    report(f"START module=persistence.variation anchor_dir={anchor_dir} config={args.config}")

    data_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    withins: list[pd.DataFrame] = []
    covariances: list[pd.DataFrame] = []
    betweens: list[pd.DataFrame] = []
    for source in config.sources:
        summary, within, covariance, between = run_source(source, anchor_dir, config)
        summaries.append(summary)
        withins.append(within)
        covariances.append(covariance)
        betweens.append(between)
        write_source_outputs(source, summary, within, covariance, between, config, data_root, image_root)

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_within = pd.concat(withins, ignore_index=True)
    combined_covariance = pd.concat(covariances, ignore_index=True)
    combined_between = pd.concat(betweens, ignore_index=True)
    meaningfulness = topology_meaningfulness_summary(combined_within, combined_between)

    source_stem = "_".join(config.sources)
    summary_path = data_root / f"{source_stem}_bootstrap_topology_summary.csv"
    within_path = data_root / f"{source_stem}_within_genre_topology_variation.csv"
    covariance_path = data_root / f"{source_stem}_cross_genre_topology_covariance.csv"
    between_path = data_root / f"{source_stem}_between_genre_topology_distance.csv"
    meaningfulness_path = data_root / f"{source_stem}_topology_meaningfulness_summary.csv"
    combined_summary.to_csv(summary_path, index=False)
    combined_within.to_csv(within_path, index=False)
    combined_covariance.to_csv(covariance_path, index=False)
    combined_between.to_csv(between_path, index=False)
    meaningfulness.to_csv(meaningfulness_path, index=False)

    save_within_genre_variation_source_panel(
        combined_within,
        image_root / f"{source_stem}_within_genre_topology_variation.png",
        list(config.sources),
    )
    save_genre_covariance_source_panel(
        combined_covariance,
        "correlation",
        image_root / f"{source_stem}_cross_genre_topology_correlation.png",
        title=f"Cross-Genre Topology Correlation by Basis ({config.summary_metric.replace('_', ' ').title()})",
        sources=list(config.sources),
    )
    save_genre_covariance_source_panel(
        combined_covariance,
        "covariance",
        image_root / f"{source_stem}_cross_genre_topology_covariance.png",
        title=f"Cross-Genre Topology Covariance by Basis ({config.summary_metric.replace('_', ' ').title()})",
        sources=list(config.sources),
    )
    save_between_genre_distance_source_panel(
        combined_between,
        image_root / f"{source_stem}_between_genre_topology_distance.png",
        list(config.sources),
    )
    save_topology_meaningfulness_plot(
        meaningfulness,
        image_root / f"{source_stem}_topology_meaningfulness.png",
    )
    log(f"summary={summary_path}")
    log(f"within={within_path}")
    log(f"covariance={covariance_path}")
    log(f"between={between_path}")
    log(f"meaningfulness={meaningfulness_path}")
    report(meaningfulness.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    report(
        f"DONE module=persistence.variation sources={','.join(config.sources)} "
        f"genres={combined_summary['genre_top'].nunique()} "
        f"bootstraps={config.bootstrap_replicates} data_dir={data_root} image_dir={image_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
