#!/usr/bin/env python3
#
# diagrams.py  Andrew Belles  May 6th, 2026
#
# Compute persistence diagrams for the selected contrastive anchor and projected embeddings.
#

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-topology-persistence-matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from ripser import ripser

from preprocess.mel import load_audio
from persistence.visualizations import (
    save_persistence_diagram,
    save_residual_persistence_diagram,
    save_residual_source_comparison,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "persistence.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_RAW_DIR = Path("preprocess/data/fma_small")
SOURCES = ("raw", "mel")


@dataclass(frozen=True)
class PersistenceConfig:
    sources: tuple[str, ...] = ("anchor",)
    reference_source: str = "anchor"
    basis_comparison_sources: tuple[str, str] = ("anchor", "anchor")
    genres: tuple[str, ...] = ()
    max_tracks_per_genre: int = 96
    seed: int = 7
    max_homology_dim: int = 1
    n_perm: int = 64
    metric: str = "euclidean"
    raw_sample_rate: int = 22_050
    raw_vector_length: int = 4_096
    mel_time_bins: int = 128
    standardize_rows: bool = True
    null_method: str = "feature_shuffle"
    null_replicates: int = 4
    residual_lifetime_quantile: float = 0.95
    projection_sources: tuple[str, ...] = ()
    representation_data_dir: str = "representation/data"
    compression_data_dir: str = "compression/data"
    projection_split: str = "training"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def report(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-genre persistence diagrams for the contrastive anchor and projected embeddings."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Raw FMA audio directory. Defaults to {DEFAULT_RAW_DIR}.",
    )
    parser.add_argument(
        "--mel-dir",
        type=Path,
        default=DEFAULT_MEL_DIR,
        help=f"Mel tensor directory with manifest_all.csv. Defaults to {DEFAULT_MEL_DIR}.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_file():
        return config_path

    example_path = config_path.with_name(f"{config_path.stem}.example{config_path.suffix}")
    if example_path.is_file():
        return example_path

    raise FileNotFoundError(f"missing config: {config_path}")


def _as_tuple(raw_value: object, default: Iterable[str]) -> tuple[str, ...]:
    if raw_value is None:
        return tuple(default)
    if isinstance(raw_value, str):
        return (raw_value,)
    if isinstance(raw_value, list | tuple):
        return tuple(str(value) for value in raw_value)
    raise ValueError("expected a string or list")


def load_config(config_path: Path) -> PersistenceConfig:
    resolved_path = resolve_config_path(config_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError(f"config must be a mapping: {resolved_path}")

    sources = _as_tuple(raw_config.get("sources"), ("anchor",))

    return PersistenceConfig(
        sources=sources,
        reference_source=str(raw_config.get("reference_source", sources[0] if sources else "anchor")),
        basis_comparison_sources=tuple(_as_tuple(raw_config.get("basis_comparison_sources"), ("anchor", "anchor"))[:2]),
        genres=_as_tuple(raw_config.get("genres"), ()),
        max_tracks_per_genre=int(raw_config.get("max_tracks_per_genre", 96)),
        seed=int(raw_config.get("seed", 7)),
        max_homology_dim=int(raw_config.get("max_homology_dim", 1)),
        n_perm=int(raw_config.get("n_perm", 64)),
        metric=str(raw_config.get("metric", "euclidean")),
        raw_sample_rate=int(raw_config.get("raw_sample_rate", 22_050)),
        raw_vector_length=int(raw_config.get("raw_vector_length", 4_096)),
        mel_time_bins=int(raw_config.get("mel_time_bins", 128)),
        standardize_rows=bool(raw_config.get("standardize_rows", True)),
        null_method=str(raw_config.get("null_method", "feature_shuffle")),
        null_replicates=int(raw_config.get("null_replicates", 4)),
        residual_lifetime_quantile=float(raw_config.get("residual_lifetime_quantile", 0.95)),
        projection_sources=_as_tuple(raw_config.get("projection_sources"), ()),
        representation_data_dir=str(raw_config.get("representation_data_dir", "representation/data")),
        compression_data_dir=str(raw_config.get("compression_data_dir", "compression/data")),
        projection_split=str(raw_config.get("projection_split", "training")),
    )


def validate_config(config: PersistenceConfig) -> None:
    if config.max_tracks_per_genre <= 1:
        raise ValueError("max_tracks_per_genre must be greater than 1")
    if config.max_homology_dim < 0:
        raise ValueError("max_homology_dim must be non-negative")
    if config.n_perm <= 1:
        raise ValueError("n_perm must be greater than 1")
    if config.raw_sample_rate <= 0:
        raise ValueError("raw_sample_rate must be positive")
    if config.raw_vector_length <= 0:
        raise ValueError("raw_vector_length must be positive")
    if config.mel_time_bins <= 0:
        raise ValueError("mel_time_bins must be positive")
    if config.null_method not in {"feature_shuffle", "row_feature_shuffle", "gaussian"}:
        raise ValueError("null_method must be one of: feature_shuffle, row_feature_shuffle, gaussian")
    if config.null_replicates <= 0:
        raise ValueError("null_replicates must be positive")
    if not 0.0 < config.residual_lifetime_quantile < 1.0:
        raise ValueError("residual_lifetime_quantile must be in (0, 1)")
    if config.projection_split not in {"training", "validation", "test"}:
        raise ValueError("projection_split must be one of: training, validation, test")
    if len(config.basis_comparison_sources) != 2:
        raise ValueError("basis_comparison_sources must contain exactly two sources")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().lower())
    return slug.strip("_") or "unknown"


def read_manifest(mel_dir: Path) -> list[dict[str, str]]:
    manifest_path = mel_dir / "manifest_all.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing mel manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_manifest_rows(
    rows: list[dict[str, str]],
    genres: tuple[str, ...],
) -> list[dict[str, str]]:
    if not genres:
        return [row for row in rows if row.get("genre_top")]

    wanted = {genre.lower() for genre in genres}
    return [row for row in rows if row.get("genre_top", "").lower() in wanted]


def sample_rows_by_genre(
    rows: list[dict[str, str]],
    max_tracks_per_genre: int,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    rng = np.random.default_rng(seed)
    rows_by_genre: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        rows_by_genre[row["genre_top"]].append(row)

    sampled: dict[str, list[dict[str, str]]] = {}
    for genre, genre_rows in sorted(rows_by_genre.items()):
        ordered = sorted(genre_rows, key=lambda row: int(row["track_id"]))
        if len(ordered) > max_tracks_per_genre:
            indices = np.sort(rng.choice(len(ordered), size=max_tracks_per_genre, replace=False))
            ordered = [ordered[index] for index in indices]
        sampled[genre] = ordered

    return sampled


def fixed_length_waveform(audio_path: Path, sample_rate: int, vector_length: int) -> torch.Tensor:
    waveform = load_audio(audio_path, sample_rate).float()
    waveform = waveform - waveform.mean()
    waveform = waveform / waveform.std().clamp_min(1e-6)
    return F.adaptive_avg_pool1d(waveform.view(1, 1, -1), output_size=vector_length).flatten()


def fixed_size_mel(mel_path: Path, time_bins: int) -> torch.Tensor:
    mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
    if mel.ndim != 2:
        raise ValueError(f"expected 2D mel tensor at {mel_path}, got shape {tuple(mel.shape)}")
    return F.adaptive_avg_pool1d(mel.unsqueeze(0), output_size=time_bins).squeeze(0).flatten()


def read_embedding_frame(source: str, mel_dir: Path, config: PersistenceConfig, root_dir: Path) -> pd.DataFrame:
    embedding_path = root_dir / f"{source}_{mel_dir.name}_{config.projection_split}.parquet"
    if embedding_path.is_file():
        frame = pd.read_parquet(embedding_path).copy()
    else:
        match = re.match(r"^(?P<method>.+)_r(?P<ratio>\d+)_s(?P<seed>\d+)$", source)
        if match is None:
            raise FileNotFoundError(f"missing embedding parquet: {embedding_path}")
        method = match.group("method")
        ratio = int(match.group("ratio"))
        seed = int(match.group("seed"))
        method_path = root_dir / f"{method}_{config.reference_source}_{mel_dir.name}.parquet"
        if not method_path.is_file():
            raise FileNotFoundError(f"missing method parquet: {method_path}")
        frame = pd.read_parquet(method_path).copy()
        frame = frame[
            (frame["split"] == config.projection_split)
            & (frame["ratio_percent"].astype(int) == ratio)
            & (frame["seed"].astype(int) == seed)
        ].copy()
        m_dim = int(frame["m_dim"].iloc[0]) if not frame.empty else 0
        feature_columns = sorted(column for column in frame.columns if column.startswith("embedding_"))[:m_dim]
        frame = frame[[column for column in frame.columns if not column.startswith("embedding_")] + feature_columns]
    if frame.empty:
        raise ValueError(f"no embedding rows matched source={source}")
    if "track_id" not in frame.columns:
        raise ValueError(f"embedding parquet is missing track_id: {embedding_path}")
    return frame.set_index(frame["track_id"].astype(int), drop=False)


def projection_vector(row: dict[str, str], projection_frame: pd.DataFrame) -> torch.Tensor:
    track_id = int(row["track_id"])
    if track_id not in projection_frame.index:
        raise KeyError(f"track_id={track_id} not present in projection parquet")
    projection_row = projection_frame.loc[track_id]
    embedding_columns = sorted(column for column in projection_frame.columns if column.startswith("embedding_"))
    if not embedding_columns:
        raise ValueError("projection parquet has no embedding columns")
    return torch.tensor(projection_row[embedding_columns].to_numpy(dtype="float32"), dtype=torch.float32)


def standardize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    mean = point_cloud.mean(axis=1, keepdims=True)
    std = point_cloud.std(axis=1, keepdims=True)
    return (point_cloud - mean) / np.maximum(std, 1e-6)


def resolve_relative_data_path(base_dir: Path, manifest_path: str) -> Path:
    relative_path = Path(manifest_path)
    if relative_path.parts and relative_path.parts[0] == base_dir.name:
        relative_path = Path(*relative_path.parts[1:])
    return base_dir / relative_path


def build_point_cloud(
    rows: list[dict[str, str]],
    source: str,
    raw_dir: Path,
    mel_dir: Path,
    config: PersistenceConfig,
    projection_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, int]:
    vectors: list[np.ndarray] = []
    skipped = 0

    for row in rows:
        try:
            if source == "raw":
                audio_path = resolve_relative_data_path(raw_dir, row["audio_path"])
                vector = fixed_length_waveform(audio_path, config.raw_sample_rate, config.raw_vector_length)
            elif source == "mel":
                mel_path = resolve_relative_data_path(mel_dir, row["mel_path"])
                vector = fixed_size_mel(mel_path, config.mel_time_bins)
            elif projection_frame is not None:
                vector = projection_vector(row, projection_frame)
            else:
                raise ValueError(f"unsupported source: {source}")
        except Exception as exc:
            skipped += 1
            log(f"[persistence] skipped source={source} track_id={row.get('track_id')} reason={exc}")
            continue

        vectors.append(vector.numpy().astype("float32", copy=False))

    if len(vectors) < 2:
        raise ValueError(f"need at least 2 valid vectors for source={source}")

    point_cloud = np.stack(vectors, axis=0)
    if config.standardize_rows:
        point_cloud = standardize_point_cloud(point_cloud)
    return point_cloud.astype("float32", copy=False), skipped


def compute_diagrams(point_cloud: np.ndarray, config: PersistenceConfig) -> list[np.ndarray]:
    n_perm = min(config.n_perm, point_cloud.shape[0])
    ripser_kwargs: dict[str, object] = {
        "maxdim": config.max_homology_dim,
        "metric": config.metric,
    }
    if n_perm < point_cloud.shape[0]:
        ripser_kwargs["n_perm"] = n_perm

    result = ripser(point_cloud, **ripser_kwargs)
    return [np.asarray(diagram, dtype="float64") for diagram in result["dgms"]]


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    if diagram.size == 0:
        return np.empty((0, 2), dtype="float64")
    return np.asarray(diagram[np.isfinite(diagram).all(axis=1)], dtype="float64")


def diagram_lifetimes(diagram: np.ndarray) -> np.ndarray:
    finite = finite_diagram(diagram)
    if finite.size == 0:
        return np.empty(0, dtype="float64")
    return finite[:, 1] - finite[:, 0]


def concatenate_diagram_replicates(diagram_replicates: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not diagram_replicates:
        return []

    combined: list[np.ndarray] = []
    for homology_dim in range(len(diagram_replicates[0])):
        parts = [finite_diagram(diagrams[homology_dim]) for diagrams in diagram_replicates]
        non_empty = [part for part in parts if len(part) > 0]
        if non_empty:
            combined.append(np.concatenate(non_empty, axis=0))
        else:
            combined.append(np.empty((0, 2), dtype="float64"))
    return combined


def build_null_point_cloud(point_cloud: np.ndarray, rng: np.random.Generator, method: str) -> np.ndarray:
    if method == "feature_shuffle":
        null_cloud = point_cloud.copy()
        for column_index in range(null_cloud.shape[1]):
            rng.shuffle(null_cloud[:, column_index])
        return null_cloud

    if method == "row_feature_shuffle":
        null_cloud = point_cloud.copy()
        for row_index in range(null_cloud.shape[0]):
            rng.shuffle(null_cloud[row_index])
        return null_cloud

    if method == "gaussian":
        means = point_cloud.mean(axis=0, keepdims=True)
        stds = point_cloud.std(axis=0, keepdims=True)
        return rng.normal(means, np.maximum(stds, 1e-6), size=point_cloud.shape).astype("float32")

    raise ValueError(f"unsupported null_method: {method}")


def compute_null_diagrams(point_cloud: np.ndarray, config: PersistenceConfig, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    null_replicates: list[list[np.ndarray]] = []

    for _ in range(config.null_replicates):
        null_cloud = build_null_point_cloud(point_cloud, rng, config.null_method)
        null_replicates.append(compute_diagrams(null_cloud.astype("float32", copy=False), config))

    return concatenate_diagram_replicates(null_replicates)


def compute_lifetime_thresholds(null_diagrams: list[np.ndarray], config: PersistenceConfig) -> dict[int, float]:
    thresholds: dict[int, float] = {}

    for homology_dim, diagram in enumerate(null_diagrams):
        lifetimes = diagram_lifetimes(diagram)
        if lifetimes.size == 0:
            thresholds[homology_dim] = 0.0
        else:
            thresholds[homology_dim] = float(np.quantile(lifetimes, config.residual_lifetime_quantile))

    return thresholds


def compute_residual_diagrams(
    diagrams: list[np.ndarray],
    thresholds: dict[int, float],
) -> list[np.ndarray]:
    residuals: list[np.ndarray] = []

    for homology_dim, diagram in enumerate(diagrams):
        finite = finite_diagram(diagram)
        if finite.size == 0:
            residuals.append(np.empty((0, 2), dtype="float64"))
            continue

        lifetimes = finite[:, 1] - finite[:, 0]
        residuals.append(finite[lifetimes > thresholds.get(homology_dim, 0.0)])

    return residuals


def write_thresholds_csv(source: str, thresholds: dict[int, float], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source": source,
            "homology_dim": homology_dim,
            "lifetime_threshold": threshold,
        }
        for homology_dim, threshold in sorted(thresholds.items())
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def save_diagrams(diagrams: list[np.ndarray], output_path: Path, metadata: dict[str, str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"H{index}": diagram for index, diagram in enumerate(diagrams)}
    payload.update({f"meta_{key}": np.array(value) for key, value in metadata.items()})
    np.savez_compressed(output_path, **payload)
    return output_path


def run_source(
    source: str,
    rows_by_genre: dict[str, list[dict[str, str]]],
    raw_dir: Path,
    mel_dir: Path,
    config: PersistenceConfig,
    data_root: Path,
    image_root: Path,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    all_rows = [row for rows in rows_by_genre.values() for row in rows]
    projection_frame = None
    if source not in SOURCES:
        root_dir = (
            Path(config.compression_data_dir).expanduser()
            if source in config.projection_sources
            else Path(config.representation_data_dir).expanduser()
        )
        projection_frame = read_embedding_frame(source, mel_dir, config, root_dir)

    background_point_cloud, background_skipped = build_point_cloud(
        all_rows,
        source,
        raw_dir,
        mel_dir,
        config,
        projection_frame=projection_frame,
    )
    null_diagrams = compute_null_diagrams(background_point_cloud, config, seed=config.seed + 10_000)
    null_thresholds = compute_lifetime_thresholds(null_diagrams, config)
    null_diagram_path = data_root / f"{source}_null_background_diagrams.npz"
    null_threshold_path = data_root / f"{source}_null_lifetime_thresholds.csv"

    save_diagrams(
        null_diagrams,
        null_diagram_path,
        {
            "source": source,
            "genre": "null_background",
            "points": str(background_point_cloud.shape[0]),
            "dimensions": str(background_point_cloud.shape[1]),
            "skipped": str(background_skipped),
            "null_method": config.null_method,
            "null_replicates": str(config.null_replicates),
            "residual_lifetime_quantile": str(config.residual_lifetime_quantile),
        },
    )
    write_thresholds_csv(source, null_thresholds, null_threshold_path)
    report(
        f"source={source} null_background points={background_point_cloud.shape[0]} "
        f"dimensions={background_point_cloud.shape[1]} method={config.null_method} "
        f"thresholds={null_threshold_path}"
    )

    for genre, rows in rows_by_genre.items():
        point_cloud, skipped = build_point_cloud(rows, source, raw_dir, mel_dir, config, projection_frame=projection_frame)
        diagrams = compute_diagrams(point_cloud, config)
        residual_diagrams = compute_residual_diagrams(diagrams, null_thresholds)
        genre_slug = slugify(genre)
        diagram_path = data_root / f"{source}_{genre_slug}_diagrams.npz"
        residual_path = data_root / f"{source}_{genre_slug}_residual_diagrams.npz"
        image_path = image_root / f"{source}_{genre_slug}_persistence.png"
        residual_image_path = image_root / f"{source}_{genre_slug}_residual_persistence.png"

        metadata = {
            "source": source,
            "genre": genre,
            "points": str(point_cloud.shape[0]),
            "dimensions": str(point_cloud.shape[1]),
            "skipped": str(skipped),
        }
        save_diagrams(diagrams, diagram_path, metadata)
        save_diagrams(
            residual_diagrams,
            residual_path,
            {
                **metadata,
                "null_method": config.null_method,
                "residual_lifetime_quantile": str(config.residual_lifetime_quantile),
                "null_threshold_path": null_threshold_path.as_posix(),
            },
        )
        save_persistence_diagram(
            diagrams,
            title=f"Persistence - {source.title()} - {genre}",
            output_path=image_path,
        )
        save_residual_persistence_diagram(
            diagrams,
            residual_diagrams,
            null_diagrams,
            null_thresholds,
            title=f"Residual Persistence - {source.title()} - {genre}",
            output_path=residual_image_path,
        )
        results.append(
            {
                "source": source,
                "genre": genre,
                "points": point_cloud.shape[0],
                "dimensions": point_cloud.shape[1],
                "skipped": skipped,
                "diagram": diagram_path,
                "image": image_path,
                "residual_diagram": residual_path,
                "residual_image": residual_image_path,
                "diagrams": diagrams,
                "residual_diagrams": residual_diagrams,
                "null_diagrams": null_diagrams,
                "thresholds": null_thresholds,
            }
        )
        report(
            f"source={source} genre={genre} points={point_cloud.shape[0]} "
            f"dimensions={point_cloud.shape[1]} skipped={skipped} image={image_path} "
            f"residual={residual_image_path}"
        )

    return results


def save_source_comparison_panels(
    results: list[dict[str, object]],
    image_root: Path,
    reference_source: str,
    skip_pairs: set[tuple[str, str]] | None = None,
) -> list[Path]:
    results_by_genre: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    written: list[Path] = []
    skip_pairs = skip_pairs or set()

    for result in results:
        results_by_genre[str(result["genre"])][str(result["source"])] = result

    for genre, source_records in sorted(results_by_genre.items()):
        if reference_source not in source_records:
            continue

        for source, record in sorted(source_records.items()):
            if source == reference_source:
                continue
            if (reference_source, source) in skip_pairs or (source, reference_source) in skip_pairs:
                continue
            output_path = image_root / f"{source}_{slugify(genre)}_{reference_source}_residual_comparison.png"
            written.append(
                save_residual_source_comparison(
                    {reference_source: source_records[reference_source], source: record},
                    genre=genre,
                    title="Reference vs Source Residual Persistence",
                    output_path=output_path,
                )
            )

    return written


def save_basis_comparison_panels(
    results: list[dict[str, object]],
    image_root: Path,
    left_source: str,
    right_source: str,
) -> list[Path]:
    if left_source == right_source:
        return []
    results_by_genre: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    written: list[Path] = []

    for result in results:
        results_by_genre[str(result["genre"])][str(result["source"])] = result

    for genre, source_records in sorted(results_by_genre.items()):
        if left_source not in source_records or right_source not in source_records:
            continue

        output_path = image_root / f"{left_source}_{right_source}_{slugify(genre)}_residual_comparison.png"
        written.append(
            save_residual_source_comparison(
                {
                    left_source: source_records[left_source],
                    right_source: source_records[right_source],
                },
                genre=genre,
                title="Anchor Basis Residual Persistence",
                output_path=output_path,
            )
        )

    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    raw_dir = args.raw_dir.expanduser().resolve()
    mel_dir = args.mel_dir.expanduser().resolve()
    data_root = Path(__file__).resolve().parent / "data"
    image_root = Path(__file__).resolve().parent / "images"

    log(f"START module=persistence raw_dir={raw_dir} mel_dir={mel_dir} config={args.config}")

    rows = filter_manifest_rows(read_manifest(mel_dir), config.genres)
    if config.projection_sources or any(source not in SOURCES for source in config.sources):
        rows = [row for row in rows if row.get("split") == config.projection_split]
    rows_by_genre = sample_rows_by_genre(rows, config.max_tracks_per_genre, config.seed)
    if not rows_by_genre:
        raise ValueError("no manifest rows matched the configured genres")

    selected_genres = ", ".join(sorted(rows_by_genre))
    report(
        f"sources={','.join(config.sources)} genres={selected_genres} "
        f"max_tracks_per_genre={config.max_tracks_per_genre} config={resolve_config_path(args.config)}"
    )

    all_results: list[dict[str, object]] = []
    for source in (*config.sources, *config.projection_sources):
        all_results.extend(run_source(source, rows_by_genre, raw_dir, mel_dir, config, data_root, image_root))

    basis_comparison_images = save_basis_comparison_panels(
        all_results,
        image_root,
        config.basis_comparison_sources[0],
        config.basis_comparison_sources[1],
    )
    comparison_images = save_source_comparison_panels(
        all_results,
        image_root,
        config.reference_source,
        skip_pairs={(config.basis_comparison_sources[0], config.basis_comparison_sources[1])},
    )

    log(
        f"DONE module=persistence diagrams={len(all_results)} "
        f"basis_comparisons={len(basis_comparison_images)} comparisons={len(comparison_images)} "
        f"image_dir={image_root} data_dir={data_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
