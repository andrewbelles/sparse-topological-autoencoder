#!/usr/bin/env python3
#
# umap.py  Andrew Belles  May 8th, 2026
#
# UMAP visualization for the selected contrastive anchor manifold.
#

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fma-representation-umap-matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/fma-representation-umap-numba")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "umap.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
SPLITS = ("training", "validation", "test")
GENRE_ORDER = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]
DEFAULT_CONFIG = {
    "seed": 7,
    "anchor_source": "anchor",
    "sources": ["anchor"],
    "dataset": "fma_small_mel",
    "splits": ["training", "validation", "test"],
    "n_neighbors": 30,
    "min_dist": 0.1,
    "metric": "cosine",
    "standardize_features": True,
    "max_points": 0,
    "point_size": 12,
    "alpha": 0.82,
    "figure_size": [8, 6],
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a 2D UMAP projection of the selected anchor manifold.")
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
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Output directory for UMAP image and coordinates. Defaults to representation/images.",
    )
    return parser.parse_args()


def validate_config(config: dict) -> None:
    splits = [str(split) for split in config["splits"]]
    invalid_splits = sorted(set(splits) - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"unsupported splits: {', '.join(invalid_splits)}")
    if not configured_sources(config):
        raise ValueError("at least one source must be configured")
    if int(config["n_neighbors"]) <= 1:
        raise ValueError("n_neighbors must be greater than 1")
    if not 0.0 <= float(config["min_dist"]) <= 1.0:
        raise ValueError("min_dist must be in [0, 1]")
    if int(config["max_points"]) < 0:
        raise ValueError("max_points must be non-negative")


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame.columns if column.startswith("embedding_"))
    if not columns:
        raise ValueError("anchor parquet has no embedding columns")
    return columns


def configured_sources(config: dict) -> list[str]:
    sources = config.get("sources")
    if sources is None:
        sources = [config.get("anchor_source", "anchor")]
    if isinstance(sources, str):
        sources = [sources]
    return [str(source) for source in sources if str(source)]


def load_source_frame(anchor_dir: Path, config: dict, source: str) -> tuple[pd.DataFrame, list[str]]:
    dataset = str(config["dataset"])
    frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    for split in [str(value) for value in config["splits"]]:
        path = anchor_dir / f"{source}_{dataset}_{split}.parquet"
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
    combined["source"] = source
    return combined, feature_columns


def sample_frame(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    max_points = int(config["max_points"])
    if max_points <= 0 or len(frame) <= max_points:
        return frame

    sampled_groups = []
    for _, group in frame.groupby("genre_top", sort=False):
        sampled_groups.append(
            group.sample(
                n=max(1, int(round(max_points * len(group) / len(frame)))),
                random_state=int(config["seed"]),
                replace=False,
            )
        )

    return (
        pd.concat(sampled_groups, ignore_index=True)
        .sample(frac=1.0, random_state=int(config["seed"]))
        .head(max_points)
        .reset_index(drop=True)
    )


def sample_keys(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    sampled = sample_frame(frame, config)
    keys = [column for column in ["track_id", "split"] if column in sampled.columns]
    if not keys:
        return sampled.reset_index().rename(columns={"index": "__row_id"})[["__row_id"]]
    return sampled[keys].drop_duplicates().reset_index(drop=True)


def apply_sample_keys(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if "__row_id" in keys.columns:
        return frame.reset_index().rename(columns={"index": "__row_id"}).merge(keys, on="__row_id", how="inner")
    key_columns = list(keys.columns)
    return frame.merge(keys, on=key_columns, how="inner")


def feature_matrix(frame: pd.DataFrame, feature_columns: list[str], standardize: bool) -> np.ndarray:
    matrix = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    if standardize:
        mean = matrix.mean(axis=0, keepdims=True)
        std = np.maximum(matrix.std(axis=0, keepdims=True), 1e-6)
        matrix = (matrix - mean) / std
    return matrix.astype("float32", copy=False)


def compute_umap(features: np.ndarray, config: dict) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=int(config["n_neighbors"]),
        min_dist=float(config["min_dist"]),
        metric=str(config["metric"]),
        random_state=int(config["seed"]),
    )
    return reducer.fit_transform(features).astype("float32", copy=False)


def source_label(source: str) -> str:
    return source.replace("_", " ").title()


def ordered_hue(frame: pd.DataFrame) -> list[str]:
    present = [genre for genre in GENRE_ORDER if genre in set(frame["genre_top"])]
    present.extend(sorted(genre for genre in frame["genre_top"].unique() if genre not in set(present)))
    return present


def save_umap_plot(frame: pd.DataFrame, config: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    figure_size = config["figure_size"]
    sources = list(frame["source"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(sources),
        figsize=(float(figure_size[0]), float(figure_size[1])),
        constrained_layout=True,
        squeeze=False,
    )

    hue_order = ordered_hue(frame)
    for index, source in enumerate(sources):
        ax = axes[0, index]
        source_frame = frame[frame["source"] == source]
        sns.scatterplot(
            data=source_frame,
            x="UMAP-1",
            y="UMAP-2",
            hue="genre_top",
            hue_order=hue_order,
            s=float(config["point_size"]),
            alpha=float(config["alpha"]),
            linewidth=0,
            palette="tab10",
            ax=ax,
            legend=index == len(sources) - 1,
        )
        ax.set_title(f"UMAP - {source_label(str(source))}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        if ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1.0), title="Genre", frameon=True)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def output_stem(sources: list[str], dataset: str) -> str:
    if len(sources) == 1:
        return f"{sources[0]}_{dataset}_umap"
    return f"{'_'.join(sources)}_{dataset}_umap"


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    validate_config(config)

    anchor_dir = args.anchor_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report(f"START module=representation.umap anchor_dir={anchor_dir} config={args.config}")

    sources = configured_sources(config)
    output_frames: list[pd.DataFrame] = []
    reference_keys: pd.DataFrame | None = None

    for source in sources:
        frame, feature_columns = load_source_frame(anchor_dir, config, source)
        if reference_keys is None:
            reference_keys = sample_keys(frame, config)
        frame = apply_sample_keys(frame, reference_keys)
        features = feature_matrix(frame, feature_columns, bool(config["standardize_features"]))
        coordinates = compute_umap(features, config)

        output = frame[
            [column for column in ["track_id", "genre_top", "split", "source"] if column in frame.columns]
        ].copy()
        output["UMAP-1"] = coordinates[:, 0]
        output["UMAP-2"] = coordinates[:, 1]
        output_frames.append(output)
        log(f"source={source} rows={len(output)} features={features.shape[1]}")

    combined = pd.concat(output_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(config["dataset"])
    stem = output_stem(sources, dataset)
    csv_path = output_dir / f"{stem}.csv"
    image_path = output_dir / f"{stem}.png"
    combined.to_csv(csv_path, index=False)
    save_umap_plot(combined, config, image_path)

    log(f"rows={len(combined)} sources={','.join(sources)} coordinates={csv_path} image={image_path}")
    report(f"DONE module=representation.umap image={image_path} coordinates={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
