#!/usr/bin/env python3
#
# Fixed CS and linear baselines over the selected ABT anchor manifold.
#

import argparse
from pathlib import Path

import numpy as np

from compression.common import (
    SPLITS,
    build_output_frame,
    dim_to_ratio,
    feature_matrix,
    fit_pca,
    gaussian_projection,
    load_anchor_splits,
    method_parquet_path,
    pca_project,
    ratio_to_dim,
    report,
    srht_project,
    write_method_parquet,
)
from compression.train_utils import load_config


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "compression.yaml"
DEFAULT_ANCHOR_DIR = Path("representation/data")
DEFAULT_OUTPUT_DIR = Path("compression/data")
METHODS = ("gaussian", "srht", "pca")
DEFAULT_CONFIG = {
    "seed": 0,
    "source": "anchor",
    "anchor_source": "anchor",
    "dataset": "fma_small_mel",
    "methods": ["gaussian", "srht", "pca"],
    "ratios": [5, 10, 20, 30, 40, 60, 80, 100],
    "dims": [],
    "convenient_dims": [16, 32, 64, 96, 128, 192, 256],
    "seeds": [0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed CS baselines over the ABT anchor manifold.")
    parser.add_argument("-a", "--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    invalid = sorted(set(str(method) for method in config["methods"]) - set(METHODS))
    if invalid:
        raise ValueError(f"unsupported fixed CS methods: {', '.join(invalid)}")
    if not config.get("ratios"):
        raise ValueError("ratios must be non-empty")
    if not config.get("seeds"):
        raise ValueError("seeds must be non-empty")


def output_dims(input_dim: int, config: dict) -> dict[int, int]:
    explicit_dims = [int(value) for value in config.get("dims", [])]
    ratios = [int(value) for value in config["ratios"]]
    if explicit_dims:
        return {dim_to_ratio(input_dim, dim): min(input_dim, max(1, dim)) for dim in explicit_dims}
    convenient_dims = [int(value) for value in config.get("convenient_dims", [])]
    return {ratio: ratio_to_dim(input_dim, ratio, convenient_dims) for ratio in ratios}


def run(config: dict, anchor_dir: Path = DEFAULT_ANCHOR_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    validate_config(config)
    source = str(config.get("source", config.get("anchor_source", "anchor")))
    dataset = str(config["dataset"])
    methods = [str(method) for method in config["methods"]]
    seeds = [int(value) for value in config["seeds"]]
    base_seed = int(config["seed"])

    split_frames = load_anchor_splits(anchor_dir, source, dataset)
    split_features = {split: feature_matrix(frame) for split, frame in split_frames.items()}
    input_dim = int(split_features["training"].shape[1])
    dims_by_ratio = output_dims(input_dim, config)
    max_dim = max(dims_by_ratio.values())

    report(
        f"compression.fixed_cs methods={','.join(methods)} ratios={','.join(str(r) for r in dims_by_ratio)} "
        f"seeds={','.join(str(s) for s in seeds)} source={source} input_dim={input_dim}"
    )

    written: list[Path] = []
    for method in methods:
        rows = []
        pca_mean = None
        pca_components = None
        if method == "pca":
            pca_mean, pca_components = fit_pca(split_features["training"])

        for ratio, m_dim in dims_by_ratio.items():
            method_seeds = [0] if method == "pca" else seeds
            for seed in method_seeds:
                projection_seed = base_seed + seed * 100_000 + ratio
                if method == "gaussian":
                    projection = gaussian_projection(input_dim, m_dim, projection_seed)

                for split in SPLITS:
                    matrix = split_features[split]
                    if method == "gaussian":
                        features = (matrix @ projection).astype("float32", copy=False)
                    elif method == "srht":
                        features = srht_project(matrix, m_dim, projection_seed)
                    elif method == "pca":
                        if pca_mean is None or pca_components is None:
                            raise RuntimeError("PCA basis was not fit")
                        features = pca_project(matrix, pca_mean, pca_components, m_dim)
                    else:
                        raise ValueError(f"unsupported method: {method}")

                    rows.append(
                        build_output_frame(
                            split_frames[split],
                            features,
                            method=method,
                            family="fixed_cs" if method in {"gaussian", "srht"} else "linear",
                            dataset=dataset,
                            source=source,
                            split=split,
                            ratio_percent=ratio,
                            input_dim=input_dim,
                            seed=seed,
                            max_dim=max_dim,
                        )
                    )

        output_path = method_parquet_path(output_dir, method, source, dataset)
        written.append(write_method_parquet(rows, output_path))
        report(f"wrote method={method} rows={sum(len(row) for row in rows)} path={output_path}")

    return written


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    run(config, args.anchor_dir.expanduser().resolve(), args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
