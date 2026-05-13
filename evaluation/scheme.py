#!/usr/bin/env python3
#
# Orchestrate compression and evaluation stages for ABT-manifold experiments.
#

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import yaml

from compression.fixed_cs import DEFAULT_CONFIG as FIXED_CS_DEFAULTS
from compression.fixed_cs import run as run_fixed_cs
from compression.manifold import DEFAULT_CONFIG as MANIFOLD_DEFAULTS
from compression.manifold import run as run_manifold
from compression.topo_cs import DEFAULT_CONFIG as TOPO_CS_DEFAULTS
from compression.topo_cs import run as run_topo_cs
from compression.train_utils import load_config, merge_config
from evaluation.linear import DEFAULT_CONFIG as LINEAR_DEFAULTS
from evaluation.topology import DEFAULT_CONFIG as TOPOLOGY_DEFAULTS


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "scheme.yaml"
DEFAULT_CONFIG = {
    "source": "anchor",
    "dataset": "fma_small_mel",
    "anchor_dir": "representation/data",
    "compression_dir": "compression/data",
    "stages": ["compress", "topology", "linear"],
    "quiet_warnings": False,
    "schemes": [
        {
            "family": "fixed_cs",
            "methods": ["gaussian", "srht", "pca"],
            "ratios": [5, 10, 20, 30, 40, 60, 80, 100],
            "seeds": [0],
        }
    ],
    "linear": {},
    "topology": {},
    "persistence": {},
}


def report(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABT-manifold compression/evaluation schemes.")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    stages = {str(stage) for stage in config["stages"]}
    invalid_stages = stages - {"compress", "linear", "topology", "persistence"}
    if invalid_stages:
        raise ValueError(f"unsupported stages: {', '.join(sorted(invalid_stages))}")
    for scheme in config["schemes"]:
        family = str(scheme["family"])
        if family not in {"fixed_cs", "manifold", "topo_cs"}:
            raise ValueError(f"unsupported scheme family: {family}")


def method_filters(config: dict) -> list[str]:
    methods: list[str] = []
    for scheme in config["schemes"]:
        methods.extend(str(method) for method in scheme.get("methods", []))
    return sorted(set(methods))


def write_yaml(directory: Path, name: str, payload: dict) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_subprocess(command: list[str], quiet_warnings: bool) -> None:
    report("RUN " + " ".join(command))
    stderr = subprocess.DEVNULL if quiet_warnings else None
    subprocess.run(command, check=True, stderr=stderr)


def run_compression(config: dict) -> list[Path]:
    anchor_dir = Path(str(config["anchor_dir"])).expanduser().resolve()
    output_dir = Path(str(config["compression_dir"])).expanduser().resolve()
    written: list[Path] = []
    for scheme in config["schemes"]:
        family = str(scheme["family"])
        if family == "fixed_cs":
            fixed_config = merge_config(
                FIXED_CS_DEFAULTS,
                {
                    "source": str(config["source"]),
                    "dataset": str(config["dataset"]),
                    **{key: value for key, value in scheme.items() if key != "family"},
                },
            )
            written.extend(run_fixed_cs(fixed_config, anchor_dir, output_dir))
        elif family == "manifold":
            manifold_config = merge_config(
                MANIFOLD_DEFAULTS,
                {
                    "source": str(config["source"]),
                    "dataset": str(config["dataset"]),
                    **{key: value for key, value in scheme.items() if key != "family"},
                },
            )
            written.extend(run_manifold(manifold_config, anchor_dir, output_dir))
        elif family == "topo_cs":
            topo_config = merge_config(
                TOPO_CS_DEFAULTS,
                {
                    "source": str(config["source"]),
                    "dataset": str(config["dataset"]),
                    **{key: value for key, value in scheme.items() if key != "family"},
                },
            )
            written.extend(run_topo_cs(topo_config, anchor_dir, output_dir))
        else:
            raise ValueError(f"unsupported family: {family}")
    return written


def persistence_projection_sources(config: dict) -> list[str]:
    ratios = [int(value) for value in config.get("persistence", {}).get("ratios", [20])]
    sources: list[str] = []
    for scheme in config["schemes"]:
        seeds = [int(value) for value in scheme.get("seeds", [0])]
        for method in [str(value) for value in scheme.get("methods", [])]:
            if method == "pca":
                method_seeds = [0]
            else:
                method_seeds = seeds
            for ratio in ratios:
                for seed in method_seeds:
                    sources.append(f"{method}_r{ratio:03d}_s{seed:02d}")
    return sources


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    validate_config(config)
    if bool(config.get("quiet_warnings", False)):
        warnings.filterwarnings("ignore")

    stages = [str(stage) for stage in config["stages"]]
    methods = method_filters(config)
    compression_dir = Path(str(config["compression_dir"])).expanduser().resolve()
    report(f"START module=evaluation.scheme stages={','.join(stages)} methods={','.join(methods)} config={args.config}")

    if "compress" in stages:
        written = run_compression(config)
        report(f"stage=compress files={len(written)} output_dir={compression_dir}")

    with tempfile.TemporaryDirectory(prefix="spiky-scheme-") as temp_name:
        temp_dir = Path(temp_name)
        if "topology" in stages:
            topology_config = merge_config(
                TOPOLOGY_DEFAULTS,
                {
                    "dataset": str(config["dataset"]),
                    "anchor_dir": str(config["anchor_dir"]),
                    "reference_source": str(config["source"]),
                    "include_methods": methods,
                    **dict(config.get("topology", {})),
                },
            )
            topology_path = write_yaml(temp_dir, "topology.yaml", topology_config)
            run_subprocess(
                [
                    sys.executable,
                    "-m",
                    "evaluation.topology",
                    "-d",
                    compression_dir.as_posix(),
                    "-c",
                    topology_path.as_posix(),
                ],
                bool(config.get("quiet_warnings", False)),
            )
            report("stage=topology status=done")

        if "linear" in stages:
            linear_overrides = dict(config.get("linear", {}))
            classifiers = [str(value) for value in linear_overrides.pop("classifiers", [])]
            if not classifiers:
                classifiers = [str(linear_overrides.get("classifier", LINEAR_DEFAULTS["classifier"]))]
            for classifier in classifiers:
                linear_config = merge_config(
                    LINEAR_DEFAULTS,
                    {
                        "classifier": classifier,
                        "source": str(config["source"]),
                        "dataset": str(config["dataset"]),
                        "anchor_dir": str(config["anchor_dir"]),
                        "include_methods": methods,
                        **linear_overrides,
                    },
                )
                linear_path = write_yaml(temp_dir, f"linear_{classifier}.yaml", linear_config)
                run_subprocess(
                    [sys.executable, "-m", "evaluation.linear", "-d", compression_dir.as_posix(), "-c", linear_path.as_posix()],
                    bool(config.get("quiet_warnings", False)),
                )
                report(f"stage=linear classifier={classifier} status=done")

        if "persistence" in stages:
            persistence_config = merge_config(
                {
                    "sources": [str(config["source"])],
                    "reference_source": str(config["source"]),
                    "basis_comparison_sources": [str(config["source"]), str(config["source"])],
                    "projection_sources": persistence_projection_sources(config),
                    "representation_data_dir": str(config["anchor_dir"]),
                    "compression_data_dir": str(config["compression_dir"]),
                    "projection_split": "training",
                    "max_tracks_per_genre": 96,
                    "n_perm": 64,
                    "max_homology_dim": 1,
                },
                dict(config.get("persistence", {})),
            )
            persistence_path = write_yaml(temp_dir, "persistence.yaml", persistence_config)
            run_subprocess(
                [
                    sys.executable,
                    "-m",
                    "persistence.diagrams",
                    "--mel-dir",
                    "preprocess/data/fma_small_mel",
                    "-c",
                    persistence_path.as_posix(),
                ],
                bool(config.get("quiet_warnings", False)),
            )
            report("stage=persistence status=done")

    report("DONE module=evaluation.scheme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
