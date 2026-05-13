#!/usr/bin/env python3
#
# extract.py  Andrew Belles  May 8th, 2026
#
# Export track-level embeddings from trained Audio Barlow Twins encoders.
#

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compression.train_utils import load_config, resolve_device, set_seed
from representation.audio import AudioCNNEncoder, crop_or_pad, load_manifest, resolve_relative_data_path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "extract.yaml"
DEFAULT_MEL_DIR = Path("preprocess/data/fma_small_mel")
DEFAULT_CONFIG = {
    "device": "cuda",
    "seed": 19,
    "checkpoint_dir": "representation/checkpoints",
    "checkpoints": [],
    "splits": ["training", "validation", "test"],
    "crop_frames": 256,
    "crops_per_track": 5,
    "batch_size": 64,
    "normalize_embeddings": True,
}


def report(message: str) -> None:
    print(message, flush=True)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export track-level Audio Barlow embeddings.")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=DEFAULT_MEL_DIR,
        help=f"Mel tensor directory with manifests. Defaults to {DEFAULT_MEL_DIR}.",
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
        default=Path(__file__).resolve().parent / "data",
        help="Output directory for representation parquets. Defaults to representation/data.",
    )
    return parser.parse_args()


def discover_checkpoints(config: dict) -> list[Path]:
    configured = [Path(str(value)).expanduser() for value in config.get("checkpoints", [])]
    if configured:
        return configured
    checkpoint_dir = Path(str(config["checkpoint_dir"])).expanduser()
    return sorted(checkpoint_dir.glob("barlow_*.pt"))


def load_encoder(checkpoint_path: Path, device: torch.device) -> tuple[AudioCNNEncoder, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = payload["model"]
    encoder = AudioCNNEncoder(
        embedding_dim=int(payload["embedding_dim"]),
        base_channels=int(model_config["base_channels"]),
        dropout=float(model_config["dropout"]),
    ).to(device)

    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    return encoder, payload


def deterministic_crops(mel: torch.Tensor, crop_frames: int, crop_count: int) -> torch.Tensor:
    if crop_count <= 1:
        return crop_or_pad(mel, crop_frames, random_crop=False).unsqueeze(0)

    if mel.size(1) <= crop_frames:
        crop = crop_or_pad(mel, crop_frames, random_crop=False)
        return crop.unsqueeze(0).repeat(crop_count, 1, 1)

    starts = torch.linspace(0, mel.size(1) - crop_frames, steps=crop_count).round().long().tolist()
    return torch.stack([mel[:, start : start + crop_frames] for start in starts], dim=0)


@torch.no_grad()
def encode_track(
    encoder: AudioCNNEncoder,
    mel_path: Path,
    crop_frames: int,
    crop_count: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    mel = torch.load(mel_path, map_location="cpu", weights_only=True).float()
    if mel.ndim != 2:
        raise ValueError(f"expected 2D mel tensor at {mel_path}, got {tuple(mel.shape)}")
    crops = deterministic_crops(mel, crop_frames, crop_count).unsqueeze(1)
    embeddings: list[torch.Tensor] = []
    for start in range(0, crops.size(0), batch_size):
        batch = crops[start : start + batch_size].to(device)
        embeddings.append(encoder(batch).detach().cpu())
    return torch.cat(embeddings, dim=0).mean(dim=0).numpy().astype("float32", copy=False)


def write_split_embeddings(
    encoder: AudioCNNEncoder,
    payload: dict,
    data_dir: Path,
    split: str,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> Path:
    frame = load_manifest(data_dir, split)
    records: list[dict[str, object]] = []
    for row_index, row in frame.iterrows():
        row_dict = row.to_dict()
        mel_path = resolve_relative_data_path(data_dir, str(row_dict["mel_path"]))
        embedding = encode_track(
            encoder,
            mel_path,
            crop_frames=int(config["crop_frames"]),
            crop_count=int(config["crops_per_track"]),
            batch_size=int(config["batch_size"]),
            device=device,
        )
        if bool(config.get("normalize_embeddings", True)):
            norm = np.linalg.norm(embedding)
            if norm > 0.0:
                embedding = (embedding / norm).astype("float32", copy=False)
        record = {key: value for key, value in row_dict.items() if key != "Unnamed: 0"}
        record["compression_method"] = str(payload["source_name"])
        record["compression_ratio_percent"] = np.nan
        record["input_dim"] = np.nan
        record["projection_dim"] = int(embedding.shape[0])
        for dim_index, value in enumerate(embedding.tolist()):
            record[f"embedding_{dim_index:04d}"] = float(value)
        records.append(record)

        if row_index == 0 or (row_index + 1) % 500 == 0 or row_index + 1 == len(frame):
            report(f"extract source={payload['source_name']} split={split} rows={row_index + 1}/{len(frame)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{payload['source_name']}_{data_dir.name}_{split}.parquet"
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)
    return output_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DEFAULT_CONFIG)
    device = resolve_device(str(config["device"]))
    set_seed(int(config["seed"]))

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_paths = discover_checkpoints(config)
    if not checkpoint_paths:
        raise FileNotFoundError("no Barlow checkpoints found")

    report(f"START module=representation.extract data_dir={data_dir} device={device} config={args.config}")
    written: list[Path] = []
    for checkpoint_path in checkpoint_paths:
        encoder, payload = load_encoder(checkpoint_path, device)
        for split in [str(value) for value in config["splits"]]:
            output_path = write_split_embeddings(encoder, payload, data_dir, split, config, output_dir, device)
            written.append(output_path)
            log(f"wrote source={payload['source_name']} split={split} path={output_path}")

    report(f"DONE module=representation.extract files={len(written)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
