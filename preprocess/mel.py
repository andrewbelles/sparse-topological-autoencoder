#!/usr/bin/env python3
# 
# mel.py  Andrew Belles  April 10th 2026 
# 
# Generates Mel-Spectrogram for 64 bins at 22.05 kHz sampling rate.
# Log-scales and per-track min-max scales for downstream representation experiments.
# 

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from PIL import Image


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "mel.yaml"


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 22_050
    n_mels: int = 64
    n_fft: int = 2_048
    hop_length: int = 512
    batch_size: int = 32
    power: float = 2.0
    top_db: float = 80.0
    device: str = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an FMA audio directory into batched log-normalized mel-spectrogram tensors."
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        required=True,
        help="Input directory containing FMA mp3 files, for example preprocess/data/fma_small.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    parser.add_argument(
        "--sample-images",
        action="store_true",
        help="Write one mel preview image per genre to preprocess/images/ using preprocess/data/tracks.csv.",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_file():
        return config_path

    example_path = config_path.with_name(f"{config_path.stem}.example{config_path.suffix}")
    if example_path.is_file():
        return example_path

    raise FileNotFoundError(f"missing config: {config_path}")


def load_config(config_path: Path) -> MelConfig:
    raw_config = {}
    resolved_path = resolve_config_path(config_path)

    with resolved_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {resolved_path}")

    raw_config.update(loaded)

    return MelConfig(
        sample_rate=int(raw_config.get("sample_rate", 22_050)),
        n_mels=int(raw_config.get("n_mels", 64)),
        n_fft=int(raw_config.get("n_fft", 2_048)),
        hop_length=int(raw_config.get("hop_length", 512)),
        batch_size=int(raw_config.get("batch_size", 32)),
        power=float(raw_config.get("power", 2.0)),
        top_db=float(raw_config.get("top_db", 80.0)),
        device=str(raw_config.get("device", "auto")),
    )


def validate_config(config: MelConfig) -> None:
    if config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.n_mels <= 0:
        raise ValueError("n_mels must be positive")
    if config.n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if config.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.power <= 0:
        raise ValueError("power must be positive")
    if config.top_db <= 0:
        raise ValueError("top_db must be positive")


def resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(config_device)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def report(message: str) -> None:
    print(message, flush=True)


def chunked(items: list[Path], size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def list_audio_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.rglob("*.mp3") if path.is_file())


def build_output_dir(data_dir: Path) -> Path:
    return data_dir.parent / f"{data_dir.name}_mel"


def build_image_dir(output_dir: Path) -> Path:
    return Path(__file__).resolve().parent / "images" / output_dir.name


def build_manifest_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "all": output_dir / "manifest_all.csv",
        "training": output_dir / "manifest_training.csv",
        "validation": output_dir / "manifest_validation.csv",
        "test": output_dir / "manifest_test.csv",
    }


def mel_frame_count(num_samples: int, hop_length: int) -> int:
    return max(1, 1 + (num_samples // hop_length))


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() != 2:
        raise ValueError(f"expected waveform with shape [channels, time], got {tuple(waveform.shape)}")

    if waveform.size(0) == 1:
        return waveform.squeeze(0)

    return waveform.mean(dim=0)


def load_audio(audio_path: Path, target_sample_rate: int) -> torch.Tensor:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        message_lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        message = message_lines[-1] if message_lines else f"ffmpeg exit code {result.returncode}"
        raise RuntimeError(f"ffmpeg failed for {audio_path}: {message}")

    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).clone()

    if waveform.numel() == 0:
        raise RuntimeError(f"ffmpeg decoded zero samples from {audio_path}")

    return waveform


def load_and_resample_batch(
    audio_paths: list[Path],
    target_sample_rate: int,
) -> tuple[torch.Tensor | None, list[Path], list[int], list[tuple[Path, str]]]:
    waveforms: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    relative_paths: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    for audio_path in audio_paths:
        try:
            mono_waveform = load_audio(audio_path, target_sample_rate)
        except Exception as exc:
            skipped.append((audio_path, str(exc)))
            continue

        waveforms.append(mono_waveform)
        sample_lengths.append(int(mono_waveform.numel()))
        relative_paths.append(audio_path)

    if not waveforms:
        return None, [], [], skipped

    max_length = max(sample_lengths)
    batch = torch.stack(
        [F.pad(waveform, (0, max_length - waveform.numel())) for waveform in waveforms],
        dim=0,
    )

    return batch, relative_paths, sample_lengths, skipped


def find_tracks_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir.parent / "tracks.csv",
        data_dir.parent / "fma_metadata" / "tracks.csv",
        data_dir.parent.parent / "fma_metadata" / "tracks.csv",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("could not find tracks.csv for genre previews")


def load_track_metadata(tracks_csv: Path) -> dict[int, dict[str, str]]:
    track_metadata: dict[int, dict[str, str]] = {}

    with tracks_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header_top = next(reader)
        header_bottom = next(reader)

        column_index = {
            (top.strip(), bottom.strip()): index
            for index, (top, bottom) in enumerate(zip(header_top, header_bottom))
        }

        split_index = column_index[("set", "split")]
        subset_index = column_index[("set", "subset")]
        genre_index = column_index[("track", "genre_top")]
        duration_index = column_index[("track", "duration")]
        title_index = column_index[("track", "title")]

        for row in reader:
            if not row:
                continue

            try:
                track_id = int(row[0])
            except ValueError:
                continue

            track_metadata[track_id] = {
                "split": row[split_index].strip(),
                "subset": row[subset_index].strip(),
                "genre_top": row[genre_index].strip(),
                "duration": row[duration_index].strip(),
                "title": row[title_index].strip(),
            }

    return track_metadata


def load_top_genres(tracks_csv: Path) -> dict[int, str]:
    return {
        track_id: values["genre_top"]
        for track_id, values in load_track_metadata(tracks_csv).items()
        if values["genre_top"]
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def save_mel_image(mel_tensor: torch.Tensor, image_path: Path) -> None:
    view = torch.flip(mel_tensor.float(), dims=[0])
    view = view - view.min()
    max_value = float(view.max())
    if max_value > 0:
        view = view / max_value

    image = Image.fromarray((view * 255).to(torch.uint8).numpy(), mode="L")
    image.save(image_path)


def log_normalize_mels(mel_batch: torch.Tensor, top_db: float) -> torch.Tensor:
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=top_db).to(mel_batch.device)
    mel_db = db_transform(mel_batch.clamp_min(1e-10))
    mel_min = mel_db.amin(dim=(-2, -1), keepdim=True)
    mel_max = mel_db.amax(dim=(-2, -1), keepdim=True)
    return (mel_db - mel_min) / (mel_max - mel_min).clamp_min(1e-6)


def write_sample_images(data_dir: Path, output_dir: Path) -> Path:
    tracks_csv = find_tracks_csv(data_dir)
    top_genres = load_top_genres(tracks_csv)
    image_dir = build_image_dir(output_dir)

    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    samples_by_genre: dict[str, Path] = {}

    for tensor_path in sorted(output_dir.rglob("*.pt")):
        try:
            track_id = int(tensor_path.stem)
        except ValueError:
            continue

        genre = top_genres.get(track_id)
        if genre and genre not in samples_by_genre:
            samples_by_genre[genre] = tensor_path

    for genre, tensor_path in sorted(samples_by_genre.items()):
        mel_tensor = torch.load(tensor_path, map_location="cpu")
        image_path = image_dir / f"{slugify(genre)}.png"
        save_mel_image(mel_tensor, image_path)

    return image_dir


def write_manifests(data_dir: Path, output_dir: Path) -> dict[str, Path]:
    tracks_csv = find_tracks_csv(data_dir)
    track_metadata = load_track_metadata(tracks_csv)
    manifest_paths = build_manifest_paths(output_dir)
    fieldnames = [
        "track_id",
        "split",
        "subset",
        "genre_top",
        "duration",
        "title",
        "audio_path",
        "mel_path",
    ]

    manifest_rows: dict[str, list[dict[str, str]]] = {
        "all": [],
        "training": [],
        "validation": [],
        "test": [],
    }

    for tensor_path in sorted(output_dir.rglob("*.pt")):
        try:
            track_id = int(tensor_path.stem)
        except ValueError:
            continue

        metadata = track_metadata.get(track_id)
        if metadata is None:
            continue

        relative_audio_path = (
            data_dir.parent / data_dir.name / tensor_path.relative_to(output_dir)
        ).with_suffix(".mp3").relative_to(data_dir.parent)
        relative_mel_path = tensor_path.relative_to(data_dir.parent)
        split = metadata["split"] or "unknown"
        row = {
            "track_id": str(track_id),
            "split": split,
            "subset": metadata["subset"],
            "genre_top": metadata["genre_top"],
            "duration": metadata["duration"],
            "title": metadata["title"],
            "audio_path": relative_audio_path.as_posix(),
            "mel_path": relative_mel_path.as_posix(),
        }

        manifest_rows["all"].append(row)
        if split in manifest_rows:
            manifest_rows[split].append(row)

    for split_name, manifest_path in manifest_paths.items():
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows[split_name])

    return manifest_paths


def convert_directory(data_dir: Path, config: MelConfig) -> tuple[Path, int, int, dict[str, Path]]:
    audio_files = list_audio_files(data_dir)
    if not audio_files:
        raise FileNotFoundError(f"no mp3 files found under {data_dir}")

    output_dir = build_output_dir(data_dir)
    staging_dir = output_dir.parent / f".{output_dir.name}.tmp"

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)

    device = resolve_device(config.device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=config.power,
        center=True,
    ).to(device)

    processed = 0
    skipped = 0

    report(
        f"START data_dir={data_dir} output_dir={output_dir} files={len(audio_files)}"
    )

    try:
        with torch.inference_mode():
            for batch_index, audio_batch in enumerate(chunked(audio_files, config.batch_size), start=1):
                batch_waveforms, batch_paths, sample_lengths, skipped_batch = load_and_resample_batch(
                    audio_batch,
                    config.sample_rate,
                )
                skipped += len(skipped_batch)

                for skipped_path, reason in skipped_batch:
                    log(f"[mel] skipped {skipped_path}: {reason}")

                if batch_waveforms is None:
                    continue

                mel_batch = mel_transform(batch_waveforms.to(device))
                mel_batch = log_normalize_mels(mel_batch, config.top_db).cpu()

                for mel_tensor, source_path, num_samples in zip(mel_batch, batch_paths, sample_lengths):
                    relative_path = source_path.relative_to(data_dir).with_suffix(".pt")
                    output_path = staging_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    frame_count = mel_frame_count(num_samples, config.hop_length)
                    torch.save(mel_tensor[:, :frame_count].contiguous(), output_path)
                    processed += 1

                if batch_index == 1 or batch_index % 10 == 0 or processed + skipped == len(audio_files):
                    log(
                        f"[mel] processed={processed} skipped={skipped} total={len(audio_files)}"
                    )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.rename(output_dir)
        manifest_paths = write_manifests(data_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return output_dir, processed, skipped, manifest_paths


def run_sample_image_export(data_dir: Path) -> Path:
    return write_sample_images(data_dir, build_output_dir(data_dir))


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {data_dir}")

    output_dir, processed, skipped, manifest_paths = convert_directory(data_dir, config)

    if args.sample_images:
        image_dir = run_sample_image_export(data_dir)
        log(f"[mel] wrote sample images to {image_dir}")
        report(
            f"DONE output_dir={output_dir} processed={processed} skipped={skipped} manifest={manifest_paths['all']} image_dir={image_dir}"
        )
        return 0

    report(
        f"DONE output_dir={output_dir} processed={processed} skipped={skipped} manifest={manifest_paths['all']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
