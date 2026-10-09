#!/usr/bin/env python3
"""Precompute causal DreamDojo LAM memories for a LeRobot v2 dataset.

Each cache row is aligned to the destination observation. With temporal stride
``s``, an episode with frames ``o[0:T]`` has shape ``(T, 32)`` and obeys::

    latent[k * s] = DreamDojoLAM(o[(k - 1) * s], o[k * s]), valid[k * s] = True
    latent[i] = 0, valid[i] = False  (i not divisible by s)

for ``k >= 1``. Thus stride one is the historical adjacent-frame cache, while
stride four contains exactly ``0->4, 4->8, ...`` and no short tail transition.

Consequently a policy at observation ``o[i]`` can consume rows ``<= i`` without
ever encoding a future frame.  The script intentionally does not import OpenPI's
model or data-loader code.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CACHE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
LATENT_DIM = 32
LATENT_STD_EPSILON = 1e-6
LATENT_NORMALIZATION_DEFINITION = "per_dimension_valid_transition_population"
CHECKPOINT_FORMAT = "dreamdojo.lightning"
LEGACY_CHECKPOINT_FORMAT = "cdlam.stage1.inference"
SUPPORTED_CHECKPOINT_FORMATS = frozenset({CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT})
CHECKPOINT_FORMAT_BY_SHA256 = {
    "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f": CHECKPOINT_FORMAT,
    "12ce318e48d5790fb3773f1027edd5a3babe8965027b4957e3ed62d5d19fc386": LEGACY_CHECKPOINT_FORMAT,
}
PREPROCESS_VERSION = "dreamdojo.lam400k.rgb_uint8.crop_4x3.resize_480x640.resize_240x320.v1"
SEQUENCE_WINDOW_CROP_VERSION = "sequence_window_crop_shared_v1"
ADJACENT_ALIGNMENT = "row_0_invalid_zero; row_i=z_mu(frame_i-1,frame_i), i>=1"
# Compatibility name for existing callers/tests of the stride-one contract.
ALIGNMENT = ADJACENT_ALIGNMENT
DEFAULT_DREAMDOJO_RUNTIME_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "dreamdojo" / "runtime"
)


def alignment_for_transition_frame_stride(stride: int) -> str:
    if stride <= 0:
        raise ValueError(f"transition frame stride must be positive, got {stride}")
    if stride == 1:
        return ADJACENT_ALIGNMENT
    return (
        f"row_{{k*{stride}}}=z_mu(frame_{{(k-1)*{stride}}},frame_{{k*{stride}}}), k>=1; "
        "all other rows invalid_zero"
    )


@dataclass(frozen=True)
class EpisodeSource:
    episode_index: int
    episode_chunk: int
    expected_length: int
    path: Path
    relative_path: str


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    info: dict[str, Any]
    episodes: tuple[EpisodeSource, ...]
    info_sha256: str
    episodes_sha256: str
    fingerprint: str


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    sha256: str
    size: int
    format: str
    step: int


@dataclass(frozen=True)
class CacheExpectation:
    checkpoint_sha256: str
    checkpoint_format: str
    checkpoint_step: int
    dataset_fingerprint: str
    image_key: str
    preprocess_version: str = PREPROCESS_VERSION
    storage_dtype: str | None = None
    transition_frame_stride: int = 1


@dataclass(frozen=True)
class CacheSummary:
    episode_index: int
    num_frames: int
    num_valid: int
    first_global_index: int
    last_global_index: int
    latent_dtype: str


@dataclass(frozen=True)
class PreparedEpisode:
    source: EpisodeSource
    episode: dict[str, np.ndarray]
    lam_frames: np.ndarray
    read_seconds: float
    crop_seconds: float
    preprocess_seconds: float


def sequence_preprocess_version(*, min_scale: float, probability: float, seed: int) -> str:
    """Return a cache identity that fully specifies sequence-level augmentation."""
    if min_scale == 1.0 or probability == 0.0:
        return PREPROCESS_VERSION
    return (
        f"{PREPROCESS_VERSION}+{SEQUENCE_WINDOW_CROP_VERSION}"
        f".min_scale={min_scale:.6f}.probability={probability:.6f}.seed={seed}"
    )


def apply_sequence_window_crop(
    frames: np.ndarray,
    *,
    episode_index: int,
    seed: int,
    min_scale: float,
    probability: float,
) -> np.ndarray:
    """Apply one deterministic crop window to every frame in an episode.

    The RNG depends only on ``seed`` and ``episode_index``. Running extraction
    separately for the external and wrist cameras therefore uses the same
    normalized crop decision and offsets for both views, while every frame of
    either view receives exactly the same spatial transform.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"sequence crop expects [T, H, W, 3] frames, got {frames.shape}")
    if min_scale == 1.0 or probability == 0.0:
        return frames
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode_index]))
    if float(rng.random()) >= probability:
        return frames
    height, width = frames.shape[1:3]
    scale = float(rng.uniform(min_scale, 1.0))
    crop_height = min(height, max(1, int(round(height * scale))))
    crop_width = min(width, max(1, int(round(width * scale))))
    top_fraction = float(rng.random())
    left_fraction = float(rng.random())
    top = int(round(top_fraction * (height - crop_height)))
    left = int(round(left_fraction * (width - crop_width)))
    return np.ascontiguousarray(frames[:, top : top + crop_height, left : left + crop_width])


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_npz(path: Path, arrays: dict[str, Any], *, compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            if compressed:
                np.savez_compressed(stream, **arrays)
            else:
                np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Atomically write an uncompressed array suitable for mmap at training time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def discover_dataset(dataset_root: Path, *, require_all_parquets: bool = True) -> DatasetSpec:
    root = dataset_root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"{root} is not a local LeRobot v2 dataset: expected meta/info.json and meta/episodes.jsonl"
        )

    info = load_json(info_path)
    codebase_version = str(info.get("codebase_version", ""))
    if not codebase_version.startswith("v2"):
        raise ValueError(f"expected LeRobot v2 metadata, got codebase_version={codebase_version!r}")
    data_path_template = info.get("data_path")
    if not isinstance(data_path_template, str):
        raise ValueError(f"missing string data_path in {info_path}")
    chunks_size = int(info.get("chunks_size", 1000))
    if chunks_size <= 0:
        raise ValueError(f"invalid chunks_size={chunks_size}")

    episode_rows: list[dict[str, Any]] = []
    with episodes_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {episodes_path}:{line_number}")
            episode_rows.append(row)
    if not episode_rows:
        raise ValueError(f"no episodes listed in {episodes_path}")

    episodes: list[EpisodeSource] = []
    seen: set[int] = set()
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        if episode_index in seen:
            raise ValueError(f"duplicate episode_index={episode_index} in {episodes_path}")
        seen.add(episode_index)
        episode_chunk = episode_index // chunks_size
        relative = data_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        source_path = root / relative
        if require_all_parquets and not source_path.is_file():
            raise FileNotFoundError(f"missing episode parquet: {source_path}")
        expected_length = int(row["length"])
        if expected_length <= 0:
            raise ValueError(f"episode {episode_index} has invalid length={expected_length}")
        episodes.append(
            EpisodeSource(
                episode_index=episode_index,
                episode_chunk=episode_chunk,
                expected_length=expected_length,
                path=source_path,
                relative_path=Path(relative).as_posix(),
            )
        )
    episodes.sort(key=lambda item: item.episode_index)

    declared_total = int(info.get("total_episodes", len(episodes)))
    if declared_total != len(episodes):
        raise ValueError(f"info.json declares {declared_total} episodes, but episodes.jsonl lists {len(episodes)}")
    declared_frames = int(info.get("total_frames", sum(source.expected_length for source in episodes)))
    episode_frames = sum(source.expected_length for source in episodes)
    if declared_frames != episode_frames:
        raise ValueError(f"info.json declares {declared_frames} frames, but episodes.jsonl lists {episode_frames}")
    info_sha256 = sha256_file(info_path)
    episodes_sha256 = sha256_file(episodes_path)
    fingerprint = sha256_text(f"{info_sha256}:{episodes_sha256}")
    return DatasetSpec(
        root=root,
        info=info,
        episodes=tuple(episodes),
        info_sha256=info_sha256,
        episodes_sha256=episodes_sha256,
        fingerprint=fingerprint,
    )


def _validate_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{label} must be 64 lowercase/uppercase hexadecimal characters")
    return normalized


def find_asset_manifest_entry(checkpoint: Path) -> tuple[Path, dict[str, Any]] | None:
    checkpoint = checkpoint.resolve()
    for parent in (checkpoint.parent, *checkpoint.parents):
        manifest_path = parent / "asset_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_json(manifest_path)
        assets = manifest.get("assets", [])
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
                continue
            if (manifest_path.parent / asset["path"]).resolve() == checkpoint:
                return manifest_path, asset
    return None


def resolve_checkpoint_identity(
    checkpoint: Path,
    *,
    explicit_sha256: str | None,
    verify_sha256: bool,
) -> CheckpointIdentity:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DreamDojo LAM checkpoint not found: {checkpoint}")
    size = checkpoint.stat().st_size
    manifest_match = find_asset_manifest_entry(checkpoint)
    manifest_sha: str | None = None
    checkpoint_format = CHECKPOINT_FORMAT
    checkpoint_step = -1
    if manifest_match is not None:
        manifest_path, asset = manifest_match
        if "bytes" in asset and int(asset["bytes"]) != size:
            raise ValueError(
                f"checkpoint size mismatch against {manifest_path}: expected {asset['bytes']}, found {size}"
            )
        if asset.get("release_sha256"):
            manifest_sha = _validate_sha256(str(asset["release_sha256"]), "manifest release_sha256")
        checkpoint_format = str(asset.get("checkpoint_format", checkpoint_format))
        checkpoint_step = int(asset.get("optimizer_step", -1))

    requested_sha = _validate_sha256(explicit_sha256, "--checkpoint-sha256") if explicit_sha256 else None
    if requested_sha and manifest_sha and requested_sha != manifest_sha:
        raise ValueError(f"--checkpoint-sha256={requested_sha} disagrees with release manifest sha256={manifest_sha}")
    expected_sha = requested_sha or manifest_sha
    if verify_sha256 or expected_sha is None:
        actual_sha = sha256_file(checkpoint)
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"checkpoint SHA256 mismatch: expected {expected_sha}, found {actual_sha}")
        expected_sha = actual_sha
    assert expected_sha is not None
    checkpoint_format = CHECKPOINT_FORMAT_BY_SHA256.get(expected_sha, checkpoint_format)
    if checkpoint_format not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(
            f"expected one of {sorted(SUPPORTED_CHECKPOINT_FORMATS)!r}, got checkpoint format {checkpoint_format!r}"
        )
    return CheckpointIdentity(
        path=checkpoint,
        sha256=expected_sha,
        size=size,
        format=checkpoint_format,
        step=checkpoint_step,
    )


def resolve_device(device_arg: str, local_rank_arg: int | None) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("DreamDojo LAM extraction requires CUDA; torch.cuda.is_available() is false")
    if device_arg != "auto":
        device = device_arg
    else:
        local_rank = local_rank_arg
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError(f"only CUDA devices are supported, got {device!r}")
    device_index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible")
    torch.cuda.set_device(device_index)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{device} does not report bfloat16 support")
    return f"cuda:{device_index}"


def load_dreamdojo_encoder(
    runtime_root: Path,
    checkpoint: CheckpointIdentity,
    device: str,
) -> tuple[Any, Callable[..., dict[str, Any]], Callable[[np.ndarray], np.ndarray], CheckpointIdentity]:
    import torch
    import torch.nn.functional as torch_f

    runtime_root = runtime_root.expanduser().resolve()
    if not (runtime_root / "external" / "lam" / "modules" / "lam.py").is_file():
        raise FileNotFoundError(f"vendored DreamDojo runtime not found under {runtime_root}")
    sys.path.insert(0, str(runtime_root))

    from dreamdojo_adapter import encode_full, official_lam_video_from_raw
    from external.lam.modules.blocks import SelfAttention
    from external.lam.modules.lam import LatentActionModel

    if not getattr(SelfAttention, "_dreamdojo_memory_sdpa_patched", False):

        def scaled_dot_product_attention(
            self,
            query,
            key,
            value,
            is_causal: bool = False,  # noqa: FBT001, FBT002
        ):
            return torch_f.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

        SelfAttention.scaled_dot_product_attention = scaled_dot_product_attention
        SelfAttention._dreamdojo_memory_sdpa_patched = True  # noqa: SLF001

    blob = torch.load(checkpoint.path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.format == CHECKPOINT_FORMAT:
        actual_format = CHECKPOINT_FORMAT
        actual_step = int(blob.get("global_step", checkpoint.step))
        model_state = blob.get("state_dict")
    else:
        actual_format = str(blob.get("format", ""))
        if actual_format != checkpoint.format:
            raise ValueError(f"checkpoint payload format mismatch: expected {checkpoint.format!r}, got {actual_format!r}")
        actual_step = int(blob.get("step", checkpoint.step))
        model_state = blob.get("model")
    if checkpoint.step >= 0 and actual_step != checkpoint.step:
        raise ValueError(f"checkpoint step mismatch: manifest={checkpoint.step}, payload={actual_step}")
    if not isinstance(model_state, dict):
        raise ValueError(f"DreamDojo LAM checkpoint format {actual_format!r} has no model state dictionary")
    cleaned_state = {(key.removeprefix("lam.")): value for key, value in model_state.items()}
    model = LatentActionModel(
        in_dim=3,
        model_dim=1024,
        latent_dim=LATENT_DIM,
        patch_size=16,
        enc_blocks=24,
        dec_blocks=24,
        num_heads=16,
    )
    model.load_state_dict(cleaned_state, strict=True, assign=True)

    # This utility only calls encode_full.  Removing decoder-only modules before
    # the CUDA transfer reduces the frozen model footprint by roughly half.
    for module_name in ("patch_up", "action_up", "decoder"):
        delattr(model, module_name)
    model = model.eval().requires_grad_(requires_grad=False).to(device=device, dtype=torch.bfloat16)
    del cleaned_state
    del model_state
    del blob
    gc.collect()

    resolved_identity = CheckpointIdentity(
        path=checkpoint.path,
        sha256=checkpoint.sha256,
        size=checkpoint.size,
        format=actual_format,
        step=actual_step,
    )
    return model, encode_full, official_lam_video_from_raw, resolved_identity


def _primitive_column(table: Any, name: str, dtype: Any) -> np.ndarray:
    column = table[name].combine_chunks()
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _resolve_external_image_path(path_value: str, dataset_root: Path, parquet_path: Path) -> Path:
    candidate = Path(path_value).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [dataset_root / candidate, parquet_path.parent / candidate]
    for item in candidates:
        if item.is_file():
            return item
    raise FileNotFoundError(f"embedded image has no bytes and path {path_value!r} could not be resolved")


@lru_cache(maxsize=None)
def _dataset_media_metadata(dataset_root: str) -> tuple[dict[str, Any], str | None]:
    info = load_json(Path(dataset_root) / "meta" / "info.json")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("dataset meta/info.json has no features object")
    video_path = info.get("video_path")
    if video_path is not None and not isinstance(video_path, str):
        raise ValueError("dataset meta/info.json video_path must be a string")
    return features, video_path


def _episode_video_path(source: EpisodeSource, dataset_root: Path, image_key: str) -> Path:
    features, template = _dataset_media_metadata(str(dataset_root))
    feature = features.get(image_key)
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise ValueError(
            f"episode parquet has no {image_key!r} column and dataset feature is not a video stream"
        )
    if not template:
        raise ValueError("video-backed dataset has no video_path template")
    path = dataset_root / template.format(
        episode_chunk=source.episode_chunk,
        episode_index=source.episode_index,
        video_key=image_key,
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing episode video: {path}")
    return path


def decode_video_episode(path: Path, expected_length: int) -> np.ndarray:
    """Decode one H.264 episode in presentation order as RGB uint8 frames."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"expected one video stream in {path}, got {len(container.streams.video)}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != expected_length:
        raise ValueError(f"decoded {len(frames)} frames from {path}, expected {expected_length}")
    if not frames:
        raise ValueError(f"decoded no frames from {path}")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"video {path} has inconsistent frame shapes {sorted(shapes)}")
    result = np.stack(frames, axis=0)
    if result.dtype != np.uint8 or result.ndim != 4 or result.shape[-1] != 3:
        raise ValueError(f"decoded video must be uint8 [T,H,W,3], got {result.dtype} {result.shape}")
    return result


def decode_image_entry(entry: Any, dataset_root: Path, parquet_path: Path) -> np.ndarray:
    payload: bytes | None = None
    path_value: str | None = None
    if isinstance(entry, dict):
        payload = entry.get("bytes")
        path_value = entry.get("path")
    elif isinstance(entry, bytes | bytearray | memoryview):
        payload = bytes(entry)
    else:
        raise TypeError(f"unsupported image parquet value: {type(entry).__name__}")
    if payload is None:
        if not path_value:
            raise ValueError("image entry contains neither bytes nor path")
        payload = _resolve_external_image_path(path_value, dataset_root, parquet_path).read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"decoded image must be HWC RGB, got {frame.shape}")
    return frame


def read_episode(source: EpisodeSource, dataset_root: Path, image_key: str) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source.path)
    available = set(parquet_file.schema_arrow.names)
    image_in_parquet = image_key in available
    required = {"frame_index", "episode_index", "index", "timestamp"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{source.path} is missing required columns: {missing}")
    columns = ["frame_index", "episode_index", "index", "timestamp"]
    if image_in_parquet:
        columns.insert(0, image_key)
    if "task_index" in available:
        columns.append("task_index")
    table = parquet_file.read(columns=columns)
    if table.num_rows != source.expected_length:
        raise ValueError(
            f"episode {source.episode_index}: metadata length={source.expected_length}, parquet rows={table.num_rows}"
        )

    frame_index = _primitive_column(table, "frame_index", np.int64)
    episode_index = _primitive_column(table, "episode_index", np.int64)
    global_index = _primitive_column(table, "index", np.int64)
    timestamp = _primitive_column(table, "timestamp", np.float32)
    task_index = (
        _primitive_column(table, "task_index", np.int64)
        if "task_index" in columns
        else np.full(table.num_rows, -1, dtype=np.int64)
    )
    if not np.all(episode_index == source.episode_index):
        unique = np.unique(episode_index).tolist()
        raise ValueError(f"{source.path} contains episode_index={unique}, expected only {source.episode_index}")

    order = np.argsort(frame_index, kind="stable")
    frame_index = frame_index[order]
    global_index = global_index[order]
    timestamp = timestamp[order]
    task_index = task_index[order]
    if len(frame_index) > 1 and not np.all(np.diff(frame_index) == 1):
        raise ValueError(f"episode {source.episode_index}: frame_index is not consecutive")
    if len(global_index) > 1 and not np.all(np.diff(global_index) == 1):
        raise ValueError(f"episode {source.episode_index}: global index is not consecutive")
    if not np.all(np.isfinite(timestamp)) or (len(timestamp) > 1 and np.any(np.diff(timestamp) < 0)):
        raise ValueError(f"episode {source.episode_index}: timestamps must be finite and nondecreasing")

    if image_in_parquet:
        image_values = table[image_key].combine_chunks().to_pylist()
        frames = [decode_image_entry(image_values[int(row)], dataset_root, source.path) for row in order]
        shapes = {frame.shape for frame in frames}
        if len(shapes) != 1:
            raise ValueError(f"episode {source.episode_index}: inconsistent image shapes {sorted(shapes)}")
        raw_frames = np.stack(frames, axis=0)
    else:
        video_path = _episode_video_path(source, dataset_root, image_key)
        raw_frames = decode_video_episode(video_path, source.expected_length)[order]
    return {
        "frames": raw_frames,
        "frame_index": frame_index,
        "global_index": global_index,
        "timestamp": timestamp,
        "task_index": task_index,
    }


def preprocess_frames(
    raw_frames: np.ndarray,
    preprocess: Callable[[np.ndarray], np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(raw_frames), chunk_size):
        chunk = preprocess(raw_frames[start : start + chunk_size])
        if chunk.dtype != np.uint8 or chunk.ndim != 4 or chunk.shape[1:] != (240, 320, 3):
            raise ValueError(f"official DreamDojo preprocessing returned unexpected {chunk.dtype} {chunk.shape}")
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def prepare_episode(
    source: EpisodeSource,
    *,
    dataset_root: Path,
    image_key: str,
    preprocess: Callable[[np.ndarray], np.ndarray],
    preprocess_chunk_size: int,
    sequence_augmentation_seed: int,
    sequence_window_crop_min_scale: float,
    sequence_window_crop_probability: float,
) -> PreparedEpisode:
    """Read and preprocess one episode without touching the CUDA encoder."""
    started = time.perf_counter()
    episode = read_episode(source, dataset_root, image_key)
    read_finished = time.perf_counter()
    augmented_frames = apply_sequence_window_crop(
        episode["frames"],
        episode_index=source.episode_index,
        seed=sequence_augmentation_seed,
        min_scale=sequence_window_crop_min_scale,
        probability=sequence_window_crop_probability,
    )
    crop_finished = time.perf_counter()
    lam_frames = preprocess_frames(augmented_frames, preprocess, preprocess_chunk_size)
    preprocess_finished = time.perf_counter()
    return PreparedEpisode(
        source=source,
        episode=episode,
        lam_frames=lam_frames,
        read_seconds=read_finished - started,
        crop_seconds=crop_finished - read_finished,
        preprocess_seconds=preprocess_finished - crop_finished,
    )


def iter_prepared_episodes(
    sources: Sequence[EpisodeSource],
    prepare: Callable[[EpisodeSource], PreparedEpisode],
    *,
    prefetch_episodes: int,
) -> Iterator[PreparedEpisode]:
    """Prepare episodes in order while keeping a bounded number of futures in flight."""
    if not 0 <= prefetch_episodes <= 10:
        raise ValueError(f"prefetch_episodes must be between 0 and 10, got {prefetch_episodes}")
    if prefetch_episodes == 0:
        for source in sources:
            yield prepare(source)
        return
    if not sources:
        return

    # Two workers help a shallow queue cover read/resize latency. For deeper
    # experimental queues, keep one producer so PyTorch's native worker pools
    # are not multiplied across every cache rank.
    worker_count = prefetch_episodes if prefetch_episodes <= 2 else 1
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="episode-prefetch")
    next_source = min(len(sources), prefetch_episodes + 1)
    futures = deque(executor.submit(prepare, source) for source in sources[:next_source])
    try:
        while futures:
            prepared = futures.popleft().result()
            yield prepared
            if next_source < len(sources):
                futures.append(executor.submit(prepare, sources[next_source]))
                next_source += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def encode_causal_latents(
    model: Any,
    encode_full: Callable[..., dict[str, Any]],
    lam_frames: np.ndarray,
    *,
    device: str,
    pair_batch_size: int,
    transition_frame_stride: int,
) -> np.ndarray:
    import torch

    if transition_frame_stride <= 0:
        raise ValueError("transition_frame_stride must be positive")
    num_frames = len(lam_frames)
    latent = np.zeros((num_frames, LATENT_DIM), dtype=np.float32)
    destination_indices = np.arange(transition_frame_stride, num_frames, transition_frame_stride, dtype=np.int64)
    if len(destination_indices) == 0:
        return latent
    with torch.inference_mode():
        for start in range(0, len(destination_indices), pair_batch_size):
            destination_batch = destination_indices[start : start + pair_batch_size]
            source_batch = destination_batch - transition_frame_stride
            # Every pair is within one episode and is causally aligned to the
            # destination cache row. For stride four these are 0->4, 4->8, ...
            pair_array = np.stack([lam_frames[source_batch], lam_frames[destination_batch]], axis=1)
            pair_tensor = torch.from_numpy(pair_array).to(device=device, dtype=torch.bfloat16)
            pair_tensor = pair_tensor / 255.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                encoded = encode_full(model, pair_tensor, sample=False, use_ckpt=False)
            z_mu = encoded["z_mu"].float()
            expected_shape = (len(destination_batch), LATENT_DIM)
            if tuple(z_mu.shape) != expected_shape:
                raise ValueError(f"DreamDojo returned z_mu shape={tuple(z_mu.shape)}, expected {expected_shape}")
            latent[destination_batch] = z_mu.cpu().numpy()
            del encoded
            del z_mu
            del pair_tensor
    if not np.all(np.isfinite(latent)):
        raise ValueError("DreamDojo produced non-finite latent values")
    return latent


def cache_path_for(output_root: Path, source: EpisodeSource) -> Path:
    return output_root / "episodes" / f"chunk-{source.episode_chunk:03d}" / f"episode_{source.episode_index:06d}.npz"


def scalar_string(value: str) -> np.ndarray:
    return np.asarray(value, dtype=np.str_)


def build_cache_arrays(
    source: EpisodeSource,
    episode: dict[str, np.ndarray],
    latent: np.ndarray,
    *,
    dataset: DatasetSpec,
    checkpoint: CheckpointIdentity,
    image_key: str,
    preprocess_version: str,
    storage_dtype: str,
    transition_frame_stride: int,
) -> dict[str, Any]:
    num_frames = len(latent)
    if transition_frame_stride <= 0:
        raise ValueError("transition_frame_stride must be positive")
    valid = np.zeros(num_frames, dtype=np.bool_)
    valid[transition_frame_stride::transition_frame_stride] = True
    source_frame_index = np.full(num_frames, -1, dtype=np.int64)
    source_global_index = np.full(num_frames, -1, dtype=np.int64)
    source_frame_index[valid] = episode["frame_index"][valid] - transition_frame_stride
    source_global_index[valid] = episode["global_index"][valid] - transition_frame_stride
    latent_dtype = np.float16 if storage_dtype == "float16" else np.float32
    stored_latent = latent.astype(latent_dtype, copy=False)
    stored_latent[~valid] = 0
    return {
        "schema_version": np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
        "latent": stored_latent,
        "valid": valid,
        "episode_index": np.asarray(source.episode_index, dtype=np.int64),
        "frame_index": episode["frame_index"].astype(np.int64, copy=False),
        "global_index": episode["global_index"].astype(np.int64, copy=False),
        "timestamp": episode["timestamp"].astype(np.float32, copy=False),
        "task_index": episode["task_index"].astype(np.int64, copy=False),
        "source_frame_index": source_frame_index,
        "source_global_index": source_global_index,
        "checkpoint_sha256": scalar_string(checkpoint.sha256),
        "checkpoint_format": scalar_string(checkpoint.format),
        "checkpoint_step": np.asarray(checkpoint.step, dtype=np.int64),
        "preprocess_version": scalar_string(preprocess_version),
        "alignment": scalar_string(alignment_for_transition_frame_stride(transition_frame_stride)),
        "transition_frame_stride": np.asarray(transition_frame_stride, dtype=np.int16),
        "image_key": scalar_string(image_key),
        "source_parquet": scalar_string(source.relative_path),
        "source_parquet_size": np.asarray(source.path.stat().st_size, dtype=np.int64),
        "dataset_fingerprint": scalar_string(dataset.fingerprint),
        "dataset_info_sha256": scalar_string(dataset.info_sha256),
        "dataset_episodes_sha256": scalar_string(dataset.episodes_sha256),
    }


def _npz_scalar(data: Any, key: str) -> Any:
    value = data[key]
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape={value.shape}")
    return value.item()


def read_cache_expectation(cache_path: Path) -> CacheExpectation:
    with np.load(cache_path, allow_pickle=False) as data:
        transition_frame_stride = (
            int(_npz_scalar(data, "transition_frame_stride")) if "transition_frame_stride" in data.files else 1
        )
        return CacheExpectation(
            checkpoint_sha256=str(_npz_scalar(data, "checkpoint_sha256")),
            checkpoint_format=str(_npz_scalar(data, "checkpoint_format")),
            checkpoint_step=int(_npz_scalar(data, "checkpoint_step")),
            dataset_fingerprint=str(_npz_scalar(data, "dataset_fingerprint")),
            image_key=str(_npz_scalar(data, "image_key")),
            preprocess_version=str(_npz_scalar(data, "preprocess_version")),
            storage_dtype=str(data["latent"].dtype),
            transition_frame_stride=transition_frame_stride,
        )


def validate_cache(
    cache_path: Path,
    source: EpisodeSource,
    expectation: CacheExpectation,
) -> CacheSummary:
    required = {
        "schema_version",
        "latent",
        "valid",
        "episode_index",
        "frame_index",
        "global_index",
        "timestamp",
        "task_index",
        "source_frame_index",
        "source_global_index",
        "checkpoint_sha256",
        "checkpoint_format",
        "checkpoint_step",
        "preprocess_version",
        "alignment",
        "image_key",
        "source_parquet",
        "source_parquet_size",
        "dataset_fingerprint",
    }
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"missing arrays: {missing}")
            if int(_npz_scalar(data, "schema_version")) != CACHE_SCHEMA_VERSION:
                raise ValueError(f"unsupported schema_version={_npz_scalar(data, 'schema_version')}")
            if int(_npz_scalar(data, "episode_index")) != source.episode_index:
                raise ValueError("episode_index mismatch")
            if str(_npz_scalar(data, "source_parquet")) != source.relative_path:
                raise ValueError("source_parquet mismatch")
            if int(_npz_scalar(data, "source_parquet_size")) != source.path.stat().st_size:
                raise ValueError("source parquet size changed")
            cache_stride = int(_npz_scalar(data, "transition_frame_stride")) if "transition_frame_stride" in data.files else 1
            if cache_stride <= 0:
                raise ValueError(f"transition_frame_stride must be positive, got {cache_stride}")
            scalar_checks = {
                "checkpoint_sha256": expectation.checkpoint_sha256,
                "checkpoint_format": expectation.checkpoint_format,
                "checkpoint_step": expectation.checkpoint_step,
                "dataset_fingerprint": expectation.dataset_fingerprint,
                "image_key": expectation.image_key,
                "preprocess_version": expectation.preprocess_version,
                "alignment": alignment_for_transition_frame_stride(cache_stride),
            }
            for key, expected in scalar_checks.items():
                actual = _npz_scalar(data, key)
                if actual != expected:
                    raise ValueError(f"{key} mismatch: expected {expected!r}, got {actual!r}")
            if cache_stride != expectation.transition_frame_stride:
                raise ValueError(
                    "transition_frame_stride mismatch: "
                    f"expected {expectation.transition_frame_stride}, got {cache_stride}"
                )

            latent = data["latent"]
            valid = data["valid"]
            frame_index = data["frame_index"]
            global_index = data["global_index"]
            timestamp = data["timestamp"]
            task_index = data["task_index"]
            source_frame_index = data["source_frame_index"]
            source_global_index = data["source_global_index"]
            num_frames = source.expected_length
            if latent.shape != (num_frames, LATENT_DIM):
                raise ValueError(f"latent shape={latent.shape}, expected {(num_frames, LATENT_DIM)}")
            if latent.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
                raise ValueError(f"latent dtype must be float16 or float32, got {latent.dtype}")
            if expectation.storage_dtype is not None and str(latent.dtype) != expectation.storage_dtype:
                raise ValueError(f"latent dtype mismatch: expected {expectation.storage_dtype}, got {latent.dtype}")
            vector_names = {
                "valid": valid,
                "frame_index": frame_index,
                "global_index": global_index,
                "timestamp": timestamp,
                "task_index": task_index,
                "source_frame_index": source_frame_index,
                "source_global_index": source_global_index,
            }
            for name, array in vector_names.items():
                if array.shape != (num_frames,):
                    raise ValueError(f"{name} shape={array.shape}, expected {(num_frames,)}")
            if valid.dtype != np.bool_:
                raise ValueError(f"valid dtype must be bool, got {valid.dtype}")
            expected_valid = np.zeros(num_frames, dtype=np.bool_)
            expected_valid[cache_stride::cache_stride] = True
            if not np.array_equal(valid, expected_valid):
                raise ValueError(
                    "valid mask must contain only complete transition-frame-stride endpoints"
                )
            if np.any(latent[~valid] != 0):
                raise ValueError("invalid latent rows must be exactly zero")
            if not np.all(np.isfinite(latent[valid])):
                raise ValueError("valid latent rows contain non-finite values")
            if num_frames > 1:
                if not np.all(np.diff(frame_index) == 1):
                    raise ValueError("frame_index is not consecutive")
                if not np.all(np.diff(global_index) == 1):
                    raise ValueError("global_index is not consecutive")
            expected_source_frame_index = np.full(num_frames, -1, dtype=np.int64)
            expected_source_global_index = np.full(num_frames, -1, dtype=np.int64)
            expected_source_frame_index[valid] = frame_index[valid] - cache_stride
            expected_source_global_index[valid] = global_index[valid] - cache_stride
            if not np.array_equal(source_frame_index, expected_source_frame_index):
                raise ValueError("source_frame_index does not identify the stride-separated source frame")
            if not np.array_equal(source_global_index, expected_source_global_index):
                raise ValueError("source_global_index does not identify the stride-separated source global row")
            if not np.all(np.isfinite(timestamp)) or (num_frames > 1 and np.any(np.diff(timestamp) < 0)):
                raise ValueError("timestamp is non-finite or decreasing")
            return CacheSummary(
                episode_index=source.episode_index,
                num_frames=num_frames,
                num_valid=int(valid.sum()),
                first_global_index=int(global_index[0]),
                last_global_index=int(global_index[-1]),
                latent_dtype=str(latent.dtype),
            )
    except Exception as error:
        raise ValueError(f"invalid cache {cache_path}: {error}") from error


def select_episodes(
    episodes: tuple[EpisodeSource, ...],
    *,
    episode_start: int | None,
    episode_stop: int | None,
) -> list[EpisodeSource]:
    return [
        source
        for source in episodes
        if (episode_start is None or source.episode_index >= episode_start)
        and (episode_stop is None or source.episode_index < episode_stop)
    ]


def resolve_rank_world(rank_arg: int | None, world_size_arg: int | None) -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0")) if rank_arg is None else rank_arg
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if world_size_arg is None else world_size_arg
    if world_size <= 0:
        raise ValueError("world size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must satisfy 0 <= rank < world_size, got rank={rank}, world_size={world_size}")
    return rank, world_size


def extract_command(args: argparse.Namespace) -> int:
    started = time.time()
    preprocess_version = sequence_preprocess_version(
        min_scale=args.sequence_window_crop_min_scale,
        probability=args.sequence_window_crop_probability,
        seed=args.sequence_augmentation_seed,
    )
    has_episode_filter = args.episode_start is not None or args.episode_stop is not None
    dataset = discover_dataset(args.dataset_root, require_all_parquets=not has_episode_filter)
    checkpoint_path = args.checkpoint
    checkpoint = resolve_checkpoint_identity(
        checkpoint_path,
        explicit_sha256=args.checkpoint_sha256,
        verify_sha256=args.verify_checkpoint_sha256,
    )
    rank, world_size = resolve_rank_world(args.rank, args.world_size)
    selected = select_episodes(
        dataset.episodes,
        episode_start=args.episode_start,
        episode_stop=args.episode_stop,
    )
    missing_selected = [source.path for source in selected if not source.path.is_file()]
    if missing_selected:
        raise FileNotFoundError(f"selected episode parquet is missing: {missing_selected[0]}")
    assigned = [source for ordinal, source in enumerate(selected) if ordinal % world_size == rank]
    if args.limit is not None:
        assigned = assigned[: args.limit]
    expectation = CacheExpectation(
        checkpoint_sha256=checkpoint.sha256,
        checkpoint_format=checkpoint.format,
        checkpoint_step=checkpoint.step,
        dataset_fingerprint=dataset.fingerprint,
        image_key=args.image_key,
        preprocess_version=preprocess_version,
        storage_dtype=args.storage_dtype,
        transition_frame_stride=args.transition_frame_stride,
    )

    pending: list[EpisodeSource] = []
    skipped = 0
    for source in assigned:
        cache_path = cache_path_for(args.output_root, source)
        if cache_path.is_file() and not args.overwrite:
            try:
                validate_cache(cache_path, source, expectation)
                skipped += 1
                continue
            except ValueError:
                if not args.repair:
                    raise
        pending.append(source)

    print(
        f"dataset={dataset.root} episodes={len(dataset.episodes)} selected={len(selected)} "
        f"rank={rank}/{world_size} assigned={len(assigned)} pending={len(pending)} skipped={skipped}"
    )
    print(
        f"checkpoint={checkpoint.path} sha256={checkpoint.sha256} format={checkpoint.format} "
        f"step={checkpoint.step} image_key={args.image_key} preprocess_version={preprocess_version}"
    )
    if args.dry_run:
        for source in pending[:10]:
            print(f"would extract episode={source.episode_index} source={source.relative_path}")
        if len(pending) > 10:
            print(f"... and {len(pending) - 10} more")
        return 0

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    total_frames = 0
    if pending:
        device = resolve_device(args.device, args.local_rank)
        model, encode_full, preprocess, checkpoint = load_dreamdojo_encoder(
            args.dreamdojo_runtime_root,
            checkpoint,
            device,
        )
        expectation = CacheExpectation(
            checkpoint_sha256=checkpoint.sha256,
            checkpoint_format=checkpoint.format,
            checkpoint_step=checkpoint.step,
            dataset_fingerprint=dataset.fingerprint,
            image_key=args.image_key,
            preprocess_version=preprocess_version,
            storage_dtype=args.storage_dtype,
            transition_frame_stride=args.transition_frame_stride,
        )
        stage_seconds = {"read": 0.0, "crop": 0.0, "preprocess": 0.0, "encode": 0.0, "write": 0.0}
        prepare = partial(
            prepare_episode,
            dataset_root=dataset.root,
            image_key=args.image_key,
            preprocess=preprocess,
            preprocess_chunk_size=args.preprocess_chunk_size,
            sequence_augmentation_seed=args.sequence_augmentation_seed,
            sequence_window_crop_min_scale=args.sequence_window_crop_min_scale,
            sequence_window_crop_probability=args.sequence_window_crop_probability,
        )
        prepared_episodes = iter_prepared_episodes(
            pending,
            prepare,
            prefetch_episodes=args.prefetch_episodes,
        )
        processing_started = time.time()
        for prepared in prepared_episodes:
            source = prepared.source
            episode_started = time.time()
            encode_started = time.perf_counter()
            latent = encode_causal_latents(
                model,
                encode_full,
                prepared.lam_frames,
                device=device,
                pair_batch_size=args.pair_batch_size,
                transition_frame_stride=args.transition_frame_stride,
            )
            encode_seconds = time.perf_counter() - encode_started
            arrays = build_cache_arrays(
                source,
                prepared.episode,
                latent,
                dataset=dataset,
                checkpoint=checkpoint,
                image_key=args.image_key,
                preprocess_version=preprocess_version,
                storage_dtype=args.storage_dtype,
                transition_frame_stride=args.transition_frame_stride,
            )
            cache_path = cache_path_for(output_root, source)
            write_started = time.perf_counter()
            atomic_write_npz(cache_path, arrays, compressed=not args.no_compress)
            summary = validate_cache(cache_path, source, expectation)
            write_seconds = time.perf_counter() - write_started
            stage_seconds["read"] += prepared.read_seconds
            stage_seconds["crop"] += prepared.crop_seconds
            stage_seconds["preprocess"] += prepared.preprocess_seconds
            stage_seconds["encode"] += encode_seconds
            stage_seconds["write"] += write_seconds
            processed += 1
            total_frames += summary.num_frames
            if processed % args.log_every == 0 or processed == 1 or processed == len(pending):
                print(
                    f"rank={rank} processed={processed}/{len(pending)} episode={source.episode_index} "
                    f"frames={summary.num_frames} seconds={time.time() - episode_started:.2f} "
                    f"stages=read:{prepared.read_seconds:.2f},crop:{prepared.crop_seconds:.2f},"
                    f"preprocess:{prepared.preprocess_seconds:.2f},encode:{encode_seconds:.2f},"
                    f"write:{write_seconds:.2f} "
                    f"pipeline_fps:{total_frames / (time.time() - processing_started):.1f} cache={cache_path}"
                )

    rank_summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "status": "complete",
        "assigned_episodes": len(assigned),
        "processed_episodes": processed,
        "skipped_episodes": skipped,
        "processed_frames": total_frames,
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_format": checkpoint.format,
        "checkpoint_step": checkpoint.step,
        "dataset_fingerprint": dataset.fingerprint,
        "image_key": args.image_key,
        "preprocess_version": preprocess_version,
        "sequence_window_crop_min_scale": args.sequence_window_crop_min_scale,
        "sequence_window_crop_probability": args.sequence_window_crop_probability,
        "sequence_augmentation_seed": args.sequence_augmentation_seed,
        "storage_dtype": args.storage_dtype,
        "transition_frame_stride": args.transition_frame_stride,
        "prefetch_episodes": args.prefetch_episodes,
        "stage_seconds": stage_seconds if pending else {},
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_root / "shards" / f"rank-{rank:05d}-of-{world_size:05d}.json", rank_summary)
    print(f"rank={rank} complete processed={processed} skipped={skipped} elapsed_seconds={time.time() - started:.1f}")
    return 0


def list_existing_cache_paths(output_root: Path) -> list[Path]:
    episodes_root = output_root / "episodes"
    return sorted(episodes_root.glob("chunk-*/episode_*.npz")) if episodes_root.is_dir() else []


def infer_expectation(
    dataset: DatasetSpec,
    output_root: Path,
    *,
    image_key: str | None,
    checkpoint_sha256: str | None,
) -> CacheExpectation:
    cache_paths = list_existing_cache_paths(output_root)
    if not cache_paths:
        raise FileNotFoundError(f"no episode caches found under {output_root / 'episodes'}")
    inferred = read_cache_expectation(cache_paths[0])
    expected_sha = (
        _validate_sha256(checkpoint_sha256, "--checkpoint-sha256") if checkpoint_sha256 else inferred.checkpoint_sha256
    )
    return CacheExpectation(
        checkpoint_sha256=expected_sha,
        checkpoint_format=inferred.checkpoint_format,
        checkpoint_step=inferred.checkpoint_step,
        dataset_fingerprint=dataset.fingerprint,
        image_key=image_key or inferred.image_key,
        preprocess_version=inferred.preprocess_version,
        storage_dtype=inferred.storage_dtype,
        transition_frame_stride=inferred.transition_frame_stride,
    )


def check_extra_caches(dataset: DatasetSpec, output_root: Path) -> None:
    expected = {cache_path_for(output_root, source).resolve() for source in dataset.episodes}
    extra = [path for path in list_existing_cache_paths(output_root) if path.resolve() not in expected]
    if extra:
        preview = ", ".join(str(path) for path in extra[:3])
        raise ValueError(f"found {len(extra)} cache files that do not belong to this dataset, e.g. {preview}")


def collect_cache_records(
    dataset: DatasetSpec,
    output_root: Path,
    expectation: CacheExpectation,
    *,
    allow_incomplete: bool,
    compute_hashes: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    missing: list[int] = []
    for source in dataset.episodes:
        cache_path = cache_path_for(output_root, source)
        if not cache_path.is_file():
            missing.append(source.episode_index)
            continue
        summary = validate_cache(cache_path, source, expectation)
        record = {
            "episode_index": source.episode_index,
            "cache_path": cache_path.relative_to(output_root).as_posix(),
            "source_parquet": source.relative_path,
            "num_frames": summary.num_frames,
            "num_valid": summary.num_valid,
            "first_global_index": summary.first_global_index,
            "last_global_index": summary.last_global_index,
            "latent_dtype": summary.latent_dtype,
        }
        if compute_hashes:
            record["cache_sha256"] = sha256_file(cache_path)
        records.append(record)
    if missing and not allow_incomplete:
        preview = ", ".join(str(index) for index in missing[:10])
        raise RuntimeError(
            f"cache is incomplete: missing {len(missing)}/{len(dataset.episodes)} episodes; first missing: {preview}"
        )
    return records, missing


def build_global_frame_index(output_root: Path, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "global_index": [],
        "episode_index": [],
        "frame_index": [],
        "cache_row": [],
        "valid": [],
    }
    for record in records:
        cache_path = output_root / record["cache_path"]
        with np.load(cache_path, allow_pickle=False) as data:
            num_frames = len(data["frame_index"])
            arrays["global_index"].append(data["global_index"].astype(np.int64, copy=False))
            arrays["episode_index"].append(np.full(num_frames, record["episode_index"], dtype=np.int64))
            arrays["frame_index"].append(data["frame_index"].astype(np.int64, copy=False))
            arrays["cache_row"].append(np.arange(num_frames, dtype=np.int32))
            arrays["valid"].append(data["valid"].astype(np.bool_, copy=False))
    if not records:
        return {
            "global_index": np.empty(0, dtype=np.int64),
            "episode_index": np.empty(0, dtype=np.int64),
            "frame_index": np.empty(0, dtype=np.int64),
            "cache_row": np.empty(0, dtype=np.int32),
            "valid": np.empty(0, dtype=np.bool_),
        }
    combined = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    order = np.argsort(combined["global_index"], kind="stable")
    combined = {name: values[order] for name, values in combined.items()}
    if len(combined["global_index"]) > 1 and np.any(np.diff(combined["global_index"]) <= 0):
        raise ValueError("global frame indices across caches are not unique and strictly increasing")
    return combined


def build_global_latents(output_root: Path, records: list[dict[str, Any]]) -> np.ndarray:
    """Build one global-index-sorted mmap-friendly latent table."""
    latent_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for record in records:
        with np.load(output_root / record["cache_path"], allow_pickle=False) as data:
            latent_chunks.append(data["latent"].copy())
            index_chunks.append(data["global_index"].astype(np.int64, copy=False))
    if not latent_chunks:
        return np.empty((0, LATENT_DIM), dtype=np.float16)
    latents = np.concatenate(latent_chunks, axis=0)
    global_indices = np.concatenate(index_chunks, axis=0)
    return latents[np.argsort(global_indices, kind="stable")]


def compute_latent_normalization(latents: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Compute stable population statistics over valid rows of the global latent table."""
    if latents.ndim != 2 or latents.shape[1] != LATENT_DIM:
        raise ValueError(f"global latents must have shape (N, {LATENT_DIM}), got {latents.shape}")
    if valid.shape != (len(latents),) or valid.dtype != np.bool_:
        raise ValueError(
            f"global valid mask must be bool with shape {(len(latents),)}, got {valid.shape}/{valid.dtype}"
        )
    count = 0
    mean = np.zeros(LATENT_DIM, dtype=np.float64)
    squared_deviation = np.zeros(LATENT_DIM, dtype=np.float64)
    for start in range(0, len(latents), 65_536):
        stop = min(start + 65_536, len(latents))
        values = np.asarray(latents[start:stop][valid[start:stop]], dtype=np.float64)
        chunk_count = len(values)
        if chunk_count == 0:
            continue
        chunk_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - chunk_mean
        chunk_squared_deviation = np.sum(centered * centered, axis=0, dtype=np.float64)
        combined_count = count + chunk_count
        delta = chunk_mean - mean
        squared_deviation += chunk_squared_deviation + delta * delta * count * chunk_count / combined_count
        mean += delta * chunk_count / combined_count
        count = combined_count
    if count <= 0:
        raise ValueError("cannot compute DreamDojo normalization without valid transitions")
    std = np.sqrt(squared_deviation / count)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("computed DreamDojo latent normalization contains non-finite values")
    if np.any(std <= LATENT_STD_EPSILON):
        bad_dimensions = np.flatnonzero(std <= LATENT_STD_EPSILON).tolist()
        raise ValueError(f"computed DreamDojo latent std is <= {LATENT_STD_EPSILON} for dimensions {bad_dimensions}")
    return {
        "definition": LATENT_NORMALIZATION_DEFINITION,
        "count": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ddof": 0,
        "epsilon": LATENT_STD_EPSILON,
    }


def parse_latent_normalization(metadata: Any, *, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(metadata, dict):
        raise ValueError(
            "manifest has no latent_normalization metadata; rerun finalize to upgrade it without re-extraction"
        )
    if metadata.get("definition") != LATENT_NORMALIZATION_DEFINITION:
        raise ValueError(f"unsupported latent normalization definition: {metadata.get('definition')!r}")
    if metadata.get("ddof") != 0:
        raise ValueError(f"latent normalization must use ddof=0, got {metadata.get('ddof')!r}")
    if metadata.get("epsilon") != LATENT_STD_EPSILON:
        raise ValueError(f"latent normalization epsilon must be {LATENT_STD_EPSILON}, got {metadata.get('epsilon')!r}")
    count = metadata.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
        raise ValueError(f"latent normalization count must be {expected_count}, got {count!r}")
    mean = np.asarray(metadata.get("mean"), dtype=np.float64)
    std = np.asarray(metadata.get("std"), dtype=np.float64)
    if mean.shape != (LATENT_DIM,) or std.shape != (LATENT_DIM,):
        raise ValueError(f"latent normalization mean/std must have shape {(LATENT_DIM,)}, got {mean.shape}/{std.shape}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("latent normalization mean/std must be finite")
    if np.any(std <= LATENT_STD_EPSILON):
        raise ValueError(f"latent normalization std must be greater than {LATENT_STD_EPSILON}")
    return mean, std


def manifest_int(manifest: dict[str, Any], key: str) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest field {key!r} must be an integer, got {value!r}")
    return value


def manifest_transition_frame_stride(manifest: dict[str, Any]) -> int:
    # Existing adjacent-frame caches predate this explicit field. Treat them
    # as stride one so their finalized manifests remain readable.
    value = manifest.get("transition_frame_stride", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "manifest field 'transition_frame_stride' must be a positive integer, "
            f"got {value!r}"
        )
    return value


def manifest_file(output_root: Path, manifest: dict[str, Any], key: str, default: str) -> Path:
    relative = manifest.get(key, default)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"manifest field {key!r} must be a non-empty path string")
    path = (output_root / relative).resolve()
    if not path.is_relative_to(output_root):
        raise ValueError(f"manifest field {key!r} escapes the output root: {relative!r}")
    return path


def load_finalized_manifest(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing finalized cache manifest: {manifest_path}; run the finalize command before validate"
        )
    return load_json(manifest_path)


def validate_manifest_against_dataset(
    manifest: dict[str, Any],
    dataset: DatasetSpec,
    *,
    allow_incomplete: bool,
) -> None:
    schema_version = manifest_int(manifest, "schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        upgrade = (
            "; rerun the finalize command to atomically upgrade existing caches without re-extracting latents"
            if schema_version < MANIFEST_SCHEMA_VERSION
            else ""
        )
        raise ValueError(
            f"unsupported manifest schema_version={schema_version}, expected {MANIFEST_SCHEMA_VERSION}{upgrade}"
        )
    if manifest_int(manifest, "cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cache_schema_version={manifest.get('cache_schema_version')!r}, "
            f"expected {CACHE_SCHEMA_VERSION}"
        )
    status = manifest.get("status")
    if status not in {"complete", "partial"}:
        raise ValueError(f"manifest status must be 'complete' or 'partial', got {status!r}")
    if status == "partial" and not allow_incomplete:
        raise ValueError("cache manifest is partial; pass --allow-incomplete only for intentional partial validation")

    root_value = manifest.get("dataset_root")
    if not isinstance(root_value, str) or Path(root_value).expanduser().resolve() != dataset.root:
        raise ValueError(f"manifest dataset_root={root_value!r} does not match current dataset {dataset.root}")
    identity_checks = {
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_info_sha256": dataset.info_sha256,
        "dataset_episodes_sha256": dataset.episodes_sha256,
    }
    for key, expected in identity_checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key}={manifest.get(key)!r} does not match current dataset {expected!r}")

    expected_episodes = len(dataset.episodes)
    declared_total_frames = int(dataset.info["total_frames"])
    cached_episodes = manifest_int(manifest, "cached_episodes")
    cached_total_frames = manifest_int(manifest, "cached_total_frames")
    cached_valid_transitions = manifest_int(manifest, "cached_valid_transitions")
    transition_frame_stride = manifest_transition_frame_stride(manifest)
    if manifest_int(manifest, "expected_episodes") != expected_episodes:
        raise ValueError("manifest expected_episodes does not match current dataset")
    if manifest_int(manifest, "declared_total_frames") != declared_total_frames:
        raise ValueError("manifest declared_total_frames does not match current dataset")
    if not 0 <= cached_episodes <= expected_episodes:
        raise ValueError(f"manifest cached_episodes is out of range: {cached_episodes}")
    if not 0 <= cached_total_frames <= declared_total_frames:
        raise ValueError(f"manifest cached_total_frames is out of range: {cached_total_frames}")
    if not 0 <= cached_valid_transitions <= cached_total_frames:
        raise ValueError(f"manifest cached_valid_transitions is out of range: {cached_valid_transitions}")

    missing = manifest.get("missing_episodes")
    if (
        not isinstance(missing, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in missing)
        or missing != sorted(set(missing))
        or any(index < 0 or index >= expected_episodes for index in missing)
    ):
        raise ValueError("manifest missing_episodes must be a sorted unique list of in-range integers")
    if cached_episodes + len(missing) != expected_episodes:
        raise ValueError("manifest cached_episodes plus missing_episodes does not equal expected_episodes")
    if status == "complete":
        if missing or cached_episodes != expected_episodes or cached_total_frames != declared_total_frames:
            raise ValueError("complete manifest has missing episodes or incomplete frame totals")
        expected_valid_transitions = sum(
            (source.expected_length - 1) // transition_frame_stride for source in dataset.episodes
        )
        if cached_valid_transitions != expected_valid_transitions:
            raise ValueError(
                "complete manifest valid-transition total is inconsistent with its transition frame stride: "
                f"cached={cached_valid_transitions}, expected={expected_valid_transitions}, "
                f"stride={transition_frame_stride}"
            )
    elif not missing:
        raise ValueError("partial manifest must list at least one missing episode")

    if manifest_int(manifest, "latent_dim") != LATENT_DIM:
        raise ValueError(f"manifest latent_dim must be {LATENT_DIM}, got {manifest.get('latent_dim')!r}")
    if manifest.get("latent_dtype") not in {"float16", "float32"}:
        raise ValueError(f"manifest latent_dtype must be float16 or float32, got {manifest.get('latent_dtype')!r}")
    if manifest.get("alignment") != alignment_for_transition_frame_stride(transition_frame_stride):
        raise ValueError(f"manifest alignment mismatch: {manifest.get('alignment')!r}")
    preprocess_version = manifest.get("preprocess_version")
    if not isinstance(preprocess_version, str) or not preprocess_version:
        raise ValueError(f"manifest preprocess_version must be a non-empty string, got {preprocess_version!r}")
    _validate_sha256(str(manifest.get("checkpoint_sha256", "")), "manifest checkpoint_sha256")
    if not isinstance(manifest.get("checkpoint_format"), str) or not manifest["checkpoint_format"]:
        raise ValueError("manifest checkpoint_format must be a non-empty string")
    manifest_int(manifest, "checkpoint_step")
    if not isinstance(manifest.get("image_key"), str) or not manifest["image_key"]:
        raise ValueError("manifest image_key must be a non-empty string")
    for key in (
        "episode_manifest_sha256",
        "global_frame_index_sha256",
        "global_latents_sha256",
    ):
        if key not in manifest:
            raise ValueError(f"schema v2 manifest is missing required field {key!r}")
        _validate_sha256(str(manifest[key]), f"manifest {key}")
    parse_latent_normalization(manifest.get("latent_normalization"), expected_count=cached_valid_transitions)


def validate_manifest_counts(manifest: dict[str, Any], records: list[dict[str, Any]], missing: list[int]) -> None:
    actual_frames = int(sum(record["num_frames"] for record in records))
    actual_valid = int(sum(record["num_valid"] for record in records))
    actual_status = "partial" if missing else "complete"
    checks = {
        "status": actual_status,
        "cached_episodes": len(records),
        "missing_episodes": missing,
        "cached_total_frames": actual_frames,
        "cached_valid_transitions": actual_valid,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest {key}={manifest.get(key)!r} does not match episode caches {expected!r}")


def validate_file_hash(manifest: dict[str, Any], key: str, path: Path) -> None:
    expected = _validate_sha256(str(manifest[key]), f"manifest {key}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: manifest={expected}, actual={actual}")


def finalize_command(args: argparse.Namespace) -> int:
    dataset = discover_dataset(args.dataset_root)
    output_root = args.output_root.expanduser().resolve()
    check_extra_caches(dataset, output_root)
    expectation = infer_expectation(
        dataset,
        output_root,
        image_key=args.image_key,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    records, missing = collect_cache_records(
        dataset,
        output_root,
        expectation,
        allow_incomplete=args.allow_incomplete,
        compute_hashes=True,
    )
    global_index = build_global_frame_index(output_root, records)
    global_latents = build_global_latents(output_root, records)
    latent_normalization = compute_latent_normalization(global_latents, global_index["valid"])
    total_frames = int(sum(record["num_frames"] for record in records))
    total_valid = int(sum(record["num_valid"] for record in records))
    declared_total_frames = int(dataset.info["total_frames"])
    expected_total_valid = sum(
        (record["num_frames"] - 1) // expectation.transition_frame_stride for record in records
    )
    if total_valid != expected_total_valid:
        raise ValueError(
            f"cached valid-transition total {total_valid} is inconsistent with "
            f"transition_frame_stride={expectation.transition_frame_stride}; expected={expected_total_valid}"
        )
    if missing:
        if total_frames >= declared_total_frames:
            raise ValueError(
                f"partial cache has {total_frames} frames, expected fewer than declared total {declared_total_frames}"
            )
    elif total_frames != declared_total_frames:
        raise ValueError(f"complete cache has {total_frames} frames, but dataset declares {declared_total_frames}")
    atomic_write_jsonl(output_root / "episodes.jsonl", records)
    atomic_write_npz(output_root / "frame_index.npz", global_index, compressed=True)
    atomic_write_npy(output_root / "latents.npy", global_latents)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "partial" if missing else "complete",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset_root": str(dataset.root),
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_info_sha256": dataset.info_sha256,
        "dataset_episodes_sha256": dataset.episodes_sha256,
        "expected_episodes": len(dataset.episodes),
        "cached_episodes": len(records),
        "missing_episodes": missing,
        "declared_total_frames": declared_total_frames,
        "cached_total_frames": total_frames,
        "cached_valid_transitions": total_valid,
        "transition_frame_stride": expectation.transition_frame_stride,
        "checkpoint_sha256": expectation.checkpoint_sha256,
        "checkpoint_format": expectation.checkpoint_format,
        "checkpoint_step": expectation.checkpoint_step,
        "preprocess_version": expectation.preprocess_version,
        "alignment": alignment_for_transition_frame_stride(expectation.transition_frame_stride),
        "image_key": expectation.image_key,
        "latent_dim": LATENT_DIM,
        "latent_dtype": expectation.storage_dtype,
        "episode_manifest": "episodes.jsonl",
        "episode_manifest_sha256": sha256_file(output_root / "episodes.jsonl"),
        "global_frame_index": "frame_index.npz",
        "global_frame_index_sha256": sha256_file(output_root / "frame_index.npz"),
        "global_latents": "latents.npy",
        "global_latents_sha256": sha256_file(output_root / "latents.npy"),
        "latent_normalization": latent_normalization,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    print(
        f"finalized status={manifest['status']} episodes={len(records)}/{len(dataset.episodes)} "
        f"frames={total_frames} valid_transitions={total_valid} output={output_root}"
    )
    return 0


def load_manifest_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"missing finalized episode manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            if episode_index in records:
                raise ValueError(f"duplicate episode {episode_index} at {path}:{line_number}")
            records[episode_index] = row
    return records


def validate_episode_manifest(records: list[dict[str, Any]], stored: dict[int, dict[str, Any]]) -> None:
    expected_indices = {int(record["episode_index"]) for record in records}
    if set(stored) != expected_indices:
        raise ValueError(
            f"episode manifest indices mismatch: expected {sorted(expected_indices)}, got {sorted(stored)}"
        )
    for record in records:
        stored_record = stored[int(record["episode_index"])]
        for key, expected in record.items():
            if stored_record.get(key) != expected:
                raise ValueError(
                    f"episode manifest entry {record['episode_index']} field {key!r}="
                    f"{stored_record.get(key)!r}, expected {expected!r}"
                )


def validate_global_frame_index(
    output_root: Path,
    records: list[dict[str, Any]],
    *,
    index_path: Path | None = None,
) -> dict[str, np.ndarray]:
    index_path = output_root / "frame_index.npz" if index_path is None else index_path
    if not index_path.is_file():
        raise FileNotFoundError(f"missing finalized global frame index: {index_path}")
    expected = build_global_frame_index(output_root, records)
    with np.load(index_path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected):
            raise ValueError(f"{index_path} arrays mismatch: expected {sorted(expected)}, got {sorted(stored.files)}")
        for name, values in expected.items():
            if not np.array_equal(stored[name], values):
                raise ValueError(f"{index_path}:{name} does not match episode caches")
        return {name: stored[name].copy() for name in stored.files}


def validate_global_latents(
    output_root: Path,
    records: list[dict[str, Any]],
    *,
    latent_path: Path | None = None,
) -> np.ndarray:
    latent_path = output_root / "latents.npy" if latent_path is None else latent_path
    if not latent_path.is_file():
        raise FileNotFoundError(f"missing finalized global latent table: {latent_path}")
    expected = build_global_latents(output_root, records)
    actual = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError("global latent table does not match episode caches")
    return actual


def validate_command(args: argparse.Namespace) -> int:
    dataset = discover_dataset(args.dataset_root)
    output_root = args.output_root.expanduser().resolve()
    manifest = load_finalized_manifest(output_root)
    validate_manifest_against_dataset(manifest, dataset, allow_incomplete=args.allow_incomplete)
    check_extra_caches(dataset, output_root)
    manifest_checkpoint_sha256 = _validate_sha256(str(manifest["checkpoint_sha256"]), "manifest checkpoint_sha256")
    requested_checkpoint_sha256 = (
        _validate_sha256(args.checkpoint_sha256, "--checkpoint-sha256") if args.checkpoint_sha256 else None
    )
    if requested_checkpoint_sha256 is not None and requested_checkpoint_sha256 != manifest_checkpoint_sha256:
        raise ValueError(
            f"--checkpoint-sha256={requested_checkpoint_sha256} disagrees with manifest={manifest_checkpoint_sha256}"
        )
    if args.image_key is not None and args.image_key != manifest["image_key"]:
        raise ValueError(f"--image-key={args.image_key!r} disagrees with manifest={manifest['image_key']!r}")
    expectation = CacheExpectation(
        checkpoint_sha256=manifest_checkpoint_sha256,
        checkpoint_format=str(manifest["checkpoint_format"]),
        checkpoint_step=manifest_int(manifest, "checkpoint_step"),
        dataset_fingerprint=dataset.fingerprint,
        image_key=str(manifest["image_key"]),
        preprocess_version=str(manifest["preprocess_version"]),
        storage_dtype=str(manifest["latent_dtype"]),
        transition_frame_stride=manifest_transition_frame_stride(manifest),
    )
    records, missing = collect_cache_records(
        dataset,
        output_root,
        expectation,
        allow_incomplete=args.allow_incomplete,
        compute_hashes=False,
    )
    validate_manifest_counts(manifest, records, missing)
    episode_manifest_path = manifest_file(output_root, manifest, "episode_manifest", "episodes.jsonl")
    validate_file_hash(manifest, "episode_manifest_sha256", episode_manifest_path)
    manifest_records = load_manifest_records(episode_manifest_path)
    validate_episode_manifest(records, manifest_records)
    if args.verify_cache_hashes:
        for record in records:
            manifest_record = manifest_records.get(int(record["episode_index"]))
            if manifest_record is None or "cache_sha256" not in manifest_record:
                raise ValueError(f"no cache hash in manifest for episode {record['episode_index']}")
            actual = sha256_file(output_root / record["cache_path"])
            if actual != manifest_record["cache_sha256"]:
                raise ValueError(f"cache SHA256 mismatch for episode {record['episode_index']}")

    index_path = manifest_file(output_root, manifest, "global_frame_index", "frame_index.npz")
    latent_path = manifest_file(output_root, manifest, "global_latents", "latents.npy")
    validate_file_hash(manifest, "global_frame_index_sha256", index_path)
    validate_file_hash(manifest, "global_latents_sha256", latent_path)
    global_index = validate_global_frame_index(output_root, records, index_path=index_path)
    global_latents = validate_global_latents(output_root, records, latent_path=latent_path)
    computed_normalization = compute_latent_normalization(global_latents, global_index["valid"])
    stored_mean, stored_std = parse_latent_normalization(
        manifest.get("latent_normalization"),
        expected_count=int(global_index["valid"].sum()),
    )
    computed_mean = np.asarray(computed_normalization["mean"], dtype=np.float64)
    computed_std = np.asarray(computed_normalization["std"], dtype=np.float64)
    if not np.allclose(stored_mean, computed_mean, rtol=1e-12, atol=1e-12):
        raise ValueError("manifest latent normalization mean does not match valid rows in global latents")
    if not np.allclose(stored_std, computed_std, rtol=1e-12, atol=1e-12):
        raise ValueError("manifest latent normalization std does not match valid rows in global latents")
    print(
        f"valid status={'partial' if missing else 'complete'} episodes={len(records)}/{len(dataset.episodes)} "
        f"frames={sum(record['num_frames'] for record in records)} checkpoint_sha256={expectation.checkpoint_sha256}"
    )
    return 0


def add_dataset_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot v2 dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing per-episode caches.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute, finalize, and validate causal DreamDojo memory caches for LeRobot v2 parquet data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract this process's deterministic episode shard.")
    add_dataset_output_arguments(extract)
    extract.add_argument(
        "--dreamdojo-runtime-root",
        type=Path,
        default=DEFAULT_DREAMDOJO_RUNTIME_ROOT,
        help="Vendored DreamDojo runtime; defaults to third_party/dreamdojo/runtime.",
    )
    extract.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="DreamDojo LAM 400k inference checkpoint.",
    )
    extract.add_argument("--checkpoint-sha256", default=None, help="Optional expected release-file SHA256.")
    extract.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Stream and hash the 2.8 GB checkpoint instead of trusting the colocated release manifest.",
    )
    extract.add_argument(
        "--image-key", default="image", help="Embedded RGB image column; default is main camera 'image'."
    )
    extract.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    extract.add_argument(
        "--transition-frame-stride",
        type=int,
        default=1,
        help="Encode only complete causal pairs (0->s, s->2s, ...) and mark other destination rows invalid.",
    )
    extract.add_argument("--pair-batch-size", type=int, default=32)
    extract.add_argument("--preprocess-chunk-size", type=int, default=64)
    extract.add_argument(
        "--prefetch-episodes",
        type=int,
        choices=range(11),
        default=0,
        help="Queue up to ten future episodes using one CPU preprocessing worker.",
    )
    extract.add_argument(
        "--sequence-window-crop-min-scale",
        type=float,
        default=1.0,
        help="Minimum episode-shared crop scale. 1.0 disables sequence window cropping.",
    )
    extract.add_argument(
        "--sequence-window-crop-probability",
        type=float,
        default=0.0,
        help="Probability that an episode receives one crop shared by every frame and camera.",
    )
    extract.add_argument(
        "--sequence-augmentation-seed",
        type=int,
        default=0,
        help="Seed combined with episode_index to reproduce the shared crop across camera extractions.",
    )
    extract.add_argument("--device", default="auto", help="CUDA device such as cuda:0; auto uses LOCAL_RANK.")
    extract.add_argument("--rank", type=int, default=None, help="Shard rank; default reads RANK, then 0.")
    extract.add_argument("--world-size", type=int, default=None, help="Shard count; default reads WORLD_SIZE, then 1.")
    extract.add_argument("--local-rank", type=int, default=None, help=argparse.SUPPRESS)
    extract.add_argument("--episode-start", type=int, default=None, help="Inclusive episode-index filter.")
    extract.add_argument("--episode-stop", type=int, default=None, help="Exclusive episode-index filter.")
    extract.add_argument("--limit", type=int, default=None, help="Limit assigned episodes, useful for a smoke run.")
    extract.add_argument("--overwrite", action="store_true", help="Recompute valid existing caches in this shard.")
    extract.add_argument(
        "--repair", action="store_true", help="Atomically replace incompatible/corrupt existing caches."
    )
    extract.add_argument("--no-compress", action="store_true", help="Use uncompressed NPZ episode caches.")
    extract.add_argument("--log-every", type=int, default=10)
    extract.add_argument(
        "--dry-run", action="store_true", help="Resolve data/checkpoint/shard without loading the model."
    )
    extract.set_defaults(func=extract_command)

    finalize = subparsers.add_parser("finalize", help="Validate caches and write manifest/global lookup files.")
    add_dataset_output_arguments(finalize)
    finalize.add_argument("--image-key", default=None, help="Expected image key; default infers from the first cache.")
    finalize.add_argument("--checkpoint-sha256", default=None, help="Expected checkpoint SHA256.")
    finalize.add_argument(
        "--allow-incomplete", action="store_true", help="Write a partial manifest instead of failing."
    )
    finalize.set_defaults(func=finalize_command)

    validate = subparsers.add_parser("validate", help="Structurally validate all available episode caches.")
    add_dataset_output_arguments(validate)
    validate.add_argument("--image-key", default=None, help="Expected image key; default infers from the first cache.")
    validate.add_argument("--checkpoint-sha256", default=None, help="Expected checkpoint SHA256.")
    validate.add_argument("--allow-incomplete", action="store_true", help="Report missing episodes without failing.")
    validate.add_argument("--verify-cache-hashes", action="store_true", help="Verify hashes written by finalize.")
    validate.set_defaults(func=validate_command)
    return parser


def validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in ("pair_batch_size", "preprocess_chunk_size", "log_every"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "transition_frame_stride") and args.transition_frame_stride <= 0:
        raise ValueError("--transition-frame-stride must be positive")
    if getattr(args, "limit", None) is not None and args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if hasattr(args, "sequence_window_crop_min_scale") and not 0.0 < args.sequence_window_crop_min_scale <= 1.0:
        raise ValueError("--sequence-window-crop-min-scale must be in (0, 1]")
    if hasattr(args, "sequence_window_crop_probability") and not 0.0 <= args.sequence_window_crop_probability <= 1.0:
        raise ValueError("--sequence-window-crop-probability must be in [0, 1]")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_positive_arguments(args)
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
