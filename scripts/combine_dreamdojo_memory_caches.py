#!/usr/bin/env python3
"""Combine aligned head/wrist DreamDojo caches into one 64-D cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Direct execution sets sys.path[0] to scripts/, so add the repository root
# before importing the scripts package. Importing this module in tests already
# has the correct package path and does not need the adjustment.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import precompute_dreamdojo_memory as cache_lib


def _load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing finalized source manifest: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"source manifest must be an object: {path}")
    return value


def _validate_sources(left: dict, right: dict) -> None:
    for label, manifest in (("external", left), ("wrist", right)):
        if manifest.get("status") != "complete":
            raise ValueError(f"{label} cache is not complete")
        if manifest.get("latent_dim") != cache_lib.LATENT_DIM:
            raise ValueError(f"{label} cache latent_dim must be {cache_lib.LATENT_DIM}")
        if manifest.get("latent_dtype") not in {"float16", "float32"}:
            raise ValueError(f"{label} cache must use float16 or float32 storage")
    if left.get("latent_dtype") != right.get("latent_dtype"):
        raise ValueError("head and wrist caches must use the same storage dtype")
    common_fields = (
        "schema_version",
        "cache_schema_version",
        "dataset_root",
        "dataset_fingerprint",
        "dataset_info_sha256",
        "dataset_episodes_sha256",
        "expected_episodes",
        "cached_episodes",
        "missing_episodes",
        "declared_total_frames",
        "cached_total_frames",
        "cached_valid_transitions",
        "checkpoint_sha256",
        "checkpoint_format",
        "checkpoint_step",
        "preprocess_version",
        "alignment",
    )
    for field in common_fields:
        if left.get(field) != right.get(field):
            raise ValueError(
                f"source caches disagree on {field}: external={left.get(field)!r}, wrist={right.get(field)!r}"
            )
    transition_frame_stride = left.get("transition_frame_stride", 1)
    if right.get("transition_frame_stride", 1) != transition_frame_stride:
        raise ValueError(
            "source caches disagree on transition_frame_stride: "
            f"external={transition_frame_stride!r}, wrist={right.get('transition_frame_stride', 1)!r}"
        )
    if (
        isinstance(transition_frame_stride, bool)
        or not isinstance(transition_frame_stride, int)
        or transition_frame_stride <= 0
    ):
        raise ValueError(f"transition_frame_stride must be a positive integer, got {transition_frame_stride!r}")
    expected_alignment = cache_lib.alignment_for_transition_frame_stride(transition_frame_stride)
    if left.get("alignment") != expected_alignment:
        raise ValueError(
            "source cache alignment does not match transition_frame_stride: "
            f"alignment={left.get('alignment')!r}, stride={transition_frame_stride}"
        )
    if left.get("image_key") != "image" or right.get("image_key") != "wrist_image":
        raise ValueError(
            f"expected image/wrist_image source order, got {left.get('image_key')!r}/{right.get('image_key')!r}"
        )


def _load_index(root: Path, manifest: dict) -> dict[str, np.ndarray]:
    path = root / manifest.get("global_frame_index", "frame_index.npz")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def _normalization(latents: np.ndarray, valid: np.ndarray) -> dict:
    latent_dim = latents.shape[1]
    count = 0
    mean = np.zeros(latent_dim, dtype=np.float64)
    squared_deviation = np.zeros(latent_dim, dtype=np.float64)
    for start in range(0, len(latents), 65_536):
        values = np.asarray(latents[start : start + 65_536][valid[start : start + 65_536]], dtype=np.float64)
        if not len(values):
            continue
        chunk_count = len(values)
        chunk_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - chunk_mean
        chunk_squared_deviation = np.sum(centered * centered, axis=0, dtype=np.float64)
        combined_count = count + chunk_count
        delta = chunk_mean - mean
        squared_deviation += chunk_squared_deviation + delta * delta * count * chunk_count / combined_count
        mean += delta * chunk_count / combined_count
        count = combined_count
    if count <= 0:
        raise ValueError("combined cache has no valid transitions")
    std = np.sqrt(squared_deviation / count)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("combined normalization is non-finite")
    if np.any(std <= cache_lib.LATENT_STD_EPSILON):
        raise ValueError("combined normalization contains a degenerate dimension")
    return {
        "definition": cache_lib.LATENT_NORMALIZATION_DEFINITION,
        "count": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "ddof": 0,
        "epsilon": cache_lib.LATENT_STD_EPSILON,
    }


def combine(external_root: Path, wrist_root: Path, output_root: Path) -> None:
    external_root = external_root.expanduser().resolve()
    wrist_root = wrist_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    external_manifest = _load_manifest(external_root)
    wrist_manifest = _load_manifest(wrist_root)
    _validate_sources(external_manifest, wrist_manifest)

    external_index = _load_index(external_root, external_manifest)
    wrist_index = _load_index(wrist_root, wrist_manifest)
    if set(external_index) != set(wrist_index):
        raise ValueError("source frame-index files contain different arrays")
    for name in external_index:
        if not np.array_equal(external_index[name], wrist_index[name]):
            raise ValueError(f"source frame-index array {name!r} is not aligned")

    external_latents = np.load(
        external_root / external_manifest.get("global_latents", "latents.npy"), mmap_mode="r", allow_pickle=False
    )
    wrist_latents = np.load(
        wrist_root / wrist_manifest.get("global_latents", "latents.npy"), mmap_mode="r", allow_pickle=False
    )
    expected_rows = int(external_manifest["cached_total_frames"])
    expected_shape = (expected_rows, cache_lib.LATENT_DIM)
    if external_latents.shape != expected_shape or wrist_latents.shape != expected_shape:
        raise ValueError(
            f"source latent shapes must both be {expected_shape}, got {external_latents.shape}/{wrist_latents.shape}"
        )
    combined = np.concatenate((external_latents, wrist_latents), axis=1)
    valid = external_index["valid"]
    if combined.shape != (expected_rows, cache_lib.LATENT_DIM * 2):
        raise AssertionError(combined.shape)
    if np.any(combined[~valid] != 0) or not np.all(np.isfinite(combined[valid])):
        raise ValueError("combined cache has invalid zero rows or non-finite valid rows")
    normalization = _normalization(combined, valid)

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "frame_index.npz"
    latent_path = output_root / "latents.npy"
    cache_lib.atomic_write_npz(index_path, external_index, compressed=True)
    cache_lib.atomic_write_npy(latent_path, combined)
    manifest = {
        key: external_manifest[key]
        for key in (
            "schema_version",
            "status",
            "cache_schema_version",
            "dataset_root",
            "dataset_fingerprint",
            "dataset_info_sha256",
            "dataset_episodes_sha256",
            "expected_episodes",
            "cached_episodes",
            "missing_episodes",
            "declared_total_frames",
            "cached_total_frames",
            "cached_valid_transitions",
            "checkpoint_sha256",
            "checkpoint_format",
            "checkpoint_step",
            "preprocess_version",
            "alignment",
        )
    }
    manifest.update(
        {
            "image_key": "image+wrist_image",
            "transition_frame_stride": external_manifest.get("transition_frame_stride", 1),
            "image_keys": ["image", "wrist_image"],
            "latent_layout": "concat(image_z_mu[32],wrist_image_z_mu[32])",
            "latent_dim": cache_lib.LATENT_DIM * 2,
            "latent_dtype": str(combined.dtype),
            "global_frame_index": index_path.name,
            "global_frame_index_sha256": cache_lib.sha256_file(index_path),
            "global_latents": latent_path.name,
            "global_latents_sha256": cache_lib.sha256_file(latent_path),
            "latent_normalization": normalization,
            "source_caches": [str(external_root), str(wrist_root)],
            "source_manifest_sha256": [
                cache_lib.sha256_file(external_root / "manifest.json"),
                cache_lib.sha256_file(wrist_root / "manifest.json"),
            ],
        }
    )
    cache_lib.atomic_write_json(output_root / "manifest.json", manifest)
    print(
        f"combined rows={expected_rows} latent_dim={combined.shape[1]} valid={int(valid.sum())} "
        f"output={output_root}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-cache", type=Path, required=True)
    parser.add_argument("--wrist-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    combine(args.external_cache, args.wrist_cache, args.output_root)


if __name__ == "__main__":
    main()
