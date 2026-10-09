"""Dataset wrapper for finalized causal CD-LAM latent-action caches."""

from collections.abc import Sized
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

EXPECTED_ALIGNMENT = "row_0_invalid_zero; row_i=z_mu(frame_i-1,frame_i), i>=1"
DELTATOK_ALIGNMENT = "row_0_invalid_zero; row_i=deltatok(frame_i-1,frame_i), i>=1"
EXPECTED_ALIGNMENTS = frozenset((EXPECTED_ALIGNMENT, DELTATOK_ALIGNMENT))
EXPECTED_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_CACHE_SCHEMA_VERSION = 1
EXPECTED_NORMALIZATION_DEFINITION = "per_dimension_valid_transition_population"
LATENT_STD_EPSILON = 1e-6
SUPPORTED_DEMO_ANCHOR_SCHEMA_VERSIONS = frozenset((1, 2))
DEMO_ANCHOR_IMAGE_DEFINITION = "episode_frame_0_when_exec_start_idx_gt_0"
DEMO_ANCHOR_FEATURE_DEFINITION = "episode_demo_siglip_patch_features_mean_when_exec_start_idx_gt_0"
DUAL_ANCHOR_IMAGE_DEFINITION = "episode_frame_0_demo_and_episode_frame_exec_start_execution"
EPISODE_ANCHOR_IMAGE_DEFINITION = "episode_first_frame"
EPISODE_ANCHOR_SCHEMA_VERSION = 1
PHASE_METADATA_SCHEMA_VERSION = 1
PHASE_IS_DEMO_DEFINITION = "frame_index < exec_start_idx"
PHASE_ACTION_LOSS_DEFINITION = "frame_index >= exec_start_idx"

MEMORY_SEGMENT_PADDING = 0
MEMORY_SEGMENT_DEMO = 1
MEMORY_SEGMENT_BOUNDARY = 2
MEMORY_SEGMENT_EXECUTION = 3


def drop_execution_lam_tail(
    latents: np.ndarray,
    mask: np.ndarray,
    segment_ids: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop up to ``count`` newest execution-LAM tokens, preserving all other segments."""
    if count < 0:
        raise ValueError("execution LAM tail drop count must be non-negative")
    execution_indices = np.flatnonzero(mask & (segment_ids == MEMORY_SEGMENT_EXECUTION))
    drop_count = min(count, len(execution_indices))
    if drop_count == 0:
        return latents, mask, segment_ids
    keep = np.ones(len(mask), dtype=np.bool_)
    keep[execution_indices[-drop_count:]] = False
    return latents[keep], mask[keep], segment_ids[keep]


def _expected_cdlam_alignment(transition_frame_stride: int) -> str:
    if transition_frame_stride <= 0:
        raise ValueError(f"transition_frame_stride must be positive, got {transition_frame_stride}")
    if transition_frame_stride == 1:
        return EXPECTED_ALIGNMENT
    return (
        f"row_{{k*{transition_frame_stride}}}=z_mu(frame_{{(k-1)*{transition_frame_stride}}},"
        f"frame_{{k*{transition_frame_stride}}}), k>=1; all other rows invalid_zero"
    )


def _manifest_transition_frame_stride(manifest: dict[str, Any]) -> int:
    # Adjacent-frame cache manifests from before gap support did not need an
    # explicit stride field, so they remain stride one by definition.
    value = manifest.get("transition_frame_stride", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "CD-LAM manifest field 'transition_frame_stride' must be a positive integer, "
            f"got {value!r}"
        )
    return value


def classify_memory_segments(frame_indices, exec_start_idx: int) -> np.ndarray:
    """Classify transition endpoints identically in offline and online paths."""
    frame_indices = np.asarray(frame_indices)
    if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
        raise ValueError(f"frame_indices must be a 1-D integer array, got {frame_indices.shape}/{frame_indices.dtype}")
    if exec_start_idx < 0:
        raise ValueError(f"exec_start_idx must be non-negative, got {exec_start_idx}")
    return np.where(
        frame_indices < exec_start_idx,
        MEMORY_SEGMENT_DEMO,
        np.where(
            (exec_start_idx > 0) & (frame_indices == exec_start_idx),
            MEMORY_SEGMENT_BOUNDARY,
            MEMORY_SEGMENT_EXECUTION,
        ),
    ).astype(np.int32, copy=False)


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _manifest_int(manifest: dict[str, Any], key: str) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CD-LAM manifest field {key!r} must be an integer, got {value!r}")
    return value


def _cache_file(cache_dir: Path, manifest: dict[str, Any], key: str, default: str) -> Path:
    relative = manifest.get(key, default)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"CD-LAM manifest field {key!r} must be a non-empty path string")
    path = (cache_dir / relative).resolve()
    if not path.is_relative_to(cache_dir):
        raise ValueError(f"CD-LAM manifest field {key!r} escapes the cache directory: {relative!r}")
    return path


def _verify_hash(manifest: dict[str, Any], key: str, path: Path) -> None:
    expected = manifest.get(key)
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected)
    ):
        raise ValueError(f"CD-LAM manifest field {key!r} is not a SHA256 digest")
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"CD-LAM cache SHA256 mismatch for {path.name}: expected {expected.lower()}, found {actual}")


def _dataset_identity(dataset_root: Path) -> tuple[int, int, str, str, str]:
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"Cannot verify CD-LAM cache against {dataset_root}: expected meta/info.json and meta/episodes.jsonl"
        )
    try:
        info = json.loads(info_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read LeRobot metadata {info_path}: {error}") from error
    if not isinstance(info, dict):
        raise ValueError(f"LeRobot metadata {info_path} must contain a JSON object")
    try:
        total_frames = int(info["total_frames"])
        total_episodes = int(info["total_episodes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"LeRobot metadata {info_path} must declare integer total_frames and total_episodes"
        ) from error
    if total_frames <= 0 or total_episodes <= 0:
        raise ValueError(
            "LeRobot metadata totals must be positive, "
            f"got total_frames={total_frames}, total_episodes={total_episodes}"
        )
    info_sha256 = _sha256_file(info_path)
    episodes_sha256 = _sha256_file(episodes_path)
    fingerprint = hashlib.sha256(f"{info_sha256}:{episodes_sha256}".encode()).hexdigest()
    return total_frames, total_episodes, info_sha256, episodes_sha256, fingerprint


def _latent_normalization(
    manifest: dict[str, Any], latent_dim: int, expected_count: int
) -> tuple[np.ndarray, np.ndarray]:
    metadata = manifest.get("latent_normalization")
    if not isinstance(metadata, dict):
        raise ValueError(
            "CD-LAM manifest has no latent_normalization metadata; rerun "
            "scripts/precompute_dreamdojo_memory.py finalize to upgrade it without re-extracting latents"
        )
    if metadata.get("definition") != EXPECTED_NORMALIZATION_DEFINITION:
        raise ValueError(f"Unsupported CD-LAM latent normalization definition: {metadata.get('definition')!r}")
    if metadata.get("ddof") != 0:
        raise ValueError(f"CD-LAM latent normalization must use ddof=0, got {metadata.get('ddof')!r}")
    if metadata.get("epsilon") != LATENT_STD_EPSILON:
        raise ValueError(
            f"CD-LAM latent normalization epsilon must be {LATENT_STD_EPSILON}, got {metadata.get('epsilon')!r}"
        )
    count = metadata.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
        raise ValueError(
            f"CD-LAM latent normalization count must equal valid transitions {expected_count}, got {count!r}"
        )
    mean = np.asarray(metadata.get("mean"), dtype=np.float64)
    std = np.asarray(metadata.get("std"), dtype=np.float64)
    if mean.shape != (latent_dim,) or std.shape != (latent_dim,):
        raise ValueError(
            f"CD-LAM latent normalization mean/std must have shape {(latent_dim,)}, got {mean.shape}/{std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("CD-LAM latent normalization mean/std must be finite")
    if np.any(std <= LATENT_STD_EPSILON):
        raise ValueError(f"CD-LAM latent normalization std must be greater than {LATENT_STD_EPSILON}")
    return mean, std


def _validate_expected_latent_normalization(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    latent_dim: int,
    expected_mean,
    expected_std,
    label: str,
) -> None:
    argument_prefix = f"{label}_" if label else ""
    display_prefix = f"{label} " if label else ""
    if (expected_mean is None) != (expected_std is None):
        raise ValueError(
            f"expected_{argument_prefix}latent_mean and expected_{argument_prefix}latent_std must be provided together"
        )
    if expected_mean is None:
        return
    configured_mean = np.asarray(expected_mean, dtype=np.float64)
    configured_std = np.asarray(expected_std, dtype=np.float64)
    if configured_mean.shape != (latent_dim,) or configured_std.shape != (latent_dim,):
        raise ValueError(
            f"Configured {display_prefix}latent mean/std must have shape {(latent_dim,)}, "
            f"got {configured_mean.shape}/{configured_std.shape}"
        )
    if not np.allclose(configured_mean, mean, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Configured {display_prefix}latent mean does not match CD-LAM cache manifest")
    if not np.allclose(configured_std, std, rtol=1e-6, atol=1e-7):
        raise ValueError(f"Configured {display_prefix}latent std does not match CD-LAM cache manifest")


def _load_aligned_future_cache(
    cache_dir: str | Path,
    *,
    expected_root: Path,
    total_frames: int,
    total_episodes: int,
    info_sha256: str,
    episodes_sha256: str,
    fingerprint: str,
    expected_valid_transitions: int,
    latent_dim: int,
    reference_indices: dict[str, np.ndarray],
    expected_latent_mean=None,
    expected_latent_std=None,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Load a future-target cache and prove row-for-row alignment with history."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing finalized future CD-LAM cache manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read future CD-LAM cache manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Future CD-LAM cache manifest {manifest_path} must contain a JSON object")
    if _manifest_int(manifest, "schema_version") != EXPECTED_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported future CD-LAM manifest schema_version={manifest.get('schema_version')!r}")
    if _manifest_int(manifest, "cache_schema_version") != EXPECTED_CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported future CD-LAM cache_schema_version={manifest.get('cache_schema_version')!r}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Future CD-LAM cache must be complete, got status={manifest.get('status')!r}")
    if manifest.get("missing_episodes") != []:
        raise ValueError(
            f"Complete future CD-LAM cache must have no missing episodes, got {manifest.get('missing_episodes')!r}"
        )
    if _manifest_int(manifest, "latent_dim") != latent_dim:
        raise ValueError(
            f"Future CD-LAM latent dimension mismatch: cache={manifest.get('latent_dim')}, model={latent_dim}"
        )
    if manifest.get("alignment") not in EXPECTED_ALIGNMENTS:
        raise ValueError(f"Unsupported future CD-LAM cache alignment: {manifest.get('alignment')!r}")

    cached_dataset_root = manifest.get("dataset_root")
    if not isinstance(cached_dataset_root, str) or not cached_dataset_root:
        raise ValueError("Future CD-LAM manifest field 'dataset_root' must be a non-empty path string")
    if Path(cached_dataset_root).expanduser().resolve() != expected_root:
        raise ValueError(f"Future CD-LAM cache dataset mismatch: cache={cached_dataset_root}, training={expected_root}")
    identity_fields = {
        "dataset_info_sha256": info_sha256,
        "dataset_episodes_sha256": episodes_sha256,
        "dataset_fingerprint": fingerprint,
    }
    for key, expected in identity_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Future CD-LAM cache is stale or belongs to different data: "
                f"{key}={manifest.get(key)!r}, current={expected!r}"
            )

    expected_episodes = _manifest_int(manifest, "expected_episodes")
    cached_episodes = _manifest_int(manifest, "cached_episodes")
    declared_total_frames = _manifest_int(manifest, "declared_total_frames")
    cached_total_frames = _manifest_int(manifest, "cached_total_frames")
    cached_valid_transitions = _manifest_int(manifest, "cached_valid_transitions")
    if expected_episodes != total_episodes or cached_episodes != total_episodes:
        raise ValueError(
            "Future CD-LAM episode totals do not match current LeRobot metadata: "
            f"expected={expected_episodes}, cached={cached_episodes}, current={total_episodes}"
        )
    if declared_total_frames != total_frames or cached_total_frames != total_frames:
        raise ValueError(
            "Future CD-LAM frame totals do not match current LeRobot metadata: "
            f"declared={declared_total_frames}, cached={cached_total_frames}, current={total_frames}"
        )
    if cached_valid_transitions != expected_valid_transitions:
        raise ValueError(
            "Future CD-LAM valid-transition total is inconsistent with episode boundaries: "
            f"cached={cached_valid_transitions}, expected={expected_valid_transitions}"
        )

    mean, std = _latent_normalization(manifest, latent_dim, expected_valid_transitions)
    _validate_expected_latent_normalization(
        mean,
        std,
        latent_dim=latent_dim,
        expected_mean=expected_latent_mean,
        expected_std=expected_latent_std,
        label="future",
    )

    index_path = _cache_file(cache_dir, manifest, "global_frame_index", "frame_index.npz")
    latent_path = _cache_file(cache_dir, manifest, "global_latents", "latents.npy")
    if not index_path.is_file() or not latent_path.is_file():
        raise FileNotFoundError(f"Finalized future CD-LAM cache is missing {index_path.name} or {latent_path.name}")
    for hash_key, path in (
        ("global_frame_index_sha256", index_path),
        ("global_latents_sha256", latent_path),
    ):
        if hash_key not in manifest:
            raise ValueError(f"Future CD-LAM schema v2 manifest is missing required field {hash_key!r}")
        _verify_hash(manifest, hash_key, path)

    with np.load(index_path, allow_pickle=False) as index:
        required = {"global_index", "episode_index", "frame_index", "valid"}
        if not required.issubset(index.files):
            raise ValueError(f"{index_path} is missing arrays {sorted(required - set(index.files))}")
        for name in required:
            array = index[name]
            reference = reference_indices[name]
            if array.shape != reference.shape or array.dtype != reference.dtype or not np.array_equal(array, reference):
                raise ValueError(f"Future CD-LAM index array {name!r} does not exactly match the history cache")

    latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
    if latents.shape != (total_frames, latent_dim):
        raise ValueError(
            f"Future CD-LAM global latent shape mismatch: got {latents.shape}, expected {(total_frames, latent_dim)}"
        )
    if not np.issubdtype(latents.dtype, np.floating):
        raise ValueError(f"Future CD-LAM latents must be floating point, got {latents.dtype}")
    valid = reference_indices["valid"]
    if np.any(latents[~valid] != 0):
        raise ValueError("Future CD-LAM latent rows marked invalid must be exactly zero")
    for start in range(0, total_frames, 65_536):
        block = latents[start : start + 65_536]
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Future CD-LAM latents contain non-finite values near global row {start}")
    return latent_path, latents, mean, std


def _load_demo_anchor_cache(
    cache_dir: str | Path,
    *,
    dataset_root: Path,
    dataset_fingerprint: str,
    dataset_episodes_sha256: str,
    total_episodes: int,
    expected_definition: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Load and fully validate an mmap-friendly demo-anchor cache."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing demo-anchor cache manifest: {manifest_path}. "
            "Run scripts/prepare_robomme_demo_anchor_cache.py first."
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read demo-anchor cache manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Demo-anchor manifest {manifest_path} must contain a JSON object")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_DEMO_ANCHOR_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported demo-anchor schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Demo-anchor cache must be complete, got status={manifest.get('status')!r}")
    cached_root = Path(str(manifest.get("dataset_root", ""))).expanduser().resolve()
    if cached_root == dataset_root:
        if manifest.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Demo-anchor cache is stale or belongs to different dataset metadata")
    elif manifest.get("dataset_episodes_sha256") != dataset_episodes_sha256:
        raise ValueError("Demo-anchor cache belongs to a row-incompatible dataset")
    definition = manifest.get("definition")
    if definition not in (
        DEMO_ANCHOR_IMAGE_DEFINITION,
        DEMO_ANCHOR_FEATURE_DEFINITION,
        DUAL_ANCHOR_IMAGE_DEFINITION,
    ):
        raise ValueError(f"Unsupported demo-anchor definition: {definition!r}")
    if schema_version == 1 and definition == DUAL_ANCHOR_IMAGE_DEFINITION:
        raise ValueError("Dual-anchor definition requires cache schema_version=2")
    if schema_version == 2 and definition != DUAL_ANCHOR_IMAGE_DEFINITION:
        raise ValueError("Demo-anchor cache schema_version=2 requires the dual-anchor definition")
    if expected_definition is not None and definition != expected_definition:
        raise ValueError(f"Demo-anchor definition is {definition!r}, expected {expected_definition!r}")
    if manifest.get("source_image_key") != "image":
        raise ValueError(
            f"Demo-anchor source must be RoboMME head camera 'image', got {manifest.get('source_image_key')!r}"
        )
    if manifest.get("total_episodes") != total_episodes:
        raise ValueError(f"Demo-anchor episode total is {manifest.get('total_episodes')!r}, expected {total_episodes}")

    if definition in (DEMO_ANCHOR_IMAGE_DEFINITION, DUAL_ANCHOR_IMAGE_DEFINITION):
        anchor_path = _cache_file(cache_dir, manifest, "images_file", "images.npy")
        anchor_hash_key = "images_sha256"
    else:
        anchor_path = _cache_file(cache_dir, manifest, "features_file", "features.npy")
        anchor_hash_key = "features_sha256"
    mapping_path = _cache_file(cache_dir, manifest, "episode_to_anchor_file", "episode_to_anchor.npy")
    exec_path = _cache_file(cache_dir, manifest, "exec_start_idx_file", "exec_start_idx.npy")
    for key, path in (
        (anchor_hash_key, anchor_path),
        ("episode_to_anchor_sha256", mapping_path),
        ("exec_start_idx_sha256", exec_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Demo-anchor cache is missing {path}")
        _verify_hash(manifest, key, path)

    anchors = np.load(anchor_path, mmap_mode="r", allow_pickle=False)
    episode_to_anchor = np.load(mapping_path, allow_pickle=False)
    exec_start_idx = np.load(exec_path, allow_pickle=False)
    if definition in (DEMO_ANCHOR_IMAGE_DEFINITION, DUAL_ANCHOR_IMAGE_DEFINITION):
        if anchors.ndim != 4 or anchors.shape[-1] != 3 or anchors.dtype != np.uint8:
            raise ValueError(f"Demo-anchor images must be uint8 [N,H,W,3], got {anchors.shape}/{anchors.dtype}")
    elif anchors.ndim != 3 or anchors.shape[1:] != (256, 2048) or anchors.dtype != np.float16:
        raise ValueError(
            f"Feature-pooled demo anchors must be float16 [N,256,2048], got {anchors.shape}/{anchors.dtype}"
        )
    if episode_to_anchor.shape != (total_episodes,) or not np.issubdtype(episode_to_anchor.dtype, np.integer):
        raise ValueError(
            f"episode_to_anchor must be integer {(total_episodes,)}, "
            f"got {episode_to_anchor.shape}/{episode_to_anchor.dtype}"
        )
    if exec_start_idx.shape != (total_episodes,) or not np.issubdtype(exec_start_idx.dtype, np.integer):
        raise ValueError(
            f"exec_start_idx must be integer {(total_episodes,)}, got {exec_start_idx.shape}/{exec_start_idx.dtype}"
        )
    episode_to_anchor = episode_to_anchor.astype(np.int64, copy=False)
    exec_start_idx = exec_start_idx.astype(np.int64, copy=False)
    has_demo = exec_start_idx > 0
    if np.any(exec_start_idx < 0) or not np.array_equal(episode_to_anchor >= 0, has_demo):
        raise ValueError("Demo-anchor mapping must exist exactly for episodes with exec_start_idx > 0")
    mapped = episode_to_anchor[has_demo]
    if not np.array_equal(np.sort(mapped), np.arange(len(anchors), dtype=np.int64)):
        raise ValueError("Demo-anchor mapping must cover every cached anchor exactly once")
    if manifest.get("demo_episodes") != int(has_demo.sum()) or len(anchors) != int(has_demo.sum()):
        raise ValueError("Demo-anchor manifest count does not match the cache arrays")
    execution_images = None
    if definition == DUAL_ANCHOR_IMAGE_DEFINITION:
        execution_path = _cache_file(cache_dir, manifest, "execution_images_file", "execution_images.npy")
        if not execution_path.is_file():
            raise FileNotFoundError(f"Dual-anchor cache is missing {execution_path}")
        _verify_hash(manifest, "execution_images_sha256", execution_path)
        execution_images = np.load(execution_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (total_episodes, *anchors.shape[1:])
        if execution_images.shape != expected_shape or execution_images.dtype != np.uint8:
            raise ValueError(
                f"Execution-anchor images must be uint8 {expected_shape}, "
                f"got {execution_images.shape}/{execution_images.dtype}"
            )
    return anchors, episode_to_anchor, exec_start_idx, definition, execution_images


def _load_episode_anchor_cache(
    cache_dir: str | Path,
    *,
    dataset_root: Path,
    dataset_fingerprint: str,
    info_sha256: str,
    episodes_sha256: str,
    total_frames: int,
    total_episodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw cam_high frame-zero anchor for every episode.

    This cache is deliberately independent of demonstration metadata: row i is
    the raw first frame of episode i, and every training sample has an anchor.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing episode-anchor cache manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read episode-anchor cache manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Episode-anchor manifest {manifest_path} must contain a JSON object")
    if manifest.get("schema_version") != EPISODE_ANCHOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported episode-anchor schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Episode-anchor cache must be complete, got status={manifest.get('status')!r}")
    if manifest.get("anchor_definition") != EPISODE_ANCHOR_IMAGE_DEFINITION:
        raise ValueError(f"Unsupported episode-anchor definition: {manifest.get('anchor_definition')!r}")
    if manifest.get("image_key") != "observation.images.cam_high":
        raise ValueError(
            "Episode anchor must come from 'observation.images.cam_high', "
            f"got {manifest.get('image_key')!r}"
        )
    if Path(str(manifest.get("dataset_root", ""))).expanduser().resolve() != dataset_root:
        raise ValueError("Episode-anchor cache belongs to a different dataset root")
    identity_fields = {
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_info_sha256": info_sha256,
        "dataset_episodes_sha256": episodes_sha256,
    }
    for key, expected in identity_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Episode-anchor cache is stale: {key} does not match the dataset")
    if manifest.get("count") != total_episodes:
        raise ValueError(f"Episode-anchor count is {manifest.get('count')!r}, expected {total_episodes}")
    if manifest.get("declared_total_frames") != total_frames:
        raise ValueError(
            f"Episode-anchor frame total is {manifest.get('declared_total_frames')!r}, expected {total_frames}"
        )

    anchors_path = _cache_file(cache_dir, manifest, "anchors", "anchors.npy")
    episode_index_path = _cache_file(cache_dir, manifest, "episode_index", "episode_index.npy")
    global_index_path = _cache_file(cache_dir, manifest, "global_index", "global_index.npy")
    for key, path in (
        ("anchors_sha256", anchors_path),
        ("episode_index_sha256", episode_index_path),
        ("global_index_sha256", global_index_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Episode-anchor cache is missing {path}")
        _verify_hash(manifest, key, path)

    anchors = np.load(anchors_path, mmap_mode="r", allow_pickle=False)
    episode_index = np.load(episode_index_path, allow_pickle=False)
    global_index = np.load(global_index_path, allow_pickle=False)
    expected_shape = (total_episodes, 224, 224, 3)
    if anchors.shape != expected_shape or anchors.dtype != np.uint8:
        raise ValueError(f"Episode anchors must be uint8 {expected_shape}, got {anchors.shape}/{anchors.dtype}")
    if episode_index.shape != (total_episodes,) or not np.issubdtype(episode_index.dtype, np.integer):
        raise ValueError("Episode-anchor episode_index must be a one-dimensional integer array")
    if global_index.shape != (total_episodes,) or not np.issubdtype(global_index.dtype, np.integer):
        raise ValueError("Episode-anchor global_index must be a one-dimensional integer array")
    episode_index = episode_index.astype(np.int64, copy=False)
    global_index = global_index.astype(np.int64, copy=False)
    if not np.array_equal(episode_index, np.arange(total_episodes, dtype=np.int64)):
        raise ValueError("Episode-anchor episode_index must be ordered exactly as [0, total_episodes)")
    if np.any(global_index < 0) or np.any(np.diff(global_index) <= 0):
        raise ValueError("Episode-anchor global_index must be non-negative and strictly increasing")
    return anchors, episode_index, global_index


def _load_phase_metadata_cache(
    cache_dir: str | Path,
    *,
    dataset_root: Path,
    info_sha256: str,
    episodes_sha256: str,
    total_frames: int,
    total_episodes: int,
) -> dict[str, np.ndarray]:
    """Load and validate row-aligned demo/execution metadata.

    Phase metadata decides which observations are eligible for action loss, so
    every identity, hash, dtype, shape, and internal alignment check is
    fail-closed.  The returned arrays are immutable.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing phase-metadata cache manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read phase-metadata cache manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Phase-metadata manifest {manifest_path} must contain a JSON object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != PHASE_METADATA_SCHEMA_VERSION:
        raise ValueError(f"Unsupported phase-metadata schema_version={manifest.get('schema_version')!r}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Phase-metadata cache must be complete, got status={manifest.get('status')!r}")
    if Path(str(manifest.get("dataset_root", ""))).expanduser().resolve() != dataset_root:
        raise ValueError("Phase-metadata cache belongs to a different dataset root")
    for key, expected in (
        ("dataset_info_sha256", info_sha256),
        ("dataset_episodes_sha256", episodes_sha256),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"Phase-metadata cache is stale: {key} does not match the dataset")
    for key, expected in (("total_frames", total_frames), ("total_episodes", total_episodes)):
        if type(manifest.get(key)) is not int or manifest[key] != expected:
            raise ValueError(f"Phase-metadata {key} is {manifest.get(key)!r}, expected {expected}")
    if manifest.get("is_demo_definition") != PHASE_IS_DEMO_DEFINITION:
        raise ValueError(f"Unsupported is_demo definition: {manifest.get('is_demo_definition')!r}")
    if manifest.get("action_loss_definition") != PHASE_ACTION_LOSS_DEFINITION:
        raise ValueError(f"Unsupported action-loss definition: {manifest.get('action_loss_definition')!r}")
    expected_segment_ids = {
        "padding": MEMORY_SEGMENT_PADDING,
        "demo": MEMORY_SEGMENT_DEMO,
        "boundary": MEMORY_SEGMENT_BOUNDARY,
        "execution": MEMORY_SEGMENT_EXECUTION,
    }
    if manifest.get("transition_segment_ids") != expected_segment_ids:
        raise ValueError(
            "Phase-metadata transition_segment_ids must be exactly "
            f"{expected_segment_ids}, got {manifest.get('transition_segment_ids')!r}"
        )

    source_path = dataset_root / "meta" / "source_episodes.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(f"Phase-metadata identity requires {source_path}")
    _verify_hash(manifest, "dataset_source_episodes_sha256", source_path)

    specs = {
        "episode_index": ((total_episodes,), np.dtype(np.int64)),
        "anchor_global_index": ((total_episodes,), np.dtype(np.int64)),
        "episode_exec_start_idx": ((total_episodes,), np.dtype(np.int32)),
        "episode_has_demo": ((total_episodes,), np.dtype(np.bool_)),
        "episode_anchor_frame_index": ((total_episodes,), np.dtype(np.int32)),
        "episode_anchor_segment_id": ((total_episodes,), np.dtype(np.uint8)),
        "row_episode_index": ((total_frames,), np.dtype(np.int32)),
        "row_frame_index": ((total_frames,), np.dtype(np.int32)),
        "row_exec_start_idx": ((total_frames,), np.dtype(np.int32)),
        "row_is_demo": ((total_frames,), np.dtype(np.bool_)),
        "row_action_loss_mask": ((total_frames,), np.dtype(np.bool_)),
        "row_transition_segment_id": ((total_frames,), np.dtype(np.uint8)),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, (expected_shape, expected_dtype) in specs.items():
        path = _cache_file(cache_dir, manifest, key, f"{key}.npy")
        if not path.is_file():
            raise FileNotFoundError(f"Phase-metadata cache is missing {path}")
        _verify_hash(manifest, f"{key}_sha256", path)
        stored = np.load(path, mmap_mode="r", allow_pickle=False)
        if stored.shape != expected_shape or stored.dtype != expected_dtype:
            raise ValueError(
                f"Phase-metadata array {key!r} must be {expected_dtype} {expected_shape}, "
                f"got {stored.dtype} {stored.shape}"
            )
        arrays[key] = np.asarray(stored).copy()

    episode_manifest_path = _cache_file(cache_dir, manifest, "episode_manifest", "episodes.jsonl")
    if not episode_manifest_path.is_file():
        raise FileNotFoundError(f"Phase-metadata cache is missing {episode_manifest_path}")
    _verify_hash(manifest, "episode_manifest_sha256", episode_manifest_path)
    try:
        episode_records = [json.loads(line) for line in episode_manifest_path.read_text().splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read phase episode manifest {episode_manifest_path}: {error}") from error
    if len(episode_records) != total_episodes or not all(isinstance(record, dict) for record in episode_records):
        raise ValueError(f"Phase episode manifest must contain exactly {total_episodes} JSON objects")

    episode_index = arrays["episode_index"]
    row_episode = arrays["row_episode_index"].astype(np.int64, copy=False)
    row_frame = arrays["row_frame_index"].astype(np.int64, copy=False)
    episode_exec = arrays["episode_exec_start_idx"].astype(np.int64, copy=False)
    row_exec = arrays["row_exec_start_idx"].astype(np.int64, copy=False)
    if not np.array_equal(episode_index, np.arange(total_episodes, dtype=np.int64)):
        raise ValueError("Phase episode_index must be ordered exactly as [0, total_episodes)")
    if np.any(row_episode < 0) or np.any(row_episode >= total_episodes) or np.any(np.diff(row_episode) < 0):
        raise ValueError("Phase row_episode_index must be ordered and within the episode range")
    episode_start_mask = np.ones(total_frames, dtype=np.bool_)
    episode_start_mask[1:] = row_episode[1:] != row_episode[:-1]
    starts = np.flatnonzero(episode_start_mask)
    ends = np.r_[starts[1:], total_frames]
    if len(starts) != total_episodes or not np.array_equal(row_episode[starts], episode_index):
        raise ValueError("Phase row_episode_index does not contain every episode exactly once")
    if np.any(row_frame[starts] != 0) or np.any(np.diff(row_frame)[~episode_start_mask[1:]] != 1):
        raise ValueError("Phase row_frame_index must start at zero and increase consecutively within each episode")
    if not np.array_equal(arrays["anchor_global_index"], starts.astype(np.int64, copy=False)):
        raise ValueError("Phase anchor_global_index must identify every episode's first global row")
    if np.any(arrays["episode_anchor_frame_index"] != 0):
        raise ValueError("Phase episode anchors must all use frame_index zero")
    if np.any(episode_exec < 0):
        raise ValueError("Phase exec_start_idx must be non-negative")

    for ordinal, (start, end) in enumerate(zip(starts, ends, strict=True)):
        boundary = int(episode_exec[ordinal])
        length = int(end - start)
        if boundary >= length:
            raise ValueError(f"Phase episode {ordinal} has exec_start_idx={boundary} outside length {length}")
        if np.any(row_exec[start:end] != boundary):
            raise ValueError(f"Phase row_exec_start_idx changes within episode {ordinal}")
        record = episode_records[ordinal]
        expected_record = {
            "episode_index": ordinal,
            "exec_start_idx": boundary,
            "has_demo": bool(boundary > 0),
            "anchor_frame_index": 0,
            "length": length,
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError(f"Phase episode manifest disagrees with arrays at episode {ordinal}")

    expected_has_demo = episode_exec > 0
    if not np.array_equal(arrays["episode_has_demo"], expected_has_demo):
        raise ValueError("Phase episode_has_demo must equal episode_exec_start_idx > 0")
    expected_anchor_segment = np.where(
        expected_has_demo, MEMORY_SEGMENT_DEMO, MEMORY_SEGMENT_EXECUTION
    ).astype(np.uint8)
    if not np.array_equal(arrays["episode_anchor_segment_id"], expected_anchor_segment):
        raise ValueError("Phase episode_anchor_segment_id does not match demo/execution anchor roles")
    if not np.array_equal(row_exec, episode_exec[row_episode]):
        raise ValueError("Phase row_exec_start_idx does not match episode_exec_start_idx")
    expected_is_demo = row_frame < row_exec
    if not np.array_equal(arrays["row_is_demo"], expected_is_demo):
        raise ValueError("Phase row_is_demo must equal row_frame_index < row_exec_start_idx")
    if not np.array_equal(arrays["row_action_loss_mask"], ~expected_is_demo):
        raise ValueError("Phase row_action_loss_mask must be the inverse of row_is_demo")
    expected_segments = np.full(total_frames, MEMORY_SEGMENT_EXECUTION, dtype=np.uint8)
    expected_segments[episode_start_mask] = MEMORY_SEGMENT_PADDING
    expected_segments[(~episode_start_mask) & expected_is_demo] = MEMORY_SEGMENT_DEMO
    expected_segments[(row_exec > 0) & (row_frame == row_exec)] = MEMORY_SEGMENT_BOUNDARY
    if not np.array_equal(arrays["row_transition_segment_id"], expected_segments):
        raise ValueError("Phase row_transition_segment_id does not match the demo/boundary/execution contract")

    expected_counts = {
        str(segment): int(np.count_nonzero(expected_segments == segment))
        for segment in (
            MEMORY_SEGMENT_PADDING,
            MEMORY_SEGMENT_DEMO,
            MEMORY_SEGMENT_BOUNDARY,
            MEMORY_SEGMENT_EXECUTION,
        )
    }
    if manifest.get("transition_segment_counts") != expected_counts:
        raise ValueError("Phase transition_segment_counts does not match row_transition_segment_id")
    for key, expected in (
        ("demo_episodes", int(expected_has_demo.sum())),
        ("demo_frames", int(expected_is_demo.sum())),
        ("execution_query_frames", int((~expected_is_demo).sum())),
    ):
        if type(manifest.get(key)) is not int or manifest[key] != expected:
            raise ValueError(f"Phase manifest {key} does not match its arrays")

    for array in arrays.values():
        array.setflags(write=False)
    return arrays


class VisionAnchorDataset(Sized):
    """Attach validated demo/execution vision anchors without any LAM tensors."""

    def __init__(
        self,
        dataset,
        anchor_cache_dir: str | Path,
        *,
        expected_dataset_root: str | Path,
    ):
        self._dataset = dataset
        dataset_root = Path(expected_dataset_root).expanduser().resolve()
        total_frames, total_episodes, _, episodes_sha256, fingerprint = _dataset_identity(dataset_root)
        if len(dataset) != total_frames:
            raise ValueError(
                f"Training dataset has {len(dataset)} rows but anchor metadata has {total_frames}; expected equality"
            )
        (
            self._demo_anchor_images,
            self._episode_to_demo_anchor,
            self._exec_start_by_episode,
            definition,
            self._execution_anchor_images,
        ) = _load_demo_anchor_cache(
            anchor_cache_dir,
            dataset_root=dataset_root,
            dataset_fingerprint=fingerprint,
            dataset_episodes_sha256=episodes_sha256,
            total_episodes=total_episodes,
            expected_definition=DUAL_ANCHOR_IMAGE_DEFINITION,
        )
        if definition != DUAL_ANCHOR_IMAGE_DEFINITION or self._execution_anchor_images is None:
            raise ValueError("vision-anchor-only training requires a schema-v2 dual-anchor image cache")
        self._demo_anchor_images_path = Path(self._demo_anchor_images.filename)
        self._execution_anchor_images_path = Path(self._execution_anchor_images.filename)
        self._empty_demo_anchor = np.zeros(self._demo_anchor_images.shape[1:], dtype=self._demo_anchor_images.dtype)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_demo_anchor_images"] = None
        state["_execution_anchor_images"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._demo_anchor_images = np.load(self._demo_anchor_images_path, mmap_mode="r", allow_pickle=False)
        self._execution_anchor_images = np.load(
            self._execution_anchor_images_path, mmap_mode="r", allow_pickle=False
        )

    @staticmethod
    def _scalar(sample: dict, key: str) -> int:
        if key not in sample:
            raise KeyError(f"Vision-anchor sample is missing scalar field {key!r}")
        value = np.asarray(sample[key]).reshape(-1)
        if value.size != 1:
            raise ValueError(f"Expected scalar {key}, got shape {np.asarray(sample[key]).shape}")
        return int(value[0])

    def __getitem__(self, index: Any) -> dict:
        position = index.__index__()
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)

        sample = self._dataset[position]
        global_index = self._scalar(sample, "index")
        if global_index != position:
            raise ValueError(f"LeRobot global index {global_index} does not match row position {position}")
        episode_index = self._scalar(sample, "episode_index")
        frame_index = self._scalar(sample, "frame_index")
        exec_start_idx = self._scalar(sample, "exec_start_idx")
        expected_exec_start = int(self._exec_start_by_episode[episode_index])
        if exec_start_idx != expected_exec_start:
            raise ValueError(
                f"Anchor exec_start_idx mismatch for episode {episode_index}: "
                f"cache={expected_exec_start}, sample={exec_start_idx}"
            )

        anchor_index = int(self._episode_to_demo_anchor[episode_index])
        has_demo_anchor = anchor_index >= 0
        sample["memory_demo_start_image"] = (
            np.asarray(self._demo_anchor_images[anchor_index]) if has_demo_anchor else self._empty_demo_anchor
        )
        sample["memory_demo_start_mask"] = np.bool_(has_demo_anchor)
        execution_visible = frame_index >= exec_start_idx
        sample["memory_execution_start_image"] = (
            np.asarray(self._execution_anchor_images[episode_index])
            if execution_visible
            else np.zeros_like(self._execution_anchor_images[episode_index])
        )
        sample["memory_execution_start_mask"] = np.bool_(execution_visible)
        for forbidden in ("memory_latents", "memory_mask", "memory_segment_ids"):
            if forbidden in sample:
                raise ValueError(f"vision-anchor-only dataset unexpectedly received {forbidden}")
        return sample


class CDLAMMemoryDataset(Sized):
    """Attach the complete causal episode latent history to each sample.

    The cache is indexed by the same global frame order as the full LeRobot
    dataset. A sample at episode frame ``t`` contains every valid transition
    row ``1..t``. Frame zero receives one masked dummy token so batches never
    have a zero-length sequence. Variable histories are left-padded by the
    PyTorch collator, and no row from another episode is ever returned.
    """

    def __init__(
        self,
        dataset,
        cache_dir: str | Path,
        *,
        memory_horizon: int,
        latent_dim: int,
        memory_stride: int = 1,
        expected_dataset_root: str | Path | None = None,
        expected_latent_mean=None,
        expected_latent_std=None,
        random_drop_execution_tail_min: int = 0,
        random_drop_execution_tail_max: int = 0,
        episode_anchor_cache_dir: str | Path | None = None,
        phase_metadata_cache_dir: str | Path | None = None,
        demo_anchor_cache_dir: str | Path | None = None,
        expected_demo_anchor_definition: str | None = None,
        execution_anchor: bool = False,
        future_prediction_horizon: int = 0,
        future_cache_dir: str | Path | None = None,
        future_latent_dim: int | None = None,
        expected_future_latent_mean=None,
        expected_future_latent_std=None,
        demo_direction_text_by_episode: tuple[str, ...] | None = None,
    ):
        if memory_horizon <= 0:
            raise ValueError("memory_horizon must be positive")
        if type(memory_stride) is not int or memory_stride <= 0:
            raise ValueError("memory_stride must be a positive integer")
        if future_prediction_horizon < 0:
            raise ValueError("future_prediction_horizon must be non-negative")
        if future_latent_dim is not None and future_latent_dim <= 0:
            raise ValueError("future_latent_dim must be positive")
        if (
            type(random_drop_execution_tail_min) is not int
            or type(random_drop_execution_tail_max) is not int
            or random_drop_execution_tail_min < 0
            or random_drop_execution_tail_max < random_drop_execution_tail_min
        ):
            raise ValueError("execution LAM tail-drop bounds must satisfy 0 <= min <= max")
        future_options = (
            future_cache_dir,
            future_latent_dim,
            expected_future_latent_mean,
            expected_future_latent_std,
        )
        if not future_prediction_horizon and any(option is not None for option in future_options):
            raise ValueError("Future-cache options require future_prediction_horizon > 0")
        self._dataset = dataset
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        self._memory_horizon = memory_horizon
        self._latent_dim = latent_dim
        self._memory_stride = memory_stride
        self._random_drop_execution_tail_min = random_drop_execution_tail_min
        self._random_drop_execution_tail_max = random_drop_execution_tail_max
        self._future_prediction_horizon = future_prediction_horizon
        self._future_latent_dim = future_latent_dim if future_latent_dim is not None else latent_dim
        self._future_latent_path = None
        self._future_latents = None
        self._demo_direction_text_by_episode = demo_direction_text_by_episode
        self.future_latent_mean = None
        self.future_latent_std = None
        self._episode_anchor_images = None
        self._episode_anchor_images_path = None
        self._episode_anchor_global_index = None
        self._phase_row_exec_start_idx = None
        self._phase_row_is_demo = None
        self._phase_row_action_loss_mask = None
        self._phase_transition_segment_ids = None
        self._scalar_columns: dict[str, np.ndarray] = {}

        manifest_path = self._cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing finalized CD-LAM cache manifest: {manifest_path}. "
                "Run scripts/precompute_dreamdojo_memory.py finalize first."
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Cannot read CD-LAM cache manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise ValueError(f"CD-LAM cache manifest {manifest_path} must contain a JSON object")
        if _manifest_int(manifest, "schema_version") != EXPECTED_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CD-LAM manifest schema_version={manifest.get('schema_version')!r}; "
                f"expected {EXPECTED_MANIFEST_SCHEMA_VERSION}"
            )
        if _manifest_int(manifest, "cache_schema_version") != EXPECTED_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CD-LAM cache_schema_version={manifest.get('cache_schema_version')!r}; "
                f"expected {EXPECTED_CACHE_SCHEMA_VERSION}"
            )
        if manifest.get("status") != "complete":
            raise ValueError(f"CD-LAM cache must be complete, got status={manifest.get('status')!r}")
        missing_episodes = manifest.get("missing_episodes")
        if missing_episodes != []:
            raise ValueError(f"Complete CD-LAM cache must have no missing episodes, got {missing_episodes!r}")
        if _manifest_int(manifest, "latent_dim") != latent_dim:
            raise ValueError(
                f"CD-LAM latent dimension mismatch: cache={manifest.get('latent_dim')}, model={latent_dim}"
            )
        transition_frame_stride = _manifest_transition_frame_stride(manifest)
        if transition_frame_stride != self._memory_stride:
            raise ValueError(
                "CD-LAM cache transition_frame_stride must match model memory_stride: "
                f"cache={transition_frame_stride}, model={self._memory_stride}"
            )
        expected_alignment = _expected_cdlam_alignment(transition_frame_stride)
        if transition_frame_stride == 1 and manifest.get("alignment") == DELTATOK_ALIGNMENT:
            pass
        elif manifest.get("alignment") != expected_alignment:
            raise ValueError(
                "Unsupported CD-LAM cache alignment for transition_frame_stride "
                f"{transition_frame_stride}: {manifest.get('alignment')!r}"
            )
        self._transition_frame_stride = transition_frame_stride

        cached_dataset_root = manifest.get("dataset_root")
        if not isinstance(cached_dataset_root, str) or not cached_dataset_root:
            raise ValueError("CD-LAM manifest field 'dataset_root' must be a non-empty path string")
        cached_root = Path(cached_dataset_root).expanduser().resolve()
        if expected_dataset_root is not None:
            expected_root = Path(expected_dataset_root).expanduser().resolve()
        else:
            expected_root = cached_root

        total_frames, total_episodes, info_sha256, episodes_sha256, fingerprint = _dataset_identity(expected_root)
        if self._demo_direction_text_by_episode is not None:
            if len(self._demo_direction_text_by_episode) != total_episodes:
                raise ValueError(
                    "demo_direction_text_by_episode must have one entry per episode: "
                    f"got {len(self._demo_direction_text_by_episode)}, expected {total_episodes}"
                )
            if any(not isinstance(value, str) for value in self._demo_direction_text_by_episode):
                raise ValueError("demo_direction_text_by_episode entries must be strings")
        identity_fields = {"dataset_episodes_sha256": episodes_sha256}
        if cached_root == expected_root:
            identity_fields.update(
                dataset_info_sha256=info_sha256,
                dataset_fingerprint=fingerprint,
            )
        for key, expected in identity_fields.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"CD-LAM cache is stale or belongs to different data: {key}="
                    f"{manifest.get(key)!r}, current={expected!r}"
                )

        expected_episodes = _manifest_int(manifest, "expected_episodes")
        cached_episodes = _manifest_int(manifest, "cached_episodes")
        declared_total_frames = _manifest_int(manifest, "declared_total_frames")
        cached_total_frames = _manifest_int(manifest, "cached_total_frames")
        cached_valid_transitions = _manifest_int(manifest, "cached_valid_transitions")
        if expected_episodes != total_episodes or cached_episodes != total_episodes:
            raise ValueError(
                "CD-LAM episode totals do not match current LeRobot metadata: "
                f"expected={expected_episodes}, cached={cached_episodes}, current={total_episodes}"
            )
        if declared_total_frames != total_frames or cached_total_frames != total_frames:
            raise ValueError(
                "CD-LAM frame totals do not match current LeRobot metadata: "
                f"declared={declared_total_frames}, cached={cached_total_frames}, current={total_frames}"
            )
        # The history cache is the count contract for any separately aligned
        # future cache. Exact endpoint placement is verified below after the
        # global index is loaded.
        expected_valid_transitions = cached_valid_transitions
        self.latent_mean, self.latent_std = _latent_normalization(manifest, latent_dim, cached_valid_transitions)
        if episode_anchor_cache_dir is not None:
            (
                self._episode_anchor_images,
                _,
                self._episode_anchor_global_index,
            ) = _load_episode_anchor_cache(
                episode_anchor_cache_dir,
                dataset_root=expected_root,
                dataset_fingerprint=fingerprint,
                info_sha256=info_sha256,
                episodes_sha256=episodes_sha256,
                total_frames=total_frames,
                total_episodes=total_episodes,
            )
            self._episode_anchor_images_path = Path(self._episode_anchor_images.filename)
        self._demo_anchor_images = None
        self._demo_anchor_images_path = None
        self._demo_anchor_features = None
        self._demo_anchor_features_path = None
        self._execution_anchor_images = None
        self._execution_anchor_images_path = None
        self._episode_to_demo_anchor = None
        self._exec_start_by_episode = None
        self._empty_demo_anchor = None
        if demo_anchor_cache_dir is not None:
            (
                anchors,
                self._episode_to_demo_anchor,
                self._exec_start_by_episode,
                definition,
                execution_images,
            ) = _load_demo_anchor_cache(
                demo_anchor_cache_dir,
                dataset_root=expected_root,
                dataset_fingerprint=fingerprint,
                dataset_episodes_sha256=episodes_sha256,
                total_episodes=total_episodes,
                expected_definition=expected_demo_anchor_definition,
            )
            if definition in (DEMO_ANCHOR_IMAGE_DEFINITION, DUAL_ANCHOR_IMAGE_DEFINITION):
                self._demo_anchor_images = anchors
                self._demo_anchor_images_path = Path(anchors.filename)
            else:
                self._demo_anchor_features = anchors
                self._demo_anchor_features_path = Path(anchors.filename)
            self._empty_demo_anchor = np.zeros(anchors.shape[1:], dtype=anchors.dtype)
            if execution_anchor:
                if execution_images is None:
                    raise ValueError("memory_execution_anchor requires a schema-v2 dual-anchor cache")
                self._execution_anchor_images = execution_images
                self._execution_anchor_images_path = Path(execution_images.filename)
        elif execution_anchor:
            raise ValueError("memory_execution_anchor requires demo_anchor_cache_dir")
        _validate_expected_latent_normalization(
            self.latent_mean,
            self.latent_std,
            latent_dim=latent_dim,
            expected_mean=expected_latent_mean,
            expected_std=expected_latent_std,
            label="",
        )

        index_path = _cache_file(self._cache_dir, manifest, "global_frame_index", "frame_index.npz")
        latent_path = _cache_file(self._cache_dir, manifest, "global_latents", "latents.npy")
        if not index_path.is_file() or not latent_path.is_file():
            raise FileNotFoundError(f"Finalized CD-LAM cache is missing {index_path.name} or {latent_path.name}")
        for hash_key, path in (
            ("global_frame_index_sha256", index_path),
            ("global_latents_sha256", latent_path),
        ):
            if hash_key not in manifest:
                raise ValueError(f"CD-LAM schema v2 manifest is missing required field {hash_key!r}")
            _verify_hash(manifest, hash_key, path)

        with np.load(index_path, allow_pickle=False) as index:
            required = {"global_index", "episode_index", "frame_index", "valid"}
            if not required.issubset(index.files):
                raise ValueError(f"{index_path} is missing arrays {sorted(required - set(index.files))}")
            arrays = {name: index[name] for name in required}
            for name, array in arrays.items():
                if array.ndim != 1 or len(array) != cached_total_frames:
                    raise ValueError(
                        f"CD-LAM index array {name!r} has shape {array.shape}; expected {(cached_total_frames,)}"
                    )
            for name in ("global_index", "episode_index", "frame_index"):
                if not np.issubdtype(arrays[name].dtype, np.integer):
                    raise ValueError(f"CD-LAM index array {name!r} must be integer, got {arrays[name].dtype}")
            if arrays["valid"].dtype != np.bool_:
                raise ValueError(f"CD-LAM index array 'valid' must be bool, got {arrays['valid'].dtype}")
            self._global_index = arrays["global_index"].astype(np.int64, copy=True)
            self._episode_index = arrays["episode_index"].astype(np.int64, copy=True)
            self._frame_index = arrays["frame_index"].astype(np.int64, copy=True)
            self._valid = arrays["valid"].copy()

        self._latent_path = latent_path
        self._latents = np.load(self._latent_path, mmap_mode="r", allow_pickle=False)

        cache_length = len(self._global_index)
        if len(dataset) != cache_length:
            raise ValueError(
                f"Training dataset has {len(dataset)} rows but cache has {cache_length}; expected equality"
            )
        if not np.array_equal(self._global_index, np.arange(cache_length, dtype=np.int64)):
            raise ValueError(f"CD-LAM global indices must cover the exact range [0, {cache_length})")
        if np.any(self._episode_index < 0) or np.any(self._episode_index >= total_episodes):
            raise ValueError(f"CD-LAM episode indices must be within [0, {total_episodes})")
        if cache_length > 1 and np.any(np.diff(self._episode_index) < 0):
            raise ValueError("CD-LAM episode indices must be nondecreasing in global frame order")
        if len(np.unique(self._episode_index)) != total_episodes:
            raise ValueError(
                f"CD-LAM index covers {len(np.unique(self._episode_index))} episodes, expected {total_episodes}"
            )
        if np.any(self._frame_index < 0):
            raise ValueError("CD-LAM frame indices must be non-negative")
        episode_start = np.ones(cache_length, dtype=np.bool_)
        if cache_length > 1:
            episode_start[1:] = self._episode_index[1:] != self._episode_index[:-1]
            within_episode = ~episode_start[1:]
            frame_deltas = np.diff(self._frame_index)
            if np.any(frame_deltas[within_episode] != 1):
                raise ValueError("CD-LAM frame indices must increase consecutively within each episode")
        if np.any(self._frame_index[episode_start] != 0):
            raise ValueError("Every CD-LAM episode must start at frame_index 0")
        episode_starts = np.flatnonzero(episode_start)
        if self._episode_anchor_global_index is not None and not np.array_equal(
            self._episode_anchor_global_index, self._global_index[episode_starts]
        ):
            raise ValueError("Episode-anchor global_index does not match CD-LAM episode starts")
        if phase_metadata_cache_dir is not None:
            phase = _load_phase_metadata_cache(
                phase_metadata_cache_dir,
                dataset_root=expected_root,
                info_sha256=info_sha256,
                episodes_sha256=episodes_sha256,
                total_frames=total_frames,
                total_episodes=total_episodes,
            )
            if not np.array_equal(phase["row_episode_index"], self._episode_index):
                raise ValueError("Phase row_episode_index does not exactly match the CD-LAM cache")
            if not np.array_equal(phase["row_frame_index"], self._frame_index):
                raise ValueError("Phase row_frame_index does not exactly match the CD-LAM cache")
            if not np.array_equal(phase["anchor_global_index"], self._global_index[episode_starts]):
                raise ValueError("Phase anchor_global_index does not match the CD-LAM episode starts")
            if self._exec_start_by_episode is not None and not np.array_equal(
                phase["episode_exec_start_idx"], self._exec_start_by_episode
            ):
                raise ValueError("Phase execution boundaries disagree with the demo-anchor cache")
            self._phase_row_exec_start_idx = phase["row_exec_start_idx"]
            self._phase_row_is_demo = phase["row_is_demo"]
            self._phase_row_action_loss_mask = phase["row_action_loss_mask"]
            self._phase_transition_segment_ids = phase["row_transition_segment_id"]
            self._scalar_columns = {
                "episode_index": phase["row_episode_index"],
                "frame_index": phase["row_frame_index"],
                "exec_start_idx": self._phase_row_exec_start_idx,
                "is_demo": self._phase_row_is_demo,
                "action_loss_mask": self._phase_row_action_loss_mask,
                "transition_segment_id": self._phase_transition_segment_ids,
            }
        self._episode_end_by_episode = np.r_[episode_starts[1:], cache_length].astype(np.int64, copy=False)
        # A valid row is an endpoint of an actual stride-sized CD-LAM pair.
        # For stride four this is exactly frame 4, 8, 12, ... with values
        # encoded from 0->4, 4->8, 8->12, ...; all other rows are intentionally
        # invalid and must never enter the memory sequence.
        expected_valid = (
            (self._frame_index >= self._transition_frame_stride)
            & ((self._frame_index % self._transition_frame_stride) == 0)
        )
        if not np.array_equal(self._valid, expected_valid):
            raise ValueError(
                "CD-LAM valid must be false outside complete transition-frame-stride endpoints"
            )
        if int(self._valid.sum()) != cached_valid_transitions:
            raise ValueError(
                f"CD-LAM valid array contains {int(self._valid.sum())} transitions, "
                f"manifest declares {cached_valid_transitions}"
            )
        max_episode_transitions = int(self._frame_index.max())
        max_memory_tokens = max_episode_transitions // self._memory_stride
        if memory_horizon < max_memory_tokens:
            raise ValueError(
                "memory_horizon must hold the complete episode history after stride selection without truncation: "
                f"configured={memory_horizon}, required={max_memory_tokens}, "
                f"memory_stride={self._memory_stride}"
            )
        self.max_episode_transitions = max_episode_transitions
        self.max_memory_tokens = max_memory_tokens
        # This is the exact sequence length produced by __getitem__ without
        # touching the underlying image dataset. Every entry is an actual
        # stride-separated CD-LAM encoding; incomplete prefix/suffix frames do
        # not synthesize a memory token. Episode starts use a masked dummy.
        self._memory_lengths = np.maximum(self._frame_index // self._memory_stride, 1)
        self._memory_lengths.setflags(write=False)
        if self._latents.shape != (cache_length, latent_dim):
            raise ValueError(
                f"CD-LAM global latent shape mismatch: got {self._latents.shape}, expected {(cache_length, latent_dim)}"
            )
        if not np.issubdtype(self._latents.dtype, np.floating):
            raise ValueError(f"CD-LAM latents must be floating point, got {self._latents.dtype}")
        if np.any(self._latents[~self._valid] != 0):
            raise ValueError("CD-LAM latent rows marked invalid must be exactly zero")
        for start in range(0, cache_length, 65_536):
            block = self._latents[start : start + 65_536]
            if not np.all(np.isfinite(block)):
                raise ValueError(f"CD-LAM latents contain non-finite values near global row {start}")

        if self._future_prediction_horizon:
            resolved_future_cache_dir = (
                self._cache_dir if future_cache_dir is None else Path(future_cache_dir).expanduser().resolve()
            )
            if resolved_future_cache_dir == self._cache_dir:
                if self._future_latent_dim != latent_dim:
                    raise ValueError(
                        "A different future_latent_dim requires a separate future_cache_dir: "
                        f"history={latent_dim}, future={self._future_latent_dim}"
                    )
                _validate_expected_latent_normalization(
                    self.latent_mean,
                    self.latent_std,
                    latent_dim=latent_dim,
                    expected_mean=expected_future_latent_mean,
                    expected_std=expected_future_latent_std,
                    label="future",
                )
                self._future_latent_path = self._latent_path
                self._future_latents = self._latents
                self.future_latent_mean = self.latent_mean
                self.future_latent_std = self.latent_std
            else:
                reference_indices = {
                    "global_index": self._global_index,
                    "episode_index": self._episode_index,
                    "frame_index": self._frame_index,
                    "valid": self._valid,
                }
                (
                    self._future_latent_path,
                    self._future_latents,
                    self.future_latent_mean,
                    self.future_latent_std,
                ) = _load_aligned_future_cache(
                    resolved_future_cache_dir,
                    expected_root=expected_root,
                    total_frames=total_frames,
                    total_episodes=total_episodes,
                    info_sha256=info_sha256,
                    episodes_sha256=episodes_sha256,
                    fingerprint=fingerprint,
                    expected_valid_transitions=expected_valid_transitions,
                    latent_dim=self._future_latent_dim,
                    reference_indices=reference_indices,
                    expected_latent_mean=expected_future_latent_mean,
                    expected_latent_std=expected_future_latent_std,
                )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getstate__(self) -> dict[str, Any]:
        """Do not serialize multi-gigabyte mmap contents into spawned workers."""
        state = self.__dict__.copy()
        state["_latents"] = None
        state["_future_latents"] = None
        state["_episode_anchor_images"] = None
        state["_demo_anchor_images"] = None
        state["_demo_anchor_features"] = None
        state["_execution_anchor_images"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for array in self._scalar_columns.values():
            array.setflags(write=False)
        self._latents = np.load(self._latent_path, mmap_mode="r", allow_pickle=False)
        if self._future_latent_path is None:
            self._future_latents = None
        elif self._future_latent_path == self._latent_path:
            self._future_latents = self._latents
        else:
            self._future_latents = np.load(self._future_latent_path, mmap_mode="r", allow_pickle=False)
        if self._episode_anchor_images_path is not None:
            self._episode_anchor_images = np.load(
                self._episode_anchor_images_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        if self._demo_anchor_images_path is not None:
            self._demo_anchor_images = np.load(
                self._demo_anchor_images_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        if self._demo_anchor_features_path is not None:
            self._demo_anchor_features = np.load(
                self._demo_anchor_features_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        if self._execution_anchor_images_path is not None:
            self._execution_anchor_images = np.load(
                self._execution_anchor_images_path,
                mmap_mode="r",
                allow_pickle=False,
            )

    @property
    def memory_lengths(self) -> np.ndarray:
        """Return read-only per-sample memory lengths for batch planning."""
        return self._memory_lengths

    @property
    def scalar_columns(self):
        """Return immutable row-aligned columns supplied by the phase sidecar."""
        return MappingProxyType(self._scalar_columns)

    def __getitem__(self, index: Any) -> dict:
        position = index.__index__()
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)

        sample = self._dataset[position]
        if "index" not in sample:
            raise KeyError("LeRobot sample is missing its global 'index' field")
        global_value = np.asarray(sample["index"]).reshape(-1)
        if global_value.size != 1:
            raise ValueError(f"Expected scalar LeRobot global index, got shape {np.asarray(sample['index']).shape}")
        global_index = int(global_value[0])
        cache_position = int(np.searchsorted(self._global_index, global_index))
        if cache_position >= len(self._global_index) or self._global_index[cache_position] != global_index:
            raise KeyError(f"Global frame index {global_index} is absent from the CD-LAM cache")
        episode_frame = int(self._frame_index[cache_position])
        if self._phase_row_exec_start_idx is not None:
            exec_start_idx = int(self._phase_row_exec_start_idx[cache_position])
            is_demo = bool(self._phase_row_is_demo[cache_position])
            for key, expected in (("exec_start_idx", exec_start_idx), ("is_demo", is_demo)):
                if key not in sample:
                    continue
                value = np.asarray(sample[key]).reshape(-1)
                if value.size != 1 or value[0] != expected:
                    raise ValueError(
                        f"LeRobot {key} disagrees with phase metadata at global frame {global_index}: "
                        f"sample={np.asarray(sample[key])!r}, phase={expected!r}"
                    )
            sample["exec_start_idx"] = np.asarray(exec_start_idx, dtype=np.int32)
            sample["is_demo"] = np.asarray(is_demo, dtype=np.bool_)
        else:
            exec_start_idx = 0
            if "exec_start_idx" in sample:
                exec_value = np.asarray(sample["exec_start_idx"]).reshape(-1)
                if exec_value.size != 1:
                    raise ValueError(
                        f"Expected scalar exec_start_idx, got {np.asarray(sample['exec_start_idx']).shape}"
                    )
                exec_start_idx = int(exec_value[0])
            is_demo = episode_frame < exec_start_idx
        if exec_start_idx < 0 or exec_start_idx > self.max_episode_transitions:
            raise ValueError(f"Invalid exec_start_idx={exec_start_idx} for episode frame {episode_frame}")
        if "is_demo" in sample:
            is_demo_value = np.asarray(sample["is_demo"]).reshape(-1)
            if is_demo_value.size != 1 or bool(is_demo_value[0]) != is_demo:
                raise ValueError(
                    f"is_demo/exec_start_idx mismatch at global frame {global_index}: "
                    f"frame={episode_frame}, exec_start_idx={exec_start_idx}"
                )
        if episode_frame == 0:
            # PyTorch attention and collation require a non-empty sequence. The
            # dummy is excluded as both a query and key by memory_mask.
            output_latents = np.zeros((1, self._latent_dim), dtype=np.float32)
            output_mask = np.zeros(1, dtype=np.bool_)
            output_segment_ids = np.full(1, MEMORY_SEGMENT_PADDING, dtype=np.int32)
        else:
            episode_start = cache_position - episode_frame
            history_positions = np.arange(episode_start, cache_position + 1, dtype=np.int64)
            source_valid = self._valid[history_positions]
            same_episode = self._episode_index[history_positions] == self._episode_index[cache_position]
            if not np.all(same_episode):
                raise RuntimeError(f"Cross-episode CD-LAM history at global frame {global_index}")
            # The sparse cache validity mask is the source of truth: selecting
            # it yields only precomputed stride-separated pairs, never an
            # adjacent-frame latent sampled every N rows.
            source_positions = history_positions[source_valid]
            if len(source_positions) > self._memory_horizon:
                raise RuntimeError(
                    f"Complete strided memory length {len(source_positions)} exceeds configured maximum "
                    f"{self._memory_horizon}"
                )
            if len(source_positions) == 0:
                output_latents = np.zeros((1, self._latent_dim), dtype=np.float32)
                output_mask = np.zeros(1, dtype=np.bool_)
                output_segment_ids = np.full(1, MEMORY_SEGMENT_PADDING, dtype=np.int32)
            else:
                output_latents = np.asarray(self._latents[source_positions], dtype=np.float32).copy()
                output_mask = np.ones(len(source_positions), dtype=np.bool_)
                transition_frames = self._frame_index[source_positions]
                if self._phase_transition_segment_ids is None:
                    output_segment_ids = classify_memory_segments(transition_frames, exec_start_idx)
                else:
                    output_segment_ids = np.asarray(
                        self._phase_transition_segment_ids[source_positions], dtype=np.int32
                    ).copy()
                if not is_demo and self._random_drop_execution_tail_max:
                    requested_drop = int(
                        np.random.randint(
                            self._random_drop_execution_tail_min,
                            self._random_drop_execution_tail_max + 1,
                        )
                    )
                    output_latents, output_mask, output_segment_ids = drop_execution_lam_tail(
                        output_latents,
                        output_mask,
                        output_segment_ids,
                        requested_drop,
                    )
                    if len(output_latents) == 0:
                        output_latents = np.zeros((1, self._latent_dim), dtype=np.float32)
                        output_mask = np.zeros(1, dtype=np.bool_)
                        output_segment_ids = np.full(1, MEMORY_SEGMENT_PADDING, dtype=np.int32)

        sample["memory_latents"] = output_latents
        sample["memory_mask"] = output_mask
        sample["memory_segment_ids"] = output_segment_ids
        if self._demo_direction_text_by_episode is not None:
            episode_index = int(self._episode_index[cache_position])
            direction_text = self._demo_direction_text_by_episode[episode_index]
            # Mixed-task datasets deliberately leave this empty outside the
            # PatternLock episode block. The tokenizer turns an empty target
            # into fully masked padding, so those examples receive action loss
            # only and cannot leak a fabricated direction condition.
            sample["demo_direction_text"] = direction_text
        if self._episode_anchor_images is not None:
            episode_index = int(self._episode_index[cache_position])
            sample["memory_episode_start_image"] = np.asarray(self._episode_anchor_images[episode_index])
            sample["memory_episode_start_mask"] = np.bool_(True)
        if self._future_prediction_horizon:
            # Cache row i is transition (frame i-1 -> frame i), hence the
            # following n rows are the n transitions after this observation.
            # Targets are right-padded at the episode boundary and never cross it.
            future_latents = np.zeros((self._future_prediction_horizon, self._future_latent_dim), dtype=np.float32)
            future_mask = np.zeros(self._future_prediction_horizon, dtype=np.bool_)
            episode_index = int(self._episode_index[cache_position])
            episode_end = int(self._episode_end_by_episode[episode_index])
            valid_count = min(self._future_prediction_horizon, episode_end - cache_position - 1)
            if valid_count:
                future_positions = np.arange(cache_position + 1, cache_position + 1 + valid_count, dtype=np.int64)
                if not np.all(self._valid[future_positions] & (self._episode_index[future_positions] == episode_index)):
                    raise RuntimeError(f"Invalid future CD-LAM target at global frame {global_index}")
                future_latents[:valid_count] = np.asarray(self._future_latents[future_positions], dtype=np.float32)
                future_mask[:valid_count] = True
            sample["future_memory_latents"] = future_latents
            sample["future_memory_mask"] = future_mask
        if self._demo_anchor_images is not None or self._demo_anchor_features is not None:
            episode_index = int(self._episode_index[cache_position])
            expected_exec_start = int(self._exec_start_by_episode[episode_index])
            if expected_exec_start != exec_start_idx:
                raise ValueError(
                    f"Demo-anchor exec_start_idx mismatch for episode {episode_index}: "
                    f"cache={expected_exec_start}, sample={exec_start_idx}"
                )
            anchor_index = int(self._episode_to_demo_anchor[episode_index])
            has_anchor = anchor_index >= 0
            if self._demo_anchor_features is not None:
                sample["memory_demo_anchor_features"] = (
                    np.asarray(self._demo_anchor_features[anchor_index]) if has_anchor else self._empty_demo_anchor
                )
            else:
                sample["memory_demo_start_image"] = (
                    np.asarray(self._demo_anchor_images[anchor_index]) if has_anchor else self._empty_demo_anchor
                )
            sample["memory_demo_start_mask"] = np.bool_(has_anchor)
        if self._execution_anchor_images is not None:
            episode_index = int(self._episode_index[cache_position])
            expected_exec_start = int(self._exec_start_by_episode[episode_index])
            if expected_exec_start != exec_start_idx:
                raise ValueError(
                    f"Execution-anchor exec_start_idx mismatch for episode {episode_index}: "
                    f"cache={expected_exec_start}, sample={exec_start_idx}"
                )
            execution_visible = episode_frame >= exec_start_idx
            sample["memory_execution_start_image"] = (
                np.asarray(self._execution_anchor_images[episode_index])
                if execution_visible
                else np.zeros_like(self._execution_anchor_images[episode_index])
            )
            sample["memory_execution_start_mask"] = np.bool_(execution_visible)
        return sample
