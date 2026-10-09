from collections.abc import Iterator, Sequence
import functools
import hashlib
import json
import logging
import multiprocessing
from multiprocessing import util as _multiprocessing_util
import os
from pathlib import Path
import time
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training.cdlam_memory_dataset import DEMO_ANCHOR_FEATURE_DEFINITION
from openpi.training.cdlam_memory_dataset import DEMO_ANCHOR_IMAGE_DEFINITION
from openpi.training.cdlam_memory_dataset import DUAL_ANCHOR_IMAGE_DEFINITION
from openpi.training.cdlam_memory_dataset import CDLAMMemoryDataset
from openpi.training.cdlam_memory_dataset import VisionAnchorDataset
from openpi.training.cdlam_memory_dataset import _dataset_identity
from openpi.training.cdlam_memory_dataset import _load_phase_metadata_cache
from openpi.training.online_cdlam_memory_dataset import OnlineCDLAMMemoryDataset
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


def _profile_event(event: str, seconds: float, *, worker_id: int | None = None) -> None:
    profile_dir = os.environ.get("OPENPI_PROFILE_DATA_DIR", "").strip()
    if not profile_dir:
        return
    path = Path(profile_dir)
    path.mkdir(parents=True, exist_ok=True)
    if worker_id is None:
        worker_id = int(os.environ["OPENPI_PROFILE_WORKER_ID"]) if "OPENPI_PROFILE_WORKER_ID" in os.environ else None
    name = f"worker-{worker_id}.jsonl" if worker_id is not None else "parent.jsonl"
    with (path / name).open("a") as stream:
        stream.write(json.dumps({"event": event, "seconds": seconds}) + "\n")


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def __len__(self) -> int:
        """Return the number of batches in one natural dataset epoch."""
        raise NotImplementedError("Subclasses of DataLoader should implement __len__.")

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        """Set the distributed sampler epoch and next batch within it."""
        raise NotImplementedError("Subclasses of DataLoader should implement set_epoch.")

    def set_adaptive_mse_block_start(self, completed_step: int) -> bool:
        """Select the next plan-gated block after restore or plan publication."""
        raise NotImplementedError("Subclasses of DataLoader should implement set_adaptive_mse_block_start.")

    def close(self) -> None:
        """Release any worker processes owned by this loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement close.")

class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class HoldLastActionTargetsDataset(Dataset[T_co]):
    """Turn right-padded LeRobot action queries into supervised hold targets."""

    def __init__(self, dataset: Dataset, action_keys: Sequence[str]):
        if not action_keys:
            raise ValueError("hold-last action targets require at least one action sequence key")
        self._dataset = dataset
        self._action_keys = tuple(action_keys)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        item = dict(self._dataset[index])
        for key in self._action_keys:
            pad_key = f"{key}_is_pad"
            if key not in item or pad_key not in item:
                raise KeyError(f"hold-last action target requires {key!r} and {pad_key!r}")
            values = item[key]
            padding = item[pad_key]
            padding_np = np.asarray(padding, dtype=np.bool_)
            if padding_np.ndim != 1 or values.shape[0] != padding_np.shape[0]:
                raise ValueError(
                    f"hold-last action shape mismatch for {key!r}: "
                    f"values={values.shape}, padding={padding_np.shape}"
                )
            padded_indices = np.flatnonzero(padding_np)
            if padded_indices.size:
                first_padded = int(padded_indices[0])
                if not bool(np.all(padding_np[first_padded:])):
                    raise ValueError(f"hold-last action padding for {key!r} must be a right-side suffix")
                hold_index = max(first_padded - 1, 0)
                if isinstance(values, torch.Tensor):
                    values = values.clone()
                    hold_value = values[hold_index].clone()
                    values[torch.as_tensor(padding_np, device=values.device)] = hold_value
                else:
                    values = np.asarray(values).copy()
                    values[padding_np] = values[hold_index]
                item[key] = values
            if isinstance(padding, torch.Tensor):
                item[pad_key] = torch.zeros_like(padding, dtype=torch.bool)
            else:
                item[pad_key] = np.zeros_like(padding_np, dtype=np.bool_)
        return item

    def __len__(self) -> int:
        return len(self._dataset)


def _find_lerobot_dataset(dataset):
    """Walk transparent wrappers until the underlying LeRobot dataset."""
    seen: set[int] = set()
    current = dataset
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "hf_dataset") and hasattr(current, "episode_data_index"):
            return current
        current = getattr(current, "_dataset", None)
        if current is None:
            break
    raise TypeError("execution_only requires an underlying LeRobotDataset")


def _lerobot_scalar_column(dataset, name: str) -> np.ndarray:
    seen: set[int] = set()
    current = dataset
    while id(current) not in seen:
        seen.add(id(current))
        scalar_columns = getattr(current, "scalar_columns", None)
        if scalar_columns is not None and name in scalar_columns:
            values = np.asarray(scalar_columns[name])
            if values.ndim != 1 or len(values) != len(current):
                raise ValueError(f"Wrapper scalar column {name!r} must be scalar per row, got {values.shape}")
            return values
        if hasattr(current, "hf_dataset") and hasattr(current, "episode_data_index"):
            base = current
            break
        current = getattr(current, "_dataset", None)
        if current is None:
            raise TypeError("execution_only requires wrapper scalar columns or an underlying LeRobotDataset")
    else:
        raise TypeError("execution_only dataset wrapper cycle detected")
    if name not in base.hf_dataset.column_names:
        raise KeyError(f"execution_only dataset is missing required column {name!r}")
    column = base.hf_dataset.data.column(name).combine_chunks()
    values = np.asarray(column.to_numpy(zero_copy_only=False))
    if values.ndim != 1 or len(values) != len(base):
        raise ValueError(f"LeRobot column {name!r} must be scalar per row, got {values.shape}")
    return values


_PATTERNLOCK_DIRECTION_NAMES = {
    "move left": "left",
    "move right": "right",
    "move forward": "up",
    "move backward": "down",
    "move forward-left": "left-up",
    "move backward-left": "left-down",
    "move forward-right": "right-up",
    "move backward-right": "right-down",
}

_PATTERNLOCK_DIRECTION_PLAN_PATH = Path(__file__).with_name("patternlock_train_direction_plans_v1.json")


@functools.lru_cache(maxsize=1)
def _load_patternlock_direction_plans() -> tuple[tuple[str, ...], ...]:
    """Load exact train paths reconstructed from the official PatternLock seeds."""
    payload = json.loads(_PATTERNLOCK_DIRECTION_PLAN_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported PatternLock direction-plan schema")
    entries = payload.get("episodes")
    if not isinstance(entries, list) or payload.get("episode_count") != len(entries):
        raise ValueError("PatternLock direction-plan manifest has an invalid episode count")
    expected_names = set(_PATTERNLOCK_DIRECTION_NAMES.values())
    plans: list[tuple[str, ...]] = []
    for expected_episode, entry in enumerate(entries):
        if entry.get("episode") != expected_episode:
            raise ValueError(
                "PatternLock direction-plan episodes must be consecutive: "
                f"expected {expected_episode}, got {entry.get('episode')}"
            )
        directions = entry.get("directions")
        path_nodes = entry.get("path_nodes")
        if not isinstance(directions, list) or not directions:
            raise ValueError(f"PatternLock episode {expected_episode} has no directions")
        if not isinstance(path_nodes, list) or len(path_nodes) != len(directions) + 1:
            raise ValueError(f"PatternLock episode {expected_episode} has an invalid path")
        unknown = set(directions) - expected_names
        if unknown:
            raise ValueError(
                f"PatternLock episode {expected_episode} has unknown directions {sorted(unknown)}"
            )
        plans.append(tuple(directions))
    return tuple(plans)


def build_patternlock_demo_direction_texts(
    dataset,
    *,
    direction_plans: Sequence[Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Build one exact comma-separated direction plan per PatternLock episode.

    The framewise ``simple_subgoal_online`` labels identify direction changes,
    but cannot distinguish one long edge from several adjacent edges in the
    same direction. Exact paths are therefore reconstructed from the official
    train seeds and stored in a checked-in manifest. The framewise labels are
    retained as an alignment check after adjacent repeats in the exact plan
    are collapsed.
    """
    if direction_plans is None:
        direction_plans = _load_patternlock_direction_plans()
    exact_plans = tuple(tuple(plan) for plan in direction_plans)
    if not exact_plans:
        raise ValueError("PatternLock direction plans cannot be empty")
    episode = _lerobot_scalar_column(dataset, "episode_index").astype(np.int64, copy=False)
    frame = _lerobot_scalar_column(dataset, "frame_index").astype(np.int64, copy=False)
    exec_start = _lerobot_scalar_column(dataset, "exec_start_idx").astype(np.int64, copy=False)
    task = _lerobot_scalar_column(dataset, "task_index").astype(np.int64, copy=False)
    raw_direction = _lerobot_scalar_column(dataset, "simple_subgoal_online")
    if not (len(episode) == len(frame) == len(exec_start) == len(task) == len(raw_direction)):
        raise ValueError("PatternLock direction columns must have identical row counts")
    if np.any(np.diff(episode) < 0):
        raise ValueError("PatternLock direction rows must be ordered by episode")
    episode_count = int(episode.max()) + 1
    plans = [""] * episode_count
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)]
    seen_names: set[str] = set()
    patternlock_episodes = 0
    for start, end in zip(starts, ends, strict=True):
        episode_index = int(episode[start])
        if np.any(episode[start:end] != episode_index) or not np.array_equal(
            frame[start:end], np.arange(end - start, dtype=np.int64)
        ):
            raise ValueError(f"Episode {episode_index} has non-consecutive frame metadata")
        if np.any(task[start:end] != task[start]) or np.any(exec_start[start:end] != exec_start[start]):
            raise ValueError(f"Episode {episode_index} changes task or exec_start_idx")
        if int(task[start]) != 0:
            continue
        patternlock_episodes += 1
        boundary = int(exec_start[start])
        if not 0 < boundary <= end - start:
            raise ValueError(f"PatternLock episode {episode_index} has invalid exec_start_idx={boundary}")
        observed_compressed: list[str] = []
        for raw_value in raw_direction[start : start + boundary]:
            raw_name = str(raw_value)
            if raw_name == "no record":
                continue
            canonical = _PATTERNLOCK_DIRECTION_NAMES.get(raw_name)
            if canonical is None:
                raise ValueError(
                    f"PatternLock episode {episode_index} contains unknown demo direction {raw_name!r}"
                )
            seen_names.add(canonical)
            if not observed_compressed or observed_compressed[-1] != canonical:
                observed_compressed.append(canonical)
        if not observed_compressed:
            raise ValueError(f"PatternLock episode {episode_index} has no labelled demo movement")
        if episode_index >= len(exact_plans):
            raise ValueError(f"PatternLock episode {episode_index} is missing from the exact-plan manifest")
        exact_plan = exact_plans[episode_index]
        exact_compressed = [
            direction
            for index, direction in enumerate(exact_plan)
            if index == 0 or direction != exact_plan[index - 1]
        ]
        if observed_compressed != exact_compressed:
            raise ValueError(
                f"PatternLock episode {episode_index} exact plan does not match frame labels: "
                f"exact_collapsed={exact_compressed}, observed={observed_compressed}"
            )
        plans[episode_index] = ", ".join(exact_plan)
    if patternlock_episodes != len(exact_plans):
        raise ValueError(
            f"Expected {len(exact_plans)} PatternLock episodes, found {patternlock_episodes}"
        )
    expected_names = set(_PATTERNLOCK_DIRECTION_NAMES.values())
    if seen_names != expected_names:
        raise ValueError(f"PatternLock direction vocabulary mismatch: found={sorted(seen_names)}")
    logging.info(
        "Prepared exact autoregressive PatternLock demo plans for %d episodes over directions %s; "
        "%d episodes retain adjacent repeated moves",
        patternlock_episodes,
        sorted(seen_names),
        sum(any(left == right for left, right in zip(plan, plan[1:])) for plan in exact_plans),
    )
    return tuple(plans)


def build_execution_query_indices(
    *,
    is_demo: np.ndarray,
    exec_start_idx: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    task_index: np.ndarray | None = None,
    allowed_task_indices: tuple[int, ...] | None = None,
    allowed_episode_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate the demo/execution contract and return execution row positions."""
    arrays = {
        "is_demo": np.asarray(is_demo),
        "exec_start_idx": np.asarray(exec_start_idx),
        "frame_index": np.asarray(frame_index),
        "episode_index": np.asarray(episode_index),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if any(value.ndim != 1 for value in arrays.values()) or len({len(value) for value in arrays.values()}) != 1:
        raise ValueError(f"Execution metadata must be equal-length 1-D arrays, got {shapes}")
    if not len(arrays["is_demo"]):
        raise ValueError("Execution metadata cannot be empty")
    if arrays["is_demo"].dtype != np.bool_:
        raise ValueError(f"is_demo must be bool, got {arrays['is_demo'].dtype}")
    for name in ("exec_start_idx", "frame_index", "episode_index"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"{name} must be integer, got {arrays[name].dtype}")

    frame = arrays["frame_index"].astype(np.int64, copy=False)
    episode = arrays["episode_index"].astype(np.int64, copy=False)
    exec_start = arrays["exec_start_idx"].astype(np.int64, copy=False)
    episode_start = np.ones(len(frame), dtype=np.bool_)
    episode_start[1:] = episode[1:] != episode[:-1]
    if np.any(np.diff(episode) < 0) or np.any(frame[episode_start] != 0):
        raise ValueError("Episodes must be ordered and every episode must begin at frame_index 0")
    within_episode = ~episode_start[1:]
    if np.any(np.diff(frame)[within_episode] != 1):
        raise ValueError("frame_index must increase consecutively within each episode")
    if np.any(exec_start < 0):
        raise ValueError("exec_start_idx must be non-negative")

    starts = np.flatnonzero(episode_start)
    ends = np.r_[starts[1:], len(frame)]
    for start, end in zip(starts, ends, strict=True):
        values = exec_start[start:end]
        if np.any(values != values[0]):
            raise ValueError(f"exec_start_idx changes within episode {episode[start]}")
        if int(values[0]) > int(frame[end - 1]):
            raise ValueError(f"episode {episode[start]} has no execution frame at exec_start_idx={values[0]}")

    expected_demo = frame < exec_start
    if not np.array_equal(arrays["is_demo"], expected_demo):
        mismatch = int(np.flatnonzero(arrays["is_demo"] != expected_demo)[0])
        raise ValueError(f"is_demo must be exactly frame_index < exec_start_idx; first mismatch at row {mismatch}")
    query_mask = ~arrays["is_demo"]
    if allowed_task_indices is not None:
        if not allowed_task_indices:
            raise ValueError("allowed_task_indices cannot be empty")
        if any(type(value) is not int or value < 0 for value in allowed_task_indices):
            raise ValueError("allowed_task_indices must contain non-negative integers")
        if len(set(allowed_task_indices)) != len(allowed_task_indices):
            raise ValueError("allowed_task_indices must be unique")
        if task_index is None:
            raise ValueError("task_index is required when allowed_task_indices is set")
        task_index = np.asarray(task_index)
        if task_index.ndim != 1 or len(task_index) != len(frame):
            raise ValueError(f"task_index must be a length-{len(frame)} 1-D array, got {task_index.shape}")
        if not np.issubdtype(task_index.dtype, np.integer):
            raise ValueError(f"task_index must be integer, got {task_index.dtype}")
        query_mask &= np.isin(task_index, np.asarray(allowed_task_indices, dtype=task_index.dtype))
        selected_tasks = set(np.unique(task_index[query_mask]).tolist())
        missing_tasks = set(allowed_task_indices) - selected_tasks
        if missing_tasks:
            raise ValueError(f"allowed_task_indices have no execution queries: {sorted(missing_tasks)}")
    if allowed_episode_indices is not None:
        if not allowed_episode_indices:
            raise ValueError("allowed_episode_indices cannot be empty")
        if any(type(value) is not int or value < 0 for value in allowed_episode_indices):
            raise ValueError("allowed_episode_indices must contain non-negative integers")
        if len(set(allowed_episode_indices)) != len(allowed_episode_indices):
            raise ValueError("allowed_episode_indices must be unique")
        query_mask &= np.isin(episode, np.asarray(allowed_episode_indices, dtype=episode.dtype))
        selected_episodes = set(np.unique(episode[query_mask]).tolist())
        missing_episodes = set(allowed_episode_indices) - selected_episodes
        if missing_episodes:
            raise ValueError(f"allowed_episode_indices have no execution queries: {sorted(missing_episodes)}")
    query_indices = np.flatnonzero(query_mask).astype(np.int64, copy=False)
    if not len(query_indices):
        raise ValueError("execution_only filtering removed every sample")
    query_indices.setflags(write=False)
    return query_indices


class ExecutionOnlyDataset(Dataset[T_co]):
    """Expose execution queries while retaining full episodes underneath."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        *,
        allowed_task_indices: tuple[int, ...] | None = None,
        allowed_episode_indices: tuple[int, ...] | None = None,
        phase_metadata_cache_dir: str | Path | None = None,
        expected_dataset_root: str | Path | None = None,
    ):
        self._dataset = dataset
        self._phase_is_demo = None
        self._phase_exec_start_idx = None
        if phase_metadata_cache_dir is None:
            is_demo = _lerobot_scalar_column(dataset, "is_demo")
            exec_start_idx = _lerobot_scalar_column(dataset, "exec_start_idx")
            frame_index = _lerobot_scalar_column(dataset, "frame_index")
            episode_index = _lerobot_scalar_column(dataset, "episode_index")
        else:
            if expected_dataset_root is None:
                raise ValueError("phase_metadata_cache_dir requires expected_dataset_root")
            dataset_root = Path(expected_dataset_root).expanduser().resolve()
            total_frames, total_episodes, info_sha256, episodes_sha256, _ = _dataset_identity(dataset_root)
            phase = _load_phase_metadata_cache(
                phase_metadata_cache_dir,
                dataset_root=dataset_root,
                info_sha256=info_sha256,
                episodes_sha256=episodes_sha256,
                total_frames=total_frames,
                total_episodes=total_episodes,
            )
            if len(dataset) != total_frames:
                raise ValueError(
                    f"Training dataset has {len(dataset)} rows but phase metadata has {total_frames}; expected equality"
                )
            frame_index = _lerobot_scalar_column(dataset, "frame_index")
            episode_index = _lerobot_scalar_column(dataset, "episode_index")
            if not np.array_equal(frame_index, phase["row_frame_index"]):
                raise ValueError("Phase row_frame_index does not exactly match the LeRobot dataset")
            if not np.array_equal(episode_index, phase["row_episode_index"]):
                raise ValueError("Phase row_episode_index does not exactly match the LeRobot dataset")
            is_demo = phase["row_is_demo"]
            exec_start_idx = phase["row_exec_start_idx"]
            self._phase_is_demo = is_demo
            self._phase_exec_start_idx = exec_start_idx
        self._source_indices = build_execution_query_indices(
            is_demo=is_demo,
            exec_start_idx=exec_start_idx,
            frame_index=frame_index,
            episode_index=episode_index,
            task_index=(
                _lerobot_scalar_column(dataset, "task_index") if allowed_task_indices is not None else None
            ),
            allowed_task_indices=allowed_task_indices,
            allowed_episode_indices=allowed_episode_indices,
        )
        lengths = getattr(dataset, "memory_lengths", None)
        self._memory_lengths = None if lengths is None else np.asarray(lengths)[self._source_indices]
        if self._memory_lengths is not None:
            self._memory_lengths.setflags(write=False)

    def __len__(self) -> int:
        return len(self._source_indices)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        position = index.__index__()
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        source_index = int(self._source_indices[position])
        sample = self._dataset[source_index]
        if self._phase_is_demo is not None:
            expected_is_demo = bool(self._phase_is_demo[source_index])
            expected_exec_start = int(self._phase_exec_start_idx[source_index])
            for key, expected in (("is_demo", expected_is_demo), ("exec_start_idx", expected_exec_start)):
                if key in sample and np.asarray(sample[key]).reshape(-1).tolist() != [expected]:
                    raise ValueError(f"LeRobot {key} disagrees with phase metadata at row {source_index}")
            sample["is_demo"] = np.asarray(expected_is_demo, dtype=np.bool_)
            sample["exec_start_idx"] = np.asarray(expected_exec_start, dtype=np.int32)
        if bool(np.asarray(sample["is_demo"]).reshape(-1)[0]):
            raise RuntimeError("ExecutionOnlyDataset exposed a demo sample")
        return sample

    @property
    def source_indices(self) -> np.ndarray:
        return self._source_indices

    @property
    def memory_lengths(self) -> np.ndarray:
        if self._memory_lengths is None:
            raise AttributeError("Underlying dataset has no memory_lengths")
        return self._memory_lengths


def _build_episode_block_sampling_weights(
    dataset: ExecutionOnlyDataset,
    blocks: tuple[tuple[int, int, float], ...],
) -> np.ndarray:
    """Convert requested task/block probability mass to query-row weights.

    The execution-only wrapper retains a source-index mapping into the full
    corpus.  Weighting each row in a block by ``mass / rows_in_block`` gives
    the exact requested block mass even when episode durations differ.
    """
    if not blocks:
        raise ValueError("task_sampling_episode_weights cannot be empty")
    source = dataset.source_indices
    all_episode_rows = _lerobot_scalar_column(dataset, "episode_index")
    episode_rows = all_episode_rows[source]
    weights = np.zeros(len(source), dtype=np.float64)
    assigned = np.zeros(len(source), dtype=np.bool_)
    masses: list[tuple[int, int, float, int]] = []
    for first_episode, last_episode, mass in blocks:
        if (
            type(first_episode) is not int
            or type(last_episode) is not int
            or first_episode < 0
            or last_episode < first_episode
            or not np.isfinite(mass)
            or mass <= 0
        ):
            raise ValueError(f"invalid task sampling block: {(first_episode, last_episode, mass)!r}")
        mask = (episode_rows >= first_episode) & (episode_rows <= last_episode)
        count = int(mask.sum())
        if not count:
            raise ValueError(f"task sampling block {first_episode}-{last_episode} has no execution queries")
        if np.any(assigned[mask]):
            raise ValueError(f"task sampling block {first_episode}-{last_episode} overlaps another block")
        weights[mask] = mass / count
        assigned[mask] = True
        masses.append((first_episode, last_episode, float(mass), count))
    if not np.all(assigned):
        missing = np.unique(episode_rows[~assigned]).tolist()
        raise ValueError(f"task sampling blocks leave execution episodes unassigned: {missing[:20]}")
    total_mass = float(sum(mass for _, _, mass, _ in masses))
    if not np.isclose(total_mass, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"task sampling block masses must sum to 1, got {total_mass}")
    logging.info("Task-mixture replacement sampling: %s", masses)
    return weights


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaselineUncertaintyDataset(Dataset[T_co]):
    """Attach a fixed uncertainty rank to each physical action target."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        cache_dir: str | Path,
        *,
        expected_dataset_root: str | Path,
        action_horizon: int,
        action_start_offset: int,
    ):
        self._dataset = dataset
        self._cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self._cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing baseline uncertainty manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Baseline uncertainty cache must be a complete schema-v1 cache")

        dataset_root = Path(expected_dataset_root).expanduser().resolve()
        total_frames, total_episodes, info_sha256, episodes_sha256, _ = _dataset_identity(dataset_root)
        expected_identity = {
            "total_frames": total_frames,
            "total_episodes": total_episodes,
            "info_sha256": info_sha256,
            "episodes_sha256": episodes_sha256,
        }
        if manifest.get("dataset") != expected_identity:
            raise ValueError("Baseline uncertainty cache dataset identity does not match the training dataset")
        if manifest.get("action_horizon") != action_horizon:
            raise ValueError(
                f"Baseline uncertainty horizon {manifest.get('action_horizon')} != model horizon {action_horizon}"
            )
        if manifest.get("target_action_start_offset") != action_start_offset:
            raise ValueError(
                "Baseline uncertainty target offset does not match action_sequence_start_offset: "
                f"{manifest.get('target_action_start_offset')} != {action_start_offset}"
            )

        score_meta = manifest.get("score", {})
        valid_meta = manifest.get("valid", {})
        self._score_path = self._cache_dir / score_meta.get("file", "")
        self._valid_path = self._cache_dir / valid_meta.get("file", "")
        for path, metadata in ((self._score_path, score_meta), (self._valid_path, valid_meta)):
            if not path.is_file():
                raise FileNotFoundError(f"Baseline uncertainty cache file is missing: {path}")
            if metadata.get("sha256") != _sha256_file(path):
                raise ValueError(f"Baseline uncertainty cache checksum mismatch: {path}")

        self._scores = np.load(self._score_path, mmap_mode="r", allow_pickle=False)
        self._valid = np.load(self._valid_path, mmap_mode="r", allow_pickle=False)
        if self._scores.shape != (total_frames, action_horizon) or self._scores.dtype != np.float32:
            raise ValueError(
                f"Expected uncertainty scores {(total_frames, action_horizon)} float32, "
                f"got {self._scores.shape} {self._scores.dtype}"
            )
        if self._valid.shape != (total_frames,) or self._valid.dtype != np.bool_:
            raise ValueError(f"Expected uncertainty valid mask {(total_frames,)} bool, got {self._valid.shape}")
        valid_scores = np.asarray(self._scores[self._valid])
        if not np.all(np.isfinite(valid_scores)) or np.any((valid_scores < 0) | (valid_scores > 1)):
            raise ValueError("Baseline uncertainty scores must be finite and in [0, 1]")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_scores"] = None
        state["_valid"] = None
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._scores = np.load(self._score_path, mmap_mode="r", allow_pickle=False)
        self._valid = np.load(self._valid_path, mmap_mode="r", allow_pickle=False)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        if "index" not in sample:
            raise KeyError("LeRobot sample is missing its global 'index' field")
        value = np.asarray(sample["index"]).reshape(-1)
        if value.size != 1:
            raise ValueError(f"Expected scalar global index, got {np.asarray(sample['index']).shape}")
        global_index = int(value[0])
        if not 0 <= global_index < len(self._valid) or not self._valid[global_index]:
            raise ValueError(f"No valid uncertainty score for global execution row {global_index}")
        sample["baseline_uncertainty_score"] = np.asarray(self._scores[global_index], dtype=np.float32).copy()
        return sample

    @property
    def memory_lengths(self) -> np.ndarray:
        return self._dataset.memory_lengths


def _empirical_midranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic [0, 1] empirical-CDF midranks for a 1-D vector."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a non-empty finite 1-D array")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranked_sorted = np.empty(len(values), dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        # Ranks are one-based for the empirical CDF and end is exclusive.
        ranked_sorted[start:end] = (start + end) / (2.0 * len(values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranked_sorted
    return ranks


_DIVERSITY_RANK_DECILE_DEFINITION = (
    "global diversity empirical-midrank intervals [j/10,(j+1)/10), final interval inclusive"
)


def _load_adaptive_mse_decile_sampling_weights(
    *,
    plan_path: str | Path,
    diversity_manifest_path: Path,
    ranks: np.ndarray,
    uniform_mass: float,
    expected_sampling_start_step: int | None = None,
) -> np.ndarray:
    """Return per-query weights from a validated normalized-MSE decile plan.

    The plan is created after a fixed 1024-query validation pack.  It carries
    one normalized physical-action MSE for each *global* diversity-rank decile.
    Within a decile, all execution queries share the MSE-driven mass equally;
    a nonzero uniform component gives every valid execution query a strict
    probability floor, including a zero-MSE decile.
    """
    if not np.isfinite(uniform_mass) or not 0.0 < uniform_mass <= 1.0:
        raise ValueError("diversity_sampling_uniform_mass must be finite and in (0, 1]")
    plan_path = Path(plan_path).expanduser().resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing adaptive diversity-MSE sampling plan: {plan_path}")
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "complete":
        raise ValueError("Adaptive diversity-MSE sampling plan must be complete schema-v1")
    if expected_sampling_start_step is not None and plan.get("sampling_start_step") != expected_sampling_start_step:
        raise ValueError(
            "Adaptive diversity-MSE sampling plan has the wrong sampling_start_step: "
            f"expected {expected_sampling_start_step}, got {plan.get('sampling_start_step')}"
        )
    if plan.get("rank_decile_definition") != _DIVERSITY_RANK_DECILE_DEFINITION:
        raise ValueError("Adaptive diversity-MSE sampling plan uses an unexpected rank-decile definition")
    if plan.get("diversity_cache_manifest_sha256") != _sha256_file(diversity_manifest_path):
        raise ValueError("Adaptive diversity-MSE sampling plan does not match the diversity cache manifest")
    plan_uniform_mass = plan.get("uniform_mass")
    if not isinstance(plan_uniform_mass, (int, float)) or not np.isclose(
        float(plan_uniform_mass), uniform_mass, rtol=0.0, atol=1e-15
    ):
        raise ValueError("Adaptive diversity-MSE sampling plan uniform mass disagrees with the training config")
    decile_mse = np.asarray(plan.get("normalized_mse_by_decile"), dtype=np.float64)
    if decile_mse.shape != (10,) or not np.all(np.isfinite(decile_mse)) or np.any(decile_mse < 0.0):
        raise ValueError("Adaptive diversity-MSE plan must contain ten finite non-negative normalized MSE values")
    if ranks.ndim != 1 or not len(ranks) or not np.all(np.isfinite(ranks)):
        raise ValueError("Adaptive diversity-MSE sampling requires a non-empty finite rank vector")

    decile = np.minimum((ranks * 10.0).astype(np.int64), 9)
    if np.any(decile < 0) or np.any(decile > 9):
        raise ValueError("Diversity ranks must be in [0, 1]")
    counts = np.bincount(decile, minlength=10).astype(np.float64)
    if np.any(counts == 0.0):
        raise ValueError("Every global diversity-rank decile must contain at least one training query")
    mse_total = float(decile_mse.sum())
    if mse_total > 0.0:
        mse_component = decile_mse[decile] / (counts[decile] * mse_total)
    else:
        # A perfectly solved validation pack has no preference signal; retain
        # the all-query eligibility invariant by falling back to uniform.
        mse_component = np.full(len(ranks), 1.0 / len(ranks), dtype=np.float64)
    weights = uniform_mass / len(ranks) + (1.0 - uniform_mass) * mse_component
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Adaptive diversity-MSE sampling produced non-positive or non-finite query weights")
    weights /= weights.sum()
    decile_mass = np.bincount(decile, weights=weights, minlength=10)
    logging.info(
        "adaptive normalized-MSE decile replacement sampling: queries=%d uniform_mass=%.4f "
        "mse=%s decile_mass=%s",
        len(weights),
        uniform_mass,
        np.array2string(decile_mse, precision=6, separator=","),
        np.array2string(decile_mass, precision=6, separator=","),
    )
    weights.setflags(write=False)
    return weights


class _AdaptiveMseDecilePlanSchedule:
    """Read an immutable plan for each fixed replacement-sampling block.

    The schedule is consumed by the parent-process batch sampler. At a block
    edge it does not issue indexes for the next block until the evaluator has
    atomically published that block's plan. Thus the worker prefetch queue can
    drain the current block but never inject future-plan samples before the
    corresponding optimizer boundary.
    """

    def __init__(
        self,
        *,
        plan_dir: str | Path,
        start_step: int,
        interval_batches: int,
        dataset: ExecutionOnlyDataset,
        diversity_cache_dir: str | Path,
        uniform_mass: float,
    ):
        if start_step < 0:
            raise ValueError(f"adaptive MSE plan start step must be non-negative, got {start_step}")
        if interval_batches <= 0:
            raise ValueError(f"adaptive MSE plan interval must be positive, got {interval_batches}")
        self._plan_dir = Path(plan_dir).expanduser().resolve()
        self._start_step = int(start_step)
        self._interval_batches = int(interval_batches)
        self._uniform_mass = float(uniform_mass)
        cache_dir = Path(diversity_cache_dir).expanduser().resolve()
        self._manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(self._manifest_path.read_text())
        uncertainty_meta = manifest["uncertainty"]
        uncertainty = np.load(cache_dir / uncertainty_meta["file"], mmap_mode="r", allow_pickle=False)
        source_indices = np.asarray(dataset.source_indices)
        query_diversity = np.asarray(uncertainty[source_indices], dtype=np.float64).mean(axis=1)
        self._ranks = _empirical_midranks(np.log(query_diversity + 1e-12))
        self._ranks.setflags(write=False)

    @property
    def start_step(self) -> int:
        return self._start_step

    @property
    def interval_batches(self) -> int:
        return self._interval_batches

    def _plan_path(self, start_step: int) -> Path:
        if start_step < self._start_step or (start_step - self._start_step) % self._interval_batches:
            raise ValueError(f"invalid adaptive MSE block start step: {start_step}")
        return self._plan_dir / f"step{start_step:06d}.json"

    def weights_for_block(self, start_step: int) -> np.ndarray:
        plan_path = self._plan_path(start_step)
        if not plan_path.is_file():
            logging.info(
                "Adaptive MSE sampler waiting for plan for steps [%d, %d): %s",
                start_step,
                start_step + self._interval_batches,
                plan_path,
            )
        while not plan_path.is_file():
            time.sleep(2.0)
        weights = _load_adaptive_mse_decile_sampling_weights(
            plan_path=plan_path,
            diversity_manifest_path=self._manifest_path,
            ranks=self._ranks,
            uniform_mass=self._uniform_mass,
            expected_sampling_start_step=start_step,
        )
        logging.info(
            "Adaptive MSE sampler activated plan for steps [%d, %d): %s",
            start_step,
            start_step + self._interval_batches,
            plan_path,
        )
        return weights


def _load_query_diversity_sampling_weights(
    dataset: ExecutionOnlyDataset,
    cache_dir: str | Path,
    *,
    expected_dataset_root: str | Path,
    action_horizon: int,
    action_start_offset: int,
    rank_threshold: float,
    rank_power: float,
    low_rank_mass: float = 0.0,
    mse_plan_path: str | Path | None = None,
    uniform_mass: float = 0.05,
) -> np.ndarray:
    """Build replacement-sampling weights from full-20-noise action spread.

    One scalar diversity is the mean over the full h20 prediction horizon of
    the robust-normalized 20-noise action spread. This deliberately differs
    from ``BaselineUncertaintyDataset``: it changes query frequency before
    batching instead of changing the loss of a uniformly sampled query.
    """
    if not np.isfinite(rank_threshold) or not 0.0 <= rank_threshold < 1.0:
        raise ValueError("diversity_sampling_rank_threshold must be finite and in [0, 1)")
    if not np.isfinite(rank_power) or rank_power <= 0.0:
        raise ValueError("diversity_sampling_rank_power must be finite and positive")
    if not np.isfinite(low_rank_mass) or not 0.0 <= low_rank_mass < 1.0:
        raise ValueError("diversity_sampling_low_rank_mass must be finite and in [0, 1)")

    cache_dir = Path(cache_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing diversity sampling manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Diversity sampling cache must be a complete schema-v1 cache")

    dataset_root = Path(expected_dataset_root).expanduser().resolve()
    total_frames, total_episodes, info_sha256, episodes_sha256, _ = _dataset_identity(dataset_root)
    expected_identity = {
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "info_sha256": info_sha256,
        "episodes_sha256": episodes_sha256,
    }
    if manifest.get("dataset") != expected_identity:
        raise ValueError("Diversity sampling cache dataset identity does not match the training dataset")
    if manifest.get("action_horizon") != action_horizon:
        raise ValueError(
            f"Diversity sampling horizon {manifest.get('action_horizon')} != model horizon {action_horizon}"
        )
    if manifest.get("target_action_start_offset") != action_start_offset:
        raise ValueError(
            "Diversity sampling target offset does not match action_sequence_start_offset: "
            f"{manifest.get('target_action_start_offset')} != {action_start_offset}"
        )
    if manifest.get("sample_horizon_slice") != [0, action_horizon]:
        raise ValueError("Diversity sampling cache must retain every supervised h20 prediction")
    source = manifest.get("source", {})
    if source.get("num_noise_samples") != 20:
        raise ValueError("Diversity sampling cache must be built from exactly 20 noise samples")

    uncertainty_meta = manifest.get("uncertainty", {})
    valid_meta = manifest.get("valid", {})
    expected_formula = "sqrt(mean_d(var_s(action_sample / robust_action_scale, ddof=1)))"
    if uncertainty_meta.get("formula") != expected_formula:
        raise ValueError("Diversity sampling cache has an unexpected 20-noise uncertainty definition")
    uncertainty_path = cache_dir / uncertainty_meta.get("file", "")
    valid_path = cache_dir / valid_meta.get("file", "")
    for path, metadata in ((uncertainty_path, uncertainty_meta), (valid_path, valid_meta)):
        if not path.is_file():
            raise FileNotFoundError(f"Diversity sampling cache file is missing: {path}")
        if metadata.get("sha256") != _sha256_file(path):
            raise ValueError(f"Diversity sampling cache checksum mismatch: {path}")

    uncertainty = np.load(uncertainty_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    if uncertainty.shape != (total_frames, action_horizon) or uncertainty.dtype != np.float32:
        raise ValueError(
            f"Expected 20-noise uncertainty {(total_frames, action_horizon)} float32, "
            f"got {uncertainty.shape} {uncertainty.dtype}"
        )
    if valid.shape != (total_frames,) or valid.dtype != np.bool_:
        raise ValueError(f"Expected diversity valid mask {(total_frames,)} bool, got {valid.shape}")

    source_indices = np.asarray(dataset.source_indices)
    if source_indices.ndim != 1 or len(source_indices) != len(dataset):
        raise ValueError("Execution-only source indices must be a length-matched 1-D array")
    if not np.issubdtype(source_indices.dtype, np.integer):
        raise ValueError("Execution-only source indices must be integer")
    if np.any(source_indices < 0) or np.any(source_indices >= total_frames):
        raise ValueError("Execution-only source indices are outside the diversity cache")
    if not np.all(valid[source_indices]):
        first = int(source_indices[np.flatnonzero(~valid[source_indices])[0]])
        raise ValueError(f"No valid 20-noise diversity for execution row {first}")

    query_diversity = np.asarray(uncertainty[source_indices], dtype=np.float64).mean(axis=1)
    if not np.all(np.isfinite(query_diversity)) or np.any(query_diversity < 0.0):
        raise ValueError("20-noise query diversity must be finite and non-negative")
    ranks = _empirical_midranks(np.log(query_diversity + 1e-12))
    if mse_plan_path is not None:
        return _load_adaptive_mse_decile_sampling_weights(
            plan_path=mse_plan_path,
            diversity_manifest_path=manifest_path,
            ranks=ranks,
            uniform_mass=uniform_mass,
        )
    weights = np.zeros_like(ranks)
    high_rank = ranks > rank_threshold
    low_rank = ~high_rank
    high_weights = ((ranks[high_rank] - rank_threshold) / (1.0 - rank_threshold)) ** rank_power
    if high_weights.size == 0 or not np.any(high_weights > 0.0):
        raise ValueError("Diversity sampling produced no positive high-rank query weights")
    if low_rank_mass:
        if not np.any(low_rank):
            raise ValueError("Cannot allocate low-rank sampling mass without low-rank queries")
        # Every low-rank query remains eligible. Their conditional distribution
        # is uniform; the high-rank conditional distribution is unchanged.
        weights[low_rank] = low_rank_mass / np.count_nonzero(low_rank)
        weights[high_rank] = (1.0 - low_rank_mass) * high_weights / high_weights.sum()
    else:
        # Preserve the old high-diversity-only behavior bit-for-bit apart from
        # the documented validation checks above.
        weights[high_rank] = high_weights
    if not np.any(weights > 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("Diversity sampling produced no positive finite query weights")
    weights /= weights.sum()
    logging.info(
        "20-noise diversity replacement sampling: queries=%d high=%d (%.2f%%), low=%d "
        "mass=%.4f, threshold=%.3f power=%.3f, diversity q50=%.6g q90=%.6g q99=%.6g",
        len(weights),
        int(np.count_nonzero(high_rank)),
        100.0 * np.count_nonzero(high_rank) / len(weights),
        int(np.count_nonzero(low_rank)),
        float(weights[low_rank].sum()),
        rank_threshold,
        rank_power,
        *np.quantile(query_diversity, [0.5, 0.9, 0.99]),
    )
    weights.setflags(write=False)
    return weights


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)
        if observation.memory_segment_ids is not None:
            observation = observation.replace(
                memory_segment_ids=jnp.ones_like(observation.memory_segment_ids, dtype=jnp.int32)
            )

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class _ProfileLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobot dataset variant that times video decoding in workers."""

    def _query_videos(self, query_timestamps, ep_idx):
        start = time.perf_counter()
        result = super()._query_videos(query_timestamps, ep_idx)
        _profile_event("video_decode", time.perf_counter() - start)
        return result


class _ProfileDataset(Dataset):
    """Time complete transformed sample retrieval in a worker."""

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __getitem__(self, index: SupportsIndex):
        start = time.perf_counter()
        result = self._dataset[index]
        _profile_event("sample_total", time.perf_counter() - start)
        return result

    def __len__(self) -> int:
        return len(self._dataset)


class DistributedLengthBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Build deterministic global batches with balanced sequence lengths.

    Every rank independently reconstructs the same shuffled global plan. A
    sortish mega-bucket keeps training order random while making each global
    batch length-homogeneous. Each sorted global batch is then striped across
    ranks, so all ranks see nearly identical padded lengths at a given step.
    """

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        *,
        local_batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        shuffle: bool = True,
        bucket_size_multiplier: int = 50,
        sampling_weights: np.ndarray | None = None,
    ):
        lengths_array = np.asarray(lengths)
        if lengths_array.ndim != 1 or len(lengths_array) == 0:
            raise ValueError(f"lengths must be a non-empty 1-D array, got shape {lengths_array.shape}")
        if not np.issubdtype(lengths_array.dtype, np.integer) or np.any(lengths_array <= 0):
            raise ValueError("lengths must contain positive integers")
        if local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be positive, got {local_batch_size}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        if bucket_size_multiplier <= 0:
            raise ValueError(f"bucket_size_multiplier must be positive, got {bucket_size_multiplier}")

        self._lengths = lengths_array.astype(np.int64, copy=True)
        self._local_batch_size = int(local_batch_size)
        self._num_replicas = int(num_replicas)
        self._rank = int(rank)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._bucket_size_multiplier = int(bucket_size_multiplier)
        self._sampling_weights = None
        if sampling_weights is not None:
            weights = np.asarray(sampling_weights, dtype=np.float64)
            if weights.shape != self._lengths.shape:
                raise ValueError(
                    f"sampling_weights must match lengths shape {self._lengths.shape}, got {weights.shape}"
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("sampling_weights must be finite and non-negative")
            total_weight = float(weights.sum())
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise ValueError("sampling_weights must have a positive finite sum")
            self._sampling_weights = weights / total_weight
        self._global_batch_size = self._local_batch_size * self._num_replicas
        self._num_batches = len(self._lengths) // self._global_batch_size
        if self._num_batches == 0:
            raise ValueError(
                f"dataset size {len(self._lengths)} is smaller than global batch size {self._global_batch_size}"
            )
        self._total_size = self._num_batches * self._global_batch_size
        self._epoch = 0
        self._batch_offset = 0

    def __len__(self) -> int:
        return self._num_batches

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def batch_offset(self) -> int:
        return self._batch_offset

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self._epoch = int(epoch)
        self._batch_offset = 0

    def set_batch_offset(self, batch_offset: int) -> None:
        """Select the first batch yielded without reading skipped dataset items."""
        if batch_offset < 0 or batch_offset > self._num_batches:
            raise ValueError(f"batch_offset must be in [0, {self._num_batches}], got {batch_offset}")
        self._batch_offset = int(batch_offset)

    def set_sampling_weights(self, sampling_weights: np.ndarray) -> None:
        """Replace the distribution used by subsequent replacement blocks."""
        weights = np.asarray(sampling_weights, dtype=np.float64)
        if weights.shape != self._lengths.shape:
            raise ValueError(f"sampling_weights must match lengths shape {self._lengths.shape}, got {weights.shape}")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("adaptive sampling_weights must be finite and strictly positive")
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("adaptive sampling_weights must have a positive finite sum")
        self._sampling_weights = weights / total_weight

    def iter_replacement_block(
        self,
        num_batches: int,
        *,
        block_start_step: int,
    ) -> Iterator[list[int]]:
        """Yield one plan-homogeneous, length-bucketed replacement block."""
        if num_batches <= 0:
            raise ValueError(f"num_batches must be positive, got {num_batches}")
        if self._sampling_weights is None:
            raise RuntimeError("adaptive replacement block requires sampling weights")
        total_size = num_batches * self._global_batch_size
        # A block-start-dependent seed makes the sequence deterministic across
        # process restarts without reusing an earlier block's draws.
        rng = np.random.default_rng(self._seed + 1_000_003 + int(block_start_step))
        indices = rng.choice(len(self._lengths), size=total_size, replace=True, p=self._sampling_weights)
        mega_bucket_size = self._global_batch_size * self._bucket_size_multiplier
        global_batches: list[np.ndarray] = []
        for start in range(0, total_size, mega_bucket_size):
            bucket = indices[start : start + mega_bucket_size]
            sorted_bucket = bucket[np.argsort(self._lengths[bucket], kind="stable")]
            global_batches.extend(np.split(sorted_bucket, len(sorted_bucket) // self._global_batch_size))
        batch_order = rng.permutation(num_batches)
        for within_block, batch_index in enumerate(batch_order):
            output_position = int(block_start_step) + within_block
            global_batch = global_batches[int(batch_index)].reshape(self._local_batch_size, self._num_replicas)
            column = (self._rank + output_position) % self._num_replicas
            local_batch = global_batch[:, column].copy()
            local_batch[1::2] = global_batch[1::2, self._num_replicas - 1 - column]
            yield local_batch.tolist()

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._seed + self._epoch)
        if self._shuffle:
            if self._sampling_weights is None:
                indices = rng.permutation(len(self._lengths))[: self._total_size]
            else:
                # Sampling with replacement is essential: a weighted
                # no-replacement epoch still visits every query exactly once.
                indices = rng.choice(
                    len(self._lengths), size=self._total_size, replace=True, p=self._sampling_weights
                )
        else:
            indices = np.arange(self._total_size, dtype=np.int64)

        mega_bucket_size = self._global_batch_size * self._bucket_size_multiplier
        global_batches: list[np.ndarray] = []
        for start in range(0, self._total_size, mega_bucket_size):
            bucket = indices[start : start + mega_bucket_size]
            sorted_bucket = bucket[np.argsort(self._lengths[bucket], kind="stable")]
            global_batches.extend(np.split(sorted_bucket, len(sorted_bucket) // self._global_batch_size))

        batch_order = rng.permutation(self._num_batches) if self._shuffle else np.arange(self._num_batches)

        # Keep output_position absolute: it participates in rank rotation, so
        # re-enumerating the sliced tail from zero would change resumed data.
        for output_position in range(self._batch_offset, self._num_batches):
            batch_index = batch_order[output_position]
            global_batch = global_batches[int(batch_index)].reshape(self._local_batch_size, self._num_replicas)
            # Adjacent sorted lengths go to different ranks. Reverse alternating
            # rows and rotate columns per step/epoch to avoid a persistent rank
            # bias while preserving a disjoint partition of the global batch.
            column = (self._rank + output_position + self._epoch) % self._num_replicas
            local_batch = global_batch[:, column].copy()
            local_batch[1::2] = global_batch[1::2, self._num_replicas - 1 - column]
            yield local_batch.tolist()


class _InfiniteLengthBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Keep one torch DataLoader iterator alive across natural epochs.

    DataLoader otherwise tears down and recreates its iterator at every epoch,
    which drains the worker prefetch queue and leaves all GPUs idle briefly.
    The wrapped sampler remains finite for direct use and keeps its natural
    epoch length; only the sampler passed to torch's DataLoader is continuous.
    """

    def __init__(
        self,
        sampler: DistributedLengthBucketBatchSampler,
        adaptive_mse_schedule: _AdaptiveMseDecilePlanSchedule | None = None,
    ):
        self._sampler = sampler
        self._adaptive_mse_schedule = adaptive_mse_schedule
        self._adaptive_mse_block_start: int | None = None

    def __len__(self) -> int:
        return len(self._sampler)

    def set_adaptive_mse_block_start(self, completed_step: int) -> bool:
        """Select the plan block that starts at ``completed_step``.

        Checkpoints preserve the optimizer/train state but deliberately do not
        serialize a Python Torch DataLoader iterator.  A resumed plan-gated
        run must therefore begin at the checkpoint's completed-step boundary,
        rather than replaying the original 70k block.
        """
        if self._adaptive_mse_schedule is None:
            return False
        start_step = self._adaptive_mse_schedule.start_step
        interval = self._adaptive_mse_schedule.interval_batches
        if completed_step < start_step or (completed_step - start_step) % interval:
            raise ValueError(
                "adaptive MSE resume step must be a plan boundary: "
                f"got {completed_step}, start={start_step}, interval={interval}"
            )
        self._adaptive_mse_block_start = int(completed_step)
        return True

    def __iter__(self) -> Iterator[list[int]]:
        if self._adaptive_mse_schedule is not None:
            block_start_step = (
                self._adaptive_mse_block_start
                if self._adaptive_mse_block_start is not None
                else self._adaptive_mse_schedule.start_step
            )
            weights = self._adaptive_mse_schedule.weights_for_block(block_start_step)
            self._sampler.set_sampling_weights(weights)
            yield from self._sampler.iter_replacement_block(
                self._adaptive_mse_schedule.interval_batches,
                block_start_step=block_start_step,
            )
            return
        # Capture the configured resume position locally. Torch may request
        # future batches well ahead of the consumer, so adapter progress must
        # never be treated as the number of completed optimizer steps.
        epoch = self._sampler.epoch
        batch_offset = self._sampler.batch_offset
        while True:
            self._sampler.set_epoch(epoch)
            self._sampler.set_batch_offset(batch_offset)
            yield from self._sampler
            epoch += 1
            batch_offset = 0


def _action_delta_timestamps(fps: float, action_horizon: int, action_start_offset: int) -> list[float]:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("dataset fps must be positive and finite")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if type(action_start_offset) is not int or action_start_offset < 0:
        raise ValueError("action_sequence_start_offset must be a non-negative integer")
    return [(action_start_offset + step) / fps for step in range(action_horizon)]


def _action_query_timestamps(
    fps: float,
    action_horizon: int,
    action_start_offset: int,
    *,
    include_previous_for_motion: bool,
) -> list[float]:
    """Build the action query, optionally prepending a[t] for motion metadata."""
    if not include_previous_for_motion:
        return _action_delta_timestamps(fps, action_horizon, action_start_offset)
    if action_start_offset < 1:
        raise ValueError(
            "action_motion_weighting requires action_sequence_start_offset >= 1 "
            "so a[t] can be queried before the first target"
        )
    return _action_delta_timestamps(fps, action_horizon + 1, action_start_offset - 1)


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # LeRobot stores one video file per episode and opens/seeks it for every
    # sample.  On the shared Ustor filesystem this turns a batch into hundreds
    # of random metadata/video reads.  A local mirror (for example a tmpfs
    # mirror under /dev/shm) can be selected without changing the config's
    # configured repo id, normalization stats, or checkpoint metadata.
    dataset_root = _resolve_dataset_root(repo_id)
    if dataset_root != repo_id:
        logging.info("Using local LeRobot dataset mirror: %s (configured repo: %s)", dataset_root, repo_id)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(dataset_root)
    dataset_cls = (
        _ProfileLeRobotDataset
        if os.environ.get("OPENPI_PROFILE_DATA_DIR", "").strip()
        else lerobot_dataset.LeRobotDataset
    )
    dataset_kwargs = {
        "delta_timestamps": {
            key: _action_query_timestamps(
                dataset_meta.fps,
                action_horizon,
                data_config.action_sequence_start_offset,
                include_previous_for_motion=bool(getattr(model_config, "action_motion_weighting", False)),
            )
            for key in data_config.action_sequence_keys
        },
        # PyAV is faster for the current one-frame-per-video access pattern,
        # while OPENPI_VIDEO_BACKEND allows benchmarking a local TorchCodec
        # runtime without changing the training configuration.
        "video_backend": os.environ.get("OPENPI_VIDEO_BACKEND", "pyav"),
    }
    parquet_num_proc = int(os.environ.get("OPENPI_PARQUET_NUM_PROC", "0"))
    if parquet_num_proc <= 0:
        dataset = dataset_cls(dataset_root, **dataset_kwargs)
    else:
        # LeRobot does not expose Hugging Face's parquet reader parallelism.
        # Apply it only while the dataset constructor materializes its Arrow
        # table; this changes I/O parallelism, not row ordering or contents.
        original_load_dataset = lerobot_dataset.load_dataset

        @functools.wraps(original_load_dataset)
        def parallel_load_dataset(*args, **kwargs):
            kwargs.setdefault("num_proc", parquet_num_proc)
            return original_load_dataset(*args, **kwargs)

        lerobot_dataset.load_dataset = parallel_load_dataset
        try:
            dataset = dataset_cls(dataset_root, **dataset_kwargs)
        finally:
            lerobot_dataset.load_dataset = original_load_dataset

    if data_config.hold_last_action_targets:
        dataset = HoldLastActionTargetsDataset(dataset, data_config.action_sequence_keys)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _resolve_dataset_root(repo_id: str) -> str:
    """Resolve an optional local mirror while retaining the configured repo id."""
    dataset_root = os.environ.get("OPENPI_DATASET_LOCAL_ROOT", "").strip() or repo_id
    if dataset_root != repo_id and not os.path.isfile(os.path.join(dataset_root, "meta", "info.json")):
        raise FileNotFoundError(f"OPENPI_DATASET_LOCAL_ROOT={dataset_root!r} is missing meta/info.json")
    return dataset_root


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        in_order=config.data_loader_in_order,
        memory_bucket_size_multiplier=config.memory_bucket_size_multiplier,
        memory_pad_to_multiple=config.memory_pad_to_multiple,
        trim_prompt_padding=config.trim_prompt_padding,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    in_order: bool = True,
    memory_bucket_size_multiplier: int = 50,
    memory_pad_to_multiple: int = 32,
    trim_prompt_padding: bool = True,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        prefetch_factor: The number of batches loaded in advance by each worker. This is
            ignored when ``num_workers`` is zero.
        in_order: Whether multiprocessing workers must return batches in sampler order.
        seed: The seed to use for shuffling the data.
    """
    use_memory = bool(getattr(model_config, "use_memory", False))
    use_lam_memory = bool(getattr(model_config, "use_lam_memory", use_memory))
    anchor_only = bool(getattr(model_config, "memory_anchor_only", False))
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    memory_lengths = None
    sampling_weights = None
    online_lam = use_memory and bool(getattr(model_config, "online_memory_lam", False))
    if online_lam:
        max_transitions = int(getattr(data_config, "online_memory_max_transitions", 0))
        if max_transitions != model_config.memory_horizon:
            raise ValueError(
                "Online unfrozen LAM must preserve the complete history: "
                f"online_memory_max_transitions={max_transitions}, "
                f"memory_horizon={model_config.memory_horizon}"
            )
        dataset = OnlineCDLAMMemoryDataset(
            dataset,
            memory_horizon=model_config.memory_horizon,
            max_transitions=max_transitions,
            expected_dataset_root=data_config.repo_id,
            demo_anchor_cache_dir=data_config.demo_anchor_cache_dir,
            expected_demo_anchor_definition=DEMO_ANCHOR_IMAGE_DEFINITION,
        )
    elif anchor_only and data_config.repo_id != "fake":
        if data_config.memory_cache_dir is not None:
            raise ValueError("memory_anchor_only must not configure DataConfig.memory_cache_dir")
        if data_config.demo_anchor_cache_dir is None:
            raise ValueError("memory_anchor_only requires DataConfig.demo_anchor_cache_dir")
        dataset = VisionAnchorDataset(
            dataset,
            data_config.demo_anchor_cache_dir,
            expected_dataset_root=data_config.repo_id,
        )
    elif use_lam_memory and data_config.repo_id != "fake":
        if data_config.memory_cache_dir is None:
            raise ValueError("A cached-memory model requires DataConfig.memory_cache_dir")
        future_prediction_horizon = getattr(model_config, "memory_flow_horizon", 0)
        dataset = CDLAMMemoryDataset(
            dataset,
            data_config.memory_cache_dir,
            memory_horizon=model_config.memory_horizon,
            latent_dim=model_config.memory_latent_dim,
            memory_stride=model_config.memory_stride,
            expected_dataset_root=data_config.repo_id,
            expected_latent_mean=model_config.memory_latent_mean,
            expected_latent_std=model_config.memory_latent_std,
            random_drop_execution_tail_min=data_config.memory_random_drop_execution_tail_min,
            random_drop_execution_tail_max=data_config.memory_random_drop_execution_tail_max,
            phase_metadata_cache_dir=data_config.phase_metadata_cache_dir,
            episode_anchor_cache_dir=(
                data_config.episode_anchor_cache_dir
                if getattr(model_config, "memory_episode_anchor", False)
                else None
            ),
            demo_anchor_cache_dir=(
                data_config.demo_anchor_cache_dir if getattr(model_config, "memory_demo_anchor", False) else None
            ),
            expected_demo_anchor_definition=(
                DUAL_ANCHOR_IMAGE_DEFINITION
                if getattr(model_config, "memory_execution_anchor", False)
                else DEMO_ANCHOR_FEATURE_DEFINITION
                if getattr(model_config, "memory_demo_anchor_feature_pool", False)
                else DEMO_ANCHOR_IMAGE_DEFINITION
            )
            if getattr(model_config, "memory_demo_anchor", False)
            else None,
            execution_anchor=getattr(model_config, "memory_execution_anchor", False),
            future_prediction_horizon=future_prediction_horizon,
            future_cache_dir=data_config.future_memory_cache_dir,
            future_latent_dim=(
                getattr(model_config, "memory_future_latent_dim", None) if future_prediction_horizon else None
            ),
            expected_future_latent_mean=getattr(model_config, "memory_future_latent_mean", None),
            expected_future_latent_std=getattr(model_config, "memory_future_latent_std", None),
            demo_direction_text_by_episode=(
                build_patternlock_demo_direction_texts(dataset)
                if getattr(model_config, "memory_demo_direction_generation", False)
                else None
            ),
        )
        if getattr(model_config, "memory_demo_anchor", False) and data_config.demo_anchor_cache_dir is None:
            raise ValueError("memory_demo_anchor requires DataConfig.demo_anchor_cache_dir")
        if getattr(model_config, "memory_episode_anchor", False) and data_config.episode_anchor_cache_dir is None:
            raise ValueError("memory_episode_anchor requires DataConfig.episode_anchor_cache_dir")
    if data_config.execution_only:
        before_filter = len(dataset)
        dataset = ExecutionOnlyDataset(
            dataset,
            allowed_task_indices=data_config.execution_only_task_indices,
            allowed_episode_indices=data_config.execution_only_episode_indices,
            # Memory datasets already expose the validated sidecar columns.
            # This direct path also supports execution-only non-memory configs.
            phase_metadata_cache_dir=None if use_memory else data_config.phase_metadata_cache_dir,
            expected_dataset_root=data_config.repo_id,
        )
        logging.info(
            "Execution-only query filtering: kept %d/%d rows for task indices %s, episode indices %s "
            "(demo rows remain available only as memory)",
            len(dataset),
            before_filter,
            data_config.execution_only_task_indices,
            data_config.execution_only_episode_indices,
        )
    if data_config.task_sampling_episode_weights is not None:
        if not isinstance(dataset, ExecutionOnlyDataset):
            raise ValueError("task_sampling_episode_weights requires execution_only query filtering")
        sampling_weights = _build_episode_block_sampling_weights(
            dataset, data_config.task_sampling_episode_weights
        )
    adaptive_mse_schedule = None
    if data_config.diversity_sampling_cache_dir is not None:
        if not isinstance(dataset, ExecutionOnlyDataset):
            raise ValueError("diversity_sampling_cache_dir requires execution_only query filtering")
        if sampling_weights is not None:
            raise ValueError("task_sampling_episode_weights cannot be combined with diversity sampling")
        plan_dir = data_config.diversity_sampling_mse_plan_dir
        if plan_dir is not None:
            start_step = data_config.diversity_sampling_mse_plan_start_step
            interval_batches = data_config.diversity_sampling_mse_plan_interval_batches
            if data_config.diversity_sampling_mse_plan_path is not None:
                raise ValueError("blockwise adaptive MSE plans cannot also set diversity_sampling_mse_plan_path")
            initial_plan_path = Path(plan_dir).expanduser() / f"step{start_step:06d}.json"
            # Retain the complete cache/dataset contract validation performed
            # by the original static-plan path before the resident iterator is
            # constructed. The schedule reloads the same validated formula at
            # each later boundary.
            sampling_weights = _load_query_diversity_sampling_weights(
                dataset,
                data_config.diversity_sampling_cache_dir,
                expected_dataset_root=data_config.repo_id,
                action_horizon=action_horizon,
                action_start_offset=data_config.action_sequence_start_offset,
                rank_threshold=data_config.diversity_sampling_rank_threshold,
                rank_power=data_config.diversity_sampling_rank_power,
                low_rank_mass=data_config.diversity_sampling_low_rank_mass,
                mse_plan_path=initial_plan_path,
                uniform_mass=data_config.diversity_sampling_uniform_mass,
            )
            adaptive_mse_schedule = _AdaptiveMseDecilePlanSchedule(
                plan_dir=plan_dir,
                start_step=start_step,
                interval_batches=interval_batches,
                dataset=dataset,
                diversity_cache_dir=data_config.diversity_sampling_cache_dir,
                uniform_mass=data_config.diversity_sampling_uniform_mass,
            )
        else:
            sampling_weights = _load_query_diversity_sampling_weights(
                dataset,
                data_config.diversity_sampling_cache_dir,
                expected_dataset_root=data_config.repo_id,
                action_horizon=action_horizon,
                action_start_offset=data_config.action_sequence_start_offset,
                rank_threshold=data_config.diversity_sampling_rank_threshold,
                rank_power=data_config.diversity_sampling_rank_power,
                low_rank_mass=data_config.diversity_sampling_low_rank_mass,
                mse_plan_path=data_config.diversity_sampling_mse_plan_path,
                uniform_mass=data_config.diversity_sampling_uniform_mass,
            )
    if getattr(model_config, "baseline_uncertainty_weighting", False):
        if data_config.baseline_uncertainty_cache_dir is None:
            raise ValueError("baseline_uncertainty_weighting requires DataConfig.baseline_uncertainty_cache_dir")
        dataset = BaselineUncertaintyDataset(
            dataset,
            data_config.baseline_uncertainty_cache_dir,
            expected_dataset_root=data_config.repo_id,
            action_horizon=action_horizon,
            action_start_offset=data_config.action_sequence_start_offset,
        )
    if use_lam_memory and data_config.repo_id != "fake":
        memory_lengths = dataset.memory_lengths
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    if os.environ.get("OPENPI_PROFILE_DATA_DIR", "").strip():
        dataset = _ProfileDataset(dataset)

    # Use TorchDataLoader for both frameworks. Memory histories are ordinary
    # JAX arrays in the JAX path and are sharded after collation below.
    # Build the same deterministic, length-balanced global batch plan for both
    # frameworks. Previously this was only enabled for PyTorch, so the JAX
    # memory path silently ignored memory_bucket_size_multiplier and padded
    # random batches close to the maximum history length.
    sampler = None
    batch_sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if batch_size % world_size != 0:
                raise ValueError(
                    f"Global batch size ({batch_size}) must be divisible by distributed world size ({world_size})."
                )
            local_batch_size = batch_size // world_size
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            local_batch_size = batch_size
            rank = 0
    else:
        world_size = jax.process_count()
        if batch_size % world_size != 0:
            raise ValueError(f"Global batch size ({batch_size}) must be divisible by JAX process count ({world_size}).")
        local_batch_size = batch_size // world_size
        rank = jax.process_index()

    if memory_lengths is not None and shuffle:
        batch_sampler = DistributedLengthBucketBatchSampler(
            memory_lengths,
            local_batch_size=local_batch_size,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            shuffle=True,
            bucket_size_multiplier=memory_bucket_size_multiplier,
            sampling_weights=sampling_weights,
        )
        if memory_bucket_size_multiplier == 1:
            logging.info(
                "Using resumable random memory batch sampler with length bucketing disabled: "
                "framework=%s, global_batch_size=%d, batches_per_epoch=%d",
                framework,
                batch_size,
                len(batch_sampler),
            )
        else:
            logging.info(
                "Using memory length-bucket sampler: framework=%s, global_batch_size=%d, "
                "bucket_batches=%d, batches_per_epoch=%d",
                framework,
                batch_size,
                memory_bucket_size_multiplier,
                len(batch_sampler),
            )
    elif framework == "pytorch" and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
            seed=seed,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        batch_sampler=batch_sampler,
        adaptive_mse_schedule=adaptive_mse_schedule,
        num_batches=num_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        in_order=in_order,
        memory_pad_to_multiple=memory_pad_to_multiple,
        memory_max_length=model_config.memory_horizon if use_lam_memory else None,
        trim_prompt_padding=trim_prompt_padding,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler[list[int]] | None = None,
        adaptive_mse_schedule: _AdaptiveMseDecilePlanSchedule | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        in_order: bool = True,
        memory_pad_to_multiple: int = 1,
        memory_max_length: int | None = None,
        trim_prompt_padding: bool = False,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            prefetch_factor: The number of batches loaded in advance by each worker. This is
                ignored when ``num_workers`` is zero.
            in_order: Whether multiprocessing workers must return batches in sampler order.
            seed: The seed to use for shuffling the data.
        """
        # The PyTorch path uses JAX only for pure Python pytree traversal and
        # must not initialize a second GPU runtime/client in every DDP rank.
        if framework != "pytorch" and jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")
        if memory_pad_to_multiple <= 0:
            raise ValueError(f"memory_pad_to_multiple must be positive, got {memory_pad_to_multiple}")
        if memory_max_length is not None and memory_max_length <= 0:
            raise ValueError(f"memory_max_length must be positive, got {memory_max_length}")
        if type(prefetch_factor) is not int or prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be a positive integer, got {prefetch_factor!r}")
        if type(in_order) is not bool:
            raise ValueError(f"in_order must be a boolean, got {in_order!r}")

        self._memory_pad_to_multiple = memory_pad_to_multiple
        self._memory_max_length = memory_max_length
        self._trim_prompt_padding = trim_prompt_padding and framework == "pytorch"
        self._as_torch = framework == "pytorch"

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        if sampler is not None and batch_sampler is not None:
            raise ValueError("sampler and batch_sampler are mutually exclusive")
        self._sampler = batch_sampler if batch_sampler is not None else sampler
        self._adaptive_mse_batch_sampler: _InfiniteLengthBucketBatchSampler | None = None
        self._epoch = 0
        self._fallback_batch_offset = 0

        mp_context = None
        if num_workers > 0:
            # Always use spawn when multiprocessing is requested.  This keeps
            # worker startup isolated from JAX/CUDA runtime state and avoids
            # inheriting live accelerator threads through fork/forkserver.
            context_name = "spawn"
            mp_context = multiprocessing.get_context(context_name)

        generator = torch.Generator()
        generator.manual_seed(seed)
        # NumPy arrays are pickled and copied through multiprocessing pipes.
        # A random full-history memory batch can exceed 500 MB, so that path
        # leaves the GPUs idle while the parent deserializes every batch.
        # Torch tensors use PyTorch's shared-memory multiprocessing reducer;
        # the JAX parent converts them to zero-copy NumPy views below before
        # starting the host-to-device transfer.
        use_torch_worker_ipc = self._as_torch or num_workers > 0
        if framework == "jax" and num_workers > 0:
            logging.info("Using shared-memory Torch IPC for JAX DataLoader workers")
        batch_collate_fn = functools.partial(
            _collate_fn,
            memory_pad_to_multiple=memory_pad_to_multiple,
            memory_max_length=memory_max_length,
            trim_prompt_padding=self._trim_prompt_padding,
            as_torch=use_torch_worker_ipc,
        )
        common_loader_kwargs = {
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "collate_fn": batch_collate_fn,
            "worker_init_fn": functools.partial(_worker_init_fn, direct_exit=framework == "jax"),
            "generator": generator,
            "pin_memory": framework == "pytorch" and torch.cuda.is_available() and torch.cuda.device_count() > 0,
        }
        if num_workers > 0:
            common_loader_kwargs["prefetch_factor"] = prefetch_factor
            common_loader_kwargs["in_order"] = in_order
        if batch_sampler is not None:
            torch_batch_sampler = (
                _InfiniteLengthBucketBatchSampler(batch_sampler, adaptive_mse_schedule)
                if isinstance(batch_sampler, DistributedLengthBucketBatchSampler)
                else batch_sampler
            )
            if isinstance(torch_batch_sampler, _InfiniteLengthBucketBatchSampler):
                self._adaptive_mse_batch_sampler = torch_batch_sampler
            batching_kwargs = {"batch_sampler": torch_batch_sampler}
        else:
            batching_kwargs = {
                "batch_size": local_batch_size,
                "shuffle": sampler is None and shuffle,
                "sampler": sampler,
                "drop_last": True,
            }
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            **batching_kwargs,
            **common_loader_kwargs,
        )
        self.set_epoch(0)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __len__(self) -> int:
        """Return batches in one pass over the underlying dataset."""
        return len(self._data_loader)

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        """Select the sampler epoch and next batch, including after resume."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        if batch_offset < 0 or batch_offset > len(self):
            raise ValueError(f"batch_offset must be in [0, {len(self)}], got {batch_offset}")
        self._epoch = int(epoch)
        set_epoch = getattr(self._sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(self._epoch)
        set_batch_offset = getattr(self._sampler, "set_batch_offset", None)
        if set_batch_offset is not None:
            set_batch_offset(batch_offset)
            self._fallback_batch_offset = 0
        else:
            # Preserve resume support for ordinary samplers. The memory length
            # sampler takes the fast path above and never reads skipped items.
            self._fallback_batch_offset = int(batch_offset)

    def set_adaptive_mse_block_start(self, completed_step: int) -> bool:
        """Prepare the next iterator to consume the specified plan block."""
        if self._adaptive_mse_batch_sampler is None:
            return False
        return self._adaptive_mse_batch_sampler.set_adaptive_mse_block_start(completed_step)

    def close(self) -> None:
        """Explicitly stop persistent Torch workers before interpreter teardown.

        With ``persistent_workers=True`` PyTorch retains its multiprocessing
        iterator on the DataLoader object. Leaving that iterator for Python's
        interpreter shutdown can race JAX runtime teardown and abort a process
        after an otherwise successful train run. This is only called when the
        enclosing process is done; block boundaries retain workers normally.
        """
        torch_iterator = getattr(self._data_loader, "_iterator", None)
        if torch_iterator is None:
            return
        shutdown_workers = getattr(torch_iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            logging.info("Stopping persistent Torch DataLoader workers")
            shutdown_workers()
        # A stopped multiprocessing iterator cannot be reused. Clear it so a
        # later intentional iteration constructs a fresh worker pool.
        self._data_loader._iterator = None

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            fallback_batch_offset = self._fallback_batch_offset
            self._fallback_batch_offset = 0
            for _ in range(fallback_batch_offset):
                try:
                    next(data_iter)
                except StopIteration as error:
                    raise RuntimeError("batch_offset exceeded the selected sampler epoch") from error
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                queue_start = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if self._adaptive_mse_batch_sampler is not None:
                        # A plan-aware sampler deliberately ends exactly at its
                        # block boundary. The trainer publishes the following
                        # plan before it asks this public iterator for another
                        # batch, so recreating it cannot prefetch across plans.
                        logging.info("Adaptive MSE block exhausted; awaiting next published plan")
                    else:
                        # The public loader is intentionally infinite. Advance
                        # the DistributedSampler seed before recreating its
                        # iterator so every natural dataset epoch gets a fresh
                        # deterministic permutation on every rank.
                        self.set_epoch(self._epoch + 1)
                    break
                _profile_event("queue_next", time.perf_counter() - queue_start)
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    device_start = time.perf_counter()

                    def to_jax_array(x):
                        if isinstance(x, torch.Tensor):
                            if x.device.type != "cpu":
                                raise ValueError("JAX DataLoader IPC tensors must remain on CPU")
                            # CPU Tensor.numpy() is a zero-copy view of the
                            # shared-memory storage received from the worker.
                            x = x.detach().numpy()
                        return jax.make_array_from_process_local_data(self._sharding, x)

                    yield jax.tree.map(to_jax_array, batch)
                    _profile_event("device_put", time.perf_counter() - device_start)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(
    items,
    *,
    memory_pad_to_multiple: int = 1,
    memory_max_length: int | None = None,
    trim_prompt_padding: bool = False,
    as_torch: bool = False,
):
    """Collate samples, left-padding variable complete memory histories."""
    collate_start = time.perf_counter()
    if memory_pad_to_multiple <= 0:
        raise ValueError(f"memory_pad_to_multiple must be positive, got {memory_pad_to_multiple}")
    batched_memory = None
    batched_memory_mask = None
    batched_memory_segments = None
    batched_online_pairs = None
    batched_online_mask = None
    batched_online_segments = None
    if items and all(isinstance(item, dict) for item in items):
        has_online = [
            (item.get("memory_lam_pairs") is not None, item.get("memory_lam_mask") is not None)
            for item in items
        ]
        if any(pairs != mask for pairs, mask in has_online):
            raise ValueError("Every online LAM sample must provide pairs and mask together")
        if any(pairs for pairs, _ in has_online):
            if not all(pairs for pairs, _ in has_online):
                raise ValueError("Either every sample must provide online LAM pairs or none may provide them")
            if any(item.get("memory_latents") is not None for item in items):
                raise ValueError("A batch cannot mix cached latents with online LAM pairs")
            pair_arrays = [np.asarray(item["memory_lam_pairs"]) for item in items]
            pair_masks = [np.asarray(item["memory_lam_mask"]) for item in items]
            segments = [np.asarray(item["memory_segment_ids"]) for item in items]
            pair_shapes = {array.shape[1:] for array in pair_arrays if array.ndim == 6}
            if (
                any(array.ndim != 6 or len(array) < 1 for array in pair_arrays)
                or any(mask.ndim != 1 or len(mask) != len(array) for mask, array in zip(pair_masks, pair_arrays, strict=True))
                or any(segment.ndim != 1 or len(segment) != len(array) for segment, array in zip(segments, pair_arrays, strict=True))
                or len(pair_shapes) != 1
            ):
                raise ValueError("Online LAM pairs must be compatible [T, 2, 2, H, W, 3] arrays")
            raw_max_length = max(len(array) for array in pair_arrays)
            if memory_max_length is not None and raw_max_length > memory_max_length:
                raise ValueError(
                    f"Online LAM batch length {raw_max_length} exceeds configured maximum {memory_max_length}"
                )
            max_length = (
                (raw_max_length + memory_pad_to_multiple - 1) // memory_pad_to_multiple
            ) * memory_pad_to_multiple
            if memory_max_length is not None:
                max_length = min(max_length, memory_max_length)
            pair_shape = next(iter(pair_shapes))
            batched_online_pairs = np.zeros((len(items), max_length, *pair_shape), dtype=np.uint8)
            batched_online_mask = np.zeros((len(items), max_length), dtype=np.bool_)
            batched_online_segments = np.zeros((len(items), max_length), dtype=np.int32)
            retained = []
            for batch_index, (item, pairs, mask, segment) in enumerate(
                zip(items, pair_arrays, pair_masks, segments, strict=True)
            ):
                left_padding = max_length - len(pairs)
                bool_mask = mask.astype(np.bool_, copy=False)
                if np.any((segment < 0) | (segment > 3)) or np.any((segment == 0) != ~bool_mask):
                    raise ValueError("Online LAM segment ids must be 0 exactly on masked entries")
                batched_online_pairs[batch_index, left_padding:] = pairs
                batched_online_mask[batch_index, left_padding:] = bool_mask
                batched_online_segments[batch_index, left_padding:] = segment.astype(np.int32, copy=False)
                retained.append(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in ("memory_lam_pairs", "memory_lam_mask", "memory_segment_ids")
                    }
                )
            items = retained
    if items and all(isinstance(item, dict) for item in items):
        has_memory = [(item.get("memory_latents") is not None, item.get("memory_mask") is not None) for item in items]
        if any(latents != mask for latents, mask in has_memory):
            raise ValueError("Every sample must provide memory_latents and memory_mask together")
        if any(latents for latents, _ in has_memory):
            if not all(latents for latents, _ in has_memory):
                raise ValueError("Either every sample in a batch must provide memory or none may provide it")

            memory_arrays = [np.asarray(item["memory_latents"]) for item in items]
            memory_masks = [np.asarray(item["memory_mask"]) for item in items]
            has_segments = [item.get("memory_segment_ids") is not None for item in items]
            if any(has_segments) and not all(has_segments):
                raise ValueError("Either every memory sample must provide memory_segment_ids or none may provide it")
            memory_segments = [np.asarray(item["memory_segment_ids"]) for item in items] if all(has_segments) else None
            latent_dims = {array.shape[1] for array in memory_arrays if array.ndim == 2}
            if (
                any(array.ndim != 2 or len(array) < 1 for array in memory_arrays)
                or any(mask.ndim != 1 for mask in memory_masks)
                or any(len(array) != len(mask) for array, mask in zip(memory_arrays, memory_masks, strict=True))
                or (
                    memory_segments is not None
                    and any(
                        segment.ndim != 1 or len(segment) != len(array)
                        for segment, array in zip(memory_segments, memory_arrays, strict=True)
                    )
                )
                or len(latent_dims) != 1
            ):
                raise ValueError("Memory samples must have compatible non-empty [T, D] latents and [T] masks")

            raw_max_memory_length = max(len(array) for array in memory_arrays)
            if memory_max_length is not None and raw_max_memory_length > memory_max_length:
                raise ValueError(
                    f"Memory batch length {raw_max_memory_length} exceeds configured maximum {memory_max_length}"
                )
            max_memory_length = (
                (raw_max_memory_length + memory_pad_to_multiple - 1) // memory_pad_to_multiple
            ) * memory_pad_to_multiple
            if memory_max_length is not None:
                max_memory_length = min(max_memory_length, memory_max_length)
            latent_dtype = np.result_type(*(memory.dtype for memory in memory_arrays))
            batched_memory = np.zeros((len(items), max_memory_length, next(iter(latent_dims))), dtype=latent_dtype)
            batched_memory_mask = np.zeros((len(items), max_memory_length), dtype=np.bool_)
            if memory_segments is not None:
                batched_memory_segments = np.zeros((len(items), max_memory_length), dtype=np.int32)

            non_memory_items = []
            segment_iter = memory_segments if memory_segments is not None else [None] * len(items)
            for batch_index, (item, memory, mask, segment) in enumerate(
                zip(items, memory_arrays, memory_masks, segment_iter, strict=True)
            ):
                left_padding = max_memory_length - len(memory)
                batched_memory[batch_index, left_padding:] = memory
                bool_mask = mask.astype(np.bool_, copy=False)
                batched_memory_mask[batch_index, left_padding:] = bool_mask
                if segment is not None:
                    if np.any((segment < 0) | (segment > 3)):
                        raise ValueError("memory_segment_ids must be in [0, 3]")
                    if np.any((segment == 0) != ~bool_mask):
                        raise ValueError("memory_segment_ids must be 0 exactly where memory_mask is false")
                    batched_memory_segments[batch_index, left_padding:] = segment.astype(np.int32, copy=False)
                non_memory_items.append(
                    {
                        key: value
                        for key, value in item.items()
                        if key not in ("memory_latents", "memory_mask", "memory_segment_ids")
                    }
                )
            items = non_memory_items

    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
    if batched_memory is not None:
        batch["memory_latents"] = batched_memory
        batch["memory_mask"] = batched_memory_mask
        if batched_memory_segments is not None:
            batch["memory_segment_ids"] = batched_memory_segments
    if batched_online_pairs is not None:
        batch["memory_lam_pairs"] = batched_online_pairs
        batch["memory_lam_mask"] = batched_online_mask
        batch["memory_segment_ids"] = batched_online_segments
    if trim_prompt_padding and isinstance(batch, dict) and "tokenized_prompt_mask" in batch:
        prompt_mask = np.asarray(batch["tokenized_prompt_mask"], dtype=np.bool_)
        if prompt_mask.ndim != 2:
            raise ValueError(f"tokenized_prompt_mask must be rank 2, got {prompt_mask.shape}")
        used_columns = np.flatnonzero(prompt_mask.any(axis=0))
        if len(used_columns) == 0:
            raise ValueError("Every prompt in a training batch is empty")
        prompt_length = int(used_columns[-1]) + 1
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            value = batch.get(key)
            if value is not None and np.ndim(value) == 2 and value.shape[1] == prompt_mask.shape[1]:
                batch[key] = value[:, :prompt_length]

    result = jax.tree.map(torch.from_numpy, batch) if as_torch else batch
    _profile_event("collate", time.perf_counter() - collate_start)
    return result


def _worker_init_fn(worker_id: int, *, direct_exit: bool = False) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["OPENPI_PROFILE_WORKER_ID"] = str(worker_id)
    if direct_exit:
        # VideoReader/PyAV can abort during normal C++ teardown in a spawned
        # JAX worker. Run this finalizer first, after PyTorch's worker loop has
        # acknowledged its shutdown sentinel but before interpreter teardown.
        _multiprocessing_util.Finalize(
            None, os._exit, args=(0,), exitpriority=10_000
        )

    # Keep video decoders from competing across all host CPUs.  The launcher
    # can provide a disjoint range per worker; otherwise leave the scheduler's
    # default affinity unchanged.
    affinity = os.environ.get("OPENPI_WORKER_CPU_AFFINITY")
    if affinity:
        try:
            os.sched_setaffinity(0, {int(cpu) for cpu in affinity.split(",") if cpu.strip()})
        except (AttributeError, OSError, ValueError) as exc:
            logging.warning("Unable to set worker %s CPU affinity to %r: %s", worker_id, affinity, exc)
    else:
        start = os.environ.get("OPENPI_WORKER_CPU_START")
        width = os.environ.get("OPENPI_WORKER_CPU_WIDTH")
        if start and width:
            try:
                first = int(start) + worker_id * int(width)
                os.sched_setaffinity(0, set(range(first, first + int(width))))
            except (AttributeError, OSError, ValueError) as exc:
                logging.warning(
                    "Unable to set worker %s CPU affinity from start=%r width=%r: %s", worker_id, start, width, exc
                )


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

    def __len__(self) -> int:
        return len(self._dataset)

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        if batch_offset:
            raise ValueError("RLDSDataLoader does not support a non-zero batch_offset")
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        set_epoch = getattr(self._dataset, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int, *, batch_offset: int = 0) -> None:
        self._data_loader.set_epoch(epoch, batch_offset=batch_offset)

    def set_adaptive_mse_block_start(self, completed_step: int) -> bool:
        configure = getattr(self._data_loader, "set_adaptive_mse_block_start", None)
        if configure is None:
            return False
        return bool(configure(completed_step))

    def close(self) -> None:
        close = getattr(self._data_loader, "close", None)
        if callable(close):
            close()

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch, normalize_torch_images=False), batch["actions"]
