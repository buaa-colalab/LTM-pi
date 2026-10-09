"""Online differentiable CD-LAM inputs for PyTorch training.

This wrapper carries causal raw RGB frame pairs instead of cached LAM latents.
The model-side encoder therefore receives action-loss gradients.
"""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training.cdlam_memory_dataset import (
    DEMO_ANCHOR_IMAGE_DEFINITION,
    DUAL_ANCHOR_IMAGE_DEFINITION,
    _dataset_identity,
    _load_demo_anchor_cache,
    classify_memory_segments,
)


def _scalar(value: Any, name: str) -> int:
    value = np.asarray(value).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar {name}, got shape {value.shape}")
    return int(value[0])


def _uint8_hwc(value: Any, *, name: str) -> np.ndarray:
    """Normalize a decoded LeRobot image to contiguous uint8 HWC."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be rank-3, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must be RGB, got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating) or not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must be finite uint8/float RGB, got {image.dtype}")
    scale = 255.0 if float(image.max(initial=0.0)) <= 1.0 + 1e-6 else 1.0
    return np.ascontiguousarray(np.clip(np.rint(image * scale), 0, 255).astype(np.uint8))


class OnlineCDLAMMemoryDataset(Sized):
    """Attach uniformly sampled causal two-view frame pairs to every query."""

    def __init__(
        self,
        dataset,
        *,
        memory_horizon: int,
        max_transitions: int,
        expected_dataset_root: str | Path,
        demo_anchor_cache_dir: str | Path | None = None,
        expected_demo_anchor_definition: str | None = None,
    ):
        if memory_horizon <= 0 or max_transitions <= 0 or max_transitions > memory_horizon:
            raise ValueError(
                f"invalid online history: horizon={memory_horizon}, max_transitions={max_transitions}"
            )
        self._dataset = dataset
        self._memory_horizon = int(memory_horizon)
        self._max_transitions = int(max_transitions)

        raw_dataset = dataset
        seen: set[int] = set()
        while id(raw_dataset) not in seen:
            seen.add(id(raw_dataset))
            if hasattr(raw_dataset, "episode_data_index") and hasattr(raw_dataset, "hf_dataset"):
                break
            raw_dataset = getattr(raw_dataset, "_dataset", None)
            if raw_dataset is None:
                raise TypeError("Online CD-LAM memory requires an underlying LeRobotDataset")
        else:
            raise TypeError("Online CD-LAM dataset wrapper cycle detected")
        if len(raw_dataset) != len(dataset):
            raise ValueError("Online CD-LAM wrapper must preserve global LeRobot row order")
        self._raw_dataset = raw_dataset

        index = raw_dataset.episode_data_index
        self._starts = np.asarray(index["from"], dtype=np.int64)
        self._stops = np.asarray(index["to"], dtype=np.int64)
        if (
            self._starts.ndim != 1
            or self._stops.ndim != 1
            or not len(self._starts)
            or self._starts.shape != self._stops.shape
            or self._starts[0] != 0
            or self._stops[-1] != len(dataset)
            or np.any(self._starts[1:] != self._stops[:-1])
            or np.any(self._stops <= self._starts)
        ):
            raise ValueError("LeRobot episode_data_index is not one contiguous positive-length row partition")
        positions = np.arange(len(dataset), dtype=np.int64)
        slots = np.searchsorted(self._stops, positions, side="right")
        self._frame_indices = (positions - self._starts[slots]).astype(np.int32, copy=False)
        self._memory_lengths = np.maximum(
            1, np.minimum(self._frame_indices, self._max_transitions)
        ).astype(np.int32, copy=False)
        self._memory_lengths.setflags(write=False)

        self._demo_anchor_images = None
        self._demo_anchor_images_path = None
        self._episode_to_demo_anchor = None
        self._empty_demo_anchor = None
        if demo_anchor_cache_dir is not None:
            root = Path(expected_dataset_root).expanduser().resolve()
            _, total_episodes, _, _, fingerprint = _dataset_identity(root)
            anchors, mapping, _, definition, _ = _load_demo_anchor_cache(
                demo_anchor_cache_dir,
                dataset_root=root,
                dataset_fingerprint=fingerprint,
                total_episodes=total_episodes,
                expected_definition=expected_demo_anchor_definition,
            )
            if definition not in (DEMO_ANCHOR_IMAGE_DEFINITION, DUAL_ANCHOR_IMAGE_DEFINITION):
                raise ValueError("Online CD-LAM requires raw-image demo anchors")
            self._demo_anchor_images = anchors
            self._demo_anchor_images_path = Path(anchors.filename)
            self._episode_to_demo_anchor = mapping.astype(np.int64, copy=False)
            self._empty_demo_anchor = np.zeros(anchors.shape[1:], dtype=anchors.dtype)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_demo_anchor_images"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._demo_anchor_images_path is not None:
            self._demo_anchor_images = np.load(self._demo_anchor_images_path, mmap_mode="r", allow_pickle=False)

    @property
    def memory_lengths(self) -> np.ndarray:
        return self._memory_lengths

    def _endpoints(self, position: int) -> np.ndarray:
        frame = int(self._frame_indices[position])
        if frame == 0:
            return np.empty(0, dtype=np.int64)
        if frame > self._memory_horizon:
            raise RuntimeError(f"History {frame} exceeds memory_horizon {self._memory_horizon}")
        slot = int(np.searchsorted(self._stops, position, side="right"))
        count = min(frame, self._max_transitions)
        offsets = np.linspace(1, frame, count, endpoint=True, dtype=np.int64)
        if len(np.unique(offsets)) != count:
            raise RuntimeError("Online uniform history sampler emitted duplicate endpoints")
        return self._starts[slot] + offsets

    def __getitem__(self, index: Any) -> dict:
        position = index.__index__()
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        sample = self._dataset[position]
        if _scalar(sample["index"], "index") != position:
            raise ValueError("LeRobot index no longer equals the row position")
        frame = int(self._frame_indices[position])
        exec_start = _scalar(sample["exec_start_idx"], "exec_start_idx")
        if exec_start < 0:
            raise ValueError(f"Invalid exec_start_idx={exec_start}")
        current_raw = self._raw_dataset[position]
        head_shape = _uint8_hwc(current_raw["image"], name="image").shape
        wrist_shape = _uint8_hwc(current_raw["wrist_image"], name="wrist_image").shape
        if head_shape != wrist_shape:
            raise ValueError(f"Online CD-LAM needs matching two-view shapes: {head_shape} vs {wrist_shape}")

        endpoints = self._endpoints(position)
        if frame == 0:
            pairs = np.zeros((1, 2, 2, *head_shape), dtype=np.uint8)
            mask = np.zeros(1, dtype=np.bool_)
            segments = np.zeros(1, dtype=np.int32)
        else:
            pairs = np.empty((len(endpoints), 2, 2, *head_shape), dtype=np.uint8)
            for i, endpoint in enumerate(endpoints):
                previous = self._raw_dataset[int(endpoint) - 1]
                current = current_raw if int(endpoint) == position else self._raw_dataset[int(endpoint)]
                pairs[i, 0, 0] = _uint8_hwc(previous["image"], name="image")
                pairs[i, 0, 1] = _uint8_hwc(current["image"], name="image")
                pairs[i, 1, 0] = _uint8_hwc(previous["wrist_image"], name="wrist_image")
                pairs[i, 1, 1] = _uint8_hwc(current["wrist_image"], name="wrist_image")
            mask = np.ones(len(endpoints), dtype=np.bool_)
            segments = classify_memory_segments(endpoints - (position - frame), exec_start)

        sample["memory_lam_pairs"] = pairs
        sample["memory_lam_mask"] = mask
        sample["memory_segment_ids"] = segments
        if self._demo_anchor_images is not None:
            episode = _scalar(sample["episode_index"], "episode_index")
            if episode < 0 or episode >= len(self._episode_to_demo_anchor):
                raise ValueError(f"Invalid episode_index={episode}")
            anchor = int(self._episode_to_demo_anchor[episode])
            sample["memory_demo_start_image"] = np.asarray(
                self._demo_anchor_images[anchor] if anchor >= 0 else self._empty_demo_anchor
            )
            sample["memory_demo_start_mask"] = np.asarray(anchor >= 0, dtype=np.bool_)
        return sample

