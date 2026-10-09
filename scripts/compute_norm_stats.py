"""Compute normalization statistics for a config."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _norm_repack_transforms(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    """Drop memory-only fields that are irrelevant to state/action statistics."""
    output = []
    for transform in data_config.repack_transforms.inputs:
        if isinstance(transform, transforms.RepackTransform):
            structure = {
                key: value
                for key, value in transform.structure.items()
                if not str(key).startswith("memory_") and key != "demo_direction_text"
            }
            output.append(dataclasses.replace(transform, structure=structure))
        else:
            output.append(transform)
    return output


def _state_action_only_transforms(data_config: _config.DataConfig) -> list[transforms.DataTransformFn]:
    """Return transforms that can change state/action normalization values."""
    output: list[transforms.DataTransformFn] = []
    for transform in data_config.data_transforms.inputs:
        if isinstance(transform, (transforms.DeltaActions, transforms.AbsoluteActions)):
            output.append(transform)
            continue
        transform_id = (transform.__class__.__module__, transform.__class__.__name__)
        if transform_id == ("openpi.policies.libero_policy", "LiberoInputs"):
            continue
        raise ValueError(
            "OPENPI_NORM_STATE_ACTION_ONLY does not know whether transform "
            f"{transform_id[0]}.{transform_id[1]} changes state/actions"
        )
    return output


def _fixed_vector_column(dataset, name: str) -> np.ndarray:
    array = dataset.hf_dataset.data.column(name).combine_chunks()
    width = getattr(array.type, "list_size", None)
    if width is None:
        raise TypeError(f"Expected {name!r} to be a fixed-size list column, got {array.type}")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False))
    return values.reshape(len(array), int(width))


class _VectorizedStateActionLoader:
    """Read norm-stat tensors directly from Arrow without decoding videos."""

    def __init__(self, dataset, data_config: _config.DataConfig, action_horizon: int, batch_size: int):
        if not isinstance(dataset, _data_loader.ExecutionOnlyDataset):
            raise TypeError("Vectorized norm stats currently require execution_only=True")
        self._source_indices = np.asarray(dataset.source_indices, dtype=np.int64)
        self._base = _data_loader._find_lerobot_dataset(dataset)
        self._states = _fixed_vector_column(self._base, "state")
        self._actions = _fixed_vector_column(self._base, "actions")
        episode_indices = np.asarray(
            self._base.hf_dataset.data.column("episode_index").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        episode_ends = np.asarray(self._base.episode_data_index["to"], dtype=np.int64)
        self._last_indices = episode_ends[episode_indices[self._source_indices]] - 1
        self._offset = int(data_config.action_sequence_start_offset)
        self._horizon = int(action_horizon)
        self._batch_size = int(batch_size)
        self._transforms = _state_action_only_transforms(data_config)
        if self._offset < 0 or self._horizon <= 0:
            raise ValueError(f"Invalid action query offset/horizon: {self._offset}/{self._horizon}")

    def __len__(self) -> int:
        return len(self._source_indices) // self._batch_size

    def batch_for_positions(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        source = self._source_indices[np.asarray(positions, dtype=np.int64)]
        last = self._last_indices[np.asarray(positions, dtype=np.int64)]
        steps = self._offset + np.arange(self._horizon, dtype=np.int64)
        action_indices = np.minimum(source[:, None] + steps[None, :], last[:, None])
        batch = {
            "state": self._states[source].copy(),
            "actions": self._actions[action_indices].copy(),
        }
        for transform in self._transforms:
            batch = transform(batch)
        return batch

    def __iter__(self):
        for start in range(0, len(self) * self._batch_size, self._batch_size):
            yield self.batch_for_positions(np.arange(start, start + self._batch_size, dtype=np.int64))


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int, dict[str, int] | None]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    selection = None
    if data_config.execution_only:
        source_rows = len(dataset)
        dataset = _data_loader.ExecutionOnlyDataset(
            dataset,
            allowed_task_indices=data_config.execution_only_task_indices,
            phase_metadata_cache_dir=data_config.phase_metadata_cache_dir,
            expected_dataset_root=data_config.repo_id,
        )
        selection = {"source_rows": source_rows, "execution_rows": len(dataset)}
        print(f"Norm-stat selection: execution rows only, kept {len(dataset)}/{source_rows}")
    full_transforms = [
        *_norm_repack_transforms(data_config),
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ]
    if os.environ.get("OPENPI_NORM_STATE_ACTION_ONLY", "").strip() == "1":
        if max_frames is not None:
            raise ValueError("Vectorized state/action norm stats require max_frames=None")
        # Check both the first execution row and an episode-tail row, where
        # horizon padding must hold the final action, against the full path.
        probe_positions = np.asarray([0, len(dataset) - 1], dtype=np.int64)
        full_dataset = _data_loader.TransformedDataset(dataset, full_transforms)
        full_probe = {
            key: np.stack([np.asarray(full_dataset[int(position)][key]) for position in probe_positions])
            for key in ("state", "actions")
        }
        data_loader = _VectorizedStateActionLoader(dataset, data_config, action_horizon, batch_size)
        fast_probe = data_loader.batch_for_positions(probe_positions)
        for key in ("state", "actions"):
            full_value = np.asarray(full_probe[key])
            fast_value = np.asarray(fast_probe[key])
            if full_value.shape != fast_value.shape or not np.array_equal(full_value, fast_value):
                max_abs = float(np.max(np.abs(full_value - fast_value)))
                raise ValueError(f"vectorized norm path mismatch for {key}: max_abs={max_abs}")
        print("Norm-stat vectorized path verified on first/tail rows: state/actions exactly match")
        return data_loader, len(data_loader), selection

    dataset = _data_loader.TransformedDataset(dataset, full_transforms)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    norm_num_workers = int(os.environ.get("OPENPI_NORM_NUM_WORKERS", str(num_workers)))
    print(f"Norm-stat DataLoader workers: {norm_num_workers}")
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=norm_num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches, selection


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    return data_loader, num_batches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        selection = None
    else:
        data_loader, num_batches, selection = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    if data_config.asset_id is None:
        raise ValueError("Data config must define asset_id before normalization stats can be saved")
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    if data_config.execution_only:
        dataset_root = Path(data_config.repo_id).expanduser().resolve()
        info_path = dataset_root / "meta" / "info.json"
        episodes_path = dataset_root / "meta" / "episodes.jsonl"
        info = json.loads(info_path.read_text())
        if selection is None:
            raise RuntimeError("execution_only norm stats lost their selection metadata")
        execution_frames = selection["execution_rows"]
        manifest = {
            "schema_version": 1,
            "selection": "is_demo_false_execution_queries_only",
            "dataset_root": str(dataset_root),
            "dataset_info_sha256": _sha256(info_path),
            "dataset_episodes_sha256": _sha256(episodes_path),
            "total_frames": int(info["total_frames"]),
            "execution_frames": execution_frames,
            "demo_frames_excluded": int(info["total_frames"]) - execution_frames,
            "action_horizon": config.model.action_horizon,
            "state_key": "state",
            "action_key": "actions",
            "delta_action_dims": getattr(config.data, "delta_action_dims", None),
            "max_frames": max_frames,
        }
        if data_config.phase_metadata_cache_dir is not None:
            phase_root = Path(data_config.phase_metadata_cache_dir).expanduser().resolve()
            phase_manifest = phase_root / "manifest.json"
            manifest["phase_metadata_cache_dir"] = str(phase_root)
            manifest["phase_metadata_manifest_sha256"] = _sha256(phase_manifest)
        manifest_path = Path(output_path) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Writing selection manifest to: {manifest_path}")


if __name__ == "__main__":
    tyro.cli(main)
