"""Resident full-history RoboMME validation for adaptive replacement sampling.

This deliberately shares the FSDP-sharded training parameters.  It does not
load a second checkpoint/model, so a 2k boundary can validate on the same
eight H100s while retaining the optimizer and compiled train step.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as model_lib
import openpi.transforms as transforms
from openpi.shared import nnx_utils
from openpi.training import config as training_config


RANK_DECILE_DEFINITION = "global diversity empirical-midrank intervals [j/10,(j+1)/10), final interval inclusive"
DECILE_LABELS = tuple(f"rank_{start:02d}_{start + 10:02d}pct" for start in range(0, 100, 10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pack(path: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, str]]:
    manifest_path = path / "manifest.json"
    ready_path = path / "_READY"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("adaptive MSE evaluation pack is not complete schema-v1")
    if ready_path.read_text().strip() != _sha256(manifest_path):
        raise ValueError("adaptive MSE evaluation pack _READY does not match manifest")
    arrays: dict[str, np.ndarray] = {}
    for name, expected in manifest["files"].items():
        file = path / name
        if not file.is_file() or _sha256(file) != expected:
            raise ValueError(f"adaptive MSE evaluation pack file failed validation: {name}")
        if name.endswith(".npy"):
            arrays[name.removesuffix(".npy")] = np.load(file, mmap_mode="r", allow_pickle=False)
    prompts = json.loads((path / "prompts.json").read_text())
    return manifest, arrays, prompts


def _batch_input(arrays: dict[str, np.ndarray], rows: np.ndarray, prompts: dict[str, str]) -> dict:
    task_indices = arrays["task_index"][rows]
    prompt_values = [prompts[str(int(task))] for task in task_indices]
    return {
        "observation/image": np.asarray(arrays["image"][rows]),
        "observation/wrist_image": np.asarray(arrays["wrist_image"][rows]),
        "observation/state": np.asarray(arrays["state"][rows]),
        "memory_latents": np.asarray(arrays["memory_latents"][rows], dtype=np.float32),
        "memory_mask": np.asarray(arrays["memory_mask"][rows]),
        "memory_segment_ids": np.asarray(arrays["memory_segment_ids"][rows], dtype=np.int32),
        "memory_demo_start_image": np.asarray(arrays["memory_demo_start_image"][rows]),
        "memory_demo_start_mask": np.asarray(arrays["memory_demo_start_mask"][rows]),
        "memory_execution_start_image": np.asarray(arrays["memory_execution_start_image"][rows]),
        "memory_execution_start_mask": np.asarray(arrays["memory_execution_start_mask"][rows]),
        "prompt": prompt_values,
    }


@dataclasses.dataclass
class ResidentAdaptiveMseEvaluator:
    """Evaluate the fixed 1024-row pack using the live sharded train params."""

    config: training_config.TrainConfig
    data_config: training_config.DataConfig
    model_def: nnx.GraphDef
    params_sharding: object
    replicated_sharding: jax.sharding.Sharding
    pack_path: Path
    output_dir: Path
    plan_dir: Path
    diversity_cache_dir: Path
    num_inference_steps: int = 10
    noise_seed: int = 0

    def __post_init__(self) -> None:
        self.pack_path = Path(self.pack_path).expanduser().resolve()
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.plan_dir = Path(self.plan_dir).expanduser().resolve()
        self.diversity_cache_dir = Path(self.diversity_cache_dir).expanduser().resolve()
        if self.num_inference_steps <= 0:
            raise ValueError("resident adaptive MSE inference steps must be positive")
        plan_start = self.data_config.diversity_sampling_mse_plan_start_step
        plan_interval = self.data_config.diversity_sampling_mse_plan_interval_batches
        if plan_start < 0 or plan_interval <= 0:
            raise ValueError("resident adaptive MSE evaluation requires a blockwise adaptive sampling schedule")
        if self.config.adaptive_mse_eval_interval_steps != plan_interval:
            raise ValueError(
                "resident adaptive MSE evaluation cadence must match the sampler plan interval: "
                f"eval={self.config.adaptive_mse_eval_interval_steps}, sampler={plan_interval}"
            )
        self._pack, self._arrays, self._prompts = _load_pack(self.pack_path)
        count = int(self._pack["samples"])
        if count != 1024:
            raise ValueError(f"resident adaptive MSE pack must contain 1024 samples, got {count}")
        required_shapes = {
            "image": (count, 256, 256, 3),
            "wrist_image": (count, 256, 256, 3),
            "memory_latents": (count, 1410, 64),
            "memory_mask": (count, 1410),
            "memory_segment_ids": (count, 1410),
            "actions": (count, 20, 8),
            "actions_is_pad": (count, 20),
        }
        for key, expected in required_shapes.items():
            if self._arrays[key].shape != expected:
                raise ValueError(f"adaptive MSE pack {key}: expected {expected}, got {self._arrays[key].shape}")
        model = self.config.model
        if model.discrete_state_input or model.memory_horizon != 1410 or model.memory_stride != 1:
            raise ValueError("resident adaptive MSE evaluator requires the no-state, gap-1 h1410 model")
        if model.memory_latent_dim != 64 or not model.memory_demo_anchor or not model.memory_execution_anchor:
            raise ValueError("resident adaptive MSE evaluator requires the 64-D dual-anchor model")
        if not np.allclose(
            self._pack["model_memory_latent_mean"], model.memory_latent_mean, rtol=0.0, atol=1e-12
        ) or not np.allclose(self._pack["model_memory_latent_std"], model.memory_latent_std, rtol=0.0, atol=1e-12):
            raise ValueError("evaluation pack memory normalization does not match the live training model")

        self._input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(None),
                *self.data_config.data_transforms.inputs,
                transforms.Normalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *self.data_config.model_transforms.outputs,
                transforms.Unnormalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
            ]
        )

        def sample(params, rng, observation, noise):
            model_instance = nnx.merge(self.model_def, params)
            model_instance.eval()
            return model_instance.sample_actions(
                rng, observation, num_steps=self.num_inference_steps, noise=noise
            )

        # Parameters retain their FSDP layout; a single validation query is
        # replicated as a cooperative eight-device inference, avoiding a full
        # second copy of the model or a per-query recompilation.
        self._sample = jax.jit(
            sample,
            in_shardings=(self.params_sharding, self.replicated_sharding, self.replicated_sharding, self.replicated_sharding),
            out_shardings=self.replicated_sharding,
        )
        logging.info(
            "Initialized resident adaptive-MSE evaluator: pack=%s samples=%d inference_steps=%d",
            self.pack_path,
            count,
            self.num_inference_steps,
        )

    def _replicate(self, value):
        def put(x):
            if x is None:
                return None
            return jax.device_put(jnp.asarray(x), self.replicated_sharding)

        return jax.tree.map(put, value, is_leaf=lambda x: x is None)

    def _write_plan(self, result_path: Path, step: int) -> Path:
        result = json.loads(result_path.read_text())
        deciles = result["rank_deciles"]
        normalized_mse = [float(deciles[label]["normalized_action_mse"]) for label in DECILE_LABELS]
        counts = [int(deciles[label]["queries"]) for label in DECILE_LABELS]
        if not all(np.isfinite(normalized_mse)) or any(value < 0.0 for value in normalized_mse):
            raise ValueError("resident adaptive MSE evaluation produced invalid decile metrics")
        diversity_manifest = self.diversity_cache_dir / "manifest.json"
        plan = {
            "schema_version": 1,
            "status": "complete",
            "rank_decile_definition": RANK_DECILE_DEFINITION,
            "diversity_cache_manifest_sha256": _sha256(diversity_manifest),
            "uniform_mass": float(self.data_config.diversity_sampling_uniform_mass),
            "sampling_start_step": int(step),
            "normalized_mse_by_decile": normalized_mse,
            "validation_queries_by_decile": counts,
            "source_evaluation": {
                "path": str(result_path),
                "sha256": _sha256(result_path),
                "checkpoint": str(self.config.checkpoint_dir / str(step)),
                "pack": str(self.pack_path),
                "num_inference_steps": self.num_inference_steps,
                "noise_seed": self.noise_seed,
                "parameter_source": "resident_train_state",
            },
            "sampling_formula": (
                "p_i = uniform_mass/N + (1-uniform_mass) * m_b/(n_b * sum_j m_j), "
                "where b is query i's global diversity-rank decile; if all m are zero, use 1/N"
            ),
        }
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        output = self.plan_dir / f"step{step:06d}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite resident adaptive MSE plan: {output}")
        temporary = output.parent / f".{output.name}.tmp-{os.getpid()}"
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
        return output

    def evaluate_and_publish(self, params, step: int) -> tuple[Path, Path]:
        plan_start = self.data_config.diversity_sampling_mse_plan_start_step
        interval = self.config.adaptive_mse_eval_interval_steps
        if step <= plan_start or (step - plan_start) % interval:
            raise ValueError(
                "resident adaptive MSE evaluation requires a completed adaptive-plan boundary: "
                f"got step={step}, start={plan_start}, interval={interval}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / f"step{step}_1024x10deciles_seed{self.noise_seed}_resident.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite resident adaptive MSE result: {result_path}")
        count = int(self._pack["samples"])
        all_predictions = np.empty_like(self._arrays["actions"], dtype=np.float32)
        noise_rng = np.random.default_rng(self.noise_seed)
        for row in range(count):
            rows = np.asarray([row], dtype=np.int64)
            raw = _batch_input(self._arrays, rows, self._prompts)
            item = {key: (value[0] if isinstance(value, np.ndarray) else value[0]) for key, value in raw.items()}
            batched = jax.tree.map(lambda x: np.asarray(x)[None, ...], self._input_transform(item))
            observation = model_lib.Observation.from_dict(batched)
            noise = noise_rng.standard_normal((1, 20, 32), dtype=np.float32)
            prediction = self._sample(
                params,
                self._replicate(jax.random.key(self.noise_seed + row)),
                self._replicate(observation),
                self._replicate(noise),
            )
            jax.block_until_ready(prediction)
            output = self._output_transform({"state": batched["state"], "actions": np.asarray(prediction)})
            predicted = np.asarray(output["actions"], dtype=np.float32)
            if predicted.shape != (1, 20, 8) or not np.all(np.isfinite(predicted)):
                raise RuntimeError(f"resident adaptive MSE bad policy output at row {row}: {predicted.shape}")
            all_predictions[row] = predicted[0]
            if (row + 1) % 32 == 0 or row + 1 == count:
                logging.info("Resident adaptive-MSE evaluation: completed=%d total=%d step=%d", row + 1, count, step)

        targets = np.asarray(self._arrays["actions"], dtype=np.float64)
        valid = ~np.asarray(self._arrays["actions_is_pad"], dtype=np.bool_)
        scale = np.asarray(self._pack["action_target"]["normalized_mse_scale"], dtype=np.float64)
        squared = (all_predictions.astype(np.float64) - targets) ** 2
        normalized_squared = ((all_predictions.astype(np.float64) - targets) / scale) ** 2
        per_query_raw = np.full(count, np.nan, dtype=np.float64)
        per_query_normalized = np.full(count, np.nan, dtype=np.float64)
        for row in range(count):
            positions = valid[row]
            per_query_raw[row] = squared[row, positions, :].mean()
            per_query_normalized[row] = normalized_squared[row, positions, :].mean()
        ranks = np.asarray(self._arrays["diversity_rank"], dtype=np.float64)
        rank_deciles = {}
        for decile, label in enumerate(DECILE_LABELS):
            lower = decile / 10.0
            upper = (decile + 1) / 10.0
            mask = (ranks >= lower) & (ranks <= upper) if decile == 9 else (ranks >= lower) & (ranks < upper)
            if not np.any(mask):
                raise ValueError(f"resident adaptive MSE pack has no query in decile {decile}")
            entry_mask = valid[mask]
            query_raw = per_query_raw[mask]
            query_normalized = per_query_normalized[mask]
            rank_deciles[label] = {
                "rank_lower_inclusive": lower,
                "rank_upper": upper,
                "rank_upper_inclusive": decile == 9,
                "queries": int(mask.sum()),
                "valid_action_positions": int(entry_mask.sum()),
                "valid_action_dimensions": int(entry_mask.sum() * 8),
                "rank_min": float(np.min(ranks[mask])),
                "rank_max": float(np.max(ranks[mask])),
                "raw_action_mse": float(squared[mask][entry_mask].mean()),
                "normalized_action_mse": float(normalized_squared[mask][entry_mask].mean()),
                "query_mean_raw_action_mse": float(query_raw.mean()),
                "query_mean_normalized_action_mse": float(query_normalized.mean()),
                "query_standard_error_raw_action_mse": float(query_raw.std(ddof=1) / np.sqrt(len(query_raw))),
                "query_standard_error_normalized_action_mse": float(
                    query_normalized.std(ddof=1) / np.sqrt(len(query_normalized))
                ),
            }
        result = {
            "schema_version": 1,
            "status": "complete",
            "checkpoint": str(self.config.checkpoint_dir / str(step)),
            "parameter_source": "resident_train_state",
            "checkpoint_step": int(step),
            "pack": str(self.pack_path),
            "pack_manifest_sha256": _sha256(self.pack_path / "manifest.json"),
            "num_inference_steps": self.num_inference_steps,
            "noise": "one deterministic standard-normal action noise tensor per query, generated in stable pack-row order",
            "noise_seed": self.noise_seed,
            "batch_size": 1,
            "metric": "mean squared error over non-padded physical action targets and 8 action dimensions",
            "normalized_metric": "same MSE after division by sidecar robust_action_scale",
            "rank_deciles": rank_deciles,
            "rank_decile_definition": RANK_DECILE_DEFINITION,
        }
        temporary = result_path.parent / f".{result_path.name}.tmp-{os.getpid()}"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, result_path)
        plan_path = self._write_plan(result_path, step)
        logging.info("Resident adaptive-MSE evaluation complete: step=%d result=%s plan=%s", step, result_path, plan_path)
        return result_path, plan_path
