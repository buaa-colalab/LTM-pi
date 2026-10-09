import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    initialize_memory_expert_from_action: bool = True
    # Ignore any expert-2 tensors stored in the source checkpoint and rebuild
    # the compatible attention/MLP tensors from expert 1. This is needed when
    # expert 2 is being repurposed (for example, from a DeltaTok history expert
    # to a future CD-LAM expert). Expert-2 norms retain the target model's
    # initialization by default because historical action and memory experts
    # used different norm parameterizations.
    force_initialize_memory_expert_from_action: bool = False
    # Copy regular or adaptive expert-2 norm tensors as well when they are
    # shape-compatible with expert 1. Keep this opt-in because historical
    # memory experts used regular RMSNorm while the action expert used
    # adaRMSNorm.
    initialize_memory_expert_norms_from_action: bool = False
    # Rebuild the future-flow input/output/time projections from their action
    # counterparts, ignoring any future-flow heads in the source checkpoint.
    # Incompatible or missing source tensors retain the target initialization.
    force_initialize_memory_flow_heads_from_action: bool = False
    skip_mismatched_shapes: bool = False

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        # Base checkpoints predate the optional CD-LAM stream. Keep all
        # initialized memory parameters (projector, normalization, and the
        # expert-id `_2` tensors) when those keys are absent from the source.
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r".*lora.*|.*memory.*|.*_2/.*|.*_2$",
            initialize_memory_expert_from_action=self.initialize_memory_expert_from_action,
            force_initialize_memory_expert_from_action=self.force_initialize_memory_expert_from_action,
            initialize_memory_expert_norms_from_action=self.initialize_memory_expert_norms_from_action,
            force_initialize_memory_flow_heads_from_action=self.force_initialize_memory_flow_heads_from_action,
            skip_mismatched_shapes=self.skip_mismatched_shapes,
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    initialize_memory_expert_from_action: bool = True,
    force_initialize_memory_expert_from_action: bool = False,
    initialize_memory_expert_norms_from_action: bool = False,
    force_initialize_memory_flow_heads_from_action: bool = False,
    skip_mismatched_shapes: bool = False,
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    pattern = re.compile(missing_regex)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            # A checkpoint may already contain expert 2 from an unrelated
            # experiment. In force mode, discard every expert-2 tensor here:
            # compatible tensors are reconstructed from expert 1 below. Norms
            # retain target initialization unless their opt-in is enabled.
            if force_initialize_memory_expert_from_action and _is_memory_expert_parameter(k):
                continue
            if force_initialize_memory_flow_heads_from_action and _memory_flow_head_source_key(k) is not None:
                continue
            if v.shape != flat_ref[k].shape:
                if not skip_mismatched_shapes:
                    result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
                    continue
                logger.info(
                    "Skipping checkpoint parameter %s with shape %s; initialized model expects %s",
                    k,
                    v.shape,
                    flat_ref[k].shape,
                )
                continue
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    # Then, merge any missing weights as defined by the missing regex.
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            # The optional JAX memory expert is deliberately named `_2` so
            # the historical action expert keeps its `_1` checkpoint keys.
            # Warm-start attention/MLP tensors from that action expert when a
            # base checkpoint does not contain the new stream. Norms remain
            # opt-in because historical streams used different parameterizations.
            source_key = k.replace("_2", "_1")
            is_memory_attention = any(
                f"/attn/{projection}_2/" in k
                for projection in ("qkv_einsum", "q_einsum", "kv_einsum", "attn_vec_einsum")
            )
            is_memory_norm = _is_memory_expert_norm_parameter(k)
            if (
                (initialize_memory_expert_from_action or force_initialize_memory_expert_from_action)
                and (
                    is_memory_attention
                    or "/mlp_2/" in k
                    or (initialize_memory_expert_norms_from_action and is_memory_norm)
                )
                and source_key in flat_loaded
            ):
                value = flat_loaded[source_key]
                if value.shape == flat_ref[k].shape:
                    result[k] = value.astype(flat_ref[k].dtype) if value.dtype != flat_ref[k].dtype else value
                    continue
            result[k] = flat_ref[k]

    # Force mode must not depend on ``missing_regex``: explicitly rebuild every
    # target expert-2 tensor. This also guarantees that a stale expert 2 in the
    # source checkpoint can never leak through. Shape-incompatible tensors keep
    # the target model's initialization.
    if force_initialize_memory_expert_from_action:
        for k, target_value in flat_ref.items():
            if not _is_memory_expert_parameter(k):
                continue
            source_key = k.replace("_2", "_1")
            can_copy = not _is_memory_expert_norm_parameter(k) or initialize_memory_expert_norms_from_action
            if can_copy:
                source_value = flat_loaded.get(source_key)
                if source_value is not None and source_value.shape == target_value.shape:
                    result[k] = (
                        source_value.astype(target_value.dtype)
                        if source_value.dtype != target_value.dtype
                        else source_value
                    )
                    continue
                logger.info(
                    "Keeping initialized expert-2 parameter %s; expert-1 source %s is missing or shape-incompatible",
                    k,
                    source_key,
                )
            result[k] = target_value

    # Future-flow heads intentionally share the action head initialization in
    # this mode. Map individual arrays (kernel and bias) so partially compatible
    # heads are handled safely instead of accepting a malformed subtree.
    if force_initialize_memory_flow_heads_from_action:
        for k, target_value in flat_ref.items():
            source_key = _memory_flow_head_source_key(k)
            if source_key is None:
                continue
            if source_key in flat_loaded and flat_loaded[source_key].shape == target_value.shape:
                source_value = flat_loaded[source_key]
                result[k] = (
                    source_value.astype(target_value.dtype)
                    if source_value.dtype != target_value.dtype
                    else source_value
                )
            else:
                logger.info(
                    "Keeping initialized future-flow parameter %s; action source %s is missing or shape-incompatible",
                    k,
                    source_key,
                )
                result[k] = target_value

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _is_memory_expert_parameter(key: str) -> bool:
    """Return whether ``key`` belongs to the optional expert-id 2."""
    return bool(
        re.search(
            r"(?:^|/)(?:qkv_einsum|q_einsum|kv_einsum|attn_vec_einsum|mlp|"
            r"pre_attention_norm|pre_ffw_norm|final_norm)_2(?:/|$)",
            key,
        )
    )


def _is_memory_expert_norm_parameter(key: str) -> bool:
    """Return whether ``key`` is an expert-2 norm tensor."""
    return bool(re.search(r"(?:^|/)(?:pre_attention_norm|pre_ffw_norm|final_norm)_2(?:/|$)", key))


def _memory_flow_head_source_key(key: str) -> str | None:
    """Map a future-flow head tensor to the corresponding action head tensor."""
    prefix_map = {
        "memory_flow_in_proj": "action_in_proj",
        "memory_flow_out_proj": "action_out_proj",
        "memory_flow_time_mlp_in": "time_mlp_in",
        "memory_flow_time_mlp_out": "time_mlp_out",
    }
    parts = key.split("/")
    for index, part in enumerate(parts):
        if source_part := prefix_map.get(part):
            parts[index] = source_part
            return "/".join(parts)
    return None
