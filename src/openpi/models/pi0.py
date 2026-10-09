import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def replace_valid_memory_with_fixed_random_latent(latents, mask, seed: int):
    """Replace every valid normalized LAM slot with one seeded shared vector."""
    fixed = jax.random.normal(
        jax.random.PRNGKey(seed),
        (latents.shape[-1],),
        dtype=latents.dtype,
    )
    return jnp.where(mask[..., None], fixed, jnp.zeros_like(latents))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # Keep a shared AR pattern batch-independent.  Broadcasting before cumsum
    # makes XLA constant-fold the same scan and NxN comparison B times whenever
    # ``mask_ar`` is a compile-time 1-D constant (the common prefix/suffix path).
    # Build that mask once as [N, N] and let the final logical_and broadcast it
    # over the dynamic [B, N, N] validity mask instead.
    if mask_ar.ndim == 1:
        cumsum = jnp.cumsum(mask_ar, axis=0)
        attn_mask = cumsum[None, :] <= cumsum[:, None]
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
        cumsum = jnp.cumsum(mask_ar, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_language_memory_vision_prefix(
    prefix_tokens: at.Float[at.Array, "b p emb"],
    prefix_mask: at.Bool[at.Array, "b p"],
    memory_tokens: at.Float[at.Array, "b m emb"],
    memory_mask: at.Bool[at.Array, "b m"],
    memory_ar_mask: at.Bool[at.Array, "*b m"],
    *,
    language_start: int,
    language_length: int,
    memory_anchor_length: int,
) -> tuple[
    at.Float[at.Array, "b n emb"],
    at.Bool[at.Array, "b n"],
    at.Bool[at.Array, " n"],
]:
    """Reorder expert-0 inputs as causal language, causal memory, then vision.

    ``prefix_tokens`` arrives in the historical layout
    ``[current vision, language, visual anchors]``. The trailing demo anchor is
    moved into the front of the causal memory group. Memory-owned anchors such
    as the execution-start image are already embedded inside ``memory_tokens``
    and remain in their temporal memory location. The first ordinary visual
    token starts a new full-attention block, preventing earlier memory queries
    from reading future visual tokens while allowing all visual queries to read
    language, memory, and one another.
    """
    prefix_length = prefix_tokens.shape[1]
    if language_length <= 0:
        raise ValueError("language_memory_vision requires a non-empty language token block")
    if language_start < 0 or language_start + language_length > prefix_length:
        raise ValueError(
            f"Invalid language slice [{language_start}, {language_start + language_length}) "
            f"for prefix length {prefix_length}"
        )
    if memory_anchor_length < 0 or language_start + language_length > prefix_length - memory_anchor_length:
        raise ValueError(
            f"Invalid trailing memory anchor length {memory_anchor_length} for prefix length {prefix_length}"
        )

    language_tokens = prefix_tokens[:, language_start : language_start + language_length]
    language_mask = prefix_mask[:, language_start : language_start + language_length]
    memory_anchor_start = prefix_length - memory_anchor_length
    memory_anchor_tokens = prefix_tokens[:, memory_anchor_start:]
    memory_anchor_mask = prefix_mask[:, memory_anchor_start:]
    visual_tokens = jnp.concatenate(
        [
            prefix_tokens[:, :language_start],
            prefix_tokens[:, language_start + language_length : memory_anchor_start],
        ],
        axis=1,
    )
    visual_mask = jnp.concatenate(
        [prefix_mask[:, :language_start], prefix_mask[:, language_start + language_length : memory_anchor_start]],
        axis=1,
    )
    if visual_tokens.shape[1] <= 0:
        raise ValueError("language_memory_vision requires a non-empty visual token block")

    memory_group_tokens = jnp.concatenate([memory_anchor_tokens, memory_tokens], axis=1)
    memory_group_mask = jnp.concatenate([memory_anchor_mask, memory_mask], axis=1)
    tokens = jnp.concatenate([language_tokens, memory_group_tokens, visual_tokens], axis=1)
    input_mask = jnp.concatenate([language_mask, memory_group_mask, visual_mask], axis=1)
    batch_size = prefix_tokens.shape[0]
    memory_ar_mask = jnp.broadcast_to(memory_ar_mask, memory_mask.shape)
    demo_anchor_ar_mask = jnp.zeros((memory_anchor_length,), dtype=jnp.bool_)
    if memory_anchor_length:
        demo_anchor_ar_mask = demo_anchor_ar_mask.at[0].set(True)
    # Every language and LAM token starts its own block. Each anchor starts one
    # block and is bidirectional internally. Vision starts one final block and
    # the remaining visual tokens share it bidirectionally.
    ar_mask = jnp.concatenate(
        [
            jnp.ones((batch_size, language_length), dtype=jnp.bool_),
            jnp.broadcast_to(demo_anchor_ar_mask, (batch_size, memory_anchor_length)),
            memory_ar_mask,
            jnp.broadcast_to(
                jnp.zeros((visual_tokens.shape[1],), dtype=jnp.bool_).at[0].set(True),
                (batch_size, visual_tokens.shape[1]),
            ),
        ],
        axis=1,
    )
    return tokens, input_mask, ar_mask


def make_memory_register_attn_mask(
    public_prefix_mask: at.Bool[at.Array, "b p"],
    private_history_mask: at.Bool[at.Array, "b h"],
    register_mask: at.Bool[at.Array, "b r"],
    action_mask: at.Bool[at.Array, "b a"],
    action_ar_mask: at.Bool[at.Array, " a"],
) -> at.Bool[at.Array, "b t t"]:
    """Build the asymmetric register-bottleneck mask for ``[P, H, R, A]``.

    ``P`` is the public current-observation prefix, ``H`` is private history
    (demo anchor followed by memory), ``R`` is the learned register bank, and
    ``A`` is the action suffix.  Action queries cannot read history directly;
    registers are the only path from history to actions.
    """

    def attend(query_mask, key_mask):
        return query_mask[:, :, None] & key_mask[:, None, :]

    batch_size = public_prefix_mask.shape[0]
    def zero(rows, cols):
        return jnp.zeros((batch_size, rows, cols), dtype=jnp.bool_)

    p, h, r, a = (
        public_prefix_mask.shape[1],
        private_history_mask.shape[1],
        register_mask.shape[1],
        action_mask.shape[1],
    )

    prefix_rows = jnp.concatenate(
        [attend(public_prefix_mask, public_prefix_mask), zero(p, h), zero(p, r), zero(p, a)], axis=-1
    )
    history_rows = jnp.concatenate(
        [
            attend(private_history_mask, public_prefix_mask),
            attend(private_history_mask, private_history_mask),
            zero(h, r),
            zero(h, a),
        ],
        axis=-1,
    )
    register_rows = jnp.concatenate(
        [
            attend(register_mask, public_prefix_mask),
            attend(register_mask, private_history_mask),
            attend(register_mask, register_mask),
            zero(r, a),
        ],
        axis=-1,
    )
    action_rows = jnp.concatenate(
        [
            attend(action_mask, public_prefix_mask),
            zero(a, h),
            attend(action_mask, register_mask),
            make_attn_mask(action_mask, action_ar_mask),
        ],
        axis=-1,
    )
    return jnp.concatenate([prefix_rows, history_rows, register_rows, action_rows], axis=1)


def make_parallel_future_attn_mask(
    history_mask: at.Bool[at.Array, "b h"],
    history_ar_mask: at.Bool[at.Array, " h"],
    future_mask: at.Bool[at.Array, "b f"],
    action_mask: at.Bool[at.Array, "b a"],
    action_ar_mask: at.Bool[at.Array, " a"],
) -> at.Bool[at.Array, "b t t"]:
    """Build a forked mask for independent action and future predictions.

    Physical token order is ``[history, future, action]``. Both prediction
    branches can read history and themselves, but neither can read the other.
    This makes future prediction a training-only auxiliary objective without
    changing the action function.
    """
    batch_size, history_length = history_mask.shape
    future_length = future_mask.shape[1]
    action_length = action_mask.shape[1]

    history_attn = make_attn_mask(history_mask, history_ar_mask)
    action_attn = make_attn_mask(action_mask, action_ar_mask)
    history_to_future = jnp.zeros((batch_size, history_length, future_length), dtype=jnp.bool_)
    history_to_action = jnp.zeros((batch_size, history_length, action_length), dtype=jnp.bool_)
    future_to_history = future_mask[:, :, None] & history_mask[:, None, :]
    future_to_future = future_mask[:, :, None] & future_mask[:, None, :]
    future_to_action = jnp.zeros((batch_size, future_length, action_length), dtype=jnp.bool_)
    action_to_history = action_mask[:, :, None] & history_mask[:, None, :]
    action_to_future = jnp.zeros((batch_size, action_length, future_length), dtype=jnp.bool_)

    history_rows = jnp.concatenate([history_attn, history_to_future, history_to_action], axis=-1)
    future_rows = jnp.concatenate([future_to_history, future_to_future, future_to_action], axis=-1)
    action_rows = jnp.concatenate([action_to_history, action_to_future, action_attn], axis=-1)
    return jnp.concatenate([history_rows, future_rows, action_rows], axis=1)


def make_parallel_direction_attn_mask(
    history_mask: at.Bool[at.Array, "b h"],
    history_ar_mask: at.Bool[at.Array, "*b h"],
    demo_context_mask: at.Bool[at.Array, "b h"],
    direction_mask: at.Bool[at.Array, "b d"],
    action_mask: at.Bool[at.Array, "b a"],
    action_ar_mask: at.Bool[at.Array, " a"],
    *,
    condition_action_on_direction: bool = False,
) -> at.Bool[at.Array, "b t t"]:
    """Build direction/action attention with optional hierarchical conditioning.

    The direction branch can read only language plus demo anchor/LAM context;
    the action branch always reads the complete history. In the legacy sibling
    mode the two branches cannot read one another. In hierarchical mode action
    tokens additionally read every valid teacher-forced direction token, which
    matches the pi0.5 ``subtask text -> action`` training factorization.
    """
    batch_size, history_length = history_mask.shape
    direction_length = direction_mask.shape[1]
    action_length = action_mask.shape[1]
    if demo_context_mask.shape != history_mask.shape:
        raise ValueError(
            f"demo_context_mask must match history_mask {history_mask.shape}, got {demo_context_mask.shape}"
        )
    history_attn = make_attn_mask(history_mask, history_ar_mask)
    direction_attn = make_attn_mask(
        direction_mask,
        jnp.ones((direction_length,), dtype=jnp.bool_),
    )
    action_attn = make_attn_mask(action_mask, action_ar_mask)
    history_to_direction = jnp.zeros((batch_size, history_length, direction_length), dtype=jnp.bool_)
    history_to_action = jnp.zeros((batch_size, history_length, action_length), dtype=jnp.bool_)
    direction_to_history = direction_mask[:, :, None] & demo_context_mask[:, None, :]
    direction_to_action = jnp.zeros((batch_size, direction_length, action_length), dtype=jnp.bool_)
    action_to_history = action_mask[:, :, None] & history_mask[:, None, :]
    action_to_direction = (
        action_mask[:, :, None] & direction_mask[:, None, :]
        if condition_action_on_direction
        else jnp.zeros((batch_size, action_length, direction_length), dtype=jnp.bool_)
    )
    history_rows = jnp.concatenate([history_attn, history_to_direction, history_to_action], axis=-1)
    direction_rows = jnp.concatenate(
        [direction_to_history, direction_attn, direction_to_action], axis=-1
    )
    action_rows = jnp.concatenate([action_to_history, action_to_direction, action_attn], axis=-1)
    return jnp.concatenate([history_rows, direction_rows, action_rows], axis=1)


def make_parallel_future_positions(
    history_mask: at.Bool[at.Array, "b h"], future_length: int, action_length: int
) -> at.Int[at.Array, "b t"]:
    """Assign sibling future/action positions without shifting action RoPE."""
    history_positions = jnp.cumsum(history_mask, axis=1) - 1
    branch_start = jnp.sum(history_mask, axis=1, keepdims=True)
    future_positions = branch_start + jnp.arange(future_length, dtype=jnp.int32)[None, :]
    action_positions = branch_start + jnp.arange(action_length, dtype=jnp.int32)[None, :]
    return jnp.concatenate([history_positions, future_positions, action_positions], axis=1)


def make_direction_action_positions(
    history_mask: at.Bool[at.Array, "b h"],
    direction_mask: at.Bool[at.Array, "b d"],
    action_length: int,
    *,
    condition_action_on_direction: bool,
) -> at.Int[at.Array, "b t"]:
    """Assign RoPE positions for sibling or direction-conditioned actions.

    Valid direction tokens occupy positions immediately after each example's
    valid history. In hierarchical mode the action stream starts after the
    final valid direction token; padding never shifts the action positions.
    """
    history_positions = jnp.cumsum(history_mask, axis=1) - 1
    history_length = jnp.sum(history_mask, axis=1, keepdims=True)
    direction_positions = history_length + jnp.cumsum(direction_mask, axis=1) - 1
    action_start = history_length
    if condition_action_on_direction:
        action_start = action_start + jnp.sum(direction_mask, axis=1, keepdims=True)
    action_positions = action_start + jnp.arange(action_length, dtype=jnp.int32)[None, :]
    return jnp.concatenate([history_positions, direction_positions, action_positions], axis=1)


def insert_tokens_at_per_example_index(
    tokens: at.Float[at.Array, "b m emb"],
    mask: at.Bool[at.Array, "b m"],
    inserted_tokens: at.Float[at.Array, "b a emb"],
    inserted_mask: at.Bool[at.Array, "b a"],
    insertion_index: at.Int[at.Array, " b"],
) -> tuple[at.Float[at.Array, "b n emb"], at.Bool[at.Array, "b n"]]:
    """Insert a fixed-width token block at a different index per example.

    Shapes remain static for JIT: every row grows by ``a`` slots, while masks
    decide whether those slots contain a visible anchor.  This is used to put
    the execution-start image after demo memory even though left padding and
    demo lengths differ across a batch.
    """
    batch_size, memory_length, width = tokens.shape
    anchor_length = inserted_tokens.shape[1]
    if inserted_tokens.shape[0] != batch_size or inserted_tokens.shape[2] != width:
        raise ValueError(f"Inserted token shape {inserted_tokens.shape} is incompatible with {tokens.shape}")
    if mask.shape != (batch_size, memory_length):
        raise ValueError(f"Token mask shape {mask.shape} is incompatible with {tokens.shape}")
    if inserted_mask.shape != (batch_size, anchor_length):
        raise ValueError(f"Inserted mask shape {inserted_mask.shape} is incompatible with {inserted_tokens.shape}")
    if insertion_index.shape != (batch_size,):
        raise ValueError(f"insertion_index must have shape {(batch_size,)}, got {insertion_index.shape}")

    output_positions = jnp.arange(memory_length + anchor_length, dtype=jnp.int32)[None, :]
    insertion_index = jnp.asarray(insertion_index, dtype=jnp.int32)[:, None]
    is_inserted = (output_positions >= insertion_index) & (output_positions < insertion_index + anchor_length)
    source_positions = jnp.where(
        output_positions < insertion_index,
        output_positions,
        output_positions - anchor_length,
    )
    source_positions = jnp.clip(source_positions, 0, memory_length - 1)
    source_tokens = jnp.take_along_axis(tokens, source_positions[..., None], axis=1)
    source_mask = jnp.take_along_axis(mask, source_positions, axis=1)
    anchor_positions = jnp.clip(output_positions - insertion_index, 0, anchor_length - 1)
    anchor_tokens = jnp.take_along_axis(inserted_tokens, anchor_positions[..., None], axis=1)
    anchor_mask = jnp.take_along_axis(inserted_mask, anchor_positions, axis=1)
    return (
        jnp.where(is_inserted[..., None], anchor_tokens, source_tokens),
        jnp.where(is_inserted, anchor_mask, source_mask),
    )


def build_recent_memory_observation(
    observation: _model.Observation,
    recent_steps: int,
    *,
    exclude_demo: bool = False,
) -> _model.Observation:
    """Keep the final ``recent_steps`` slots of a left-padded memory sequence.

    Memory batches in this repository are left padded, so a fixed tail slice
    contains the last K temporal transitions for every sample while retaining
    a static shape. When requested, demo and reset-boundary entries inside that
    tail are masked out; the visual demo anchor is handled separately in the
    prefix. Prompt, current observations, and future targets are untouched.
    """
    if observation.memory_latents is None or observation.memory_mask is None:
        raise ValueError("recent-memory scoring requires memory_latents and memory_mask")
    if recent_steps <= 0:
        raise ValueError("recent_steps must be positive")

    memory_length = observation.memory_latents.shape[1]
    start = max(memory_length - recent_steps, 0)
    recent_mask = jnp.asarray(observation.memory_mask[:, start:], dtype=jnp.bool_)
    recent_latents = jnp.asarray(observation.memory_latents[:, start:])
    left_padding = recent_steps - recent_latents.shape[1]
    if left_padding:
        recent_mask = jnp.pad(recent_mask, ((0, 0), (left_padding, 0)), constant_values=False)
        recent_latents = jnp.pad(recent_latents, ((0, 0), (left_padding, 0), (0, 0)))
    recent_latents = jnp.where(recent_mask[..., None], recent_latents, 0)
    recent_segment_ids = observation.memory_segment_ids
    if recent_segment_ids is not None:
        recent_segment_ids = jnp.asarray(recent_segment_ids[:, start:], dtype=jnp.int32)
        if left_padding:
            recent_segment_ids = jnp.pad(recent_segment_ids, ((0, 0), (left_padding, 0)), constant_values=0)
        if exclude_demo:
            # Segment 3 is execution history. Segment 1 is demonstration and
            # segment 2 is its reset boundary; neither belongs in the recent
            # counterfactual requested here.
            recent_mask = recent_mask & (recent_segment_ids == 3)
        recent_segment_ids = jnp.where(recent_mask, recent_segment_ids, 0)
    elif exclude_demo:
        raise ValueError("exclude_demo requires memory_segment_ids")
    recent_latents = jnp.where(recent_mask[..., None], recent_latents, 0)
    return observation.replace(
        memory_latents=recent_latents,
        memory_mask=recent_mask,
        memory_segment_ids=recent_segment_ids,
    )


def per_action_flow_error(
    prediction: at.Float[at.Array, "b h d"],
    target: at.Float[at.Array, "b h d"],
    score_action_dims: int,
) -> at.Float[at.Array, "b h"]:
    """Return per-position flow error over the real control dimensions."""
    prediction = jnp.asarray(prediction[..., :score_action_dims], dtype=jnp.float32)
    target = jnp.asarray(target[..., :score_action_dims], dtype=jnp.float32)
    return jnp.mean(jnp.square(prediction - target), axis=-1)


def action_motion_weights(
    action_motion_delta: at.Float[at.Array, "b h joint"],
    action_gripper_flip: at.Bool[at.Array, "b h"],
    *,
    joint_scales: tuple[float, ...],
    min_weight: float,
    max_weight: float,
    eps: float = 1e-6,
) -> tuple[
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
]:
    """Return detached per-position weights from raw adjacent GT actions.

    A joint reaches score 1 at its execution-split p95 absolute transition.
    Taking the maximum preserves a decisive move in any joint. Gripper flips
    receive score 1 independently because their binary scale is not
    comparable to joint radians. The final mean-one normalization keeps the
    optimizer's average loss/LR scale unchanged.
    """
    motion_delta = jax.lax.stop_gradient(jnp.asarray(action_motion_delta, dtype=jnp.float32))
    gripper_flip = jax.lax.stop_gradient(jnp.asarray(action_gripper_flip, dtype=jnp.bool_))
    scales = jnp.asarray(joint_scales, dtype=jnp.float32)
    if motion_delta.shape[-1] != scales.shape[0]:
        raise ValueError(f"action_motion_delta has {motion_delta.shape[-1]} joint dims, expected {scales.shape[0]}")
    if motion_delta.shape[:-1] != gripper_flip.shape:
        raise ValueError(f"action motion/gripper shapes must agree, got {motion_delta.shape} and {gripper_flip.shape}")

    joint_score = jnp.max(jnp.abs(motion_delta) / jnp.maximum(scales, eps), axis=-1)
    motion_score = jnp.maximum(jnp.clip(joint_score, 0.0, 1.0), gripper_flip.astype(jnp.float32))
    raw_weight = min_weight + (max_weight - min_weight) * motion_score
    weight = raw_weight / jnp.maximum(jnp.mean(raw_weight), eps)
    return tuple(jax.lax.stop_gradient(value) for value in (weight, motion_score, raw_weight))


def baseline_uncertainty_weights(
    score: at.Float[at.Array, "b h"],
    *,
    min_weight: float,
    max_weight: float,
    rank_power: float,
) -> at.Float[at.Array, "b h"]:
    """Map detached global uncertainty ranks to fixed per-position weights."""
    score = jax.lax.stop_gradient(jnp.asarray(score, dtype=jnp.float32))
    score = jnp.clip(score, 0.0, 1.0)
    weight = min_weight + (max_weight - min_weight) * jnp.power(score, rank_power)
    return jax.lax.stop_gradient(weight)


def flow_error_hardness_score(
    full_error: at.Float[at.Array, "b h"],
    flow_time: at.Float[at.Array, "b"],
    *,
    threshold: float = 0.5,
    temperature: float = 0.5,
    time_bins: int = 4,
    eps: float = 1e-6,
) -> at.Float[at.Array, "b h"]:
    """Return a detached focal-style score for hard full-branch positions.

    Flow-matching error changes systematically with the sampled flow timestep.
    Standardizing log error inside coarse timestep buckets prevents the score
    from merely selecting one noisy part of the flow trajectory.
    """
    full_error = jax.lax.stop_gradient(jnp.asarray(full_error, dtype=jnp.float32))
    error_time = jnp.broadcast_to(jnp.asarray(flow_time, dtype=jnp.float32)[..., None], full_error.shape)
    time_bin = jnp.minimum(
        jnp.floor(jnp.clip(error_time, 0.0, 1.0) * time_bins).astype(jnp.int32),
        time_bins - 1,
    )
    flat_log_error = jnp.log(full_error + eps).reshape(-1)
    flat_time_bin = time_bin.reshape(-1)
    membership = jax.nn.one_hot(flat_time_bin, time_bins, dtype=jnp.float32)
    bin_count = jnp.maximum(jnp.sum(membership, axis=0), 1.0)
    bin_mean = jnp.sum(membership * flat_log_error[:, None], axis=0) / bin_count
    centered = flat_log_error - bin_mean[flat_time_bin]
    bin_variance = jnp.sum(membership * jnp.square(centered[:, None]), axis=0) / bin_count
    position_variance = bin_variance[flat_time_bin]
    standardized_error = centered / jnp.sqrt(position_variance + eps)
    score = jax.nn.sigmoid((standardized_error.reshape(full_error.shape) - threshold) / temperature)
    score = jnp.where(position_variance.reshape(full_error.shape) > eps, score, 0.0)
    return jax.lax.stop_gradient(score)


def flow_error_hardness_weights(
    full_error: at.Float[at.Array, "b h"],
    flow_time: at.Float[at.Array, "b"],
    *,
    min_weight: float,
    max_weight: float,
    threshold: float = 0.5,
    temperature: float = 0.5,
    time_bins: int = 4,
    eps: float = 1e-6,
) -> tuple[
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
]:
    """Convert detached full-branch hardness into mean-one loss weights."""
    hardness_score = flow_error_hardness_score(
        full_error,
        flow_time,
        threshold=threshold,
        temperature=temperature,
        time_bins=time_bins,
        eps=eps,
    )
    raw_weight = min_weight + (max_weight - min_weight) * hardness_score
    weight = raw_weight / jnp.maximum(jnp.mean(raw_weight), eps)
    return tuple(jax.lax.stop_gradient(value) for value in (weight, hardness_score, raw_weight))


def memory_advantage_weights(
    full_error: at.Float[at.Array, "b h"],
    recent_error: at.Float[at.Array, "b h"],
    *,
    flow_time: at.Array | None = None,
    alpha: at.Float[at.Array, ""] | float,
    min_weight: float,
    max_weight: float,
    threshold: float,
    temperature: float,
    hardness_weight: float = 0.0,
    hardness_threshold: float = 0.5,
    hardness_temperature: float = 0.5,
    hardness_time_bins: int = 4,
    eps: float = 1e-6,
) -> tuple[
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
    at.Float[at.Array, "b h"],
]:
    """Create detached, globally normalized per-action memory weights.

    ``jnp.mean`` acts on the logical global array under this training code's
    NamedSharding/GSPMD setup, so it reduces across both the global batch and
    action-chunk axes without a pmap axis name.
    """
    full_error = jax.lax.stop_gradient(jnp.asarray(full_error, dtype=jnp.float32))
    recent_error = jax.lax.stop_gradient(jnp.asarray(recent_error, dtype=jnp.float32))
    signed_advantage = (recent_error - full_error) / (recent_error + eps)
    # Keep the bounded positive-only value for the existing diagnostics, but
    # let the sigmoid see the signed counterfactual result.  This separates a
    # genuinely harmful full-memory branch from the neutral A=0 boundary.
    advantage = jnp.clip(signed_advantage, 0.0, 1.0)
    memory_score = jax.nn.sigmoid((signed_advantage - threshold) / temperature)
    if hardness_weight > 0.0:
        if flow_time is None:
            raise ValueError("flow_time is required when memory advantage hardness weighting is enabled")
        hardness_score = flow_error_hardness_score(
            full_error,
            flow_time,
            threshold=hardness_threshold,
            temperature=hardness_temperature,
            time_bins=hardness_time_bins,
            eps=eps,
        )
        # Smooth OR: preserve the raw score of memory-sensitive positions,
        # while a hard-only position can contribute at most hardness_weight.
        score = 1.0 - (1.0 - memory_score) * (1.0 - hardness_weight * hardness_score)
    else:
        hardness_score = jnp.zeros_like(memory_score)
        score = memory_score
    raw_weight = min_weight + (max_weight - min_weight) * score
    normalized_weight = raw_weight / jnp.mean(raw_weight)
    alpha = jnp.clip(jnp.asarray(alpha, dtype=jnp.float32), 0.0, 1.0)
    weight = 1.0 + alpha * (normalized_weight - 1.0)
    return tuple(
        jax.lax.stop_gradient(value) for value in (weight, advantage, raw_weight, memory_score, hardness_score, score)
    )


def make_full_memory_positions_for_recent_action(
    prefix_mask: at.Bool[at.Array, "b p"],
    recent_memory_mask: at.Bool[at.Array, "b m"],
    full_memory_mask: at.Bool[at.Array, "b fm"],
    action_mask: at.Bool[at.Array, "b a"],
    *,
    full_prefix_mask: at.Bool[at.Array, "b fp"] | None = None,
) -> at.Int[at.Array, "b t"]:
    """Position a short recent branch exactly where it lived in full history."""
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    full_prefix_length = jnp.sum(
        prefix_mask if full_prefix_mask is None else full_prefix_mask,
        axis=1,
        keepdims=True,
    )
    full_memory_length = jnp.sum(full_memory_mask, axis=1, keepdims=True)
    recent_memory_length = jnp.sum(recent_memory_mask, axis=1, keepdims=True)
    dropped_memory_length = full_memory_length - recent_memory_length
    memory_positions = full_prefix_length + dropped_memory_length + jnp.cumsum(recent_memory_mask, axis=1) - 1
    action_positions = full_prefix_length + full_memory_length + jnp.cumsum(action_mask, axis=1) - 1
    return jnp.concatenate([prefix_positions, memory_positions, action_positions], axis=1)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        auxiliary_config = None
        if config.use_memory and (not config.memory_use_vlm_expert or config.memory_future_use_separate_expert):
            auxiliary_config = _gemma.Config(
                width=action_expert_config.width,
                depth=action_expert_config.depth,
                mlp_dim=action_expert_config.mlp_dim,
                num_heads=action_expert_config.num_heads,
                num_kv_heads=action_expert_config.num_kv_heads,
                head_dim=action_expert_config.head_dim,
            )
        llm_configs = (
            [paligemma_config, auxiliary_config, action_expert_config]
            if auxiliary_config
            else [
                paligemma_config,
                action_expert_config,
            ]
        )
        expert_ids = (0, 2, 1) if auxiliary_config else ()
        attention_expert_ids = (0, 1, 1) if auxiliary_config and config.share_memory_attention else expert_ids
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
                expert_ids=expert_ids,
                attention_expert_ids=attention_expert_ids,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=(
                [False, config.memory_future_use_adarms_time_conditioning, True]
                if config.pi05
                else [False, False, False]
            )
            if auxiliary_config
            else ([False, True] if config.pi05 else [False, False]),
        )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.config = config
        self.use_memory = config.use_memory
        self.use_lam_memory = config.use_lam_memory
        self.use_memory_expert = config.use_lam_memory and not config.memory_use_vlm_expert
        self.use_future_expert = config.memory_future_use_separate_expert
        if self.use_memory:
            memory_width = paligemma_config.width if config.memory_use_vlm_expert else action_expert_config.width
            if config.memory_segment_embedding and (
                not config.memory_use_vlm_expert or config.memory_segment_embedding_after_projection
            ):
                self.memory_segment_embed = nnx.Embed(config.memory_segment_vocab_size, memory_width, rngs=rngs)
        if self.use_lam_memory:
            projector_input_width = (
                config.memory_latent_dim
                if config.memory_pooling_mode == "mean"
                else config.memory_latent_dim * config.memory_frames_per_token
            )
            if (
                config.memory_use_vlm_expert
                and config.memory_segment_embedding
                and not config.memory_segment_embedding_after_projection
            ):
                # Keep the type of every concatenated frame slot. This makes a
                # grouped token unambiguous even when it contains a demo,
                # reset-boundary, and execution transition.
                projector_input_width += 4 * config.memory_frames_per_token
            self.memory_latent_mean = nnx.Variable(
                jnp.asarray(
                    config.memory_latent_mean
                    if config.memory_latent_mean is not None
                    else (0.0,) * config.memory_latent_dim,
                    dtype=jnp.float32,
                )
            )
            self.memory_latent_std = nnx.Variable(
                jnp.asarray(
                    config.memory_latent_std
                    if config.memory_latent_std is not None
                    else (1.0,) * config.memory_latent_dim,
                    dtype=jnp.float32,
                )
            )
            self.memory_projector_in = nnx.Linear(
                projector_input_width,
                config.memory_projector_hidden_dim,
                rngs=rngs,
            )
            self.memory_projector_out = nnx.Linear(config.memory_projector_hidden_dim, memory_width, rngs=rngs)
            self.memory_projector_norm = nnx.LayerNorm(memory_width, rngs=rngs)
            if config.memory_register_count:
                self.memory_register_embed = nnx.Embed(config.memory_register_count, memory_width, rngs=rngs)
            if config.memory_flow_horizon:
                future_width = action_expert_config.width if self.use_future_expert else memory_width
                future_latent_dim = config.memory_future_latent_dim
                self.memory_future_latent_mean = nnx.Variable(
                    jnp.asarray(
                        config.memory_future_latent_mean
                        if config.memory_future_latent_mean is not None
                        else config.memory_latent_mean
                        if future_latent_dim == config.memory_latent_dim and config.memory_latent_mean is not None
                        else (0.0,) * future_latent_dim,
                        dtype=jnp.float32,
                    )
                )
                self.memory_future_latent_std = nnx.Variable(
                    jnp.asarray(
                        config.memory_future_latent_std
                        if config.memory_future_latent_std is not None
                        else config.memory_latent_std
                        if future_latent_dim == config.memory_latent_dim and config.memory_latent_std is not None
                        else (1.0,) * future_latent_dim,
                        dtype=jnp.float32,
                    )
                )
                if config.memory_future_use_learned_queries:
                    self.memory_flow_queries = nnx.Embed(config.memory_flow_horizon, future_width, rngs=rngs)
                if config.memory_future_prediction_mode == "flow_matching":
                    self.memory_flow_in_proj = nnx.Linear(future_latent_dim, future_width, rngs=rngs)
                    self.memory_flow_time_mlp_in = nnx.Linear(future_width, future_width, rngs=rngs)
                    self.memory_flow_time_mlp_out = nnx.Linear(future_width, future_width, rngs=rngs)
                self.memory_flow_out_proj = nnx.Linear(future_width, future_latent_dim, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask, _, _, _ = self._embed_prefix_with_anchor_length(obs)
        return tokens, input_mask, ar_mask

    def _embed_prefix_with_anchor_length(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        int,
        int,
        int,
    ]:
        """Embed the prefix and expose anchor and language slice metadata."""
        input_mask = []
        ar_mask = []
        tokens = []
        demo_anchor_length = 0
        language_start = 0
        language_length = 0
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            language_start = sum(token_block.shape[1] for token_block in tokens)
            language_length = tokenized_inputs.shape[1]
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        # RoboDojo uses the raw first head-camera frame as an absolute episode
        # anchor. It is a normal expert-0 prefix image: there is no separate
        # expert and no attention boundary before or after it.
        if self.config.memory_episode_anchor:
            if obs.memory_episode_start_image is None or obs.memory_episode_start_mask is None:
                raise ValueError(
                    "memory_episode_anchor requires memory_episode_start_image and memory_episode_start_mask"
                )
            episode_anchor_tokens, _ = self.PaliGemma.img(obs.memory_episode_start_image, train=False)
            episode_anchor_tokens = jnp.asarray(episode_anchor_tokens, dtype=tokens[0].dtype)
            tokens.append(episode_anchor_tokens)
            input_mask.append(
                einops.repeat(
                    jnp.asarray(obs.memory_episode_start_mask, dtype=jnp.bool_),
                    "b -> b s",
                    s=episode_anchor_tokens.shape[1],
                )
            )
            ar_mask += [False] * episode_anchor_tokens.shape[1]
        # The demonstration anchor is kept in the VLM prefix, directly before
        # the memory stream, so every memory/action token can attend to it.
        if self.config.memory_demo_anchor:
            if obs.memory_demo_start_mask is None:
                raise ValueError("memory_demo_anchor requires memory_demo_start_mask")
            if self.config.memory_demo_anchor_feature_pool:
                if obs.memory_demo_anchor_features is None:
                    raise ValueError("feature-pooled demo anchor requires memory_demo_anchor_features")
                anchor_tokens = jnp.asarray(obs.memory_demo_anchor_features, dtype=tokens[0].dtype)
            else:
                if obs.memory_demo_start_image is None:
                    raise ValueError("image demo anchor requires memory_demo_start_image")
                anchor_tokens, _ = self.PaliGemma.img(obs.memory_demo_start_image, train=False)
            if self.config.memory_execution_anchor:
                demo_type_ids = jnp.full(
                    anchor_tokens.shape[:2], self.config.memory_demo_anchor_type_id, dtype=jnp.int32
                )
                demo_type_embedding = jnp.asarray(self.memory_segment_embed(demo_type_ids), dtype=anchor_tokens.dtype)
                anchor_tokens = anchor_tokens + demo_type_embedding
            tokens.append(anchor_tokens)
            input_mask.append(einops.repeat(obs.memory_demo_start_mask, "b -> b s", s=anchor_tokens.shape[1]))
            ar_mask += [False] * anchor_tokens.shape[1]
            demo_anchor_length = anchor_tokens.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, demo_anchor_length, language_start, language_length

    @at.typecheck
    def embed_memory(
        self,
        obs: _model.Observation,
        *,
        train: bool = False,
        dropout_rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "b m emb"], at.Bool[at.Array, "b m"], at.Bool[at.Array, "*m"]]:
        """Project and normalize LAM memory tokens for their configured MoT stream."""
        if not self.use_memory:
            raise RuntimeError("embed_memory called while memory is disabled")
        if self.config.memory_anchor_only:
            if obs.memory_latents is not None or obs.memory_mask is not None or obs.memory_segment_ids is not None:
                raise ValueError("memory_anchor_only forbids latent memory tensors")
            if obs.memory_execution_start_image is None or obs.memory_execution_start_mask is None:
                raise ValueError(
                    "memory_anchor_only requires memory_execution_start_image and memory_execution_start_mask"
                )
            execution_anchor_tokens, _ = self.PaliGemma.img(obs.memory_execution_start_image, train=False)
            execution_type_ids = jnp.full(
                execution_anchor_tokens.shape[:2], self.config.memory_execution_anchor_type_id, dtype=jnp.int32
            )
            execution_anchor_tokens = jnp.asarray(execution_anchor_tokens, dtype=self.config.dtype)
            execution_anchor_tokens = execution_anchor_tokens + jnp.asarray(
                self.memory_segment_embed(execution_type_ids), dtype=execution_anchor_tokens.dtype
            )
            execution_anchor_mask = einops.repeat(
                jnp.asarray(obs.memory_execution_start_mask, dtype=jnp.bool_),
                "b -> b s",
                s=execution_anchor_tokens.shape[1],
            )
            execution_anchor_tokens = jnp.where(
                execution_anchor_mask[..., None], execution_anchor_tokens, jnp.zeros_like(execution_anchor_tokens)
            )
            execution_anchor_ar_mask = jnp.zeros(
                (execution_anchor_tokens.shape[1],), dtype=jnp.bool_
            ).at[0].set(True)
            return execution_anchor_tokens, execution_anchor_mask, execution_anchor_ar_mask
        if obs.memory_latents is None or obs.memory_mask is None:
            raise ValueError(
                "Memory-enabled PI0 requires memory_latents and memory_mask. "
                "Pass an explicit all-false mask for a no-memory ablation."
            )
        latents = obs.memory_latents
        mask = jnp.asarray(obs.memory_mask, dtype=jnp.bool_)
        if (
            latents.ndim != 3
            or latents.shape[0] != obs.state.shape[0]
            or latents.shape[2] != self.config.memory_latent_dim
        ):
            raise ValueError(
                f"Expected memory_latents shape [B, T, {self.config.memory_latent_dim}] with "
                f"B={obs.state.shape[0]}, got {latents.shape}"
            )
        if mask.shape != latents.shape[:2]:
            raise ValueError(f"Expected memory_mask shape {latents.shape[:2]}, got {mask.shape}")
        segment_ids = obs.memory_segment_ids
        if self.config.memory_segment_embedding:
            if segment_ids is None or segment_ids.shape != mask.shape:
                actual = None if segment_ids is None else segment_ids.shape
                raise ValueError(f"Expected memory_segment_ids shape {mask.shape}, got {actual}")
            segment_ids = jnp.asarray(segment_ids, dtype=jnp.int32)
            if not isinstance(segment_ids, jax.core.Tracer):
                if bool(jnp.any((segment_ids < 0) | (segment_ids > 3))):
                    raise ValueError("memory_segment_ids must be in [0, 3]")
        if latents.shape[1] < 1 or latents.shape[1] > self.config.memory_horizon:
            raise ValueError(f"Memory length must be in [1, {self.config.memory_horizon}], got {latents.shape[1]}")
        # Normalize valid frames independently, then group consecutive frames
        # in temporal order. Left padding keeps the newest transition aligned
        # to the final slot of the final token. Mask after arithmetic so
        # NaN/Inf sentinels and normalization offsets in padded slots cannot
        # contaminate a partially valid group.
        latents = jnp.asarray(latents, jnp.float32)
        latents = (latents - self.memory_latent_mean.value) / self.memory_latent_std.value
        if self.config.memory_fixed_random_latent_seed is not None:
            latents = replace_valid_memory_with_fixed_random_latent(
                latents,
                mask,
                self.config.memory_fixed_random_latent_seed,
            )
        else:
            latents = jnp.where(mask[..., None], latents, 0.0)
        frames_per_token = self.config.memory_frames_per_token
        left_padding = (-latents.shape[1]) % frames_per_token
        if left_padding:
            latents = jnp.pad(latents, ((0, 0), (left_padding, 0), (0, 0)))
            mask = jnp.pad(mask, ((0, 0), (left_padding, 0)), constant_values=False)
            if segment_ids is not None:
                segment_ids = jnp.pad(
                    segment_ids,
                    ((0, 0), (left_padding, 0)),
                    constant_values=0,
                )
        batch_size, padded_length, latent_dim = latents.shape
        phase_safe_mean_pooling = (
            self.config.memory_pooling_mode == "mean" and frames_per_token > 1 and segment_ids is not None
        )
        if phase_safe_mean_pooling:
            # Never merge demo, demo->execution boundary, and execution
            # latents. Each contiguous phase is chunked from its own start;
            # the final chunk in a phase may therefore contain 1--N-1 frames.
            # At most three non-padding phases exist, so ceil(T/N)+2 is a
            # static upper bound. Right-aligning each example retains the
            # repository's left-padded memory convention.
            token_count = (padded_length + frames_per_token - 1) // frames_per_token + 2
            frame_segment_one_hot = jax.nn.one_hot(segment_ids, 4, dtype=jnp.int32)
            frame_segment_one_hot = frame_segment_one_hot * mask[..., None]
            segment_frame_counts = frame_segment_one_hot.sum(axis=1)
            segment_token_counts = (segment_frame_counts + frames_per_token - 1) // frames_per_token
            segment_token_counts = segment_token_counts.at[:, 0].set(0)
            segment_token_offsets = jnp.cumsum(segment_token_counts, axis=1) - segment_token_counts
            segment_running_counts = jnp.cumsum(frame_segment_one_hot, axis=1)
            frame_index_in_segment = jnp.sum(
                (segment_running_counts - 1) * frame_segment_one_hot,
                axis=-1,
            )
            frame_segment_offsets = jnp.sum(
                segment_token_offsets[:, None, :] * frame_segment_one_hot,
                axis=-1,
            )
            total_token_count = segment_token_counts.sum(axis=1)
            token_index = (
                token_count
                - total_token_count[:, None]
                + frame_segment_offsets
                + frame_index_in_segment // frames_per_token
            )
            token_index = jnp.where(mask, token_index, 0).astype(jnp.int32)
            batch_index = jnp.arange(batch_size, dtype=jnp.int32)[:, None]

            grouped_latent_sum = jnp.zeros((batch_size, token_count, latent_dim), dtype=latents.dtype)
            grouped_latent_sum = grouped_latent_sum.at[batch_index, token_index].add(
                jnp.where(mask[..., None], latents, 0)
            )
            grouped_frame_count = jnp.zeros((batch_size, token_count), dtype=jnp.int32)
            grouped_frame_count = grouped_frame_count.at[batch_index, token_index].add(mask.astype(jnp.int32))
            grouped_mask = grouped_frame_count > 0
            grouped_latents = grouped_latent_sum / jnp.maximum(grouped_frame_count[..., None], 1)
            grouped_segment_ids = jnp.zeros((batch_size, token_count), dtype=jnp.int32)
            grouped_segment_ids = grouped_segment_ids.at[batch_index, token_index].max(jnp.where(mask, segment_ids, 0))
            grouped_frame_mask = None
        else:
            token_count = padded_length // frames_per_token
            grouped_frame_mask = mask.reshape(batch_size, token_count, frames_per_token)
            grouped_mask = grouped_frame_mask.any(axis=2)
            grouped_frame_latents = latents.reshape(batch_size, token_count, frames_per_token, latent_dim)
            if self.config.memory_pooling_mode == "mean":
                # Padded slots are already zero. Divide by the number of valid
                # frames so a partial 1--3 frame group has the same scale as a
                # complete group of four.
                valid_count = jnp.maximum(grouped_frame_mask.sum(axis=2, keepdims=True), 1)
                grouped_latents = grouped_frame_latents.sum(axis=2) / valid_count
            else:
                grouped_latents = grouped_frame_latents.reshape(batch_size, token_count, frames_per_token * latent_dim)
            grouped_segment_ids = (
                segment_ids.reshape(batch_size, token_count, frames_per_token) if segment_ids is not None else None
            )

        if (
            self.config.memory_segment_embedding
            and self.config.memory_use_vlm_expert
            and not self.config.memory_segment_embedding_after_projection
        ):
            # Keep the projector as the only memory-specific learned module.
            # Each frame slot contributes a separate one-hot projector input,
            # preserving typed boundaries inside a concatenated token.
            grouped_segment_one_hot = jax.nn.one_hot(grouped_segment_ids, 4, dtype=grouped_latents.dtype).reshape(
                batch_size, token_count, frames_per_token * 4
            )
            grouped_latents = jnp.concatenate([grouped_latents, grouped_segment_one_hot], axis=-1)

        tokens = self.memory_projector_in(grouped_latents)
        tokens = nnx.gelu(tokens)
        tokens = self.memory_projector_out(tokens)
        if self.config.memory_segment_embedding and (
            not self.config.memory_use_vlm_expert or self.config.memory_segment_embedding_after_projection
        ):
            grouped_segment_embeddings = self.memory_segment_embed(grouped_segment_ids)
            if phase_safe_mean_pooling:
                # Phase-aware grouping gives every token one exact type.
                pass
            elif self.config.memory_pooling_mode == "mean":
                # The only non-phase-aware typed mean case is one frame per
                # token, but keep the masked reduction explicit.
                valid_count = jnp.maximum(grouped_frame_mask.sum(axis=2, keepdims=True), 1)
                grouped_segment_embeddings = (
                    jnp.where(grouped_frame_mask[..., None], grouped_segment_embeddings, 0).sum(axis=2) / valid_count
                )
            else:
                # Grouped concatenate mode is rejected by the config, so this
                # path is exactly one typed frame per token.
                grouped_segment_embeddings = grouped_segment_embeddings[..., 0, :]
            tokens = tokens + grouped_segment_embeddings
        tokens = self.memory_projector_norm(tokens)
        if train and self.config.memory_token_dropout_rate:
            if dropout_rng is None:
                raise ValueError("dropout_rng is required when training with memory token dropout")
            keep_mask = jax.random.bernoulli(
                dropout_rng,
                1.0 - self.config.memory_token_dropout_rate,
                grouped_mask.shape,
            )
            grouped_mask = grouped_mask & keep_mask
        tokens = jnp.where(grouped_mask[..., None], tokens, 0.0)
        if self.config.memory_demo_lam_boundary_token:
            if obs.memory_demo_start_mask is None:
                raise ValueError("memory_demo_lam_boundary_token requires memory_demo_start_mask")
            demo_boundary_type_ids = jnp.full(
                (batch_size, 1), self.config.memory_demo_lam_boundary_type_id, dtype=jnp.int32
            )
            demo_boundary_tokens = jnp.asarray(
                self.memory_projector_norm(self.memory_segment_embed(demo_boundary_type_ids)), dtype=tokens.dtype
            )
            demo_boundary_mask = jnp.asarray(obs.memory_demo_start_mask, dtype=jnp.bool_)[:, None]
            demo_boundary_tokens = jnp.where(
                demo_boundary_mask[..., None], demo_boundary_tokens, jnp.zeros_like(demo_boundary_tokens)
            )
            # Put the separator after left padding and immediately before the
            # first valid demo LAM token. Invalid padding remains invisible.
            demo_boundary_insertion_index = jnp.sum(grouped_segment_ids[..., 0] == 0, axis=1)
            tokens, grouped_mask = insert_tokens_at_per_example_index(
                tokens,
                grouped_mask,
                demo_boundary_tokens,
                demo_boundary_mask,
                demo_boundary_insertion_index,
            )
        if self.config.memory_execution_anchor:
            if obs.memory_execution_start_image is None or obs.memory_execution_start_mask is None:
                raise ValueError(
                    "memory_execution_anchor requires memory_execution_start_image and memory_execution_start_mask"
                )
            execution_anchor_tokens, _ = self.PaliGemma.img(obs.memory_execution_start_image, train=False)
            execution_anchor_tokens = jnp.asarray(execution_anchor_tokens, dtype=tokens.dtype)
            execution_type_ids = jnp.full(
                execution_anchor_tokens.shape[:2], self.config.memory_execution_anchor_type_id, dtype=jnp.int32
            )
            execution_type_embedding = jnp.asarray(
                self.memory_segment_embed(execution_type_ids), dtype=execution_anchor_tokens.dtype
            )
            execution_anchor_tokens = execution_anchor_tokens + execution_type_embedding
            execution_anchor_mask = einops.repeat(
                jnp.asarray(obs.memory_execution_start_mask, dtype=jnp.bool_),
                "b -> b s",
                s=execution_anchor_tokens.shape[1],
            )
            execution_anchor_tokens = jnp.where(execution_anchor_mask[..., None], execution_anchor_tokens, 0.0)
            # Left-padded memory is ordered [padding, demo, boundary,
            # execution].  Insert after padding/demo so the physical and RoPE
            # order is exactly demo anchor -> optional separator -> demo
            # memory -> execution anchor -> boundary/execution memory for
            # every example in the batch.
            insertion_index = jnp.sum(
                (grouped_segment_ids[..., 0] == 0) | (grouped_segment_ids[..., 0] == 1),
                axis=1,
            )
            if self.config.memory_demo_lam_boundary_token:
                insertion_index = insertion_index + 1
            tokens, grouped_mask = insert_tokens_at_per_example_index(
                tokens,
                grouped_mask,
                execution_anchor_tokens,
                execution_anchor_mask,
                insertion_index,
            )
            token_count = tokens.shape[1]
        if self.config.memory_causal_attention:
            # Every LAM token is causal. The execution anchor occupies a
            # per-example insertion range: its first patch starts one new
            # block and the remaining patches share that block, giving the
            # anchor full internal attention without allowing preceding demo
            # LAM tokens to see it.
            ar_mask = jnp.ones((batch_size, token_count), dtype=jnp.bool_)
            if self.config.memory_execution_anchor:
                output_positions = jnp.arange(token_count, dtype=jnp.int32)[None, :]
                in_execution_anchor = (output_positions >= insertion_index[:, None]) & (
                    output_positions < insertion_index[:, None] + execution_anchor_tokens.shape[1]
                )
                execution_anchor_start = output_positions == insertion_index[:, None]
                ar_mask = jnp.where(in_execution_anchor, execution_anchor_start, ar_mask)
            return tokens, grouped_mask, ar_mask
        # The RoboDojo history configuration extends the original expert-0
        # prefix block, so current images, language, episode anchor, and every
        # memory token are mutually visible. Other memory experiments retain
        # the historical separate-block behavior.
        ar_mask = jnp.zeros((token_count,), dtype=jnp.bool_)
        if not self.config.memory_same_prefix_block:
            ar_mask = ar_mask.at[0].set(True)
        return tokens, grouped_mask, ar_mask

    def make_demo_memory_context_mask(
        self,
        obs: _model.Observation,
        embedded_memory_length: int,
    ) -> at.Bool[at.Array, "b m"]:
        """Align a demo-only key mask with boundary/anchor-expanded memory."""
        if obs.memory_mask is None or obs.memory_segment_ids is None:
            raise ValueError("Demo direction generation requires memory_mask and memory_segment_ids")
        if self.config.memory_frames_per_token != 1:
            raise ValueError("Demo context alignment requires one memory frame per token")
        segment_ids = jnp.asarray(obs.memory_segment_ids, dtype=jnp.int32)
        source_mask = jnp.asarray(obs.memory_mask, dtype=jnp.bool_)
        demo_mask = source_mask & (segment_ids == 1)
        batch_size, source_length = demo_mask.shape
        dummy = jnp.zeros((batch_size, source_length, 1), dtype=jnp.float32)
        if self.config.memory_demo_lam_boundary_token:
            insertion_index = jnp.sum(segment_ids == 0, axis=1)
            dummy, demo_mask = insert_tokens_at_per_example_index(
                dummy,
                demo_mask,
                jnp.zeros((batch_size, 1, 1), dtype=dummy.dtype),
                jnp.zeros((batch_size, 1), dtype=jnp.bool_),
                insertion_index,
            )
        if self.config.memory_execution_anchor:
            anchor_length = embedded_memory_length - dummy.shape[1]
            if anchor_length <= 0:
                raise ValueError(
                    "Execution-anchor memory length must exceed source memory length after the demo boundary"
                )
            insertion_index = jnp.sum((segment_ids == 0) | (segment_ids == 1), axis=1)
            if self.config.memory_demo_lam_boundary_token:
                insertion_index = insertion_index + 1
            dummy, demo_mask = insert_tokens_at_per_example_index(
                dummy,
                demo_mask,
                jnp.zeros((batch_size, anchor_length, 1), dtype=dummy.dtype),
                jnp.zeros((batch_size, anchor_length), dtype=jnp.bool_),
                insertion_index,
            )
        if demo_mask.shape[1] != embedded_memory_length:
            raise ValueError(
                f"Aligned demo memory mask length {demo_mask.shape[1]} != embedded memory length "
                f"{embedded_memory_length}"
            )
        return demo_mask

    def embed_memory_registers(
        self, batch_size: int, dtype: jnp.dtype
    ) -> tuple[at.Float[at.Array, "b r emb"], at.Bool[at.Array, "b r"]]:
        """Broadcast the learned register bank across a batch."""
        if not self.config.memory_register_count:
            raise RuntimeError("embed_memory_registers called while memory registers are disabled")
        register_ids = jnp.arange(self.config.memory_register_count, dtype=jnp.int32)
        tokens = jnp.asarray(self.memory_register_embed(register_ids), dtype=dtype)
        tokens = jnp.broadcast_to(tokens[None, :, :], (batch_size, *tokens.shape))
        mask = jnp.ones((batch_size, self.config.memory_register_count), dtype=jnp.bool_)
        return tokens, mask

    @at.typecheck
    def embed_memory_flow(
        self,
        noisy_latents: at.Float[at.Array, "b fm d"],
        timestep: at.Float[at.Array, " b"],
        target_mask: at.Bool[at.Array, "b fm"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b fm emb"],
        at.Bool[at.Array, "b fm"],
        at.Bool[at.Array, " fm"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed noisy future latents and return an optional expert AdaRMS condition."""
        horizon = self.config.memory_flow_horizon
        if not horizon:
            raise RuntimeError("embed_memory_flow called while memory flow prediction is disabled")
        expected_shape = (timestep.shape[0], horizon, self.config.memory_future_latent_dim)
        if noisy_latents.shape != expected_shape:
            raise ValueError(f"Expected noisy future memory shape {expected_shape}, got {noisy_latents.shape}")

        tokens = self.memory_flow_in_proj(noisy_latents)
        if self.config.memory_future_use_learned_queries:
            queries = self.memory_flow_queries(jnp.arange(horizon, dtype=jnp.int32))
            tokens = tokens + queries[None, :, :]
        time_emb = posemb_sincos(
            timestep,
            self.memory_flow_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_emb = self.memory_flow_time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.memory_flow_time_mlp_out(time_emb)
        if self.config.memory_future_use_adarms_time_conditioning:
            # This is intentionally identical to pi0.5's action time path.
            # Expert 2 receives the time embedding through each AdaRMSNorm;
            # it is not added to the future input tokens.
            adarms_cond = nnx.swish(time_emb)
        else:
            tokens = tokens + time_emb[:, None, :]
            adarms_cond = None

        # All n future tokens form one bidirectional block. The block starts
        # after causal history, so history cannot attend to future tokens.
        mask = (
            jnp.ones((timestep.shape[0], horizon), dtype=jnp.bool_)
            if target_mask is None
            else jnp.asarray(target_mask, dtype=jnp.bool_)
        )
        if mask.shape != expected_shape[:2]:
            raise ValueError(f"Expected future memory mask shape {expected_shape[:2]}, got {mask.shape}")
        tokens = jnp.where(mask[..., None], tokens, 0.0)
        ar_mask = jnp.zeros((horizon,), dtype=jnp.bool_).at[0].set(True)
        return tokens, mask, ar_mask, adarms_cond

    @at.typecheck
    def embed_memory_direct_queries(
        self,
        batch_size: int,
        target_mask: at.Bool[at.Array, "b fm"] | None = None,
    ) -> tuple[at.Float[at.Array, "b fm emb"], at.Bool[at.Array, "b fm"], at.Bool[at.Array, " fm"]]:
        """Create target-free learned queries for direct future-latent prediction."""
        horizon = self.config.memory_flow_horizon
        if not horizon or self.config.memory_future_prediction_mode != "direct":
            raise RuntimeError("direct future-memory prediction is disabled")
        queries = self.memory_flow_queries(jnp.arange(horizon, dtype=jnp.int32))
        tokens = jnp.broadcast_to(queries[None, :, :], (batch_size, *queries.shape))
        # The future queries jointly read the complete causal history. Whether
        # another prediction branch may read them is decided by the caller's
        # attention mask; training-only auxiliary futures use a sibling mask
        # that isolates them from action tokens in both directions.
        mask = (
            jnp.ones((batch_size, horizon), dtype=jnp.bool_)
            if target_mask is None
            else jnp.asarray(target_mask, dtype=jnp.bool_)
        )
        if mask.shape != (batch_size, horizon):
            raise ValueError(f"Expected future memory mask shape {(batch_size, horizon)}, got {mask.shape}")
        tokens = jnp.where(mask[..., None], tokens, 0.0)
        ar_mask = jnp.zeros((horizon,), dtype=jnp.bool_).at[0].set(True)
        return tokens, mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _predict_action_only(
        self,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
        memory_tokens: at.Float[at.Array, "b m emb"],
        memory_mask: at.Bool[at.Array, "b m"],
        memory_ar_mask: at.Bool[at.Array, " m"],
        suffix_tokens: at.Float[at.Array, "b s emb"],
        suffix_mask: at.Bool[at.Array, "b s"],
        suffix_ar_mask: at.Bool[at.Array, " s"],
        adarms_cond: at.Float[at.Array, "b emb"] | None,
        *,
        full_memory_mask_for_positions: at.Bool[at.Array, "b fm"] | None = None,
        full_prefix_mask_for_positions: at.Bool[at.Array, "b fp"] | None = None,
    ) -> at.Float[at.Array, "b h d"]:
        """Run an action-only branch, optionally preserving full-history RoPE.

        The counterfactual recent branch is physically short for efficiency,
        but its retained memory and action tokens keep the same position IDs
        they had with full memory. Thus its prediction difference measures
        removed content rather than an unrelated position shift. Training-only
        future tokens/expert are intentionally absent from this scorer.
        """
        input_mask = jnp.concatenate([prefix_mask, memory_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, memory_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        if full_memory_mask_for_positions is None:
            positions = jnp.cumsum(input_mask, axis=1) - 1
        else:
            positions = make_full_memory_positions_for_recent_action(
                prefix_mask,
                memory_mask,
                full_memory_mask_for_positions,
                suffix_mask,
                full_prefix_mask=full_prefix_mask_for_positions,
            )

        if self.use_memory_expert:
            stream_tokens = [prefix_tokens, memory_tokens, suffix_tokens]
            stream_conds = [None, None, adarms_cond]
        elif self.use_future_expert:
            stream_tokens = [jnp.concatenate([prefix_tokens, memory_tokens], axis=1), None, suffix_tokens]
            stream_conds = [None, None, adarms_cond]
        else:
            stream_tokens = [jnp.concatenate([prefix_tokens, memory_tokens], axis=1), suffix_tokens]
            stream_conds = [None, adarms_cond]
        outputs, _ = self.PaliGemma.llm(
            stream_tokens,
            mask=attn_mask,
            positions=positions,
            adarms_cond=stream_conds,
        )
        return self.action_out_proj(outputs[-1][:, -self.action_horizon :])

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        return_components: bool = False,
        return_advantage_values: bool = False,
        memory_advantage_alpha: at.Float[at.Array, ""] | float = 1.0,
        hardness_alpha: at.Float[at.Array, ""] | float = 1.0,
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        (
            preprocess_rng,
            noise_rng,
            memory_noise_rng,
            time_rng,
            memory_time_rng,
            memory_dropout_rng,
            recent_memory_dropout_rng,
        ) = jax.random.split(rng, 7)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, image_keys=self.config.active_image_keys
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        (
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            demo_anchor_length,
            language_start,
            language_length,
        ) = self._embed_prefix_with_anchor_length(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        direction_tokens = None
        if self.use_memory:
            memory_tokens, memory_mask, memory_ar_mask = self.embed_memory(
                observation,
                train=train,
                dropout_rng=memory_dropout_rng,
            )
            if self.config.memory_register_count:
                register_tokens, register_mask = self.embed_memory_registers(
                    actions.shape[0], prefix_tokens.dtype
                )
            # Keep the history-only mask for the recent-memory scorer. Some
            # legacy joint-prediction modes append future tokens below.
            full_history_memory_mask = memory_mask
            memory_prediction_target = None
            memory_prediction_target_mask = None
            memory_flow_velocity = None
            if self.config.memory_flow_horizon:
                if observation.future_memory_latents is None or observation.future_memory_mask is None:
                    raise ValueError("future-memory prediction requires future_memory_latents and future_memory_mask")
                expected_shape = (
                    actions.shape[0],
                    self.config.memory_flow_horizon,
                    self.config.memory_future_latent_dim,
                )
                if observation.future_memory_latents.shape != expected_shape:
                    raise ValueError(
                        f"Expected future_memory_latents shape {expected_shape}, "
                        f"got {observation.future_memory_latents.shape}"
                    )
                if observation.future_memory_mask.shape != expected_shape[:2]:
                    raise ValueError(
                        f"Expected future_memory_mask shape {expected_shape[:2]}, "
                        f"got {observation.future_memory_mask.shape}"
                    )
                memory_prediction_target_mask = jnp.asarray(observation.future_memory_mask, dtype=jnp.bool_)
                memory_prediction_target = jnp.asarray(observation.future_memory_latents, dtype=jnp.float32)
                memory_prediction_target = (
                    memory_prediction_target - self.memory_future_latent_mean.value
                ) / self.memory_future_latent_std.value
                memory_prediction_target = jnp.where(
                    memory_prediction_target_mask[..., None], memory_prediction_target, 0.0
                )
                if self.config.memory_future_prediction_mode == "flow_matching":
                    memory_time = jax.random.beta(memory_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
                    memory_time_expanded = memory_time[..., None, None]
                    memory_noise = jax.random.normal(memory_noise_rng, expected_shape)
                    memory_x_t = (
                        memory_time_expanded * memory_noise + (1 - memory_time_expanded) * memory_prediction_target
                    )
                    memory_flow_velocity = memory_noise - memory_prediction_target
                    future_tokens, future_mask, future_ar_mask, future_adarms_cond = self.embed_memory_flow(
                        memory_x_t,
                        memory_time,
                        memory_prediction_target_mask,
                    )
                else:
                    future_tokens, future_mask, future_ar_mask = self.embed_memory_direct_queries(
                        actions.shape[0], memory_prediction_target_mask
                    )
                    future_adarms_cond = None
                if not self.config.memory_future_training_only:
                    memory_tokens = jnp.concatenate([memory_tokens, future_tokens], axis=1)
                    memory_mask = jnp.concatenate([memory_mask, future_mask], axis=1)
                    memory_ar_mask = jnp.concatenate([memory_ar_mask, future_ar_mask], axis=0)
            if self.config.memory_future_training_only:
                # Treat future prediction and action prediction as sibling
                # branches over the same history. Future tokens exist only in
                # this training forward, and the explicit mask prevents either
                # prediction branch from reading the other.
                history_tokens = jnp.concatenate([prefix_tokens, memory_tokens], axis=1)
                history_mask = jnp.concatenate([prefix_mask, memory_mask], axis=1)
                history_ar_mask = jnp.concatenate([prefix_ar_mask, memory_ar_mask], axis=0)
                if self.use_future_expert:
                    stream_tokens = [history_tokens, future_tokens, suffix_tokens]
                    stream_conds = [None, future_adarms_cond, adarms_cond]
                else:
                    stream_tokens = [jnp.concatenate([history_tokens, future_tokens], axis=1), suffix_tokens]
                    stream_conds = [None, adarms_cond]
                attn_mask = make_parallel_future_attn_mask(
                    history_mask,
                    history_ar_mask,
                    future_mask,
                    suffix_mask,
                    suffix_ar_mask,
                )
                positions = make_parallel_future_positions(
                    history_mask,
                    future_tokens.shape[1],
                    suffix_tokens.shape[1],
                )
            elif self.config.memory_use_vlm_expert:
                if self.config.memory_prefix_order == "language_memory_vision":
                    ordered_tokens, ordered_mask, ordered_ar_mask = make_language_memory_vision_prefix(
                        prefix_tokens,
                        prefix_mask,
                        memory_tokens,
                        memory_mask,
                        memory_ar_mask,
                        language_start=language_start,
                        language_length=language_length,
                        memory_anchor_length=demo_anchor_length,
                    )
                    if self.config.memory_demo_direction_generation:
                        if (
                            observation.demo_direction_tokens is None
                            or observation.demo_direction_mask is None
                            or observation.demo_direction_loss_mask is None
                        ):
                            raise ValueError(
                                "Demo direction generation requires tokens, mask, and loss mask during training"
                            )
                        expected_direction_shape = (
                            actions.shape[0],
                            self.config.memory_demo_direction_max_token_len,
                        )
                        if observation.demo_direction_tokens.shape != expected_direction_shape:
                            raise ValueError(
                                f"Expected demo_direction_tokens {expected_direction_shape}, "
                                f"got {observation.demo_direction_tokens.shape}"
                            )
                        if observation.demo_direction_mask.shape != expected_direction_shape:
                            raise ValueError("demo_direction_mask shape must match demo_direction_tokens")
                        if observation.demo_direction_loss_mask.shape != expected_direction_shape:
                            raise ValueError("demo_direction_loss_mask shape must match demo_direction_tokens")
                        direction_tokens = self.PaliGemma.llm(
                            observation.demo_direction_tokens,
                            method="embed",
                        )
                        direction_mask = jnp.asarray(observation.demo_direction_mask, dtype=jnp.bool_)
                        embedded_demo_memory_mask = self.make_demo_memory_context_mask(
                            observation,
                            memory_tokens.shape[1],
                        )
                        language_mask = prefix_mask[:, language_start : language_start + language_length]
                        demo_anchor_mask = prefix_mask[:, -demo_anchor_length:]
                        visual_length = (
                            ordered_tokens.shape[1]
                            - language_length
                            - demo_anchor_length
                            - memory_tokens.shape[1]
                        )
                        demo_context_mask = jnp.concatenate(
                            [
                                language_mask,
                                demo_anchor_mask,
                                embedded_demo_memory_mask,
                                jnp.zeros((actions.shape[0], visual_length), dtype=jnp.bool_),
                            ],
                            axis=1,
                        )
                        input_mask = jnp.concatenate(
                            [ordered_mask, direction_mask, suffix_mask], axis=1
                        )
                        attn_mask = make_parallel_direction_attn_mask(
                            ordered_mask,
                            ordered_ar_mask,
                            demo_context_mask,
                            direction_mask,
                            suffix_mask,
                            suffix_ar_mask,
                            condition_action_on_direction=(
                                self.config.memory_demo_direction_condition_action
                            ),
                        )
                        positions = make_direction_action_positions(
                            ordered_mask,
                            direction_mask,
                            suffix_tokens.shape[1],
                            condition_action_on_direction=(
                                self.config.memory_demo_direction_condition_action
                            ),
                        )
                        stream_tokens = [
                            jnp.concatenate([ordered_tokens, direction_tokens], axis=1),
                            suffix_tokens,
                        ]
                    else:
                        input_mask = jnp.concatenate([ordered_mask, suffix_mask], axis=1)
                        # Per-example execution-anchor positions make the ordered
                        # prefix mask batch-shaped. Broadcast the shared action
                        # suffix pattern before concatenating along sequence.
                        ar_mask = jnp.concatenate(
                            [ordered_ar_mask, jnp.broadcast_to(suffix_ar_mask, suffix_mask.shape)], axis=1
                        )
                        stream_tokens = [ordered_tokens, suffix_tokens]
                elif self.config.memory_register_count:
                    # Physical order is [public prefix, demo anchor, memory,
                    # registers, actions].  Anchor + memory are private
                    # history: actions can consume them only through the
                    # learned registers.
                    public_prefix_tokens = prefix_tokens[:, :-demo_anchor_length]
                    private_anchor_tokens = prefix_tokens[:, -demo_anchor_length:]
                    public_prefix_mask = prefix_mask[:, :-demo_anchor_length]
                    private_anchor_mask = prefix_mask[:, -demo_anchor_length:]
                    private_history_tokens = jnp.concatenate([private_anchor_tokens, memory_tokens], axis=1)
                    private_history_mask = jnp.concatenate([private_anchor_mask, memory_mask], axis=1)
                    input_mask = jnp.concatenate(
                        [public_prefix_mask, private_history_mask, register_mask, suffix_mask], axis=1
                    )
                    stream_tokens = [
                        jnp.concatenate(
                            [public_prefix_tokens, private_history_tokens, register_tokens], axis=1
                        ),
                        suffix_tokens,
                    ]
                    attn_mask = make_memory_register_attn_mask(
                        public_prefix_mask,
                        private_history_mask,
                        register_mask,
                        suffix_mask,
                        suffix_ar_mask,
                    )
                    positions = jnp.cumsum(input_mask, axis=1) - 1
                else:
                    # Expert-0 memory is appended after the ordinary prefix.
                    # Its configured AR mask decides whether it extends that
                    # full-attention block or begins a separate block.
                    input_mask = jnp.concatenate([prefix_mask, memory_mask, suffix_mask], axis=1)
                    ar_mask = jnp.concatenate([prefix_ar_mask, memory_ar_mask, suffix_ar_mask], axis=0)
                    stream_tokens = [jnp.concatenate([prefix_tokens, memory_tokens], axis=1), suffix_tokens]
                stream_conds = [None, adarms_cond]
            else:
                input_mask = jnp.concatenate([prefix_mask, memory_mask, suffix_mask], axis=1)
                ar_mask = jnp.concatenate([prefix_ar_mask, memory_ar_mask, suffix_ar_mask], axis=0)
                stream_tokens = [prefix_tokens, memory_tokens, suffix_tokens]
                stream_conds = [None, None, adarms_cond]
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            stream_tokens = [prefix_tokens, suffix_tokens]
            stream_conds = [None, adarms_cond]
        if (
            not (self.use_memory and self.config.memory_future_training_only)
            and not self.config.memory_register_count
            and not self.config.memory_demo_direction_generation
        ):
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
        outputs, _ = self.PaliGemma.llm(stream_tokens, mask=attn_mask, positions=positions, adarms_cond=stream_conds)
        suffix_out = outputs[-1]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        weighted_action_loss = action_loss
        recent_action_loss = None
        loss_components = {"loss/action_flow": jnp.mean(action_loss)}
        if self.config.baseline_uncertainty_weighting:
            if observation.baseline_uncertainty_score is None:
                raise ValueError(
                    "baseline_uncertainty_weighting requires baseline_uncertainty_score metadata"
                )
            expected_score_shape = (actions.shape[0], self.action_horizon)
            if observation.baseline_uncertainty_score.shape != expected_score_shape:
                raise ValueError(
                    f"Expected baseline_uncertainty_score shape {expected_score_shape}, "
                    f"got {observation.baseline_uncertainty_score.shape}"
                )
            weight = baseline_uncertainty_weights(
                observation.baseline_uncertainty_score,
                min_weight=self.config.baseline_uncertainty_min_weight,
                max_weight=self.config.baseline_uncertainty_max_weight,
                rank_power=self.config.baseline_uncertainty_rank_power,
            )
            weighted_action_loss = weight * action_loss
            loss_components.update(
                {
                    "unweighted_loss": jnp.mean(action_loss),
                    "weighted_loss": jnp.mean(weighted_action_loss),
                    "weighted_loss_delta": jnp.mean(weighted_action_loss) - jnp.mean(action_loss),
                    "baseline_uncertainty_score_mean": jnp.mean(observation.baseline_uncertainty_score),
                    "baseline_uncertainty_score_p90": jnp.quantile(observation.baseline_uncertainty_score, 0.9),
                    "baseline_uncertainty_score_max": jnp.max(observation.baseline_uncertainty_score),
                    "baseline_uncertainty_weight_min": jnp.min(weight),
                    "baseline_uncertainty_weight_mean": jnp.mean(weight),
                    "baseline_uncertainty_weight_p90": jnp.quantile(weight, 0.9),
                    "baseline_uncertainty_weight_max": jnp.max(weight),
                }
            )
        elif self.config.action_motion_weighting:
            if observation.action_motion_delta is None or observation.action_gripper_flip is None:
                raise ValueError(
                    "action_motion_weighting requires action_motion_delta and action_gripper_flip metadata"
                )
            expected_delta_shape = (
                actions.shape[0],
                self.action_horizon,
                self.config.action_motion_joint_dims,
            )
            expected_flip_shape = (actions.shape[0], self.action_horizon)
            if observation.action_motion_delta.shape != expected_delta_shape:
                raise ValueError(
                    f"Expected action_motion_delta shape {expected_delta_shape}, "
                    f"got {observation.action_motion_delta.shape}"
                )
            if observation.action_gripper_flip.shape != expected_flip_shape:
                raise ValueError(
                    f"Expected action_gripper_flip shape {expected_flip_shape}, "
                    f"got {observation.action_gripper_flip.shape}"
                )
            weight, motion_score, raw_weight = action_motion_weights(
                observation.action_motion_delta,
                observation.action_gripper_flip,
                joint_scales=self.config.action_motion_joint_scales,
                min_weight=self.config.action_motion_min_weight,
                max_weight=self.config.action_motion_max_weight,
            )
            weighted_action_loss = weight * action_loss
            motion_rms = jnp.sqrt(jnp.mean(jnp.square(observation.action_motion_delta), axis=-1))
            decision_mask = motion_score >= 1.0
            decision_count = jnp.maximum(jnp.sum(decision_mask), 1)
            loss_components.update(
                {
                    "unweighted_loss": jnp.mean(action_loss),
                    "weighted_loss": jnp.mean(weighted_action_loss),
                    "weighted_loss_delta": jnp.mean(weighted_action_loss) - jnp.mean(action_loss),
                    "action_motion_rms_mean": jnp.mean(motion_rms),
                    "action_motion_rms_p90": jnp.quantile(motion_rms, 0.9),
                    "action_motion_rms_max": jnp.max(motion_rms),
                    "action_motion_score_mean": jnp.mean(motion_score),
                    "action_motion_score_p90": jnp.quantile(motion_score, 0.9),
                    "action_motion_score_max": jnp.max(motion_score),
                    "action_motion_decision_fraction": jnp.mean(decision_mask.astype(jnp.float32)),
                    "action_gripper_flip_fraction": jnp.mean(observation.action_gripper_flip.astype(jnp.float32)),
                    "action_motion_raw_weight_min": jnp.min(raw_weight),
                    "action_motion_raw_weight_mean": jnp.mean(raw_weight),
                    "action_motion_raw_weight_p90": jnp.quantile(raw_weight, 0.9),
                    "action_motion_raw_weight_max": jnp.max(raw_weight),
                    "action_motion_weight_min": jnp.min(weight),
                    "action_motion_weight_mean": jnp.mean(weight),
                    "action_motion_weight_p90": jnp.quantile(weight, 0.9),
                    "action_motion_weight_max": jnp.max(weight),
                    "decision_loss_unweighted": jnp.sum(action_loss * decision_mask) / decision_count,
                    "decision_loss_weighted": jnp.sum(weighted_action_loss * decision_mask) / decision_count,
                }
            )
        elif self.config.hardness_weighting:
            score_action_dims = self.config.hardness_score_action_dims or self.action_dim
            full_error = per_action_flow_error(v_t, u_t, score_action_dims)
            weight, hardness_score, raw_weight = flow_error_hardness_weights(
                full_error,
                time,
                min_weight=self.config.hardness_min_weight,
                max_weight=self.config.hardness_max_weight,
                threshold=self.config.hardness_threshold,
                temperature=self.config.hardness_temperature,
                time_bins=self.config.hardness_time_bins,
            )
            hardness_alpha = jnp.asarray(hardness_alpha, dtype=jnp.float32)
            effective_weight = 1.0 + hardness_alpha * (weight - 1.0)
            weighted_action_loss = effective_weight * action_loss
            loss_components.update(
                {
                    "hardness_weight_alpha": hardness_alpha,
                    "unweighted_loss": jnp.mean(action_loss),
                    "weighted_loss": jnp.mean(weighted_action_loss),
                    "weighted_loss_delta": jnp.mean(weighted_action_loss) - jnp.mean(action_loss),
                    "full_error": jnp.mean(full_error),
                    "full_hardness_mean": jnp.mean(hardness_score),
                    "full_hardness_p90": jnp.quantile(hardness_score, 0.9),
                    "full_hardness_max": jnp.max(hardness_score),
                    "hardness_raw_weight_min": jnp.min(raw_weight),
                    "hardness_raw_weight_mean": jnp.mean(raw_weight),
                    "hardness_raw_weight_p90": jnp.quantile(raw_weight, 0.9),
                    "hardness_raw_weight_max": jnp.max(raw_weight),
                    "hardness_weight_min": jnp.min(weight),
                    "hardness_weight_mean": jnp.mean(weight),
                    "hardness_weight_p90": jnp.quantile(weight, 0.9),
                    "hardness_weight_max": jnp.max(weight),
                    "hardness_effective_weight_min": jnp.min(effective_weight),
                    "hardness_effective_weight_mean": jnp.mean(effective_weight),
                    "hardness_effective_weight_p90": jnp.quantile(effective_weight, 0.9),
                    "hardness_effective_weight_max": jnp.max(effective_weight),
                }
            )
        elif self.config.memory_advantage_weighting:
            recent_observation = build_recent_memory_observation(
                observation,
                self.config.memory_advantage_recent_steps,
                exclude_demo=self.config.memory_advantage_recent_exclude_demo,
            )
            recent_memory_tokens, recent_memory_mask, recent_memory_ar_mask = self.embed_memory(
                recent_observation,
                train=train,
                dropout_rng=recent_memory_dropout_rng,
            )
            recent_prefix_tokens = prefix_tokens
            recent_prefix_mask = prefix_mask
            recent_prefix_ar_mask = prefix_ar_mask
            if self.config.memory_advantage_recent_exclude_demo and demo_anchor_length:
                recent_prefix_tokens = recent_prefix_tokens[:, :-demo_anchor_length]
                recent_prefix_mask = recent_prefix_mask[:, :-demo_anchor_length]
                recent_prefix_ar_mask = recent_prefix_ar_mask[:-demo_anchor_length]
            recent_v_t = self._predict_action_only(
                recent_prefix_tokens,
                recent_prefix_mask,
                recent_prefix_ar_mask,
                recent_memory_tokens,
                recent_memory_mask,
                recent_memory_ar_mask,
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
                adarms_cond,
                full_memory_mask_for_positions=full_history_memory_mask,
                full_prefix_mask_for_positions=prefix_mask,
            )
            recent_action_loss = jnp.mean(jnp.square(recent_v_t - u_t), axis=-1)
            score_action_dims = self.config.memory_advantage_score_action_dims or self.action_dim
            full_memory_error = per_action_flow_error(v_t, u_t, score_action_dims)
            recent_memory_error = per_action_flow_error(recent_v_t, u_t, score_action_dims)
            signed_advantage = jax.lax.stop_gradient(
                (recent_memory_error - full_memory_error) / (recent_memory_error + 1e-6)
            )
            weight, advantage, raw_weight, memory_score, hardness_score, combined_score = memory_advantage_weights(
                full_memory_error,
                recent_memory_error,
                flow_time=time,
                alpha=memory_advantage_alpha,
                min_weight=self.config.memory_advantage_min_weight,
                max_weight=self.config.memory_advantage_max_weight,
                threshold=self.config.memory_advantage_threshold,
                temperature=self.config.memory_advantage_temperature,
                hardness_weight=self.config.memory_advantage_hardness_weight,
                hardness_threshold=self.config.memory_advantage_hardness_threshold,
                hardness_temperature=self.config.memory_advantage_hardness_temperature,
                hardness_time_bins=self.config.memory_advantage_hardness_time_bins,
            )
            weighted_action_loss = weight * action_loss
            weighted_recent_action_loss = self.config.memory_advantage_recent_loss_weight * recent_action_loss
            loss_components.update(
                {
                    "unweighted_loss": jnp.mean(action_loss),
                    "weighted_loss": jnp.mean(weighted_action_loss),
                    "full_memory_error": jnp.mean(full_memory_error),
                    "recent_memory_error": jnp.mean(recent_memory_error),
                    "memory_advantage_mean": jnp.mean(advantage),
                    "memory_advantage_p90": jnp.quantile(advantage, 0.9),
                    "memory_advantage_max": jnp.max(advantage),
                    "memory_advantage_signed_mean": jnp.mean(signed_advantage),
                    "memory_advantage_signed_p10": jnp.quantile(signed_advantage, 0.1),
                    "memory_advantage_signed_min": jnp.min(signed_advantage),
                    "memory_negative_fraction": jnp.mean(signed_advantage < 0.0),
                    "memory_sensitive_fraction": jnp.mean(advantage > self.config.memory_advantage_threshold),
                    "weight_mean": jnp.mean(weight),
                    "weight_min": jnp.min(weight),
                    "weight_p90": jnp.quantile(weight, 0.9),
                    "weight_max": jnp.max(weight),
                    "raw_weight_min": jnp.min(raw_weight),
                    "raw_weight_mean": jnp.mean(raw_weight),
                    "raw_weight_p90": jnp.quantile(raw_weight, 0.9),
                    "raw_weight_max": jnp.max(raw_weight),
                    "memory_weight_score_mean": jnp.mean(memory_score),
                    "full_hardness_mean": jnp.mean(hardness_score),
                    "full_hardness_p90": jnp.quantile(hardness_score, 0.9),
                    "combined_weight_score_mean": jnp.mean(combined_score),
                    "weight_alpha": jnp.asarray(memory_advantage_alpha, dtype=jnp.float32),
                    "recent_branch_loss": jnp.mean(recent_action_loss),
                    "recent_branch_loss_weighted": jnp.mean(weighted_recent_action_loss),
                    "weighted_loss_delta": jnp.mean(weighted_action_loss) - jnp.mean(action_loss),
                }
            )
            if return_advantage_values:
                loss_components["memory_advantage_signed_values"] = signed_advantage
                loss_components["memory_advantage_values"] = advantage
        total_loss = weighted_action_loss
        if recent_action_loss is not None:
            total_loss = total_loss + self.config.memory_advantage_recent_loss_weight * recent_action_loss
        if self.config.memory_demo_direction_generation:
            if direction_tokens is None:
                raise RuntimeError("Demo direction branch was not constructed")
            direction_hidden = outputs[0][:, -direction_tokens.shape[1] :]
            direction_logits = self.PaliGemma.llm(
                direction_hidden[:, :-1],
                method="decode",
            )
            direction_targets = jnp.asarray(observation.demo_direction_tokens[:, 1:], dtype=jnp.int32)
            direction_loss_mask = jnp.asarray(
                observation.demo_direction_loss_mask[:, 1:], dtype=jnp.bool_
            ) & jnp.asarray(observation.demo_direction_mask[:, 1:], dtype=jnp.bool_)
            direction_log_probs = jax.nn.log_softmax(direction_logits.astype(jnp.float32), axis=-1)
            target_log_probs = jnp.take_along_axis(
                direction_log_probs,
                direction_targets[..., None],
                axis=-1,
            )[..., 0]
            valid_direction_count = jnp.maximum(jnp.sum(direction_loss_mask, axis=-1), 1)
            direction_loss = -jnp.sum(target_log_probs * direction_loss_mask, axis=-1) / valid_direction_count
            weighted_direction_loss = self.config.memory_demo_direction_loss_weight * direction_loss
            total_loss = total_loss + weighted_direction_loss[:, None]
            direction_predictions = jnp.argmax(direction_logits, axis=-1)
            direction_correct = (direction_predictions == direction_targets) & direction_loss_mask
            global_direction_count = jnp.maximum(jnp.sum(direction_loss_mask), 1)
            loss_components.update(
                {
                    "loss/demo_direction_lm": jnp.mean(direction_loss),
                    "loss/demo_direction_lm_weighted": jnp.mean(weighted_direction_loss),
                    "demo_direction_token_accuracy": (
                        jnp.sum(direction_correct) / global_direction_count
                    ),
                    "demo_direction_supervised_tokens": jnp.sum(direction_loss_mask),
                }
            )
        if self.config.memory_flow_horizon:
            future_stream_index = 1 if self.use_future_expert else 0 if self.config.memory_future_training_only else 1
            memory_out = outputs[future_stream_index][:, -self.config.memory_flow_horizon :]
            memory_prediction = self.memory_flow_out_proj(memory_out)
            memory_supervision_target = (
                memory_flow_velocity
                if self.config.memory_future_prediction_mode == "flow_matching"
                else memory_prediction_target
            )
            memory_token_loss = jnp.mean(jnp.square(memory_prediction - memory_supervision_target), axis=-1)
            valid_count = jnp.maximum(jnp.sum(memory_prediction_target_mask, axis=-1), 1)
            memory_loss = jnp.sum(memory_token_loss * memory_prediction_target_mask, axis=-1) / valid_count
            weighted_memory_loss = self.config.memory_flow_loss_weight * memory_loss
            total_loss = total_loss + weighted_memory_loss[:, None]
            memory_loss_name = (
                "memory_flow" if self.config.memory_future_prediction_mode == "flow_matching" else "memory_direct"
            )
            loss_components[f"loss/{memory_loss_name}"] = jnp.mean(memory_loss)
            loss_components[f"loss/{memory_loss_name}_weighted"] = jnp.mean(weighted_memory_loss)
        if return_components:
            return total_loss, loss_components
        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=self.config.active_image_keys
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        use_memory_flow = (
            self.config.memory_flow_horizon > 0
            and self.config.memory_future_prediction_mode == "flow_matching"
            and not self.config.memory_future_training_only
        )
        use_memory_direct = (
            self.config.memory_flow_horizon > 0
            and self.config.memory_future_prediction_mode == "direct"
            and not self.config.memory_future_training_only
        )
        if use_memory_flow:
            action_rng, memory_rng = jax.random.split(rng)
        else:
            action_rng, memory_rng = rng, None
        if noise is None:
            noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))
        memory_noise = (
            jax.random.normal(
                memory_rng,
                (batch_size, self.config.memory_flow_horizon, self.config.memory_future_latent_dim),
            )
            if use_memory_flow
            else None
        )

        # first fill KV cache with a forward pass of the prefix
        (
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            demo_anchor_length,
            language_start,
            language_length,
        ) = self._embed_prefix_with_anchor_length(observation)
        if self.use_memory:
            memory_tokens, memory_mask, memory_ar_mask = self.embed_memory(observation)
            if use_memory_direct:
                future_tokens, future_mask, future_ar_mask = self.embed_memory_direct_queries(batch_size)
                memory_tokens = jnp.concatenate([memory_tokens, future_tokens], axis=1)
                memory_mask = jnp.concatenate([memory_mask, future_mask], axis=1)
                memory_ar_mask = jnp.concatenate([memory_ar_mask, future_ar_mask], axis=0)
            if self.config.memory_prefix_order == "language_memory_vision":
                prefix_tokens_full, prefix_mask_full, prefix_ar_mask_full = make_language_memory_vision_prefix(
                    prefix_tokens,
                    prefix_mask,
                    memory_tokens,
                    memory_mask,
                    memory_ar_mask,
                    language_start=language_start,
                    language_length=language_length,
                    memory_anchor_length=demo_anchor_length,
                )
            elif self.config.memory_register_count:
                register_tokens, register_mask = self.embed_memory_registers(batch_size, prefix_tokens.dtype)
                public_prefix_tokens = prefix_tokens[:, :-demo_anchor_length]
                private_anchor_tokens = prefix_tokens[:, -demo_anchor_length:]
                public_prefix_mask = prefix_mask[:, :-demo_anchor_length]
                private_anchor_mask = prefix_mask[:, -demo_anchor_length:]
                private_history_tokens = jnp.concatenate([private_anchor_tokens, memory_tokens], axis=1)
                private_history_mask = jnp.concatenate([private_anchor_mask, memory_mask], axis=1)
                prefix_mask_full = jnp.concatenate(
                    [public_prefix_mask, private_history_mask, register_mask], axis=1
                )
                prefix_tokens_full = jnp.concatenate(
                    [public_prefix_tokens, private_history_tokens, register_tokens], axis=1
                )
                empty_action_mask = jnp.zeros((batch_size, 0), dtype=jnp.bool_)
                prefix_attn_mask = make_memory_register_attn_mask(
                    public_prefix_mask,
                    private_history_mask,
                    register_mask,
                    empty_action_mask,
                    jnp.zeros((0,), dtype=jnp.bool_),
                )
            else:
                prefix_mask_full = jnp.concatenate([prefix_mask, memory_mask], axis=1)
                prefix_ar_mask_full = jnp.concatenate([prefix_ar_mask, memory_ar_mask], axis=0)
                prefix_tokens_full = (
                    jnp.concatenate([prefix_tokens, memory_tokens], axis=1)
                    if self.config.memory_use_vlm_expert
                    else prefix_tokens
                )
        else:
            prefix_mask_full = prefix_mask
            prefix_ar_mask_full = prefix_ar_mask
            prefix_tokens_full = prefix_tokens
        if not self.config.memory_register_count:
            prefix_attn_mask = make_attn_mask(prefix_mask_full, prefix_ar_mask_full)
        positions = jnp.cumsum(prefix_mask_full, axis=1) - 1
        if self.use_memory_expert:
            # Prefix and memory are both invariant across the denoising loop.
            # Cache their joint three-stream forward once, including the
            # memory expert's K/V, so each denoising step only runs the action
            # suffix. The block mask already prevents prefix tokens from
            # attending to memory, matching the full joint forward exactly.
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens, memory_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
                adarms_cond=[None, None, None],
            )
        elif self.use_future_expert:
            # The third stream is a training-only future expert. Leave it
            # empty so inference is identical to expert-0 history + action.
            _, kv_cache = self.PaliGemma.llm(
                [prefix_tokens_full, None, None],
                mask=prefix_attn_mask,
                positions=positions,
                adarms_cond=[None, None, None],
            )
        else:
            # Expert-0 memory is part of the VLM prefix and uses the same
            # one-time KV-cache fill as the non-memory model.
            _, kv_cache = self.PaliGemma.llm([prefix_tokens_full, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, memory_x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            if use_memory_flow:
                memory_flow_tokens, memory_flow_mask, memory_flow_ar_mask, _ = self.embed_memory_flow(
                    memory_x_t, jnp.broadcast_to(time, batch_size)
                )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            if use_memory_flow:
                current_mask = jnp.concatenate([memory_flow_mask, suffix_mask], axis=1)
                current_ar_mask = jnp.concatenate([memory_flow_ar_mask, suffix_ar_mask], axis=0)
            else:
                current_mask = suffix_mask
                current_ar_mask = suffix_ar_mask
            all_mask = jnp.concatenate([prefix_mask_full, current_mask], axis=1)
            current_length = current_mask.shape[1]
            if self.config.memory_register_count:
                full_attn_mask = make_memory_register_attn_mask(
                    public_prefix_mask,
                    private_history_mask,
                    register_mask,
                    current_mask,
                    current_ar_mask,
                )[:, -current_length:, :]
            else:
                if prefix_ar_mask_full.ndim == 2:
                    current_ar_mask = jnp.broadcast_to(current_ar_mask, current_mask.shape)
                    all_ar_mask = jnp.concatenate([prefix_ar_mask_full, current_ar_mask], axis=1)
                else:
                    all_ar_mask = jnp.concatenate([prefix_ar_mask_full, current_ar_mask], axis=0)
                full_attn_mask = make_attn_mask(all_mask, all_ar_mask)[:, -current_length:, :]
            positions = (jnp.cumsum(all_mask, axis=1) - 1)[:, -current_length:]

            if self.use_memory_expert:
                outputs, _ = self.PaliGemma.llm(
                    [None, memory_flow_tokens if use_memory_flow else None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, None, adarms_cond],
                )
                suffix_out = outputs[-1]
            elif self.use_future_expert:
                outputs, _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, None, adarms_cond],
                )
                suffix_out = outputs[-1]
            else:
                outputs, _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                suffix_out = outputs[-1]
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            if use_memory_flow:
                memory_out = outputs[1][:, -self.config.memory_flow_horizon :]
                memory_v_t = self.memory_flow_out_proj(memory_out)
                memory_x_t = memory_x_t + dt * memory_v_t

            return x_t + dt * v_t, memory_x_t, time + dt

        def cond(carry):
            _, _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, memory_noise, 1.0))
        return x_0
