import dataclasses
import math
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"
    # Joint MoT streams cannot use the stock single-stream attention module.
    # SDPA lets PyTorch select a fused attention kernel on supported GPUs while
    # retaining a portable math implementation on CPU.
    mot_attention_backend: Literal["sdpa", "eager"] = "sdpa"
    # Views that are structurally absent from a dataset should be omitted
    # instead of running a vision tower and transformer FFN for masked zeros.
    active_image_keys: tuple[str, ...] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    # Optional CD-LAM latent-action memory. A positive horizon enables memory
    # tokens. By default they use a third MoT stream; memory_use_vlm_expert can
    # instead append them to expert 0 after a learned projector.
    memory_horizon: int = 0
    # Strict visual-anchor ablation: keep the demo/execution anchor images as
    # ordered context, but do not construct latent memory inputs or a memory
    # projector. This is intentionally distinct from an all-false LAM mask.
    memory_anchor_only: bool = False
    memory_latent_dim: int = 32
    # Required temporal separation of valid cached transitions. With stride
    # four, cache rows encode 0->4, 4->8, ... and all other rows are invalid;
    # this is not subsampling adjacent-frame latents. The dataset checks this
    # field against the cache manifest before training.
    memory_stride: int = 1
    # Consecutive latent frames are grouped in temporal order before the
    # projector. ``concatenate`` preserves the historical behavior (for
    # example, four 32-D deltas become a 128-D projector input), while
    # ``mean`` keeps the projector input at 32-D and averages only valid
    # frames in each group. With typed memory, demo, boundary, and execution
    # are grouped independently, so no token averages across phases; each
    # phase's final group may contain 1--N-1 frames.
    memory_frames_per_token: int = 1
    memory_pooling_mode: Literal["concatenate", "mean"] = "concatenate"
    memory_projector_hidden_dim: int = 512
    # Independently hide projected memory tokens during training. This changes
    # both the attention mask and token value; inference always keeps all
    # valid tokens.
    memory_token_dropout_rate: float = 0.0
    # Replace every valid normalized LAM vector with one deterministic random
    # vector shared by all examples and timesteps. Masks, temporal positions,
    # segment IDs, and visual anchors are preserved for content ablations.
    memory_fixed_random_latent_seed: int | None = None
    # Add typed segment information for padding/demo/reset-boundary/execution.
    # With grouped expert-0 memory, every frame slot contributes its own 4-D
    # one-hot type to the projector so grouping does not erase boundaries.
    memory_segment_embedding: bool = False
    # Number of learned memory type embeddings. Dataset LAM segment ids remain
    # 0=padding, 1=demo, 2=boundary, 3=execution; extra ids may be reserved for
    # visually encoded anchors without changing the cache schema.
    memory_segment_vocab_size: int = 4
    memory_demo_anchor_type_id: int = 1
    memory_execution_anchor_type_id: int = 2
    # Insert one learned boundary token between the visual demo anchor and the
    # first valid demo LAM token. The token is represented by the configured
    # segment embedding only and carries no future transition latent.
    memory_demo_lam_boundary_token: bool = False
    memory_demo_lam_boundary_type_id: int = 2
    # Keep the projector input width equal to the latent width by adding a
    # learned type embedding after projection. This is intended for framewise
    # expert-0 memory such as DeltaTok's 768-D token. Grouped legacy memory
    # continues to concatenate a per-frame one-hot type before projection.
    memory_segment_embedding_after_projection: bool = False
    # Encode the first demonstration frame through the shared vision tower and
    # place it immediately before the latent memory block.
    memory_demo_anchor: bool = False
    # Encode the raw head-camera frame at episode timestep zero and place it
    # after language but immediately before the latent history. Unlike the
    # demo anchor, this is present for every episode and does not imply that
    # demonstrations exist in the dataset.
    memory_episode_anchor: bool = False
    # Consume a cached temporal mean of per-frame SigLIP patch features instead
    # of encoding one raw anchor image. This preserves the same anchor token
    # count and embedding width as the original first-frame design.
    memory_demo_anchor_feature_pool: bool = False
    # Add the absolute head-camera frame at exec_start_idx between the demo
    # and execution portions of the latent history.  The anchor reuses the
    # shared vision tower and the existing typed-memory embedding table, so it
    # introduces no new trainable parameters.
    memory_execution_anchor: bool = False
    # Reuse the action expert's Q/K/V/output attention projections for memory
    # tokens while retaining separate memory norms and FFN parameters.
    share_memory_attention: bool = False
    # Route projected memory tokens through the expert-0 VLM stream. This
    # removes the separate memory Transformer entirely: attention, norms and
    # FFNs are all the exact expert-0 parameters. Type IDs, when enabled, are
    # concatenated as one-hot inputs to the memory projector rather than using
    # a separate embedding table.
    memory_use_vlm_expert: bool = False
    # Keep expert-0 memory in the existing image/language prefix attention
    # block. When false, the first memory token starts a new causal block (the
    # historical LIBERO-Mem behavior).
    memory_same_prefix_block: bool = False
    # Physical expert-0 prefix order. The default preserves the historical
    # image/language/anchor prefix followed by memory. The RoboMME LMV variant
    # moves prompt tokens first, memory second, and all ordinary visual prefix
    # tokens (current observations and image anchors) last.
    memory_prefix_order: str = "vision_language_memory"
    # These flags are intentionally explicit rather than inferred from token
    # order so checkpoints record the exact attention contract. Memory anchors
    # remain full-attention blocks internally; causal boundaries apply between
    # successive LAM tokens and around each anchor block.
    language_causal_attention: bool = False
    memory_causal_attention: bool = False
    # Append learned expert-0 register tokens after anchor + memory and route
    # all historical information to the action expert through those registers.
    # Zero preserves the legacy attention behavior.
    memory_register_count: int = 0
    # Optional fixed normalization applied before the memory projector. These
    # values must be computed from valid CD-LAM transitions only. Keeping them
    # in the model config makes training and online inference use the same
    # latent scale; omitted values mean identity normalization.
    memory_latent_mean: tuple[float, ...] | None = None
    memory_latent_std: tuple[float, ...] | None = None
    # Train CD-LAM inside the PyTorch PI0.5 graph instead of reading cached
    # latents. The data loader supplies raw [history, view, pair, H, W, C]
    # tensors and the trainable encoder concatenates head32 and wrist32.
    online_memory_lam: bool = False
    online_memory_lam_checkpoint: str | None = None
    online_memory_lam_pair_batch_size: int = 4
    online_memory_lam_gradient_checkpointing: bool = True
    # Number of future per-frame memory latents jointly generated with the
    # action chunk. Zero disables the auxiliary stream.
    memory_flow_horizon: int = 0
    # Future supervision may come from a different latent model than the
    # history memory. When omitted, preserve the historical behavior of using
    # memory_latent_dim and its normalization for both.
    memory_future_latent_dim: int | None = None
    memory_future_latent_mean: tuple[float, ...] | None = None
    memory_future_latent_std: tuple[float, ...] | None = None
    # Flow matching jointly denoises future latents with the action chunk.
    # Direct prediction uses learned queries that read the observation and
    # memory history, then regress future latents without target/noise inputs.
    memory_future_prediction_mode: str = "flow_matching"
    # Keep future prediction as a training-only sibling objective. The
    # future queries read the same prefix/history as the action branch, but the
    # action branch cannot attend to them and inference omits them entirely.
    memory_future_training_only: bool = False
    # Route the training-only future branch through its own 311M expert while
    # history memory remains on expert 0. Its attention/MLP weights can then be
    # warm-started from the action expert without coupling their activations.
    memory_future_use_separate_expert: bool = False
    # Learned per-horizon queries are retained by default for checkpoint and
    # behavior compatibility. A separate flow-matching future expert can omit
    # them and rely on RoPE for horizon order, matching the action branch.
    memory_future_use_learned_queries: bool = True
    # Match pi0.5 action timestep conditioning exactly: project the sinusoidal
    # timestep through in -> swish -> out -> swish, then pass it to expert 2 as
    # its AdaRMS condition instead of adding it to the input token.
    memory_future_use_adarms_time_conditioning: bool = False
    memory_flow_loss_weight: float = 1.0

    # Training-only PaliGemma language branch for PatternLock demonstration
    # decomposition. The backbone autoregressively emits the full move
    # sequence, including adjacent repeated directions, while the action
    # expert predicts its flow-matching target.
    memory_demo_direction_generation: bool = False
    memory_demo_direction_max_token_len: int = 32
    memory_demo_direction_loss_weight: float = 1.0
    # When enabled, action tokens attend to the teacher-forced direction plan
    # and start after its valid tokens in RoPE position space.
    memory_demo_direction_condition_action: bool = False
    memory_demo_direction_seed_token_len: int = 3
    memory_demo_direction_eos_token_id: int = 1

    # Training-only counterfactual weighting for long-horizon memory. The
    # normal action branch sees the complete history; a counterfactual branch
    # sees only recent execution memory under the exact same flow noise and
    # timestep. Its score is detached, while a small auxiliary loss keeps the
    # shared policy competent when only recent memory is available.
    memory_advantage_weighting: bool = False
    memory_advantage_recent_steps: int = 8
    memory_advantage_min_weight: float = 0.3
    memory_advantage_max_weight: float = 3.0
    memory_advantage_threshold: float = 0.1
    memory_advantage_temperature: float = 0.05
    # Add a focal-style score for positions the full-memory branch still gets
    # wrong. The log error is standardized inside coarse flow-time buckets;
    # the weight controls its maximum contribution to the combined score.
    memory_advantage_hardness_weight: float = 0.0
    memory_advantage_hardness_threshold: float = 0.5
    memory_advantage_hardness_temperature: float = 0.5
    memory_advantage_hardness_time_bins: int = 4
    memory_advantage_warmup_fraction: float = 0.1
    memory_advantage_ramp_fraction: float = 0.1
    memory_advantage_recent_loss_weight: float = 0.1
    # Remove both the visual demo anchor and demo/boundary DeltaTok entries
    # from the recent branch. The normal full-memory branch is unchanged.
    memory_advantage_recent_exclude_demo: bool = True
    # Only these leading action dimensions contribute to the counterfactual
    # score. The main action loss remains unchanged over all action_dim values.
    # This excludes padded dimensions in datasets such as RoboMME (8 of 32).
    memory_advantage_score_action_dims: int | None = None

    # Training-only focal-style weighting from the full policy branch's own
    # detached flow error. Unlike memory_advantage_weighting, this performs no
    # counterfactual recent-memory forward and adds no auxiliary branch loss.
    hardness_weighting: bool = False
    # Keep loss uniform for this many optimizer steps before enabling hardness
    # reweighting. The training loop passes the corresponding gate into
    # compute_loss; keeping this distinct from LR warmup makes the contract
    # explicit even when schedules change independently.
    hardness_warmup_steps: int = 0
    hardness_min_weight: float = 0.3
    hardness_max_weight: float = 5.0
    hardness_threshold: float = 0.5
    hardness_temperature: float = 0.5
    hardness_time_bins: int = 4
    hardness_score_action_dims: int | None = None

    # Training-only weighting based purely on the ground-truth raw action
    # transition |a[t+h+1] - a[t+h]|. Each joint is divided by a fixed
    # execution-split p95 before taking the maximum, so a meaningful change in
    # any joint is retained despite different per-joint motion scales.
    action_motion_weighting: bool = False
    action_motion_joint_dims: int = 7
    action_motion_joint_scales: tuple[float, ...] | None = None
    action_motion_gripper_index: int | None = 7
    action_motion_gripper_flip_threshold: float = 0.5
    action_motion_min_weight: float = 0.3
    action_motion_max_weight: float = 5.0

    # Training-only fixed weighting from a no-memory policy's sampling
    # dispersion. Scores are global empirical-CDF quantiles in [0, 1], one
    # per physical action position, and never receive gradients.
    baseline_uncertainty_weighting: bool = False
    baseline_uncertainty_min_weight: float = 0.25
    baseline_uncertainty_max_weight: float = 3.25
    baseline_uncertainty_rank_power: float = 3.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.mot_attention_backend not in ("sdpa", "eager"):
            raise ValueError(f"Unsupported MoT attention backend: {self.mot_attention_backend}")
        supported_image_keys = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        if not self.active_image_keys or len(set(self.active_image_keys)) != len(self.active_image_keys):
            raise ValueError("active_image_keys must be non-empty and unique")
        if unknown_image_keys := set(self.active_image_keys) - supported_image_keys:
            raise ValueError(f"Unsupported active image keys: {sorted(unknown_image_keys)}")
        if self.memory_horizon < 0:
            raise ValueError("memory_horizon must be non-negative")
        if self.memory_anchor_only:
            if self.memory_horizon != 0:
                raise ValueError("memory_anchor_only requires memory_horizon=0")
            if not self.memory_demo_anchor or not self.memory_execution_anchor:
                raise ValueError("memory_anchor_only requires demo and execution anchors")
            if not self.memory_use_vlm_expert:
                raise ValueError("memory_anchor_only requires memory_use_vlm_expert")
            if self.memory_prefix_order != "language_memory_vision":
                raise ValueError("memory_anchor_only requires language_memory_vision order")
            if not self.language_causal_attention or not self.memory_causal_attention:
                raise ValueError("memory_anchor_only requires causal language and anchor blocks")
            if self.memory_demo_lam_boundary_token:
                raise ValueError("memory_anchor_only cannot insert a demo/LAM boundary token")
            if self.memory_flow_horizon or self.memory_register_count or self.online_memory_lam:
                raise ValueError("memory_anchor_only is incompatible with latent-memory auxiliary paths")
        if self.memory_flow_horizon < 0:
            raise ValueError("memory_flow_horizon must be non-negative")
        if self.memory_future_latent_dim is None:
            object.__setattr__(self, "memory_future_latent_dim", self.memory_latent_dim)
        if self.memory_future_latent_dim <= 0:
            raise ValueError("memory_future_latent_dim must be positive")
        if self.memory_future_prediction_mode not in ("flow_matching", "direct"):
            raise ValueError("memory_future_prediction_mode must be 'flow_matching' or 'direct'")
        if not self.memory_flow_horizon and self.memory_future_prediction_mode != "flow_matching":
            raise ValueError("memory_future_prediction_mode requires memory_flow_horizon > 0")
        if self.memory_flow_horizon and self.memory_horizon == 0:
            raise ValueError("memory_flow_horizon requires memory_horizon > 0")
        if self.memory_future_training_only and not self.memory_flow_horizon:
            raise ValueError("memory_future_training_only requires memory_flow_horizon > 0")
        if self.memory_future_training_only and not self.memory_use_vlm_expert:
            raise ValueError("training-only future prediction requires expert-0 memory")
        if self.memory_future_use_separate_expert and not self.memory_future_training_only:
            raise ValueError("a separate future expert requires training-only future prediction")
        if self.memory_future_use_separate_expert and not self.memory_flow_horizon:
            raise ValueError("a separate future expert requires memory_flow_horizon > 0")
        if not self.memory_future_use_learned_queries:
            if self.memory_future_prediction_mode != "flow_matching":
                raise ValueError("future learned queries can only be disabled for flow matching")
            if not self.memory_future_use_separate_expert:
                raise ValueError("disabling future learned queries requires a separate future expert")
        if self.memory_future_use_adarms_time_conditioning:
            if not self.pi05:
                raise ValueError("future AdaRMS time conditioning requires pi05")
            if self.memory_future_prediction_mode != "flow_matching":
                raise ValueError("future AdaRMS time conditioning requires flow matching")
            if not self.memory_future_use_separate_expert:
                raise ValueError("future AdaRMS time conditioning requires a separate future expert")
        if self.memory_flow_horizon and self.memory_use_vlm_expert and not self.memory_future_training_only:
            raise ValueError("memory prediction with expert-0 memory requires training-only mode")
        if self.memory_flow_horizon and self.memory_frames_per_token != 1:
            raise ValueError("memory flow prediction requires memory_frames_per_token=1")
        if not math.isfinite(self.memory_flow_loss_weight) or self.memory_flow_loss_weight < 0:
            raise ValueError("memory_flow_loss_weight must be finite and non-negative")
        if type(self.memory_demo_direction_max_token_len) is not int or self.memory_demo_direction_max_token_len < 3:
            raise ValueError("memory_demo_direction_max_token_len must be an integer >= 3")
        if type(self.memory_demo_direction_seed_token_len) is not int or self.memory_demo_direction_seed_token_len < 1:
            raise ValueError("memory_demo_direction_seed_token_len must be a positive integer")
        if self.memory_demo_direction_seed_token_len >= self.memory_demo_direction_max_token_len:
            raise ValueError("direction seed must be shorter than memory_demo_direction_max_token_len")
        if type(self.memory_demo_direction_eos_token_id) is not int or self.memory_demo_direction_eos_token_id < 0:
            raise ValueError("memory_demo_direction_eos_token_id must be a non-negative integer")
        if not math.isfinite(self.memory_demo_direction_loss_weight) or self.memory_demo_direction_loss_weight < 0:
            raise ValueError("memory_demo_direction_loss_weight must be finite and non-negative")
        if self.memory_demo_direction_generation:
            if not self.memory_use_vlm_expert:
                raise ValueError("demo direction generation requires expert-0/VLM memory")
            if not self.memory_demo_anchor:
                raise ValueError("demo direction generation requires the demonstration visual anchor")
            if self.memory_prefix_order != "language_memory_vision":
                raise ValueError("demo direction generation requires language_memory_vision prefix order")
            if self.memory_frames_per_token != 1:
                raise ValueError("demo direction generation requires one LAM frame per token")
            if not self.memory_segment_embedding:
                raise ValueError("demo direction generation requires typed memory segments")
        if self.memory_demo_direction_condition_action and not self.memory_demo_direction_generation:
            raise ValueError("conditioning actions on direction text requires demo direction generation")
        if self.memory_advantage_weighting and not self.use_memory:
            raise ValueError("memory_advantage_weighting requires memory_horizon > 0")
        if (
            self.memory_advantage_weighting
            and self.memory_advantage_recent_exclude_demo
            and not self.memory_segment_embedding
        ):
            raise ValueError("excluding demo from recent memory requires memory_segment_embedding")
        if self.memory_advantage_weighting and self.memory_flow_horizon and not self.memory_future_training_only:
            raise ValueError(
                "memory_advantage_weighting requires future prediction to be training-only so the action "
                "counterfactual changes only memory history"
            )
        if type(self.memory_advantage_recent_steps) is not int or self.memory_advantage_recent_steps <= 0:
            raise ValueError("memory_advantage_recent_steps must be a positive integer")
        if self.memory_advantage_weighting and self.memory_advantage_recent_steps > self.memory_horizon:
            raise ValueError("memory_advantage_recent_steps cannot exceed memory_horizon")
        if not math.isfinite(self.memory_advantage_min_weight) or self.memory_advantage_min_weight <= 0:
            raise ValueError("memory_advantage_min_weight must be finite and positive")
        if (
            not math.isfinite(self.memory_advantage_max_weight)
            or self.memory_advantage_max_weight < self.memory_advantage_min_weight
        ):
            raise ValueError("memory_advantage_max_weight must be finite and >= memory_advantage_min_weight")
        if not math.isfinite(self.memory_advantage_threshold) or not 0 <= self.memory_advantage_threshold <= 1:
            raise ValueError("memory_advantage_threshold must be finite and in [0, 1]")
        if not math.isfinite(self.memory_advantage_temperature) or self.memory_advantage_temperature <= 0:
            raise ValueError("memory_advantage_temperature must be finite and positive")
        if (
            not math.isfinite(self.memory_advantage_hardness_weight)
            or not 0 <= self.memory_advantage_hardness_weight <= 1
        ):
            raise ValueError("memory_advantage_hardness_weight must be finite and in [0, 1]")
        if not math.isfinite(self.memory_advantage_hardness_threshold):
            raise ValueError("memory_advantage_hardness_threshold must be finite")
        if (
            not math.isfinite(self.memory_advantage_hardness_temperature)
            or self.memory_advantage_hardness_temperature <= 0
        ):
            raise ValueError("memory_advantage_hardness_temperature must be finite and positive")
        if type(self.memory_advantage_hardness_time_bins) is not int or self.memory_advantage_hardness_time_bins <= 0:
            raise ValueError("memory_advantage_hardness_time_bins must be a positive integer")
        if not math.isfinite(self.memory_advantage_recent_loss_weight) or self.memory_advantage_recent_loss_weight < 0:
            raise ValueError("memory_advantage_recent_loss_weight must be finite and non-negative")
        if type(self.memory_advantage_recent_exclude_demo) is not bool:
            raise ValueError("memory_advantage_recent_exclude_demo must be a boolean")
        for name, value in (
            ("memory_advantage_warmup_fraction", self.memory_advantage_warmup_fraction),
            ("memory_advantage_ramp_fraction", self.memory_advantage_ramp_fraction),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.memory_advantage_warmup_fraction + self.memory_advantage_ramp_fraction > 1:
            raise ValueError("memory advantage warmup and ramp fractions must sum to at most 1")
        if self.memory_advantage_score_action_dims is not None and (
            type(self.memory_advantage_score_action_dims) is not int
            or not 1 <= self.memory_advantage_score_action_dims <= self.action_dim
        ):
            raise ValueError("memory_advantage_score_action_dims must be in [1, action_dim]")
        if sum(
            (
                self.action_motion_weighting,
                self.memory_advantage_weighting,
                self.hardness_weighting,
                self.baseline_uncertainty_weighting,
            )
        ) > 1:
            raise ValueError(
                "action_motion_weighting, memory_advantage_weighting, hardness_weighting, and "
                "baseline_uncertainty_weighting are mutually exclusive"
            )
        if (
            not math.isfinite(self.baseline_uncertainty_min_weight)
            or self.baseline_uncertainty_min_weight <= 0
        ):
            raise ValueError("baseline_uncertainty_min_weight must be positive and finite")
        if (
            not math.isfinite(self.baseline_uncertainty_max_weight)
            or self.baseline_uncertainty_max_weight < self.baseline_uncertainty_min_weight
        ):
            raise ValueError("baseline_uncertainty_max_weight must be finite and >= min_weight")
        if (
            not math.isfinite(self.baseline_uncertainty_rank_power)
            or self.baseline_uncertainty_rank_power <= 0
        ):
            raise ValueError("baseline_uncertainty_rank_power must be positive and finite")
        if type(self.hardness_warmup_steps) is not int or self.hardness_warmup_steps < 0:
            raise ValueError("hardness_warmup_steps must be a non-negative integer")
        if not math.isfinite(self.hardness_min_weight) or self.hardness_min_weight <= 0:
            raise ValueError("hardness_min_weight must be finite and positive")
        if not math.isfinite(self.hardness_max_weight) or self.hardness_max_weight < self.hardness_min_weight:
            raise ValueError("hardness_max_weight must be finite and >= hardness_min_weight")
        if not math.isfinite(self.hardness_threshold):
            raise ValueError("hardness_threshold must be finite")
        if not math.isfinite(self.hardness_temperature) or self.hardness_temperature <= 0:
            raise ValueError("hardness_temperature must be finite and positive")
        if type(self.hardness_time_bins) is not int or self.hardness_time_bins <= 0:
            raise ValueError("hardness_time_bins must be a positive integer")
        if self.hardness_score_action_dims is not None and (
            type(self.hardness_score_action_dims) is not int
            or not 1 <= self.hardness_score_action_dims <= self.action_dim
        ):
            raise ValueError("hardness_score_action_dims must be in [1, action_dim]")
        if self.action_motion_weighting:
            if (
                type(self.action_motion_joint_dims) is not int
                or not 1 <= self.action_motion_joint_dims <= self.action_dim
            ):
                raise ValueError("action_motion_joint_dims must be in [1, action_dim]")
            if self.action_motion_joint_scales is None:
                raise ValueError("action_motion_weighting requires action_motion_joint_scales")
            if len(self.action_motion_joint_scales) != self.action_motion_joint_dims:
                raise ValueError("action_motion_joint_scales must match action_motion_joint_dims")
            if any(not math.isfinite(value) or value <= 0 for value in self.action_motion_joint_scales):
                raise ValueError("action_motion_joint_scales must contain positive finite values")
            if self.action_motion_gripper_index is not None and (
                type(self.action_motion_gripper_index) is not int
                or not 0 <= self.action_motion_gripper_index < self.action_dim
            ):
                raise ValueError("action_motion_gripper_index must be in [0, action_dim)")
            if (
                not math.isfinite(self.action_motion_gripper_flip_threshold)
                or self.action_motion_gripper_flip_threshold <= 0
            ):
                raise ValueError("action_motion_gripper_flip_threshold must be positive and finite")
            if not math.isfinite(self.action_motion_min_weight) or self.action_motion_min_weight <= 0:
                raise ValueError("action_motion_min_weight must be positive and finite")
            if (
                not math.isfinite(self.action_motion_max_weight)
                or self.action_motion_max_weight < self.action_motion_min_weight
            ):
                raise ValueError("action_motion_max_weight must be finite and >= action_motion_min_weight")
        if (self.memory_future_latent_mean is None) != (self.memory_future_latent_std is None):
            raise ValueError("memory future latent mean/std must be provided together")
        if self.memory_future_latent_mean is not None:
            if len(self.memory_future_latent_mean) != self.memory_future_latent_dim:
                raise ValueError("memory_future_latent_mean must match memory_future_latent_dim")
            if len(self.memory_future_latent_std) != self.memory_future_latent_dim:
                raise ValueError("memory_future_latent_std must match memory_future_latent_dim")
        if self.use_lam_memory and self.memory_latent_dim <= 0:
            raise ValueError("memory_latent_dim must be positive when memory is enabled")
        if type(self.memory_stride) is not int or self.memory_stride <= 0:
            raise ValueError("memory_stride must be a positive integer")
        if self.memory_frames_per_token <= 0:
            raise ValueError("memory_frames_per_token must be positive")
        if self.memory_pooling_mode not in ("concatenate", "mean"):
            raise ValueError("memory_pooling_mode must be 'concatenate' or 'mean'")
        if self.use_lam_memory and self.memory_projector_hidden_dim <= 0:
            raise ValueError("memory_projector_hidden_dim must be positive when memory is enabled")
        if type(self.memory_segment_vocab_size) is not int or self.memory_segment_vocab_size < 4:
            raise ValueError("memory_segment_vocab_size must be an integer >= 4")
        for name, type_id in (
            ("memory_demo_anchor_type_id", self.memory_demo_anchor_type_id),
            ("memory_execution_anchor_type_id", self.memory_execution_anchor_type_id),
            ("memory_demo_lam_boundary_type_id", self.memory_demo_lam_boundary_type_id),
        ):
            if type(type_id) is not int or not 0 <= type_id < self.memory_segment_vocab_size:
                raise ValueError(f"{name} must be in [0, memory_segment_vocab_size)")
        if self.memory_demo_lam_boundary_token:
            if not self.memory_demo_anchor:
                raise ValueError("memory_demo_lam_boundary_token requires memory_demo_anchor")
            if not self.memory_segment_embedding or not self.memory_segment_embedding_after_projection:
                raise ValueError(
                    "memory_demo_lam_boundary_token requires typed memory_segment_embedding_after_projection"
                )
            if not self.memory_causal_attention:
                raise ValueError("memory_demo_lam_boundary_token requires memory_causal_attention")
            if self.memory_frames_per_token != 1:
                raise ValueError("memory_demo_lam_boundary_token requires memory_frames_per_token=1")
        if self.online_memory_lam:
            if self.memory_horizon <= 0:
                raise ValueError("online_memory_lam requires memory_horizon > 0")
            if not self.online_memory_lam_checkpoint:
                raise ValueError("online_memory_lam requires online_memory_lam_checkpoint")
            if type(self.online_memory_lam_pair_batch_size) is not int or self.online_memory_lam_pair_batch_size <= 0:
                raise ValueError("online_memory_lam_pair_batch_size must be a positive integer")
            if self.memory_latent_dim != 64:
                raise ValueError("online two-view CD-LAM requires memory_latent_dim=64")
            if self.memory_frames_per_token != 1:
                raise ValueError("online two-view CD-LAM requires framewise memory_frames_per_token=1")
            if self.memory_stride != 1:
                raise ValueError("online two-view CD-LAM does not support memory_stride != 1")
        if not math.isfinite(self.memory_token_dropout_rate) or not 0 <= self.memory_token_dropout_rate < 1:
            raise ValueError("memory_token_dropout_rate must be finite and in [0, 1)")
        if self.memory_fixed_random_latent_seed is not None:
            if type(self.memory_fixed_random_latent_seed) is not int or not (
                0 <= self.memory_fixed_random_latent_seed < 2**32
            ):
                raise ValueError("memory_fixed_random_latent_seed must be an integer in [0, 2**32)")
            if self.memory_horizon <= 0:
                raise ValueError("memory_fixed_random_latent_seed requires latent memory")
        if self.share_memory_attention and self.memory_horizon == 0:
            raise ValueError("share_memory_attention requires memory_horizon > 0")
        if self.memory_use_vlm_expert and not self.use_memory:
            raise ValueError("memory_use_vlm_expert requires latent memory or memory_anchor_only")
        if self.memory_use_vlm_expert and self.share_memory_attention:
            raise ValueError("memory_use_vlm_expert is incompatible with share_memory_attention")
        if self.memory_same_prefix_block and not self.memory_use_vlm_expert:
            raise ValueError("memory_same_prefix_block requires memory_use_vlm_expert")
        if self.memory_prefix_order not in ("vision_language_memory", "language_memory_vision"):
            raise ValueError(
                "memory_prefix_order must be 'vision_language_memory' or 'language_memory_vision'"
            )
        if self.memory_prefix_order == "language_memory_vision":
            if not self.memory_use_vlm_expert:
                raise ValueError("language_memory_vision requires memory_use_vlm_expert")
            if not self.language_causal_attention or not self.memory_causal_attention:
                raise ValueError("language_memory_vision requires causal language and memory attention")
            if self.memory_same_prefix_block:
                raise ValueError("language_memory_vision is incompatible with memory_same_prefix_block")
            if self.memory_register_count:
                raise ValueError("language_memory_vision does not support memory registers")
            if self.memory_flow_horizon:
                raise ValueError("language_memory_vision does not support future-memory prediction")
            if self.memory_advantage_weighting:
                raise ValueError("language_memory_vision does not support memory_advantage_weighting")
        if type(self.memory_register_count) is not int or self.memory_register_count < 0:
            raise ValueError("memory_register_count must be a non-negative integer")
        if self.memory_register_count:
            if not self.memory_use_vlm_expert:
                raise ValueError("memory registers require memory_use_vlm_expert")
            if not self.memory_demo_anchor:
                raise ValueError("memory registers require memory_demo_anchor")
            if self.memory_episode_anchor:
                raise ValueError("memory registers do not yet support memory_episode_anchor")
            if self.memory_same_prefix_block:
                raise ValueError(
                    "memory registers use their own attention mask and require memory_same_prefix_block=False"
                )
            if self.memory_flow_horizon:
                raise ValueError("memory registers do not yet support future-memory prediction")
            if self.memory_advantage_weighting:
                raise ValueError("memory registers do not yet support memory_advantage_weighting")
        if (
            self.memory_segment_embedding
            or self.memory_demo_anchor
            or self.memory_episode_anchor
            or self.memory_execution_anchor
        ) and not self.use_memory:
            raise ValueError("memory segment/anchor inputs require memory_horizon > 0")
        if self.memory_demo_anchor_feature_pool and not self.memory_demo_anchor:
            raise ValueError("memory_demo_anchor_feature_pool requires memory_demo_anchor")
        if self.memory_execution_anchor:
            if not self.memory_demo_anchor:
                raise ValueError("memory_execution_anchor requires memory_demo_anchor")
            if not self.memory_use_vlm_expert:
                raise ValueError("memory_execution_anchor requires memory_use_vlm_expert")
            if not self.memory_segment_embedding or not self.memory_segment_embedding_after_projection:
                raise ValueError("memory_execution_anchor requires typed memory_segment_embedding_after_projection")
            if self.memory_frames_per_token != 1:
                raise ValueError("memory_execution_anchor requires memory_frames_per_token=1")
        if self.memory_segment_embedding_after_projection and not self.memory_segment_embedding:
            raise ValueError("memory_segment_embedding_after_projection requires memory_segment_embedding")
        if (
            self.memory_segment_embedding_after_projection
            and self.memory_frames_per_token != 1
            and self.memory_pooling_mode != "mean"
        ):
            raise ValueError(
                "memory_segment_embedding_after_projection requires memory_frames_per_token=1 "
                "unless memory_pooling_mode='mean'"
            )
        if (
            self.memory_pooling_mode == "mean"
            and self.memory_segment_embedding
            and not self.memory_segment_embedding_after_projection
        ):
            raise ValueError("typed mean-pooled memory requires memory_segment_embedding_after_projection")
        if self.memory_segment_embedding and self.memory_frames_per_token != 1 and not self.memory_use_vlm_expert:
            raise ValueError(
                "grouped memory_segment_embedding requires memory_use_vlm_expert; "
                "the separate memory expert only supports memory_frames_per_token=1"
            )
        if (self.memory_latent_mean is None) != (self.memory_latent_std is None):
            raise ValueError("memory_latent_mean and memory_latent_std must be provided together")
        if self.memory_latent_mean is not None:
            if len(self.memory_latent_mean) != self.memory_latent_dim:
                raise ValueError(
                    f"memory_latent_mean must have {self.memory_latent_dim} values, got {len(self.memory_latent_mean)}"
                )
            if len(self.memory_latent_std) != self.memory_latent_dim:
                raise ValueError(
                    f"memory_latent_std must have {self.memory_latent_dim} values, got {len(self.memory_latent_std)}"
                )
            if not all(math.isfinite(value) for value in (*self.memory_latent_mean, *self.memory_latent_std)):
                raise ValueError("memory latent normalization values must be finite")
            if not all(value > 0 for value in self.memory_latent_std):
                raise ValueError("memory_latent_std values must be positive")

    @property
    def use_memory(self) -> bool:
        return self.use_lam_memory or self.memory_anchor_only

    @property
    def use_lam_memory(self) -> bool:
        return self.memory_horizon > 0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                memory_latents=(
                    jax.ShapeDtypeStruct([batch_size, self.memory_horizon, self.memory_latent_dim], jnp.float32)
                    if self.use_lam_memory
                    else None
                ),
                memory_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.memory_horizon], jnp.bool_)
                    if self.use_lam_memory
                    else None
                ),
                future_memory_latents=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.memory_flow_horizon, self.memory_future_latent_dim], jnp.float32
                    )
                    if self.memory_flow_horizon
                    else None
                ),
                future_memory_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.memory_flow_horizon], jnp.bool_)
                    if self.memory_flow_horizon
                    else None
                ),
                memory_segment_ids=(
                    jax.ShapeDtypeStruct([batch_size, self.memory_horizon], jnp.int32)
                    if self.memory_segment_embedding and self.use_lam_memory
                    else None
                ),
                demo_direction_tokens=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.memory_demo_direction_max_token_len], jnp.int32
                    )
                    if self.memory_demo_direction_generation
                    else None
                ),
                demo_direction_mask=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.memory_demo_direction_max_token_len], jnp.bool_
                    )
                    if self.memory_demo_direction_generation
                    else None
                ),
                demo_direction_loss_mask=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.memory_demo_direction_max_token_len], jnp.bool_
                    )
                    if self.memory_demo_direction_generation
                    else None
                ),
                memory_demo_start_image=(
                    image_spec if self.memory_demo_anchor and not self.memory_demo_anchor_feature_pool else None
                ),
                memory_demo_anchor_features=(
                    jax.ShapeDtypeStruct(
                        [batch_size, 256, _gemma.get_config(self.paligemma_variant).width], jnp.float16
                    )
                    if self.memory_demo_anchor_feature_pool
                    else None
                ),
                memory_demo_start_mask=(image_mask_spec if self.memory_demo_anchor else None),
                memory_episode_start_image=(image_spec if self.memory_episode_anchor else None),
                memory_episode_start_mask=(image_mask_spec if self.memory_episode_anchor else None),
                memory_execution_start_image=(image_spec if self.memory_execution_anchor else None),
                memory_execution_start_mask=(image_mask_spec if self.memory_execution_anchor else None),
                action_motion_delta=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_motion_joint_dims], jnp.float32)
                    if self.action_motion_weighting
                    else None
                ),
                action_gripper_flip=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.bool_)
                    if self.action_motion_weighting
                    else None
                ),
                baseline_uncertainty_score=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.float32)
                    if self.baseline_uncertainty_weighting
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

    def get_action_memory_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze everything except the action and memory experts/heads."""
        if not self.use_memory:
            raise ValueError("action-memory-only training requires memory_horizon > 0")
        trainable = nnx_utils.PathRegex(
            r"(?:"
            r"PaliGemma/llm/.*(?:_1|_2).*"
            r"|(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
            r"|memory_projector_in|memory_projector_out|memory_projector_norm|memory_segment_embed"
            r"|memory_register_embed"
            r"|memory_flow_in_proj|memory_flow_out_proj|memory_flow_time_mlp_in"
            r"|memory_flow_time_mlp_out|memory_flow_queries)/.*"
            r")"
        )
        return nnx.Not(trainable)

    def get_memory_auxiliary_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze everything except the memory adapter and future-prediction head."""
        if not self.use_memory:
            raise ValueError("memory-only warmup requires memory_horizon > 0")
        trainable = nnx_utils.PathRegex(
            r"(?:PaliGemma/llm/.*_2.*|memory_projector_in|memory_projector_out|memory_projector_norm|memory_segment_embed"
            r"|memory_register_embed"
            r"|memory_flow_in_proj|memory_flow_out_proj|memory_flow_time_mlp_in"
            r"|memory_flow_time_mlp_out|memory_flow_queries)/.*"
        )
        return nnx.Not(trainable)
