"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import math
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Offset, in dataset frames, between the query observation and the first
    # supervised action. Most LeRobot datasets use 0. RoboMME execution
    # training starts at the following action (offset 1).
    action_sequence_start_offset: int = 0
    # Keep every requested action position supervised near episode end. The
    # LeRobot query is right-padded with the final in-episode action and its
    # padding flags are cleared before transforms, giving an explicit
    # hold-last-state target for the complete action horizon.
    hold_last_action_targets: bool = False

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # Optional finalized cache produced by scripts/precompute_dreamdojo_memory.py.
    # This is consumed only by memory-enabled PyTorch PI0.5 models.
    memory_cache_dir: str | None = None
    # Training-only temporal-lag augmentation. For each execution query, drop
    # a uniformly sampled number of the newest execution LAM tokens while
    # preserving all demo memory, boundaries, and visual anchors.
    memory_random_drop_execution_tail_min: int = 0
    memory_random_drop_execution_tail_max: int = 0
    # Online CD-LAM carries raw frame pairs into the PyTorch model rather than
    # reading memory_cache_dir. It must equal the full model history horizon.
    online_memory_max_transitions: int = 0
    # Optional second aligned cache used only as future-latent supervision.
    # This allows, for example, DeltaTok history with 32-D CD-LAM futures.
    future_memory_cache_dir: str | None = None
    # Optional row-aligned cache of no-memory policy uncertainty quantiles.
    # Each execution query contributes one score per supervised physical
    # action position. The model turns these fixed scores into loss weights.
    baseline_uncertainty_cache_dir: str | None = None
    # Optional sidecar built from the full 20-noise baseline export. When set,
    # execution queries are sampled with replacement in proportion to their
    # aggregated action-trajectory diversity rather than being loss-reweighted.
    diversity_sampling_cache_dir: str | None = None
    # Global empirical-diversity rank separating the low and high sampling
    # groups. High-group ranks are rescaled to [0, 1] before exponent.
    diversity_sampling_rank_threshold: float = 0.5
    diversity_sampling_rank_power: float = 1.0
    # Optional total replacement-sampling mass allocated uniformly to the
    # rank <= diversity_sampling_rank_threshold group. Zero reproduces the
    # original high-diversity-only sampler exactly.
    diversity_sampling_low_rank_mass: float = 0.0
    # Optional, atomically published plan produced from a fixed diversity-rank
    # validation pack. When present it replaces the static diversity-rank
    # formula with normalized-action-MSE decile weighting. A uniform component
    # keeps every execution query eligible for replacement sampling.
    diversity_sampling_mse_plan_path: str | None = None
    diversity_sampling_uniform_mass: float = 0.05
    # When set, reload the atomically published plan named ``step%06d.json``
    # at fixed physical-global-batch boundaries. This changes only the
    # dataloader's replacement distribution; model, optimizer, and compiled
    # train step remain resident.
    diversity_sampling_mse_plan_dir: str | None = None
    diversity_sampling_mse_plan_start_step: int = 0
    diversity_sampling_mse_plan_interval_batches: int = 0
    # Restrict training/norm-stat queries to execution frames. Demo frames stay
    # available to the memory wrapper but can never become action-loss targets.
    execution_only: bool = False
    # Optional task-index subset for execution queries. The underlying dataset
    # remains complete so demo/history memory and aligned caches are untouched.
    execution_only_task_indices: tuple[int, ...] | None = None
    # Optional episode-index subset for execution queries. This is needed when
    # different environments reuse the same natural-language task indices.
    execution_only_episode_indices: tuple[int, ...] | None = None
    # Optional per-episode-block probability mass for replacement sampling of
    # execution queries. Each entry is (first_episode, last_episode, mass).
    # Within a block every eligible execution query is uniform, so blocks with
    # different durations still receive exactly their requested total mass.
    task_sampling_episode_weights: tuple[tuple[int, int, float], ...] | None = None
    # Optional mmap cache containing either one decoded first-demo frame, one
    # temporal mean of per-frame SigLIP patch features, or schema-v2 demo plus
    # execution-start images.
    demo_anchor_cache_dir: str | None = None
    # Optional mmap cache containing the raw cam_high frame at timestep zero
    # for every episode. This is independent of demonstration metadata.
    episode_anchor_cache_dir: str | None = None
    # Optional strictly validated sidecar containing row-aligned demo/execution
    # boundaries and transition segment ids. Demo rows remain available as
    # history but are excluded from action-loss queries.
    phase_metadata_cache_dir: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                            demo_direction_max_token_len=(
                                model_config.memory_demo_direction_max_token_len
                                if model_config.memory_demo_direction_generation
                                else None
                            ),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    # Number of leading absolute-action dimensions converted to deltas. The
    # final dimension is normally an absolute gripper command.
    delta_action_dims: int = 6
    action_output_dim: int = 7

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        # LeRobot v2.1 uses the standard `action` and
        # `observation.images.*` fields. Older converted LIBERO exports used
        # openpi-specific aliases, so retain that mapping for those datasets.
        dataset_root = pathlib.Path(self.repo_id) if self.repo_id else None
        is_lerobot_v21 = bool(
            dataset_root
            and (dataset_root / "meta" / "info.json").is_file()
            and "observation.images.front"
            in json.loads((dataset_root / "meta" / "info.json").read_text()).get("features", {})
        )
        if is_lerobot_v21:
            repack_structure = {
                "observation/image": "observation.images.front",
                "observation/wrist_image": "observation.images.wrist",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            }
            action_sequence_keys = ("action",)
        else:
            repack_structure = {
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }
            action_sequence_keys = ("actions",)
        if getattr(model_config, "use_memory", False):
            if getattr(model_config, "use_lam_memory", True):
                repack_structure["memory_latents"] = "memory_latents"
                repack_structure["memory_mask"] = "memory_mask"
            if getattr(model_config, "memory_flow_horizon", 0):
                repack_structure["future_memory_latents"] = "future_memory_latents"
                repack_structure["future_memory_mask"] = "future_memory_mask"
            if getattr(model_config, "memory_segment_embedding", False) and getattr(
                model_config, "use_lam_memory", True
            ):
                repack_structure["memory_segment_ids"] = "memory_segment_ids"
            if getattr(model_config, "memory_demo_anchor", False):
                if getattr(model_config, "memory_demo_anchor_feature_pool", False):
                    repack_structure["memory_demo_anchor_features"] = "memory_demo_anchor_features"
                else:
                    repack_structure["memory_demo_start_image"] = "memory_demo_start_image"
                repack_structure["memory_demo_start_mask"] = "memory_demo_start_mask"
            if getattr(model_config, "memory_execution_anchor", False):
                repack_structure["memory_execution_start_image"] = "memory_execution_start_image"
                repack_structure["memory_execution_start_mask"] = "memory_execution_start_mask"
            if getattr(model_config, "memory_demo_direction_generation", False):
                repack_structure["demo_direction_text"] = "demo_direction_text"
        if getattr(model_config, "baseline_uncertainty_weighting", False):
            repack_structure["baseline_uncertainty_score"] = "baseline_uncertainty_score"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        input_transforms: list[_transforms.DataTransformFn] = [
            libero_policy.LiberoInputs(model_type=model_config.model_type)
        ]
        if getattr(model_config, "action_motion_weighting", False):
            input_transforms.append(
                _transforms.ActionMotionTargets(
                    joint_action_dims=model_config.action_motion_joint_dims,
                    gripper_index=model_config.action_motion_gripper_index,
                    gripper_flip_threshold=model_config.action_motion_gripper_flip_threshold,
                )
            )
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[libero_policy.LiberoOutputs(action_dim=self.action_output_dim)],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            if self.delta_action_dims <= 0:
                raise ValueError("delta_action_dims must be positive when extra_delta_transform is enabled")
            delta_action_mask = _transforms.make_bool_mask(self.delta_action_dims, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of equal per-rank microbatches used for one optimizer step. The
    # data loader still emits the full global batch so sampler semantics stay
    # unchanged; PyTorch training slices each local batch before the forward.
    gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of batches loaded in advance by each data loader worker. This is
    # only used when num_workers is greater than zero.
    prefetch_factor: int = 2
    # Preserve the sampler's exact batch order when using multiprocessing.
    # Disabling this avoids head-of-line blocking when one worker finishes a
    # much heavier memory batch later than workers producing subsequent batches.
    data_loader_in_order: bool = True
    # Number of global batches in each sortish memory-length bucket. This keeps
    # full-history memory batches balanced across DDP ranks without a curriculum.
    memory_bucket_size_multiplier: int = 50
    # Quantize dynamically padded memory lengths to a small set of shapes. This
    # lets CUDA reuse allocator blocks/kernels instead of remapping memory for
    # nearly every full-history batch.
    memory_pad_to_multiple: int = 1
    # Drop prompt columns that are padding for every sample in a PyTorch batch.
    trim_prompt_padding: bool = False
    # Require dataset-specific latent normalization before entering the
    # training loop. Launchers may inject it from a finalized cache manifest.
    require_memory_latent_normalization: bool = False
    # If positive, only the memory projector and the memory expert RMSNorms are
    # updated for this many initial JAX optimizer steps by default. A config
    # may provide warmup_freeze_filter to select a broader static subset (for
    # example, memory projector plus action expert). All other parameters and
    # their optimizer moments remain unchanged until the staged warmup ends.
    memory_adapter_warmup_steps: int = 0
    # Multiplies the base schedule only for newly initialized memory projector
    # and typed segment-embedding parameters. The backbone schedule is unchanged.
    memory_projector_lr_multiplier: float = 1.0
    # Optional staged-warmup overrides. These are config-owned rather than CLI
    # flags because nnx filters cannot be represented reliably by tyro.
    warmup_freeze_filter: tyro.conf.Suppress[Filter | None] = None
    warmup_lr_schedule: tyro.conf.Suppress[_optimizer.LRScheduleConfig | None] = None
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # Optional exact completed-step checkpoints. When non-empty, these replace
    # save_interval and use update counts as directory names (no off-by-one).
    save_steps: tuple[int, ...] = ()
    # Optional resident fixed-pack validation. When configured, training uses
    # the live FSDP parameters at each interval to publish the next adaptive
    # replacement-sampling plan, without a separate policy process.
    adaptive_mse_eval_pack_path: str | None = None
    adaptive_mse_eval_result_dir: str | None = None
    adaptive_mse_eval_diversity_cache_dir: str | None = None
    adaptive_mse_eval_interval_steps: int = 0
    # Optional immutable checkpoint source. Unlike ``resume``, this restores
    # state from a different experiment while all new checkpoints are written
    # under ``checkpoint_dir``. This is useful for a fully isolated smoke run
    # that must exercise the exact optimizer/model state of a production
    # boundary without adding artifacts to the production experiment.
    restore_checkpoint_dir: str | None = None
    restore_checkpoint_step: int | None = None
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.memory_bucket_size_multiplier <= 0:
            raise ValueError("memory_bucket_size_multiplier must be positive")
        if self.memory_pad_to_multiple <= 0:
            raise ValueError("memory_pad_to_multiple must be positive")
        if type(self.prefetch_factor) is not int or self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer")
        if type(self.data_loader_in_order) is not bool:
            raise ValueError("data_loader_in_order must be a boolean")
        if type(self.gradient_accumulation_steps) is not int or self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if type(self.memory_adapter_warmup_steps) is not int or self.memory_adapter_warmup_steps < 0:
            raise ValueError("memory_adapter_warmup_steps must be a non-negative integer")
        if self.memory_adapter_warmup_steps and not getattr(self.model, "use_memory", False):
            raise ValueError("memory_adapter_warmup_steps requires a memory-enabled model")
        if not math.isfinite(self.memory_projector_lr_multiplier) or self.memory_projector_lr_multiplier <= 0:
            raise ValueError("memory_projector_lr_multiplier must be a positive finite number")
        if self.memory_adapter_warmup_steps and self.memory_projector_lr_multiplier != 1.0:
            raise ValueError("memory_projector_lr_multiplier cannot be combined with memory_adapter_warmup_steps")
        if any(type(step) is not int or step <= 0 for step in self.save_steps):
            raise ValueError("save_steps must contain only positive integers")
        if tuple(sorted(set(self.save_steps))) != self.save_steps:
            raise ValueError("save_steps must be strictly increasing and unique")
        if self.save_steps and self.save_steps[-1] > self.num_train_steps:
            raise ValueError("save_steps cannot exceed num_train_steps")
        adaptive_paths = (
            self.adaptive_mse_eval_pack_path,
            self.adaptive_mse_eval_result_dir,
            self.adaptive_mse_eval_diversity_cache_dir,
        )
        if any(path is not None for path in adaptive_paths):
            if any(path is None for path in adaptive_paths):
                raise ValueError("resident adaptive MSE evaluation requires pack, result, and diversity paths")
            if self.adaptive_mse_eval_interval_steps <= 0:
                raise ValueError("resident adaptive MSE evaluation interval must be positive")
        elif self.adaptive_mse_eval_interval_steps:
            raise ValueError("resident adaptive MSE evaluation interval requires its paths")
        if self.restore_checkpoint_dir is not None:
            if self.resume:
                raise ValueError("restore_checkpoint_dir and resume are mutually exclusive")
            if self.restore_checkpoint_step is None or self.restore_checkpoint_step <= 0:
                raise ValueError("restore_checkpoint_dir requires a positive restore_checkpoint_step")
        elif self.restore_checkpoint_step is not None:
            raise ValueError("restore_checkpoint_step requires restore_checkpoint_dir")


_LIBERO_PLUS_DATASET_ROOT = "Sylvest/libero_plus_lerobot"
_LIBERO_PLUS_MEMORY_CACHE_ROOT = "external/libero/libero_plus_lerobot_v2_cdlam_memory"
_LIBERO_PLUS_MEMORY_WRIST64_CACHE_ROOT = (
    "external/libero/libero_plus_lerobot_v2_cdlam_memory_wrist64_seqcrop_v1"
)
_LIBERO_PLUS_ASSETS_ROOT = "external/memory/assets/pi05_libero_plus_default"
_LIBERO_PLUS_ASSET_ID = "lerobot/libero_plus"
_LIBERO_PLUS_PI05_BASE_PARAMS = os.environ.get("PI05_BASE_PARAMS", "external/pi05-params")

_LIBERO_MEM_DATASET_ROOT = "external/libero/LIBERO-Mem_lerobot_v21"
_LIBERO_MEM_MEMORY_CACHE_ROOT = "external/libero/LIBERO-Mem_lerobot_v21_cdlam_agent_clean_v1"
_LIBERO_MEM_ASSETS_ROOT = "external/memory/assets/pi05_libero_mem_memory_clean32_framewise_default"
_LIBERO_MEM_ASSET_ID = "lerobot/libero_mem"

_ROBOMME_DATASET_ROOT = "external/robomme/patternlock_unique_paths_1000_lerobot_v1"
_ROBOMME_MEMORY_CACHE_ROOT = "external/robomme/robomme_data_lerobot_cdlam_head_clean_v1"
_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT = (
    "external/robomme/robomme_data_lerobot_dreamdojo_lam400k_head_wrist64_fp32_v1"
)
_ROBOMME_CDLAM_HEAD_WRIST64_MEMORY_CACHE_ROOT = (
    "external/robomme/patternlock_unique_paths_1000_cdlam_head_wrist64_v1"
)
_ROBOMME_HEAD_WRIST64_GAP4_MEMORY_CACHE_ROOT = (
    "external/robomme/robomme_data_lerobot_cdlam_head_wrist64_gap4_clean_v1"
)
_ROBOMME_DELTATOK_CACHE_ROOT = "external/robomme/robomme_data_lerobot_deltatok_kinetics_head_v1"
_ROBOMME_DELTATOK_FP32_CACHE_ROOT = "external/robomme/robomme_data_lerobot_deltatok_kinetics_head_fp32_v1"
_ROBOMME_DELTATOK_HEAD_WRIST1536_FP32_CACHE_ROOT = (
    "external/robomme/robomme_data_lerobot_deltatok_kinetics_head_wrist1536_fp32_v1"
)
_ROBOMME_ASSETS_ROOT = "external/memory/assets/pi05_robomme_memory_clean32_framewise_default"
_ROBOMME_ASSET_ID = "lerobot/robomme"
_ROBOMME_DEMO_ANCHOR_CACHE_ROOT = "external/robomme/robomme_demo_anchor_head_v1"
_ROBOMME_CDLAM_ANCHOR_H20_100K_PARAMS = (
    "external/memory/checkpoints/"
    "pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_h20_offset1_default/"
    "pi05_robomme_frame1_h512_h20_offset1_uniform_warmup1k_cosine100k_seed0_v2/100000/params"
)
_ROBOMME_SWAP_EPISODE_INDICES = tuple(range(100, 200)) + tuple(range(400, 500))
_ROBOMME_BASELINE_UNCERTAINTY_CACHE_ROOT = (
    "external/robomme/analysis/pi05_baseline_full20noise_uncertainty_physical10_v1"
)
_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT = (
    "external/robomme/analysis/pi05_baseline_full20noise_uncertainty_physical20_offset1_v2"
)
_ROBOMME_ADAPTIVE_MSE_DECILE_PLAN_PATH = (
    "external/memory/artifacts/"
    "robomme_cdlam_head_wrist64_nostate_gap1_adaptive_mse_deciles_resume70k_v1/"
    "adaptive_decile_sampling_plan.json"
)
_ROBOMME_ADAPTIVE_MSE_DECILE_PLAN_DIR = (
    "external/memory/artifacts/"
    "robomme_cdlam_head_wrist64_nostate_gap1_adaptive_mse_deciles_resume70k_v1/"
    "adaptive_decile_sampling_plans"
)
_ROBOMME_ADAPTIVE_MSE_EVAL_PACK = (
    "external/robomme/evaluation_artifacts/"
    "pi05_robomme_headwrist64_nostate_dualanchor_gap1_adaptive_mse10decile_uniform5pct_"
    "projector4x_frame1_h512_h20_offset1_warmup1k_cosine100k_seed0_resume70k_v1/"
    "20000-msepack512x2"
)
_ROBOMME_ADAPTIVE_MSE_EVAL_RESULT_DIR = (
    "external/robomme/evaluations/"
    "pi05_robomme_headwrist64_nostate_dualanchor_gap1_adaptive_mse10decile_uniform5pct_"
    "projector4x_frame1_h512_h20_offset1_warmup1k_cosine100k_seed0_resume70k_v1/"
    "resident_train_state"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_PLAN_DIR = (
    "external/memory/artifacts/"
    "robomme_cdlam_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100full_v9_blockprefetch/"
    "adaptive_decile_sampling_plans"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_EVAL_RESULT_DIR = (
    "external/robomme/evaluations/"
    "pi05_robomme_headwrist64_nostate_dualanchor_gap1_adaptive_mse10decile_uniform5pct_"
    "projector4x_frame1_h512_h20_offset1_warmup1k_cosine100k_seed0_resume70k_smoke100full_v9_blockprefetch/"
    "resident_train_state"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_V10_PLAN_DIR = (
    "external/memory/artifacts/"
    "robomme_cdlam_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100full_v10_blockprefetch_shutdown/"
    "adaptive_decile_sampling_plans"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_V10_EVAL_RESULT_DIR = (
    "external/robomme/evaluations/"
    "pi05_robomme_headwrist64_nostate_dualanchor_gap1_adaptive_mse10decile_uniform5pct_"
    "projector4x_frame1_h512_h20_offset1_warmup1k_cosine100k_seed0_resume70k_smoke100full_v10_blockprefetch_shutdown/"
    "resident_train_state"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_V11_PLAN_DIR = (
    "external/memory/artifacts/"
    "robomme_cdlam_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100full_v11_blockprefetch_workerexit/"
    "adaptive_decile_sampling_plans"
)
_ROBOMME_ADAPTIVE_MSE_SMOKE_V11_EVAL_RESULT_DIR = (
    "external/robomme/evaluations/"
    "pi05_robomme_headwrist64_nostate_dualanchor_gap1_adaptive_mse10decile_uniform5pct_"
    "projector4x_frame1_h512_h20_offset1_warmup1k_cosine100k_seed0_resume70k_smoke100full_v11_blockprefetch_workerexit/"
    "resident_train_state"
)
_ROBOMME_DUAL_ANCHOR_CACHE_ROOT = "external/robomme/robomme_dual_anchor_head_v2"
_ROBOMME_PATTERNLOCK_UNIQUE1000_DUAL_ANCHOR_CACHE_ROOT = (
    "external/robomme/patternlock_unique_paths_1000_dual_anchor_head_v1"
)
_ROBOMME_16X1000_DATASET_ROOT = os.environ.get(
    "ROBOMME_VIDEO_DATASET", "external/robomme_16x1000_lerobot_video_h264_v1"
)
_ROBOMME_16X1000_DREAMDOJO_HEAD_WRIST64_MEMORY_CACHE_ROOT = (
    os.environ.get("ROBOMME_MEMORY_CACHE", "external/robomme_16x1000_dreamdojo_lam400k_head_wrist64_fp32_v1")
)
_ROBOMME_16X1000_DUAL_ANCHOR_CACHE_ROOT = (
    os.environ.get("ROBOMME_ANCHOR_CACHE", "external/robomme_16x1000_dual_anchor_head_v1")
)
_ROBOMME_16X1000_ASSETS_ROOT = (
    os.environ.get("ROBOMME_ASSETS_ROOT", "external/assets")
)
_ROBOMME_16X1000_ASSET_ID = os.environ.get("ROBOMME_ASSET_ID", "lerobot/robomme_16x1000")
_ROBOMME_DEMO_ANCHOR_FEATURE_MEAN_CACHE_ROOT = (
    "external/robomme/robomme_demo_anchor_head_siglip_feature_mean_v1"
)
_ROBODOJO_DATASET_ROOT = (
    "external/robodojo/data/lerobot_v21/robodojo_memory_all_arx_x5_cdlam_anchor_v21"
)
_ROBODOJO_MEMORY_CACHE_ROOT = "external/robodojo/data/cdlam_adjacent_cam_high_av1_v1"
_ROBODOJO_EPISODE_ANCHOR_CACHE_ROOT = "external/robodojo/data/cam_high_frame0_anchor_png224_v1"
_ROBODOJO_PHASE_METADATA_CACHE_ROOT = "external/robodojo/data/robodojo_phase_anchor_v1"
_ROBODOJO_ASSETS_ROOT = (
    "external/robodojo/cdlam_pi05/openpi_history_expert0/"
    "assets/pi05_robodojo_cdlam_history_expert0"
)
_ROBODOJO_ASSET_ID = "robodojo_memory_all_arx_x5_cdlam_anchor_v21"


def _finalized_memory_normalization(cache_root: str) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Read normalization only after the adjacent-frame cache is complete."""
    manifest_path = pathlib.Path(cache_root) / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        return None, None
    if manifest.get("alignment") != "row_0_invalid_zero; row_i=z_mu(frame_i-1,frame_i), i>=1":
        raise ValueError(f"Unexpected RoboDojo CD-LAM alignment: {manifest.get('alignment')!r}")
    normalization = manifest.get("latent_normalization")
    if not isinstance(normalization, dict):
        return None, None
    mean = tuple(float(value) for value in normalization.get("mean", ()))
    std = tuple(float(value) for value in normalization.get("std", ()))
    if len(mean) != 32 or len(std) != 32:
        raise ValueError("RoboDojo CD-LAM normalization must contain 32-D mean/std")
    return mean, std


_ROBODOJO_MEMORY_MEAN, _ROBODOJO_MEMORY_STD = _finalized_memory_normalization(_ROBODOJO_MEMORY_CACHE_ROOT)
_ROBOMME_DELTATOK_INDEPENDENT_ACTION1_30K_PARAMS = (
    "external/memory/checkpoints/"
    "pi05_robomme_memory_exec_only_anchor_deltatok768_framewise_independent_h1024_default/"
    "pi05_robomme_deltatok768_exec_anchor_action1_full_dp_b128_notriton_fromscratch_nocmd_v4/"
    "30000/params"
)
_ROBOMME_MAW_PI05BASE_ACTION1_30K_PARAMS = (
    "external/memory/checkpoints/"
    "pi05_robomme_memory_exec_only_anchor_deltatok768_expert0_nofuture_h1024_memory_advantage_k8_default/"
    "pi05_robomme_deltatok768_expert0_nofuture_maw_k8_action1_pi05base_fsdp8_b128_mem090_30k_v1/"
    "30000/params"
)
_ROBOMME_EXEC_ASSETS_ROOT = (
    "external/memory/assets/pi05_robomme_memory_exec_only_anchor_clean32_framewise_action1_default"
)


def _pi05_libero_plus_default_config(
    name: str,
    *,
    use_memory: bool,
    share_memory_attention: bool = False,
    memory_latent_dim: int = 32,
    memory_frames_per_token: int = 1,
    memory_cache_dir: str = _LIBERO_PLUS_MEMORY_CACHE_ROOT,
    train_action_memory_only: bool = False,
) -> TrainConfig:
    """Build the paired LIBERO-plus configs from one shared hyperparameter definition."""
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        # The official 40 LIBERO-plus training prompts need at most 21
        # PaliGemma tokens, including BOS/newline.
        max_token_len=21,
        pytorch_compile_mode=None,
        # The longest official training episode has 505 frames.
        memory_horizon=504 if use_memory else 0,
        share_memory_attention=share_memory_attention,
        active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        memory_latent_dim=memory_latent_dim,
        memory_frames_per_token=memory_frames_per_token,
        memory_projector_hidden_dim=512,
        # The sequential launcher injects these from the finalized cache
        # manifest. They must never be copied from another dataset.
        memory_latent_mean=None,
        memory_latent_std=None,
    )
    return TrainConfig(
        name=name,
        model=model,
        data=LeRobotLiberoDataConfig(
            repo_id=_LIBERO_PLUS_DATASET_ROOT,
            assets=AssetsConfig(
                assets_dir=_LIBERO_PLUS_ASSETS_ROOT,
                asset_id=_LIBERO_PLUS_ASSET_ID,
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=memory_cache_dir if use_memory else None,
            ),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=model.get_action_memory_freeze_filter() if train_action_memory_only else nnx.Nothing,
        weight_loader=weight_loaders.CheckpointWeightLoader(_LIBERO_PLUS_PI05_BASE_PARAMS),
        pytorch_weight_path="external/memory/checkpoints/pi05_base_pytorch",
        require_memory_latent_normalization=use_memory,
        num_train_steps=30_000,
    )


def _pi05_libero_mem_memory_clean32_framewise_default_config() -> TrainConfig:
    """LIBERO-Mem with clean front-camera 32D CD-LAM memory, one token per frame."""
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        # The longest of the ten LIBERO-Mem prompts occupies 18 tokens.
        max_token_len=18,
        pytorch_compile_mode=None,
        # The longest episode has 677 frames, hence 676 causal transitions.
        memory_horizon=676,
        share_memory_attention=False,
        active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        memory_latent_dim=32,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=512,
        # Injected from the finalized clean-cache manifest by the launcher.
        memory_latent_mean=None,
        memory_latent_std=None,
    )
    return TrainConfig(
        name="pi05_libero_mem_memory_clean32_framewise_default",
        model=model,
        data=LeRobotLiberoDataConfig(
            repo_id=_LIBERO_MEM_DATASET_ROOT,
            assets=AssetsConfig(
                assets_dir=_LIBERO_MEM_ASSETS_ROOT,
                asset_id=_LIBERO_MEM_ASSET_ID,
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=_LIBERO_MEM_MEMORY_CACHE_ROOT,
            ),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=nnx.Nothing,
        weight_loader=weight_loaders.CheckpointWeightLoader(_LIBERO_PLUS_PI05_BASE_PARAMS),
        require_memory_latent_normalization=True,
        num_train_steps=30_000,
    )


def _pi05_robomme_memory_clean32_framewise_default_config(
    *,
    name: str = "pi05_robomme_memory_clean32_framewise_default",
    memory_latent_dim: int = 32,
    memory_projector_hidden_dim: int = 512,
    memory_cache_dir: str = _ROBOMME_MEMORY_CACHE_ROOT,
) -> TrainConfig:
    """RoboMME with clean head-camera framewise memory, one token per frame."""
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        # Exact maximum over the 116 released RoboMME task strings.
        max_token_len=48,
        pytorch_compile_mode=None,
        # The longest released episode has 1411 frames.
        memory_horizon=1410,
        share_memory_attention=False,
        active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        memory_latent_dim=memory_latent_dim,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=memory_projector_hidden_dim,
        # Injected from the finalized dataset-specific cache by the launcher.
        memory_latent_mean=None,
        memory_latent_std=None,
    )
    return TrainConfig(
        name=name,
        model=model,
        data=LeRobotLiberoDataConfig(
            repo_id=_ROBOMME_DATASET_ROOT,
            assets=AssetsConfig(
                assets_dir=_ROBOMME_ASSETS_ROOT,
                asset_id=_ROBOMME_ASSET_ID,
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=memory_cache_dir,
            ),
            # RoboMME stores seven absolute joint targets followed by one
            # absolute gripper command. PI0.5 learns joint deltas.
            extra_delta_transform=True,
            delta_action_dims=7,
            action_output_dim=8,
        ),
        # At the 1410-token maximum, 16 samples/device is comparable to the
        # validated LIBERO-Mem token budget while retaining pure 8-way DP.
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=nnx.Nothing,
        weight_loader=weight_loaders.CheckpointWeightLoader(_LIBERO_PLUS_PI05_BASE_PARAMS),
        require_memory_latent_normalization=True,
        num_train_steps=30_000,
    )


def _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
    *,
    name: str = "pi05_robomme_memory_exec_only_anchor_clean32_framewise_default",
    dataset_root: str = _ROBOMME_DATASET_ROOT,
    assets_root: str = _ROBOMME_EXEC_ASSETS_ROOT,
    asset_id: str = _ROBOMME_ASSET_ID,
    memory_use_vlm_expert: bool = False,
    discrete_state_input: bool = False,
    memory_latent_dim: int = 32,
    max_token_len: int = 48,
    memory_horizon: int = 1410,
    memory_anchor_only: bool = False,
    memory_stride: int = 1,
    memory_frames_per_token: int = 1,
    memory_projector_hidden_dim: int = 512,
    memory_fixed_random_latent_seed: int | None = None,
    memory_prefix_order: str = "vision_language_memory",
    language_causal_attention: bool = False,
    memory_causal_attention: bool = False,
    memory_register_count: int = 0,
    memory_segment_vocab_size: int = 4,
    memory_demo_anchor_type_id: int = 1,
    memory_execution_anchor_type_id: int = 2,
    memory_demo_lam_boundary_token: bool = False,
    memory_demo_lam_boundary_type_id: int = 2,
    memory_segment_embedding_after_projection: bool = False,
    memory_cache_dir: str | None = _ROBOMME_MEMORY_CACHE_ROOT,
    memory_random_drop_execution_tail_min: int = 0,
    memory_random_drop_execution_tail_max: int = 0,
    future_memory_cache_dir: str | None = None,
    demo_anchor_cache_dir: str = _ROBOMME_DEMO_ANCHOR_CACHE_ROOT,
    memory_demo_anchor: bool = True,
    memory_demo_anchor_feature_pool: bool = False,
    memory_execution_anchor: bool = False,
    normalize_memory_latents: bool = True,
    action_memory_warmup_steps: int = 0,
    memory_flow_horizon: int = 0,
    memory_future_latent_dim: int | None = None,
    memory_future_prediction_mode: str = "flow_matching",
    memory_future_training_only: bool = False,
    memory_future_use_separate_expert: bool = False,
    memory_future_use_learned_queries: bool = True,
    memory_future_use_adarms_time_conditioning: bool = False,
    memory_flow_loss_weight: float = 1.0,
    memory_demo_direction_generation: bool = False,
    memory_demo_direction_max_token_len: int = 32,
    memory_demo_direction_loss_weight: float = 1.0,
    memory_demo_direction_condition_action: bool = False,
    random_init_memory_expert: bool = False,
    memory_auxiliary_warmup_only: bool = False,
    initial_params_path: str = _LIBERO_PLUS_PI05_BASE_PARAMS,
    skip_mismatched_checkpoint_shapes: bool = False,
    force_initialize_memory_expert_from_action: bool = False,
    initialize_memory_expert_norms_from_action: bool = False,
    force_initialize_memory_flow_heads_from_action: bool = False,
    memory_advantage_weighting: bool = False,
    memory_advantage_recent_steps: int = 8,
    memory_advantage_score_action_dims: int | None = None,
    memory_advantage_max_weight: float = 3.0,
    memory_advantage_hardness_weight: float = 0.0,
    memory_advantage_hardness_threshold: float = 0.5,
    memory_advantage_hardness_temperature: float = 0.5,
    memory_advantage_hardness_time_bins: int = 4,
    memory_advantage_warmup_fraction: float = 0.1,
    memory_advantage_ramp_fraction: float = 0.1,
    memory_advantage_recent_loss_weight: float = 0.1,
    hardness_weighting: bool = False,
    hardness_min_weight: float = 0.3,
    hardness_max_weight: float = 5.0,
    hardness_threshold: float = 0.5,
    hardness_temperature: float = 0.5,
    hardness_time_bins: int = 4,
    hardness_score_action_dims: int | None = None,
    execution_only_task_indices: tuple[int, ...] | None = None,
    execution_only_episode_indices: tuple[int, ...] | None = None,
    task_sampling_episode_weights: tuple[tuple[int, int, float], ...] | None = None,
    action_motion_weighting: bool = False,
    action_motion_joint_dims: int = 7,
    action_motion_joint_scales: tuple[float, ...] | None = None,
    action_motion_gripper_index: int | None = 7,
    action_motion_gripper_flip_threshold: float = 0.5,
    action_motion_min_weight: float = 0.3,
    action_motion_max_weight: float = 5.0,
    baseline_uncertainty_weighting: bool = False,
    baseline_uncertainty_cache_dir: str | None = None,
    baseline_uncertainty_min_weight: float = 0.25,
    baseline_uncertainty_max_weight: float = 3.25,
    baseline_uncertainty_rank_power: float = 3.0,
    diversity_sampling_cache_dir: str | None = None,
    diversity_sampling_rank_threshold: float = 0.5,
    diversity_sampling_rank_power: float = 1.0,
    diversity_sampling_low_rank_mass: float = 0.0,
    diversity_sampling_mse_plan_path: str | None = None,
    diversity_sampling_uniform_mass: float = 0.05,
    diversity_sampling_mse_plan_dir: str | None = None,
    diversity_sampling_mse_plan_start_step: int = 0,
    diversity_sampling_mse_plan_interval_batches: int = 0,
    action_horizon: int = 10,
    action_sequence_start_offset: int = 1,
    hold_last_action_targets: bool = False,
    num_train_steps: int = 30_000,
    save_steps: tuple[int, ...] = (),
) -> TrainConfig:
    """Corrected RoboMME: demo as typed memory, execution-only action loss."""
    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        discrete_state_input=discrete_state_input,
        max_token_len=max_token_len,
        pytorch_compile_mode=None,
        memory_horizon=memory_horizon,
        memory_anchor_only=memory_anchor_only,
        share_memory_attention=False,
        memory_use_vlm_expert=memory_use_vlm_expert,
        active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        memory_latent_dim=memory_latent_dim,
        memory_stride=memory_stride,
        memory_frames_per_token=memory_frames_per_token,
        memory_projector_hidden_dim=memory_projector_hidden_dim,
        memory_fixed_random_latent_seed=memory_fixed_random_latent_seed,
        memory_prefix_order=memory_prefix_order,
        language_causal_attention=language_causal_attention,
        memory_causal_attention=memory_causal_attention,
        memory_register_count=memory_register_count,
        memory_segment_embedding=True,
        memory_segment_vocab_size=memory_segment_vocab_size,
        memory_demo_anchor_type_id=memory_demo_anchor_type_id,
        memory_execution_anchor_type_id=memory_execution_anchor_type_id,
        memory_demo_lam_boundary_token=memory_demo_lam_boundary_token,
        memory_demo_lam_boundary_type_id=memory_demo_lam_boundary_type_id,
        memory_segment_embedding_after_projection=memory_segment_embedding_after_projection,
        memory_demo_anchor=memory_demo_anchor,
        memory_demo_anchor_feature_pool=memory_demo_anchor_feature_pool,
        memory_execution_anchor=memory_execution_anchor,
        memory_latent_mean=None,
        memory_latent_std=None,
        memory_flow_horizon=memory_flow_horizon,
        memory_future_latent_dim=memory_future_latent_dim,
        memory_future_prediction_mode=memory_future_prediction_mode,
        memory_future_training_only=memory_future_training_only,
        memory_future_use_separate_expert=memory_future_use_separate_expert,
        memory_future_use_learned_queries=memory_future_use_learned_queries,
        memory_future_use_adarms_time_conditioning=memory_future_use_adarms_time_conditioning,
        memory_flow_loss_weight=memory_flow_loss_weight,
        memory_demo_direction_generation=memory_demo_direction_generation,
        memory_demo_direction_max_token_len=memory_demo_direction_max_token_len,
        memory_demo_direction_loss_weight=memory_demo_direction_loss_weight,
        memory_demo_direction_condition_action=memory_demo_direction_condition_action,
        memory_advantage_weighting=memory_advantage_weighting,
        memory_advantage_recent_steps=memory_advantage_recent_steps,
        memory_advantage_score_action_dims=memory_advantage_score_action_dims,
        memory_advantage_max_weight=memory_advantage_max_weight,
        memory_advantage_hardness_weight=memory_advantage_hardness_weight,
        memory_advantage_hardness_threshold=memory_advantage_hardness_threshold,
        memory_advantage_hardness_temperature=memory_advantage_hardness_temperature,
        memory_advantage_hardness_time_bins=memory_advantage_hardness_time_bins,
        memory_advantage_warmup_fraction=memory_advantage_warmup_fraction,
        memory_advantage_ramp_fraction=memory_advantage_ramp_fraction,
        memory_advantage_recent_loss_weight=memory_advantage_recent_loss_weight,
        hardness_weighting=hardness_weighting,
        hardness_min_weight=hardness_min_weight,
        hardness_max_weight=hardness_max_weight,
        hardness_threshold=hardness_threshold,
        hardness_temperature=hardness_temperature,
        hardness_time_bins=hardness_time_bins,
        hardness_score_action_dims=hardness_score_action_dims,
        action_motion_weighting=action_motion_weighting,
        action_motion_joint_dims=action_motion_joint_dims,
        action_motion_joint_scales=action_motion_joint_scales,
        action_motion_gripper_index=action_motion_gripper_index,
        action_motion_gripper_flip_threshold=action_motion_gripper_flip_threshold,
        action_motion_min_weight=action_motion_min_weight,
        action_motion_max_weight=action_motion_max_weight,
        baseline_uncertainty_weighting=baseline_uncertainty_weighting,
        baseline_uncertainty_min_weight=baseline_uncertainty_min_weight,
        baseline_uncertainty_max_weight=baseline_uncertainty_max_weight,
        baseline_uncertainty_rank_power=baseline_uncertainty_rank_power,
    )
    return TrainConfig(
        name=name,
        model=model,
        data=LeRobotLiberoDataConfig(
            repo_id=dataset_root,
            assets=AssetsConfig(
                assets_dir=assets_root,
                asset_id=asset_id,
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=memory_cache_dir,
                memory_random_drop_execution_tail_min=memory_random_drop_execution_tail_min,
                memory_random_drop_execution_tail_max=memory_random_drop_execution_tail_max,
                future_memory_cache_dir=future_memory_cache_dir,
                baseline_uncertainty_cache_dir=baseline_uncertainty_cache_dir,
                diversity_sampling_cache_dir=diversity_sampling_cache_dir,
                diversity_sampling_rank_threshold=diversity_sampling_rank_threshold,
                diversity_sampling_rank_power=diversity_sampling_rank_power,
                diversity_sampling_low_rank_mass=diversity_sampling_low_rank_mass,
                diversity_sampling_mse_plan_path=diversity_sampling_mse_plan_path,
                diversity_sampling_uniform_mass=diversity_sampling_uniform_mass,
                diversity_sampling_mse_plan_dir=diversity_sampling_mse_plan_dir,
                diversity_sampling_mse_plan_start_step=diversity_sampling_mse_plan_start_step,
                diversity_sampling_mse_plan_interval_batches=diversity_sampling_mse_plan_interval_batches,
                execution_only=True,
                execution_only_task_indices=execution_only_task_indices,
                execution_only_episode_indices=execution_only_episode_indices,
                task_sampling_episode_weights=task_sampling_episode_weights,
                demo_anchor_cache_dir=demo_anchor_cache_dir,
                action_sequence_start_offset=action_sequence_start_offset,
                hold_last_action_targets=hold_last_action_targets,
            ),
            extra_delta_transform=True,
            delta_action_dims=7,
            action_output_dim=8,
        ),
        batch_size=128,
        lr_schedule=(
            _optimizer.ConstantSchedule(learning_rate=5e-6)
            if action_memory_warmup_steps
            else _optimizer.CosineDecaySchedule(
                warmup_steps=10_000,
                peak_lr=5e-5,
                decay_steps=1_000_000,
                decay_lr=5e-5,
            )
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=nnx.Nothing,
        memory_adapter_warmup_steps=action_memory_warmup_steps,
        warmup_freeze_filter=(
            model.get_memory_auxiliary_freeze_filter()
            if action_memory_warmup_steps and memory_auxiliary_warmup_only
            else model.get_action_memory_freeze_filter()
            if action_memory_warmup_steps
            else None
        ),
        warmup_lr_schedule=(_optimizer.ConstantSchedule(learning_rate=5e-5) if action_memory_warmup_steps else None),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            initial_params_path,
            initialize_memory_expert_from_action=not random_init_memory_expert,
            skip_mismatched_shapes=skip_mismatched_checkpoint_shapes,
            force_initialize_memory_expert_from_action=force_initialize_memory_expert_from_action,
            initialize_memory_expert_norms_from_action=initialize_memory_expert_norms_from_action,
            force_initialize_memory_flow_heads_from_action=force_initialize_memory_flow_heads_from_action,
        ),
        require_memory_latent_normalization=normalize_memory_latents,
        num_train_steps=num_train_steps,
        save_steps=save_steps,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi05_robodojo_cdlam_history_expert0",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=200,
            active_image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            # Longest episode is 1,374 frames, hence 1,373 adjacent transitions.
            memory_horizon=1373,
            memory_latent_dim=32,
            memory_frames_per_token=1,
            memory_projector_hidden_dim=512,
            memory_use_vlm_expert=True,
            memory_same_prefix_block=True,
            # Keep demo, demo->execution boundary, and execution transitions
            # distinguishable after the 32-D CD-LAM projector. One latent is
            # mapped to one token, so the segment label remains exact.
            memory_segment_embedding=True,
            memory_segment_embedding_after_projection=True,
            memory_episode_anchor=True,
            memory_latent_mean=_ROBODOJO_MEMORY_MEAN,
            memory_latent_std=_ROBODOJO_MEMORY_STD,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=_ROBODOJO_DATASET_ROOT,
            assets=AssetsConfig(
                assets_dir=_ROBODOJO_ASSETS_ROOT,
                asset_id=_ROBODOJO_ASSET_ID,
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "memory_latents": "memory_latents",
                            "memory_mask": "memory_mask",
                            "memory_segment_ids": "memory_segment_ids",
                            "memory_episode_start_image": "memory_episode_start_image",
                            "memory_episode_start_mask": "memory_episode_start_mask",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=_ROBODOJO_MEMORY_CACHE_ROOT,
                episode_anchor_cache_dir=_ROBODOJO_EPISODE_ANCHOR_CACHE_ROOT,
                phase_metadata_cache_dir=_ROBODOJO_PHASE_METADATA_CACHE_ROOT,
                # Keep demo observations in history, but never sample them as
                # action-loss queries.
                execution_only=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("external/ckpt/params"),
        seed=0,
        batch_size=128,
        fsdp_devices=8,
        memory_pad_to_multiple=32,
        require_memory_latent_normalization=True,
        num_train_steps=60_000,
        # Upload through the credential-safe sidecar; the training process
        # never reads or persists a W&B key.
        wandb_enabled=False,
        policy_metadata={
            "memory_type": "cdlam_adjacent_full_history",
            "memory_expert": 0,
            "memory_attention_block": "pi05_prefix",
            "memory_latent_dim": 32,
            "memory_projector": "32-512-2048-gelu-layernorm",
            "anchor": "raw_cam_high_episode_frame_0",
            "uses_demo": True,
            "demo_action_supervision": False,
            "memory_segment_ids": {
                "padding": 0,
                "demo": 1,
                "demo_execution_boundary": 2,
                "execution": 3,
            },
        },
    ),
    TrainConfig(
        name="pi05_robodojo_cdlam_history_expert0_pool4_h2048",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=50,
            max_token_len=200,
            active_image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            # Longest episode is 1,374 frames, hence 1,373 adjacent
            # transitions before phase-aligned mean pooling.
            memory_horizon=1373,
            memory_latent_dim=32,
            memory_frames_per_token=4,
            memory_pooling_mode="mean",
            memory_projector_hidden_dim=2048,
            memory_token_dropout_rate=0.1,
            memory_use_vlm_expert=True,
            memory_same_prefix_block=True,
            # Pool within demo, boundary, and execution segments separately so
            # every pooled token retains one exact type embedding.
            memory_segment_embedding=True,
            memory_segment_embedding_after_projection=True,
            memory_episode_anchor=True,
            # Warm-up steps use unit weights. Detached flow-error hardness is
            # enabled beginning with optimizer step 1000.
            hardness_weighting=True,
            hardness_warmup_steps=1_000,
            hardness_min_weight=0.3,
            hardness_max_weight=3.0,
            hardness_threshold=0.5,
            hardness_temperature=0.5,
            hardness_time_bins=4,
            hardness_score_action_dims=14,
            memory_latent_mean=_ROBODOJO_MEMORY_MEAN,
            memory_latent_std=_ROBODOJO_MEMORY_STD,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=_ROBODOJO_DATASET_ROOT,
            assets=AssetsConfig(
                assets_dir=_ROBODOJO_ASSETS_ROOT,
                asset_id=_ROBODOJO_ASSET_ID,
            ),
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "memory_latents": "memory_latents",
                            "memory_mask": "memory_mask",
                            "memory_segment_ids": "memory_segment_ids",
                            "memory_episode_start_image": "memory_episode_start_image",
                            "memory_episode_start_mask": "memory_episode_start_mask",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir=_ROBODOJO_MEMORY_CACHE_ROOT,
                episode_anchor_cache_dir=_ROBODOJO_EPISODE_ANCHOR_CACHE_ROOT,
                phase_metadata_cache_dir=_ROBODOJO_PHASE_METADATA_CACHE_ROOT,
                # Demo frames remain available to memory but can never become
                # action-loss queries.
                execution_only=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("external/ckpt/params"),
        seed=0,
        batch_size=256,
        gradient_accumulation_steps=1,
        fsdp_devices=8,
        # Three static raw-memory buckets (512, 1024, 1373) avoid dozens of
        # expensive JAX recompiles while preserving the same 1373-frame cap.
        memory_pad_to_multiple=512,
        require_memory_latent_normalization=True,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=30_000,
        # Upload through the credential-safe sidecar; the trainer itself never
        # receives or persists W&B/proxy credentials.
        wandb_enabled=False,
        policy_metadata={
            "memory_type": "cdlam_adjacent_full_history_pool4_mean",
            "memory_expert": 0,
            "memory_attention_block": "pi05_prefix",
            "memory_latent_dim": 32,
            "memory_pooling": "phase_aligned_mean_4_tail_1_to_3",
            "memory_projector": "32-2048-2048-gelu-layernorm",
            "memory_token_dropout_rate": 0.1,
            "anchor": "raw_cam_high_episode_frame_0",
            "uses_demo": True,
            "demo_action_supervision": False,
            "hardness_reweight": {
                "warmup_steps": 1_000,
                "raw_min_weight": 0.3,
                "raw_max_weight": 3.0,
                "score_action_dims": 14,
            },
            "memory_segment_ids": {
                "padding": 0,
                "demo": 1,
                "demo_execution_boundary": 2,
                "execution": 3,
            },
        },
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    _pi05_libero_plus_default_config(
        "pi05_libero_plus_nomemory_default",
        use_memory=False,
    ),
    _pi05_libero_plus_default_config(
        "pi05_libero_plus_memory_default",
        use_memory=True,
    ),
    _pi05_libero_plus_default_config(
        "pi05_libero_plus_memory_shared_attention_default",
        use_memory=True,
        share_memory_attention=True,
    ),
    _pi05_libero_plus_default_config(
        "pi05_libero_plus_memory_wrist64_default",
        use_memory=True,
        memory_latent_dim=64,
        memory_frames_per_token=16,
        memory_cache_dir=_LIBERO_PLUS_MEMORY_WRIST64_CACHE_ROOT,
    ),
    _pi05_libero_plus_default_config(
        "pi05_libero_plus_memory_wrist64_framewise_actionmemory_default",
        use_memory=True,
        memory_latent_dim=64,
        memory_frames_per_token=1,
        memory_cache_dir=_LIBERO_PLUS_MEMORY_WRIST64_CACHE_ROOT,
        train_action_memory_only=True,
    ),
    _pi05_libero_mem_memory_clean32_framewise_default_config(),
    _pi05_robomme_memory_clean32_framewise_default_config(),
    _pi05_robomme_memory_clean32_framewise_default_config(
        name="pi05_robomme_memory_deltatok768_framewise_controlled_default",
        memory_latent_dim=768,
        memory_projector_hidden_dim=1024,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_siglip_mean_anchor_clean32_framewise_action1_default",
        demo_anchor_cache_dir=_ROBOMME_DEMO_ANCHOR_FEATURE_MEAN_CACHE_ROOT,
        memory_demo_anchor_feature_pool=True,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_default",
        memory_use_vlm_expert=True,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_uncertainty_default",
        memory_use_vlm_expert=True,
        baseline_uncertainty_weighting=True,
        baseline_uncertainty_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_CACHE_ROOT,
        baseline_uncertainty_min_weight=0.25,
        baseline_uncertainty_max_weight=3.25,
        baseline_uncertainty_rank_power=3.0,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_h20_offset1_default",
        memory_use_vlm_expert=True,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    # This ablation keeps the h20 execution-only anchor recipe fixed while
    # replacing head-only 32-D CD-LAM with aligned [head32, wrist32] memory
    # and enabling PI0.5 discrete state tokens.
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_head_wrist64_state_framewise_expert0_shared_h20_offset1_default",
        memory_use_vlm_expert=True,
        discrete_state_input=True,
        memory_latent_dim=64,
        max_token_len=200,
        memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    # Same two-view/state h20 recipe, but expose the cached execution-start
    # image after the demo history and before boundary/execution memory.
    # Execution anchors require the schema-v2 dual-anchor cache and typed
    # embeddings after the memory projection (validated by Pi0Config).
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_state_framewise_expert0_shared_h20_offset1_default",
        memory_use_vlm_expert=True,
        discrete_state_input=True,
        memory_latent_dim=64,
        max_token_len=200,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
        demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
        memory_execution_anchor=True,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    # Keep the 64-D transition projector framewise. Memory rows are actual
    # four-frame CD-LAM encodings 0->4, 4->8, ...; incomplete tails are absent
    # rather than adjacent-frame latents sampled every four rows.
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_state_gap4_framewise_expert0_shared_h20_offset1_default",
        memory_use_vlm_expert=True,
        discrete_state_input=True,
        memory_latent_dim=64,
        max_token_len=200,
        memory_horizon=352,
        memory_stride=4,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_HEAD_WRIST64_GAP4_MEMORY_CACHE_ROOT,
        demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
        memory_execution_anchor=True,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    # Identical to the gap-4 two-view/state recipe above, except the randomly
    # initialized projector and typed segment embedding use a 4x LR schedule.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_state_gap4_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=True,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=352,
            memory_stride=4,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_GAP4_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # Hierarchical PatternLock decomposition: first autoregressively predict
    # the demonstration direction plan, then condition action flow matching on
    # all valid plan tokens. This is the pi0.5 subtask-text -> action topology.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_patternlock_cdlam_lmv_causal_demo_direction_conditioned_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_CDLAM_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=0,
            memory_random_drop_execution_tail_max=10,
            demo_anchor_cache_dir=_ROBOMME_PATTERNLOCK_UNIQUE1000_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            memory_demo_direction_generation=True,
            memory_demo_direction_max_token_len=32,
            memory_demo_direction_loss_weight=1.0,
            memory_demo_direction_condition_action=True,
            execution_only_task_indices=(0,),
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
            num_train_steps=120_000,
            save_steps=tuple(range(10_000, 120_001, 10_000)),
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "task": "PatternLock_unique_paths_only_task_index_0_episodes_0_999",
            "memory_type": "cdlam_head_wrist64_float16_storage_float32_input",
            "prefix_order": "language_memory_vision",
            "direction_output": "autoregressive_paligemma_text",
            "direction_vocabulary": (
                "left,right,up,down,left-up,left-down,right-up,right-down"
            ),
            "direction_target": "official_seed_path_preserving_adjacent_repeated_moves",
            "direction_attention": "language_plus_demo_anchor_plus_demo_lam_only",
            "direction_loss_weight": 1.0,
            "direction_action_branch": "hierarchical_teacher_forced_direction_conditions_action",
            "direction_action_positions": "action_starts_after_valid_direction_tokens",
            "memory_random_drop_execution_tail": "uniform_integer_0_to_10_per_sample",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Balanced 16 x 1000 RoboMME training without the optional direction-text
    # branch. All 16 task blocks contribute execution action queries.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_16x1000_dreamdojo_lmv_causal_h50_default",
            dataset_root=_ROBOMME_16X1000_DATASET_ROOT,
            assets_root=_ROBOMME_16X1000_ASSETS_ROOT,
            asset_id=_ROBOMME_16X1000_ASSET_ID,
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            # The generated 16x1000 corpus contains episodes up to 1800
            # frames, so full-history execution memory requires 1799 slots.
            memory_horizon=1800,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_16X1000_DREAMDOJO_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=0,
            memory_random_drop_execution_tail_max=10,
            demo_anchor_cache_dir=_ROBOMME_16X1000_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            memory_demo_direction_generation=False,
            memory_demo_direction_max_token_len=32,
            memory_demo_direction_loss_weight=0.0,
            memory_demo_direction_condition_action=False,
            # This generated dataset has 16 benchmark blocks but 134 language
            # task IDs. Do not filter on task_index: retain execution queries
            # from every instruction in all 16 blocks.
            execution_only_task_indices=None,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
            num_train_steps=120_000,
            save_steps=tuple(range(10_000, 120_001, 10_000)),
        ),
        memory_projector_lr_multiplier=4.0,
        num_workers=32,
        policy_metadata={
            "task": "RoboMME_balanced_16_tasks_x_1000_generated_episodes",
            "memory_type": "dreamdojo_lam400k_head_wrist64_fp32",
            "memory_encoder_checkpoint_sha256": "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f",
            "dreamdojo_upstream_commit": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
            "prefix_order": "language_memory_vision",
            "direction_output": "disabled",
            "direction_action_branch": "disabled",
            "memory_random_drop_execution_tail": "uniform_integer_0_to_10_per_sample",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Balanced 16 x 1000 RoboMME training with a fixed 10-transition lag.
    # This is identical to the configuration above except that every
    # execution query drops exactly the newest 10 execution LAM tokens.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default",
            dataset_root=_ROBOMME_16X1000_DATASET_ROOT,
            assets_root=_ROBOMME_16X1000_ASSETS_ROOT,
            asset_id=_ROBOMME_16X1000_ASSET_ID,
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            # The generated 16x1000 corpus contains episodes up to 1800
            # frames, so full-history execution memory requires 1799 slots.
            memory_horizon=1800,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_16X1000_DREAMDOJO_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=10,
            memory_random_drop_execution_tail_max=10,
            demo_anchor_cache_dir=_ROBOMME_16X1000_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            memory_demo_direction_generation=False,
            memory_demo_direction_max_token_len=32,
            memory_demo_direction_loss_weight=0.0,
            memory_demo_direction_condition_action=False,
            # This generated dataset has 16 benchmark blocks but 134 language
            # task IDs. Do not filter on task_index: retain execution queries
            # from every instruction in all 16 blocks.
            execution_only_task_indices=None,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
            num_train_steps=120_000,
            save_steps=tuple(range(10_000, 120_001, 10_000)),
        ),
        memory_projector_lr_multiplier=4.0,
        num_workers=32,
        policy_metadata={
            "task": "RoboMME_balanced_16_tasks_x_1000_generated_episodes",
            "memory_type": "dreamdojo_lam400k_head_wrist64_fp32",
            "memory_encoder_checkpoint_sha256": "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f",
            "dreamdojo_upstream_commit": "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff",
            "prefix_order": "language_memory_vision",
            "direction_output": "disabled",
            "direction_action_branch": "disabled",
            "memory_random_drop_execution_tail": "fixed_integer_10_per_sample",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # h50 DreamDojo continuation: the eight requested environment blocks each
    # receive 10% replacement-sampling mass (80% total); the other eight
    # blocks each receive 2.5% (20% total). The block-aware sampler converts
    # this task-level specification to row weights after execution filtering.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_taskmix80_target8_projector4x_framewise_expert0_shared_h50_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
            task_sampling_episode_weights=(
                (0, 99, 0.10),             # PatternLock
                (100, 199, 0.025),         # ButtonUnmaskSwap
                (200, 299, 0.025),         # ButtonUnmask
                (300, 399, 0.10),          # VideoPlaceButton
                (400, 499, 0.025),         # VideoUnmaskSwap
                (500, 599, 0.025),         # PickXtimes
                (600, 699, 0.025),         # StopCube
                (700, 799, 0.025),         # SwingXtimes
                (800, 899, 0.10),          # PickHighlight
                (900, 999, 0.10),          # MoveCube
                (1000, 1099, 0.10),        # InsertPeg
                (1100, 1199, 0.10),        # RouteStick
                (1200, 1299, 0.025),       # BinFill
                (1300, 1399, 0.10),        # VideoPlaceOrder
                (1400, 1499, 0.10),        # VideoRepick
                (1500, 1599, 0.025),       # VideoUnmask
            ),
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # Keep the dual anchors, two-view 64-D transition memory, h20 loss, and
    # 4x projector LR from the gap-4/state run. Restore adjacent transitions
    # and remove state tokens; action queries are sampled from the high-half
    # of full-20-noise trajectory diversity with replacement.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_diversity_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_rank_threshold=0.5,
            diversity_sampling_rank_power=1.0,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # Resume the high-diversity-only run with every low-half query restored to
    # the replacement sampler. The low half has a fixed 5% global mass while
    # the high half retains its original rank-linear relative weights.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_diversity_low50p5pct_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_rank_threshold=0.5,
            diversity_sampling_rank_power=1.0,
            diversity_sampling_low_rank_mass=0.05,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # Starting from the 70k diversity run, replacement sampling is controlled
    # by the most recent fixed-pack validation: each global diversity decile
    # receives mass proportional to its normalized action MSE, plus a 5%
    # uniform component over all execution queries.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_adaptive_mse_deciles_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_uniform_mass=0.05,
            diversity_sampling_mse_plan_dir=_ROBOMME_ADAPTIVE_MSE_DECILE_PLAN_DIR,
            diversity_sampling_mse_plan_start_step=70_000,
            diversity_sampling_mse_plan_interval_batches=2_000,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
        adaptive_mse_eval_pack_path=_ROBOMME_ADAPTIVE_MSE_EVAL_PACK,
        adaptive_mse_eval_result_dir=_ROBOMME_ADAPTIVE_MSE_EVAL_RESULT_DIR,
        adaptive_mse_eval_diversity_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
        adaptive_mse_eval_interval_steps=2_000,
    ),
    # Uniform-sampling h50 control: retain the full-history, dual-anchor,
    # two-view 64-D CD-LAM recipe while removing all diversity/MSE sampling
    # inputs and resident evaluation. The regular length-bucket sampler then
    # applies one uniformly shuffled pass over every execution query per epoch.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_uniform_projector4x_framewise_expert0_shared_h50_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # DreamDojo LMV experiment: prompt tokens first, framewise memory second,
    # and the ordinary visual prefix last. Language and memory are both causal;
    # the visual block is bidirectional and can read all preceding context.
    # Terminal action chunks remain fully supervised by holding the final
    # in-episode absolute target before the existing delta transform.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_dreamdojo_lmv_causal_holdlast_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "memory_type": "dreamdojo_lam400k_head_wrist64_fp32",
            "prefix_order": "language_memory_vision",
            "language_attention": "causal",
            "memory_attention": "causal",
            "memory_type_ids": {
                "padding": 0,
                "demo_lam": 1,
                "demo_to_execution_boundary": 2,
                "execution_lam": 3,
                "demo_anchor": 4,
                "execution_anchor": 5,
            },
            "vision_attention": "full_block_after_memory",
            "demo_anchor_location": "causal_memory_group_before_latent_memory",
            "demo_anchor_to_lam_boundary": "learned_type_token_id_2",
            "execution_anchor_location": "causal_memory_group_at_execution_boundary",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Async-LAM training variant of the LMV-causal baseline. Each execution
    # query independently drops the newest 1--10 execution LAM tokens, while
    # keeping demo LAM, boundaries, and both visual anchors unchanged.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_dreamdojo_lmv_causal_holdlast_random_drop_exec_lam_1to10_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=1,
            memory_random_drop_execution_tail_max=10,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "memory_type": "dreamdojo_lam400k_head_wrist64_fp32",
            "prefix_order": "language_memory_vision",
            "language_attention": "causal",
            "memory_attention": "causal",
            "memory_type_ids": {
                "padding": 0,
                "demo_lam": 1,
                "demo_to_execution_boundary": 2,
                "execution_lam": 3,
                "demo_anchor": 4,
                "execution_anchor": 5,
            },
            "vision_attention": "full_block_after_memory",
            "demo_anchor_location": "causal_memory_group_before_latent_memory",
            "demo_anchor_to_lam_boundary": "learned_type_token_id_2",
            "execution_anchor_location": "causal_memory_group_at_execution_boundary",
            "terminal_action_target": "hold_last_fully_supervised",
            "memory_random_drop_execution_tail": "uniform_integer_1_to_10_per_sample",
        },
    ),
    # CD-LAM counterpart of the DreamDojo async-memory training run above.
    # Preserve the complete LMV/causal/anchor/h50/hold-last recipe and change
    # only the memory cache plus the uniformly sampled execution-tail lag.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_cdlam_lmv_causal_holdlast_random_drop_exec_lam_0to20_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_CDLAM_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=0,
            memory_random_drop_execution_tail_max=20,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "memory_type": "cdlam_head_wrist64_float16_storage_float32_input",
            "prefix_order": "language_memory_vision",
            "language_attention": "causal",
            "memory_attention": "causal",
            "memory_type_ids": {
                "padding": 0,
                "demo_lam": 1,
                "demo_to_execution_boundary": 2,
                "execution_lam": 3,
                "demo_anchor": 4,
                "execution_anchor": 5,
            },
            "vision_attention": "full_block_after_memory",
            "demo_anchor_location": "causal_memory_group_before_latent_memory",
            "demo_anchor_to_lam_boundary": "learned_type_token_id_2",
            "execution_anchor_location": "causal_memory_group_at_execution_boundary",
            "terminal_action_target": "hold_last_fully_supervised",
            "memory_random_drop_execution_tail": "uniform_integer_0_to_20_per_sample",
        },
    ),
    # PatternLock-only joint action + language experiment. The action branch
    # retains the CD-LAM LMV-causal h50 recipe. A sibling PaliGemma branch
    # teacher-forces the compressed demo move plan and predicts it with the
    # tied language-model head; action queries cannot attend target tokens.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_patternlock_cdlam_lmv_causal_demo_direction_lm_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_CDLAM_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            memory_random_drop_execution_tail_min=0,
            memory_random_drop_execution_tail_max=10,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            memory_demo_direction_generation=True,
            memory_demo_direction_max_token_len=32,
            memory_demo_direction_loss_weight=1.0,
            execution_only_task_indices=(0,),
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
            num_train_steps=120_000,
            save_steps=tuple(range(10_000, 120_001, 10_000)),
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "task": "PatternLock_only_task_index_0_episodes_0_99",
            "memory_type": "cdlam_head_wrist64_float16_storage_float32_input",
            "prefix_order": "language_memory_vision",
            "direction_output": "autoregressive_paligemma_text",
            "direction_vocabulary": (
                "left,right,up,down,left-up,left-down,right-up,right-down"
            ),
            "direction_target": "official_seed_path_preserving_adjacent_repeated_moves",
            "direction_attention": "language_plus_demo_anchor_plus_demo_lam_only",
            "direction_loss_weight": 1.0,
            "direction_action_branch": "parallel_siblings_no_teacher_forcing_leakage",
            "memory_random_drop_execution_tail": "uniform_integer_0_to_10_per_sample",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Content-only LAM ablation of the dual-anchor DreamDojo LMV baseline.
    # Retain both visual anchors, temporal length/mask, segment IDs, boundary
    # tokens, causal order, and projector. Replace every valid normalized LAM
    # vector with the same deterministic 64-D standard-normal vector (seed 0).
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_dreamdojo_anchor_fixed_random_lam_lmv_causal_holdlast_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_fixed_random_latent_seed=0,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=True,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_demo_anchor=True,
            memory_execution_anchor=True,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "ablation": "dual_anchor_fixed_random_lam",
            "memory_type": "fixed_random_normalized_64d_seed0",
            "prefix_order": "language_demo_anchor_demo_lam_execution_anchor_execution_lam_vision",
            "language_attention": "causal",
            "memory_attention": "causal",
            "memory_mask_and_history": "preserved_from_dreamdojo_cache",
            "memory_type_ids": {
                "padding": 0,
                "demo_lam": 1,
                "demo_to_execution_boundary": 2,
                "execution_lam": 3,
                "demo_anchor": 4,
                "execution_anchor": 5,
            },
            "vision_attention": "full_block_after_memory",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Symmetric LAM-only ablation of the DreamDojo LMV run. Preserve the
    # complete typed demo/boundary/execution latent stream and every training
    # hyperparameter, but remove both absolute visual anchors. The model sees
    # language -> demo LAM -> execution boundary -> execution LAM -> current
    # vision -> action. The learned demo-anchor/LAM separator is disabled
    # because no demo anchor remains; the dataset boundary (segment ID 2)
    # between demonstration and execution memory is retained.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_dreamdojo_lam_only_lmv_causal_holdlast_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=False,
            memory_demo_lam_boundary_type_id=2,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_demo_anchor=False,
            memory_execution_anchor=False,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=4.0,
        policy_metadata={
            "ablation": "lam_only_no_vision_anchor",
            "memory_type": "dreamdojo_lam400k_head_wrist64_fp32",
            "prefix_order": "language_demo_lam_execution_boundary_execution_lam_vision",
            "language_attention": "causal",
            "memory_attention": "causal",
            "memory_type_ids": {
                "padding": 0,
                "demo_lam": 1,
                "demo_to_execution_boundary": 2,
                "execution_lam": 3,
                "demo_anchor": 4,
                "execution_anchor": 5,
            },
            "vision_anchors": "absent",
            "vision_attention": "full_block_after_memory",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # Strict vision-anchor ablation of the DreamDojo LMV run. Preserve both
    # absolute visual anchors and the LMV causal ordering, but do not read a
    # LAM cache, create latent tensors, instantiate a projector, or insert LAM
    # boundary/tokens. The sequence is language -> demo anchor -> execution
    # anchor -> current vision -> action.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_vision_anchor_only_lmv_causal_holdlast_h50_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_horizon=0,
            memory_anchor_only=True,
            max_token_len=200,
            memory_prefix_order="language_memory_vision",
            language_causal_attention=True,
            memory_causal_attention=True,
            memory_segment_vocab_size=6,
            memory_demo_anchor_type_id=4,
            memory_execution_anchor_type_id=5,
            memory_demo_lam_boundary_token=False,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=None,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            normalize_memory_latents=False,
            action_horizon=50,
            action_sequence_start_offset=1,
            hold_last_action_targets=True,
        ),
        memory_projector_lr_multiplier=1.0,
        policy_metadata={
            "memory_type": "vision_anchor_only",
            "prefix_order": "language_demo_anchor_execution_anchor_vision",
            "language_attention": "causal",
            "anchor_attention": "full_within_each_anchor_block",
            "memory_type_ids": {"demo_anchor": 4, "execution_anchor": 5},
            "lam_tokens": "absent",
            "vision_attention": "full_block_after_anchors",
            "terminal_action_target": "hold_last_fully_supervised",
        },
    ),
    # DeltaTok counterpart of the h50 uniform CD-LAM control above. Preserve
    # its data/query/anchor/optimizer semantics, while replacing the two-view
    # 64-D CD-LAM history with concatenated head+wrist DeltaTok tokens and the
    # established 2048-wide DeltaTok projector.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist_deltatok1536_fp32_nostate_gap1_uniform_projector4x_framewise_expert0_shared_h50_offset1_h2048_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=1536,
            memory_frames_per_token=1,
            memory_projector_hidden_dim=2048,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_DELTATOK_HEAD_WRIST1536_FP32_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            normalize_memory_latents=True,
            action_horizon=50,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
    ),
    # End-to-end isolated smoke configuration. It intentionally preserves the
    # production model and sampling formula; only the plan/evaluation cadence
    # and all produced artifact paths differ. The 100-step boundary therefore
    # covers the same checkpoint -> resident-eval -> atomic-plan-publication ->
    # next-plan-consumption control flow as the 2k production boundary.
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_uniform_mass=0.05,
            diversity_sampling_mse_plan_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_PLAN_DIR,
            diversity_sampling_mse_plan_start_step=70_000,
            diversity_sampling_mse_plan_interval_batches=100,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
        adaptive_mse_eval_pack_path=_ROBOMME_ADAPTIVE_MSE_EVAL_PACK,
        adaptive_mse_eval_result_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_EVAL_RESULT_DIR,
        adaptive_mse_eval_diversity_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
        adaptive_mse_eval_interval_steps=100,
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100_v10_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_uniform_mass=0.05,
            diversity_sampling_mse_plan_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_V10_PLAN_DIR,
            diversity_sampling_mse_plan_start_step=70_000,
            diversity_sampling_mse_plan_interval_batches=100,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
        adaptive_mse_eval_pack_path=_ROBOMME_ADAPTIVE_MSE_EVAL_PACK,
        adaptive_mse_eval_result_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_V10_EVAL_RESULT_DIR,
        adaptive_mse_eval_diversity_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
        adaptive_mse_eval_interval_steps=100,
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_dual_anchor_head_wrist64_nostate_gap1_adaptive_mse_deciles_smoke100_v11_projector4x_framewise_expert0_shared_h20_offset1_default",
            memory_use_vlm_expert=True,
            discrete_state_input=False,
            memory_latent_dim=64,
            max_token_len=200,
            memory_horizon=1410,
            memory_stride=1,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_HEAD_WRIST64_MEMORY_CACHE_ROOT,
            demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
            memory_execution_anchor=True,
            diversity_sampling_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
            diversity_sampling_uniform_mass=0.05,
            diversity_sampling_mse_plan_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_V11_PLAN_DIR,
            diversity_sampling_mse_plan_start_step=70_000,
            diversity_sampling_mse_plan_interval_batches=100,
            action_horizon=20,
            action_sequence_start_offset=1,
        ),
        memory_projector_lr_multiplier=4.0,
        adaptive_mse_eval_pack_path=_ROBOMME_ADAPTIVE_MSE_EVAL_PACK,
        adaptive_mse_eval_result_dir=_ROBOMME_ADAPTIVE_MSE_SMOKE_V11_EVAL_RESULT_DIR,
        adaptive_mse_eval_diversity_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
        adaptive_mse_eval_interval_steps=100,
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_h20_offset1_swap2_from100k_default",
            memory_use_vlm_expert=True,
            execution_only_episode_indices=_ROBOMME_SWAP_EPISODE_INDICES,
            action_horizon=20,
            action_sequence_start_offset=1,
            initial_params_path=_ROBOMME_CDLAM_ANCHOR_H20_100K_PARAMS,
            num_train_steps=30_000,
            save_steps=(5_000, 10_000, 20_000, 30_000),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _ROBOMME_CDLAM_ANCHOR_H20_100K_PARAMS,
            initialize_memory_expert_from_action=False,
            skip_mismatched_shapes=False,
        ),
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_framewise_expert0_shared_h20_offset1_h2048_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=2048,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    # Register-bottleneck variant: the demo anchor and DeltaTok history are
    # private to expert 0.  The action expert can read the learned registers,
    # but cannot attend either historical source directly.
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_register8_h20_offset1_h2048_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=2048,
        memory_register_count=8,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_horizon=20,
        action_sequence_start_offset=1,
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_anchor_deltatok768_fp32_framewise_expert0_shared_h20_offset1_h2048_default",
            memory_use_vlm_expert=True,
            memory_latent_dim=768,
            memory_frames_per_token=1,
            memory_projector_hidden_dim=2048,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_DELTATOK_FP32_CACHE_ROOT,
            normalize_memory_latents=True,
            action_horizon=20,
            action_sequence_start_offset=1,
            num_train_steps=120_000,
            save_steps=(110_000, 120_000),
        ),
        lr_schedule=_optimizer.ConstantSchedule(learning_rate=2.5e-6),
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_clean32_framewise_expert0_shared_h20_offset1_uncertainty_default",
        memory_use_vlm_expert=True,
        action_horizon=20,
        action_sequence_start_offset=1,
        baseline_uncertainty_weighting=True,
        baseline_uncertainty_cache_dir=_ROBOMME_BASELINE_UNCERTAINTY_H20_CACHE_ROOT,
        baseline_uncertainty_min_weight=0.25,
        baseline_uncertainty_max_weight=3.25,
        baseline_uncertainty_rank_power=3.0,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_raw32_group4_expert0_shared_default",
        memory_use_vlm_expert=True,
        memory_frames_per_token=4,
        normalize_memory_latents=False,
        action_memory_warmup_steps=1_000,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_framewise_expert0_shared_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=2048,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=False,
        action_memory_warmup_steps=1_000,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_framewise_independent_h1024_default",
        memory_use_vlm_expert=False,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=0,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_dual_anchor_deltatok768_expert0_h1024_action_motion_p95_pi05base_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        demo_anchor_cache_dir=_ROBOMME_DUAL_ANCHOR_CACHE_ROOT,
        memory_execution_anchor=True,
        normalize_memory_latents=True,
        action_memory_warmup_steps=0,
        initial_params_path=_LIBERO_PLUS_PI05_BASE_PARAMS,
        skip_mismatched_checkpoint_shapes=False,
        memory_advantage_weighting=False,
        action_motion_weighting=True,
        action_motion_joint_dims=7,
        action_motion_joint_scales=(
            0.011260396242141721,
            0.026021263003349303,
            0.0079950595274567594,
            0.031784105300903309,
            0.0070857495069503711,
            0.02788858413696288,
            0.02521220445632932,
        ),
        action_motion_gripper_index=7,
        action_motion_gripper_flip_threshold=0.5,
        action_motion_min_weight=0.3,
        action_motion_max_weight=5.0,
        num_train_steps=30_000,
        save_steps=(10_000, 20_000, 30_000),
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_future10_flow_random_h1024_default",
        memory_use_vlm_expert=False,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=0,
        memory_flow_horizon=10,
        memory_flow_loss_weight=1.0,
        random_init_memory_expert=True,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_future10_direct_random_h1024_default",
        memory_use_vlm_expert=False,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=0,
        memory_flow_horizon=10,
        memory_future_prediction_mode="direct",
        memory_flow_loss_weight=1.0,
        random_init_memory_expert=True,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_future10_direct_expert0_trainonly_h1024_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=1_000,
        memory_flow_horizon=10,
        memory_future_prediction_mode="direct",
        memory_future_training_only=True,
        memory_flow_loss_weight=0.005,
        memory_auxiliary_warmup_only=True,
        initial_params_path=_ROBOMME_DELTATOK_INDEPENDENT_ACTION1_30K_PARAMS,
        skip_mismatched_checkpoint_shapes=True,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_expert0_future10_cdlam32_flow_actioninit_h1024_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        future_memory_cache_dir=_ROBOMME_MEMORY_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=1_000,
        memory_flow_horizon=10,
        memory_future_latent_dim=32,
        memory_future_prediction_mode="flow_matching",
        memory_future_training_only=True,
        memory_future_use_separate_expert=True,
        memory_flow_loss_weight=0.005,
        memory_auxiliary_warmup_only=True,
        initial_params_path=_ROBOMME_DELTATOK_INDEPENDENT_ACTION1_30K_PARAMS,
        skip_mismatched_checkpoint_shapes=True,
        force_initialize_memory_expert_from_action=True,
        num_train_steps=20_000,
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_expert0_future10_cdlam32_flow_noquery_adarms_fullactioninit_h1024_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        future_memory_cache_dir=_ROBOMME_MEMORY_CACHE_ROOT,
        normalize_memory_latents=True,
        action_memory_warmup_steps=1_000,
        memory_flow_horizon=10,
        memory_future_latent_dim=32,
        memory_future_prediction_mode="flow_matching",
        memory_future_training_only=True,
        memory_future_use_separate_expert=True,
        memory_future_use_learned_queries=False,
        memory_future_use_adarms_time_conditioning=True,
        memory_flow_loss_weight=0.005,
        memory_auxiliary_warmup_only=True,
        initial_params_path=_ROBOMME_DELTATOK_INDEPENDENT_ACTION1_30K_PARAMS,
        skip_mismatched_checkpoint_shapes=True,
        force_initialize_memory_expert_from_action=True,
        initialize_memory_expert_norms_from_action=True,
        force_initialize_memory_flow_heads_from_action=True,
        num_train_steps=20_000,
        save_steps=(4_000, 8_000, 12_000, 16_000, 20_000),
    ),
    _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
        name="pi05_robomme_memory_exec_only_anchor_deltatok768_expert0_nofuture_h1024_memory_advantage_k8_default",
        memory_use_vlm_expert=True,
        memory_latent_dim=768,
        memory_frames_per_token=1,
        memory_projector_hidden_dim=1024,
        memory_segment_embedding_after_projection=True,
        memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
        normalize_memory_latents=True,
        initial_params_path=_ROBOMME_DELTATOK_INDEPENDENT_ACTION1_30K_PARAMS,
        skip_mismatched_checkpoint_shapes=True,
        memory_advantage_weighting=True,
        memory_advantage_recent_steps=8,
        memory_advantage_score_action_dims=8,
        # Memory usefulness and residual full-branch hardness jointly produce
        # a bounded raw weight in [0.3, 5.0].
        memory_advantage_max_weight=5.0,
        # Also retain weight on positions that remain hard for the full-memory
        # branch. Hard-only positions contribute at most half of the score.
        memory_advantage_hardness_weight=0.5,
        memory_advantage_hardness_threshold=0.5,
        memory_advantage_hardness_temperature=0.5,
        memory_advantage_hardness_time_bins=4,
        memory_advantage_warmup_fraction=5_000 / 30_000,
        memory_advantage_ramp_fraction=5_000 / 30_000,
        memory_advantage_recent_loss_weight=0.1,
        num_train_steps=30_000,
        save_steps=(5_000, 10_000, 20_000, 30_000),
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_imitation4_anchor_deltatok768_expert0_h1024_hardness_only_from_maw30k_default",
            memory_use_vlm_expert=True,
            memory_latent_dim=768,
            memory_frames_per_token=1,
            memory_projector_hidden_dim=1024,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
            normalize_memory_latents=True,
            initial_params_path=_ROBOMME_MAW_PI05BASE_ACTION1_30K_PARAMS,
            skip_mismatched_checkpoint_shapes=False,
            memory_advantage_weighting=False,
            hardness_weighting=True,
            hardness_min_weight=0.3,
            hardness_max_weight=5.0,
            hardness_threshold=0.5,
            hardness_temperature=0.5,
            hardness_time_bins=4,
            hardness_score_action_dims=8,
            action_motion_weighting=False,
            execution_only_task_indices=(0, 62, 63, 64),
            num_train_steps=5_000,
            save_steps=(1_000, 5_000),
        ),
        # The source checkpoint has already completed the base schedule's
        # warmup; reset optimizer state but continue at its trained LR.
        lr_schedule=_optimizer.ConstantSchedule(learning_rate=5e-5),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            _ROBOMME_MAW_PI05BASE_ACTION1_30K_PARAMS,
            initialize_memory_expert_from_action=False,
            skip_mismatched_shapes=False,
        ),
    ),
    dataclasses.replace(
        _pi05_robomme_memory_exec_only_anchor_clean32_framewise_default_config(
            name="pi05_robomme_memory_exec_only_anchor_deltatok768_expert0_nofuture_h1024_action_motion_p95_default",
            memory_use_vlm_expert=True,
            memory_latent_dim=768,
            memory_frames_per_token=1,
            memory_projector_hidden_dim=1024,
            memory_segment_embedding_after_projection=True,
            memory_cache_dir=_ROBOMME_DELTATOK_CACHE_ROOT,
            normalize_memory_latents=True,
            initial_params_path=_ROBOMME_MAW_PI05BASE_ACTION1_30K_PARAMS,
            skip_mismatched_checkpoint_shapes=False,
            memory_advantage_weighting=False,
            action_motion_weighting=True,
            action_motion_joint_dims=7,
            action_motion_joint_scales=(
                0.011260396242141721,
                0.026021263003349303,
                0.0079950595274567594,
                0.031784105300903309,
                0.0070857495069503711,
                0.02788858413696288,
                0.02521220445632932,
            ),
            action_motion_gripper_index=7,
            action_motion_gripper_flip_threshold=0.5,
            action_motion_min_weight=0.3,
            action_motion_max_weight=5.0,
            num_train_steps=10_000,
            save_steps=(1_000, 5_000, 10_000),
        ),
        # The source checkpoint already completed the pi0.5 10k warmup and
        # trained at 5e-5. Keep that current LR while resetting optimizer state
        # so the experiment changes only the loss weighting objective.
        lr_schedule=_optimizer.ConstantSchedule(learning_rate=5e-5),
    ),
    TrainConfig(
        name="pi05_libero_nomemory_default",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            # Match the memory baseline exactly apart from disabling its
            # memory stream. The 40 fixed LIBERO prompts need at most 21
            # tokens, and the dataset contains only these two real cameras.
            max_token_len=21,
            pytorch_compile_mode=None,
            memory_horizon=0,
            active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="external/libero/physical-intelligence_libero",
            assets=AssetsConfig(
                assets_dir="external/memory/assets/pi05_libero_memory_default",
                asset_id="physical-intelligence/libero",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="external/memory/checkpoints/pi05_base_pytorch",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_memory_default",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            # Exact maximum over the 40 fixed LIBERO task prompts (including
            # BOS/newline). Keeping this static across ranks avoids 179 masked
            # language tokens without introducing per-rank sequence skew.
            max_token_len=21,
            # torch.compile only wraps sample_actions, not the training
            # forward. Disable it here because each DDP rank otherwise starts
            # a large Inductor worker pool for variable-length memory inference.
            pytorch_compile_mode=None,
            # The official LIBERO dataset's longest episode has 505 frames,
            # hence at most 504 causal adjacent-frame transition latents.
            memory_horizon=504,
            # LIBERO provides only one external and one wrist camera. The
            # right-wrist tensor is an all-zero, all-masked compatibility pad.
            active_image_keys=("base_0_rgb", "left_wrist_0_rgb"),
            memory_latent_dim=32,
            memory_projector_hidden_dim=512,
            # Population statistics over all 271,772 valid transitions in the
            # finalized physical-intelligence/libero CD-LAM cache. Invalid
            # episode-start rows are excluded.
            memory_latent_mean=(
                0.04626310759453199,
                0.2594701785569497,
                -0.05373315532346309,
                0.07646829593837459,
                -1.0841681751304741,
                -0.5310980378405904,
                -0.641513140227571,
                -0.034336143821596755,
                0.10054984477927914,
                0.08046076114010531,
                -0.02693429100322739,
                0.06165823028917798,
                -0.15707363297421603,
                -0.5413241220501274,
                0.23744331148665646,
                0.9291702438555838,
                -0.1481879460765646,
                -0.12135923736249969,
                0.07901516528403849,
                0.20400838448413258,
                0.09024587546057168,
                0.2061961278497665,
                0.19887375577391858,
                -0.20340795938269965,
                0.19055807457621435,
                0.06608455951881828,
                -0.20351730567035192,
                0.09691796784546183,
                -0.03260832242230683,
                -0.3314430693800848,
                -0.23292225702259006,
                -0.6301026841154557,
            ),
            memory_latent_std=(
                0.5128081878653022,
                0.45694138170924636,
                0.6261776025514394,
                0.5213203621651317,
                0.7155646267390123,
                1.0457691441125676,
                0.8219476369618608,
                0.537351623648782,
                0.45115372923928454,
                0.41483553729693184,
                0.6875650550184012,
                0.37379220866209234,
                0.40071008461184293,
                0.6884223386691443,
                0.5098141733367554,
                0.3927348220487438,
                0.6318080947981333,
                0.43767560267893757,
                0.5059550705582427,
                0.48280385381334,
                0.5471246205573911,
                0.6696593736049797,
                0.5137167499038829,
                0.9405252231599663,
                0.7665829131491189,
                0.8642246232185816,
                0.5687185050550126,
                0.4421917518532816,
                0.55947890157251,
                0.7226461496414661,
                0.29489960553887734,
                0.9810828378404,
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="external/libero/physical-intelligence_libero",
            assets=AssetsConfig(asset_id="physical-intelligence/libero"),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir="external/libero/physical-intelligence_libero_cdlam_memory",
            ),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="external/memory/checkpoints/pi05_base_pytorch",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero_memory",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            pytorch_compile_mode=None,
            # LIBERO-90's longest episode has 353 frames, hence at most 352
            # causal adjacent-frame transition latents.
            memory_horizon=352,
            memory_latent_dim=32,
            memory_projector_hidden_dim=512,
            # Population statistics over the 570,612 valid transitions in the
            # finalized LIBERO-90 CD-LAM cache. Invalid episode-start rows are
            # excluded. The same values are stored in the cache manifest.
            memory_latent_mean=(
                0.06729129555011812,
                0.18464855857702542,
                -0.06588686245739647,
                0.10339990907419587,
                -0.955590107060757,
                -0.6488303955550279,
                -0.5572408567419009,
                -0.048948413896661616,
                0.10426438857850347,
                0.09724228484806842,
                -0.08569239014541018,
                0.07952256235649885,
                -0.1436110077274753,
                -0.5708902539421757,
                0.259029496408167,
                0.9044873228400481,
                -0.14373026278106563,
                -0.10818012641502717,
                0.12039357815173773,
                0.19734476200858792,
                0.06921372786096909,
                0.17248549002537714,
                0.15638770886238693,
                -0.3025907070505108,
                0.1277574504104851,
                0.0018563590674365465,
                -0.15543074416311858,
                0.09087521929865018,
                -0.021215778203445702,
                -0.305306916704224,
                -0.20497077956440432,
                -0.7046097898553597,
            ),
            memory_latent_std=(
                0.5056678370679775,
                0.459394001595067,
                0.6666007791900611,
                0.5403166483319359,
                0.7230113331958855,
                0.9772929947265402,
                0.8141895872891823,
                0.5678720763949829,
                0.4224032974493984,
                0.40471230697001276,
                0.7202046019472853,
                0.36749000280262306,
                0.38109110137062613,
                0.7061612085966144,
                0.4815942106794864,
                0.40103842429828285,
                0.6087965064173791,
                0.4356142413420833,
                0.5229204786599735,
                0.49643358191826104,
                0.5260164580680599,
                0.6288368220512935,
                0.4773192315195827,
                0.9573395227593086,
                0.8785057511220067,
                0.7406668267948912,
                0.5447775521840829,
                0.47870244299085424,
                0.47735202233745433,
                0.6714024331090871,
                0.29524238812543263,
                0.9486643340573422,
            ),
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="external/libero/libero_90_lerobot",
            assets=AssetsConfig(asset_id="libero_90_lerobot"),
            base_config=DataConfig(
                prompt_from_task=True,
                memory_cache_dir="external/libero/libero_90_cdlam_memory",
            ),
            extra_delta_transform=False,
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("external/ckpt/params"),
        pytorch_weight_path="external/memory/checkpoints/pi05_base_pytorch",
        num_workers=0,
        num_train_steps=30_000,
        wandb_enabled=False,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
