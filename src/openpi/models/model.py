import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images are normally float32 in [-1, 1]. PyTorch training may defer
    # normalization until after the uint8 NHWC tensors reach the GPU.
    images: dict[str, at.Float[ArrayT, "*b h w c"] | at.UInt8[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Optional latent-action history. Each token represents the deterministic
    # LAM z_mu for one past frame transition (o_{i-1}, o_i).
    memory_latents: at.Float[ArrayT, "*b m d"] | None = None
    memory_mask: at.Bool[ArrayT, "*b m"] | None = None
    # Optional fixed-length future latent targets used only by the memory
    # expert's flow-matching objective. They are absent during inference.
    future_memory_latents: at.Float[ArrayT, "*b fm fd"] | None = None
    future_memory_mask: at.Bool[ArrayT, "*b fm"] | None = None
    # Semantic type of every latent transition: 0=padding, 1=demo,
    # 2=demo-to-execution reset boundary, 3=execution.
    memory_segment_ids: at.Int[ArrayT, "*b m"] | None = None
    # Optional training-only autoregressive text branch. These tokens encode
    # the compressed PatternLock demonstration plan, for example
    # "Directions: left-up, right, down". The VLM predicts the next token
    # while the action expert predicts the action chunk in a sibling branch.
    demo_direction_tokens: at.Int[ArrayT, "*b dt"] | None = None
    demo_direction_mask: at.Bool[ArrayT, "*b dt"] | None = None
    demo_direction_loss_mask: at.Bool[ArrayT, "*b dt"] | None = None
    # Absolute visual anchor for transition-only memory. It is either one
    # head-camera image or a precomputed mean of per-frame SigLIP patch
    # features. Exactly one representation is present when the anchor is used.
    memory_demo_start_image: (
        at.Float[ArrayT, "*b anchor_h anchor_w c"] | at.UInt8[ArrayT, "*b anchor_h anchor_w c"] | None
    ) = None
    memory_demo_anchor_features: at.Float[ArrayT, "*b anchor_tokens emb"] | None = None
    memory_demo_start_mask: at.Bool[ArrayT, "*b"] | None = None
    # Raw head-camera frame at episode timestep zero. This anchor is present
    # for every RoboDojo sample, including timestep zero, and is kept separate
    # from the optional demonstration/execution anchors above and below.
    memory_episode_start_image: (
        at.Float[ArrayT, "*b anchor_h anchor_w c"] | at.UInt8[ArrayT, "*b anchor_h anchor_w c"] | None
    ) = None
    memory_episode_start_mask: at.Bool[ArrayT, "*b"] | None = None
    # Absolute head-camera image at the first execution frame.  Unlike the
    # demo anchor this is inserted inside the memory stream, immediately after
    # demo transitions and before the reset-boundary/execution transitions.
    memory_execution_start_image: (
        at.Float[ArrayT, "*b anchor_h anchor_w c"] | at.UInt8[ArrayT, "*b anchor_h anchor_w c"] | None
    ) = None
    memory_execution_start_mask: at.Bool[ArrayT, "*b"] | None = None

    # Training-only metadata for motion-aware action-loss weighting.  These
    # values are computed from raw adjacent actions before delta conversion or
    # normalization and are absent during policy inference.
    action_motion_delta: at.Float[ArrayT, "*b ah joint"] | None = None
    action_gripper_flip: at.Bool[ArrayT, "*b ah"] | None = None
    # Global rank-CDF score of no-memory action-sampling dispersion, aligned
    # to the supervised physical action positions.
    baseline_uncertainty_score: at.Float[ArrayT, "*b ah"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(
        cls,
        data: at.PyTree[ArrayT],
        *,
        normalize_torch_images: bool = True,
    ) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if (data.get("tokenized_prompt") is not None) != (data.get("tokenized_prompt_mask") is not None):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        if (data.get("memory_latents") is not None) != (data.get("memory_mask") is not None):
            raise ValueError("memory_latents and memory_mask must be provided together.")
        if (data.get("future_memory_latents") is not None) != (data.get("future_memory_mask") is not None):
            raise ValueError("future_memory_latents and future_memory_mask must be provided together.")
        if data.get("memory_segment_ids") is not None and data.get("memory_latents") is None:
            raise ValueError("memory_segment_ids requires memory_latents and memory_mask.")
        direction_fields = (
            data.get("demo_direction_tokens"),
            data.get("demo_direction_mask"),
            data.get("demo_direction_loss_mask"),
        )
        if any(value is not None for value in direction_fields) and not all(
            value is not None for value in direction_fields
        ):
            raise ValueError(
                "demo_direction_tokens, demo_direction_mask, and demo_direction_loss_mask must be provided together."
            )
        has_anchor_image = data.get("memory_demo_start_image") is not None
        has_anchor_features = data.get("memory_demo_anchor_features") is not None
        has_anchor_mask = data.get("memory_demo_start_mask") is not None
        if has_anchor_image and has_anchor_features:
            raise ValueError("Provide only one demo-anchor representation: image or SigLIP features.")
        if (has_anchor_image or has_anchor_features) != has_anchor_mask:
            raise ValueError("A demo-anchor representation and memory_demo_start_mask must be provided together.")
        has_episode_anchor = data.get("memory_episode_start_image") is not None
        has_episode_anchor_mask = data.get("memory_episode_start_mask") is not None
        if has_episode_anchor != has_episode_anchor_mask:
            raise ValueError("memory_episode_start_image and memory_episode_start_mask must be provided together.")
        has_execution_anchor = data.get("memory_execution_start_image") is not None
        has_execution_anchor_mask = data.get("memory_execution_start_mask") is not None
        if has_execution_anchor != has_execution_anchor_mask:
            raise ValueError("memory_execution_start_image and memory_execution_start_mask must be provided together.")
        if (data.get("action_motion_delta") is not None) != (data.get("action_gripper_flip") is not None):
            raise ValueError("action_motion_delta and action_gripper_flip must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif (
                normalize_torch_images
                and hasattr(data["image"][key], "dtype")
                and data["image"][key].dtype == torch.uint8
            ):
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        memory_demo_start_image = data.get("memory_demo_start_image")
        if memory_demo_start_image is not None:
            if memory_demo_start_image.dtype == np.uint8:
                memory_demo_start_image = memory_demo_start_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif normalize_torch_images and memory_demo_start_image.dtype == torch.uint8:
                memory_demo_start_image = (
                    memory_demo_start_image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
        memory_episode_start_image = data.get("memory_episode_start_image")
        if memory_episode_start_image is not None:
            if memory_episode_start_image.dtype == np.uint8:
                memory_episode_start_image = memory_episode_start_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif normalize_torch_images and memory_episode_start_image.dtype == torch.uint8:
                memory_episode_start_image = (
                    memory_episode_start_image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
        memory_execution_start_image = data.get("memory_execution_start_image")
        if memory_execution_start_image is not None:
            if memory_execution_start_image.dtype == np.uint8:
                memory_execution_start_image = memory_execution_start_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            elif normalize_torch_images and memory_execution_start_image.dtype == torch.uint8:
                memory_execution_start_image = (
                    memory_execution_start_image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            memory_latents=data.get("memory_latents"),
            memory_mask=data.get("memory_mask"),
            future_memory_latents=data.get("future_memory_latents"),
            future_memory_mask=data.get("future_memory_mask"),
            memory_segment_ids=data.get("memory_segment_ids"),
            demo_direction_tokens=data.get("demo_direction_tokens"),
            demo_direction_mask=data.get("demo_direction_mask"),
            demo_direction_loss_mask=data.get("demo_direction_loss_mask"),
            memory_demo_start_image=memory_demo_start_image,
            memory_demo_anchor_features=data.get("memory_demo_anchor_features"),
            memory_demo_start_mask=data.get("memory_demo_start_mask"),
            memory_episode_start_image=memory_episode_start_image,
            memory_episode_start_mask=data.get("memory_episode_start_mask"),
            memory_execution_start_image=memory_execution_start_image,
            memory_execution_start_mask=data.get("memory_execution_start_mask"),
            action_motion_delta=data.get("action_motion_delta"),
            action_gripper_flip=data.get("action_gripper_flip"),
            baseline_uncertainty_score=data.get("baseline_uncertainty_score"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        memory_latents=observation.memory_latents,
        memory_mask=observation.memory_mask,
        future_memory_latents=observation.future_memory_latents,
        future_memory_mask=observation.future_memory_mask,
        memory_segment_ids=observation.memory_segment_ids,
        demo_direction_tokens=observation.demo_direction_tokens,
        demo_direction_mask=observation.demo_direction_mask,
        demo_direction_loss_mask=observation.demo_direction_loss_mask,
        memory_demo_start_image=(
            _preprocess_memory_demo_start_image(
                rng,
                observation.memory_demo_start_image,
                train=train,
                image_resolution=image_resolution,
            )
            if observation.memory_demo_start_image is not None
            else None
        ),
        memory_demo_anchor_features=observation.memory_demo_anchor_features,
        memory_demo_start_mask=observation.memory_demo_start_mask,
        memory_episode_start_image=(
            _preprocess_memory_anchor_image(
                rng,
                observation.memory_episode_start_image,
                train=train,
                image_resolution=image_resolution,
            )
            if observation.memory_episode_start_image is not None
            else None
        ),
        memory_episode_start_mask=observation.memory_episode_start_mask,
        memory_execution_start_image=(
            _preprocess_memory_anchor_image(
                rng,
                observation.memory_execution_start_image,
                train=train,
                image_resolution=image_resolution,
            )
            if observation.memory_execution_start_image is not None
            else None
        ),
        memory_execution_start_mask=observation.memory_execution_start_mask,
        action_motion_delta=observation.action_motion_delta,
        action_gripper_flip=observation.action_gripper_flip,
        baseline_uncertainty_score=observation.baseline_uncertainty_score,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def _preprocess_memory_anchor_image(
    rng: at.KeyArrayLike | None,
    image,
    *,
    train: bool,
    image_resolution: tuple[int, int],
):
    """Apply the same head-camera preprocessing to an absolute memory anchor.

    Reusing the batch RNG intentionally gives the anchor and current head view
    the same sampled geometric/color transform in the JAX training path.
    """
    if image.shape[1:3] != image_resolution:
        image = image_tools.resize_with_pad(image, *image_resolution)
    if not train:
        return image
    if rng is None:
        raise ValueError("Training image augmentation requires an RNG")
    image = image / 2.0 + 0.5
    height, width = image.shape[1:3]
    transforms = augmax.Chain(
        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
        augmax.Resize(width, height),
        augmax.Rotate((-5, 5)),
        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
    )
    sub_rngs = jax.random.split(rng, image.shape[0])
    return jax.vmap(transforms)(sub_rngs, image) * 2.0 - 1.0


# Backward-compatible private name used by out-of-tree callers.
_preprocess_memory_demo_start_image = _preprocess_memory_anchor_image


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
