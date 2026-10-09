from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


class SimpleProcessedObservation:
    """Lightweight preprocessed view without per-step dynamic class garbage."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _random_crop_and_rotate(image: torch.Tensor, *, crop_fraction: float = 0.95, max_degrees: float = 5.0):
    """Apply the existing batch-shared crop/rotation as one async GPU op.

    Keeping crop offsets and the angle as device tensors avoids the implicit
    CUDA synchronizations caused by using GPU scalars as Python slice indices
    and branch conditions.
    """
    batch_size, height, width, channels = image.shape
    crop_height = int(height * crop_fraction)
    crop_width = int(width * crop_fraction)
    max_h = height - crop_height
    max_w = width - crop_width

    start_h = torch.randint(0, max_h + 1, (), device=image.device).to(dtype=image.dtype)
    start_w = torch.randint(0, max_w + 1, (), device=image.device).to(dtype=image.dtype)
    angle = (torch.rand((), device=image.device) * (2 * max_degrees) - max_degrees) * torch.pi / 180.0
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    scale_y = crop_height / height
    scale_x = crop_width / width
    translate_y = (2 * start_h + crop_height) / height - 1
    translate_x = (2 * start_w + crop_width) / width - 1
    theta = torch.stack(
        (
            torch.stack((scale_x * cos_a, -scale_x * sin_a, translate_x)),
            torch.stack((scale_y * sin_a, scale_y * cos_a, translate_y)),
        )
    ).unsqueeze(0)
    theta = theta.expand(batch_size, -1, -1)

    channels_first = image.permute(0, 3, 1, 2)
    grid = torch.nn.functional.affine_grid(theta, channels_first.shape, align_corners=False)
    transformed = torch.nn.functional.grid_sample(
        channels_first,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return transformed.permute(0, 2, 3, 1).reshape(batch_size, height, width, channels)


def _color_augment(image: torch.Tensor) -> torch.Tensor:
    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
    image = image * brightness_factor
    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast_factor + mean
    saturation_factor = 0.5 + torch.rand(1, device=image.device)
    gray = image.mean(dim=-1, keepdim=True)
    return torch.clamp(gray + (image - gray) * saturation_factor, 0, 1)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # The training loader keeps uint8 NHWC images in pinned host memory so
        # the small tensor can be copied asynchronously. Normalize only after
        # it reaches the GPU instead of creating a new unpinned float32 tensor
        # in the main process.
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32).mul_(2.0 / 255.0).add_(-1.0)

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                image = _random_crop_and_rotate(image)

            image = _color_augment(image)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Vision tower always consumes [B, C, H, W].
        image = image.permute(0, 3, 1, 2)

        out_images[key] = image

    memory_demo_start_image = observation.memory_demo_start_image
    if memory_demo_start_image is not None:
        anchor = memory_demo_start_image
        if anchor.dtype == torch.uint8:
            anchor = anchor.to(dtype=torch.float32).mul_(2.0 / 255.0).add_(-1.0)
        if anchor.shape[1] == 3:
            anchor = anchor.permute(0, 2, 3, 1)
        if anchor.shape[1:3] != image_resolution:
            anchor = image_tools.resize_with_pad_torch(anchor, *image_resolution)
        if train:
            anchor = _random_crop_and_rotate(anchor / 2.0 + 0.5)
            anchor = _color_augment(anchor) * 2.0 - 1.0
        memory_demo_start_image = anchor.permute(0, 3, 1, 2)

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        memory_latents=observation.memory_latents,
        memory_mask=observation.memory_mask,
        memory_segment_ids=observation.memory_segment_ids,
        memory_demo_start_image=memory_demo_start_image,
        memory_demo_start_mask=observation.memory_demo_start_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
