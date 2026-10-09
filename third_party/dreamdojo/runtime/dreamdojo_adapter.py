"""Minimal DreamDojo LAM inference adapter used by LTM-Pi.

This project-specific adapter is not part of upstream DreamDojo. It exposes
only the deterministic encoder and image preprocessing needed to build and
serve the RoboMME memory cache.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch import Tensor

OFFICIAL_WM_HW = (480, 640)
OFFICIAL_LAM_HW = (240, 320)
OFFICIAL_ASPECT = OFFICIAL_WM_HW[1] / OFFICIAL_WM_HW[0]


def _as_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) RGB video, got shape={frames.shape}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    return np.clip(frames, 0, 255).astype(np.uint8, copy=False)


def _resize_uint8_video(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    frames = _as_uint8_rgb(frames)
    height, width = hw
    if frames.shape[1:3] == (height, width):
        return np.ascontiguousarray(frames)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
    tensor = torch_f.interpolate(tensor, size=hw, mode="bilinear", align_corners=False)
    return tensor.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def official_lam_video_from_raw(raw_frames: np.ndarray) -> np.ndarray:
    """Apply DreamDojo's 4:3 center crop and two-stage LAM resize."""
    frames = _as_uint8_rgb(raw_frames)
    height, width = frames.shape[1:3]
    if width / height > OFFICIAL_ASPECT:
        crop_height, crop_width = height, int(height * OFFICIAL_ASPECT)
    elif width / height < OFFICIAL_ASPECT:
        crop_height, crop_width = int(width / OFFICIAL_ASPECT), width
    else:
        crop_height, crop_width = height, width
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = np.ascontiguousarray(frames[:, top : top + crop_height, left : left + crop_width])
    wm_video = _resize_uint8_video(cropped, OFFICIAL_WM_HW)
    return _resize_uint8_video(wm_video, OFFICIAL_LAM_HW)


def encode_full(lam_inner, videos: Tensor, sample: bool = False, use_ckpt: bool = False):
    """Run the upstream DreamDojo LAM encoder and return its 32-D posterior mean."""
    if sample:
        raise ValueError("LTM-Pi cache extraction requires deterministic DreamDojo posterior means")
    if use_ckpt:
        raise ValueError("gradient checkpointing is unavailable in inference-only mode")
    from external.lam.modules.blocks import patchify

    batch, timesteps = videos.shape[:2]
    if timesteps != 2:
        raise ValueError(f"DreamDojo LAM expects frame pairs, got T={timesteps}")
    patches = patchify(videos, lam_inner.patch_size)
    action_pad = lam_inner.action_prompt.expand(batch, timesteps, -1, -1)
    encoded = lam_inner.encoder(torch.cat([action_pad, patches], dim=2))
    moments = lam_inner.fc(encoded[:, 1:, 0].reshape(batch * (timesteps - 1), lam_inner.model_dim))
    z_mu, z_var = torch.chunk(moments, 2, dim=1)
    z_mu = z_mu.float()
    z_var = z_var.float()
    return {
        "patches": patches,
        "z_mu": z_mu,
        "z_var": z_var,
        "z_rep": z_mu.reshape(batch, timesteps - 1, 1, lam_inner.latent_dim),
        "z_rep_flat": z_mu,
    }
