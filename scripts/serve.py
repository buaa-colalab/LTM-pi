#!/usr/bin/env python3
"""Serve the fixed-lag-10 RoboMME policy and its DreamDojo visual encoder.

Each websocket connection owns an independent causal memory history.  The
shared JAX policy and DreamDojo encoder are serialized on one GPU so several
simulator workers can overlap environment stepping without duplicating model
weights.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import http
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_RUNTIME_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "src" / "openpi").is_dir()
)
for _path in (str(_RUNTIME_ROOT), str(_RUNTIME_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


DEFAULT_CONFIG = "pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default"
DEFAULT_DREAMDOJO_RUNTIME_ROOT = _RUNTIME_ROOT / "third_party" / "dreamdojo" / "runtime"
EXPECTED_ENCODER_SHA256 = "d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f"
EXPECTED_CHECKPOINT_FORMAT = "dreamdojo.lightning"
LEGACY_ENCODER_SHA256 = "12ce318e48d5790fb3773f1027edd5a3babe8965027b4957e3ed62d5d19fc386"
LEGACY_CHECKPOINT_FORMAT = "cdlam.stage1.inference"
COMPATIBLE_ENCODER_IDENTITIES = frozenset(
    {
        (EXPECTED_CHECKPOINT_FORMAT, EXPECTED_ENCODER_SHA256),
        (LEGACY_CHECKPOINT_FORMAT, LEGACY_ENCODER_SHA256),
    }
)
EXPECTED_ALIGNMENT = "row_0_invalid_zero; row_i=z_mu(frame_i-1,frame_i), i>=1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--memory-manifest",
        type=Path,
        required=True,
        help="Path to the matching 64-D head+wrist FP32 memory-cache manifest.",
    )
    parser.add_argument(
        "--dreamdojo-runtime-root",
        type=Path,
        default=DEFAULT_DREAMDOJO_RUNTIME_ROOT,
        help="Vendored DreamDojo runtime; defaults to third_party/dreamdojo/runtime.",
    )
    parser.add_argument(
        "--dreamdojo-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=200,
        help="Eval-only static prompt length. Must not be lower than the training config.",
    )
    parser.add_argument(
        "--memory-lag",
        type=int,
        default=10,
        help="Hide this many newest execution LAM tokens at inference; must match fixed-lag training.",
    )
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument(
        "--transition-frame-stride",
        type=int,
        default=None,
        help="Expected causal DreamDojo endpoint stride; defaults to the cache manifest value.",
    )
    parser.add_argument(
        "--verify-dreamdojo-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify the DreamDojo checkpoint SHA-256 (enabled by default).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config, checkpoint, manifest and encoder identity without loading GPU models.",
    )
    return parser.parse_args()


def _load_manifest(path: Path, latent_dim: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    path = path.expanduser().resolve()
    manifest = json.loads(path.read_text())
    normalization = manifest.get("latent_normalization")
    if not isinstance(normalization, dict):
        raise ValueError(f"No latent_normalization in {path}")
    mean = np.asarray(normalization.get("mean"), dtype=np.float64)
    std = np.asarray(normalization.get("std"), dtype=np.float64)
    if mean.shape != (latent_dim,) or std.shape != (latent_dim,):
        raise ValueError(f"LAM stats shape mismatch: mean={mean.shape}, std={std.shape}, expected {(latent_dim,)}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("LAM normalization must be finite with positive standard deviations")
    if manifest.get("status") != "complete" or manifest.get("missing_episodes") not in ([], None):
        raise ValueError(f"LAM cache manifest is not complete: {path}")
    if int(manifest.get("latent_dim", -1)) != latent_dim:
        raise ValueError(f"LAM manifest latent_dim={manifest.get('latent_dim')} does not match model {latent_dim}")
    expected = {
        "alignment": EXPECTED_ALIGNMENT,
        "image_keys": ["image", "wrist_image"],
        "latent_layout": "concat(image_z_mu[32],wrist_image_z_mu[32])",
        "latent_dtype": "float32",
        "preprocess_version": "dreamdojo.lam400k.rgb_uint8.crop_4x3.resize_480x640.resize_240x320.v1",
        "transition_frame_stride": 1,
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"Memory manifest is not the fixed-lag-10 training manifest: {mismatches}")
    manifest_encoder = (
        str(manifest.get("checkpoint_format", "")),
        str(manifest.get("checkpoint_sha256", "")),
    )
    if manifest_encoder not in COMPATIBLE_ENCODER_IDENTITIES:
        raise ValueError(f"Memory manifest has an unsupported DreamDojo encoder identity: {manifest_encoder}")
    return manifest, mean, std


def _checkpoint_is_committed(path: Path) -> None:
    path = path.expanduser().resolve()
    required = (
        path / "_CHECKPOINT_METADATA",
        path / "params" / "_METADATA",
        path / "params" / "manifest.ocdbt",
    )
    missing = [str(item) for item in required if not item.is_file()]
    norm_stats = tuple((path / "assets").rglob("norm_stats.json")) if (path / "assets").is_dir() else ()
    if missing or len(norm_stats) != 1:
        raise FileNotFoundError(
            f"Checkpoint is incomplete: missing={missing}, norm_stats={[str(item) for item in norm_stats]}"
        )
    metadata = json.loads((path / "_CHECKPOINT_METADATA").read_text())
    if not int(metadata.get("commit_timestamp_nsecs", 0)):
        raise ValueError(f"Checkpoint has no Orbax commit timestamp: {path}")


class MemorySession:
    def __init__(
        self,
        *,
        policy: Any,
        dreamdojo_model: Any,
        encode_full: Any,
        preprocess: Any,
        device: str,
        memory_horizon: int,
        latent_dim: int,
        action_horizon: int,
        action_dim: int,
        prompt_tokenizer: Any,
        max_token_len: int,
        pair_batch_size: int,
        use_memory_segments: bool,
        use_demo_anchor: bool,
        use_execution_anchor: bool,
        transition_frame_stride: int,
        two_view_memory: bool,
        memory_lag: int,
        latent_storage_dtype: str,
    ) -> None:
        self.policy = policy
        self.dreamdojo_model = dreamdojo_model
        self.encode_full = encode_full
        self.preprocess = preprocess
        self.device = device
        self.memory_horizon = memory_horizon
        self.latent_dim = latent_dim
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.prompt_tokenizer = prompt_tokenizer
        self.max_token_len = max_token_len
        self.pair_batch_size = pair_batch_size
        self.use_memory_segments = use_memory_segments
        self.use_demo_anchor = use_demo_anchor
        self.use_execution_anchor = use_execution_anchor
        self.transition_frame_stride = transition_frame_stride
        self.two_view_memory = two_view_memory
        self.memory_lag = memory_lag
        self.latent_storage_dtype = latent_storage_dtype
        self.reset(0)

    def reset(self, seed: int) -> dict[str, Any]:
        self.previous_stride_head_frame: np.ndarray | None = None
        self.previous_stride_wrist_frame: np.ndarray | None = None
        self.latents: list[np.ndarray] = []
        self.segment_ids: list[int] = []
        self.exec_start_idx: int | None = None
        self.demo_start_image: np.ndarray | None = None
        self.execution_start_image: np.ndarray | None = None
        self.total_frames = 0
        self.total_transitions = 0
        self.dropped_transitions = 0
        self.inference_calls = 0
        self.rng = np.random.default_rng(int(seed))
        return {
            "reset_finished": True,
            "seed": int(seed),
            "memory_horizon": self.memory_horizon,
            "memory_lag": self.memory_lag,
        }

    @staticmethod
    def _parse_images(message: dict[str, Any]) -> np.ndarray:
        images = np.asarray(message.get("images"))
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
            raise ValueError(f"add_buffer images must be [T,1,H,W,3] or [T,H,W,3], got {images.shape}")
        if np.issubdtype(images.dtype, np.floating):
            images = np.clip(images * (255.0 if float(np.nanmax(images)) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8, copy=False)
        return np.ascontiguousarray(images)

    def _encode_dreamdojo_pairs(self, frames: np.ndarray) -> np.ndarray:
        """Encode explicit causal pairs represented by a [previous, endpoints...] stream."""
        from scripts.precompute_dreamdojo_memory import encode_causal_latents

        encoded = encode_causal_latents(
            self.dreamdojo_model,
            self.encode_full,
            frames,
            device=self.device,
            pair_batch_size=self.pair_batch_size,
            # `frames` is the compressed causal endpoint stream
            # [frame_0, frame_4, frame_8, ...].  Adjacent elements therefore
            # already represent the training pairs 0->4, 4->8, ... .
            transition_frame_stride=1,
        )
        return encoded[1:]

    def add_buffer(self, message: dict[str, Any]) -> dict[str, Any]:
        from openpi.training.cdlam_memory_dataset import classify_memory_segments

        start = time.monotonic()
        images = self._parse_images(message)
        wrist_images = self._parse_images({"images": message.get("wrist_images")}) if self.two_view_memory else None
        if wrist_images is not None and wrist_images.shape != images.shape:
            raise ValueError(
                "wrist_images must have exactly the same [T,H,W,3] shape as images: "
                f"head={images.shape}, wrist={wrist_images.shape}"
            )
        semantic_inputs = self.use_memory_segments or self.use_demo_anchor or self.use_execution_anchor
        if self.total_frames == 0 and semantic_inputs:
            if "exec_start_idx" not in message:
                raise ValueError("The first add_buffer must provide exec_start_idx for typed RoboMME memory")
            exec_start_idx = int(message["exec_start_idx"])
            if exec_start_idx < 0:
                raise ValueError(f"exec_start_idx must be non-negative, got {exec_start_idx}")
            if self.use_execution_anchor and exec_start_idx >= len(images):
                raise ValueError(
                    "The first add_buffer must contain images[exec_start_idx] for the execution anchor: "
                    f"exec_start_idx={exec_start_idx}, frames={len(images)}"
                )
            # Commit semantic state only after validating the complete first
            # chunk, so a rejected request cannot leave a half-initialized
            # anchor session behind.
            self.exec_start_idx = exec_start_idx
            if self.use_demo_anchor:
                self.demo_start_image = images[0].copy() if self.exec_start_idx > 0 else np.zeros_like(images[0])
            if self.use_execution_anchor:
                self.execution_start_image = images[self.exec_start_idx].copy()
        elif semantic_inputs and "exec_start_idx" in message:
            # Evaluator chunks after reset use 0 to state that every newly
            # appended frame is execution. The session retains the original
            # global boundary for endpoint classification.
            supplied_exec_start = int(message["exec_start_idx"])
            if supplied_exec_start not in (0, self.exec_start_idx):
                raise ValueError(
                    f"exec_start_idx changed within a session: {self.exec_start_idx} -> {supplied_exec_start}"
                )
        from scripts.precompute_dreamdojo_memory import preprocess_frames

        head_current = preprocess_frames(images, self.preprocess, chunk_size=min(32, len(images)))
        wrist_current = (
            preprocess_frames(wrist_images, self.preprocess, chunk_size=min(32, len(wrist_images)))
            if wrist_images is not None
            else None
        )
        # The training cache contains actual pairs 0->4, 4->8, ... rather
        # than every fourth adjacent-frame latent. Preserve those endpoints
        # even when evaluator messages split the frame stream arbitrarily.
        left_head = self.previous_stride_head_frame
        left_wrist = self.previous_stride_wrist_frame
        head_stream: list[np.ndarray] = []
        wrist_stream: list[np.ndarray] = []
        endpoints: list[int] = []
        for local_index, head_frame in enumerate(head_current):
            endpoint = self.total_frames + local_index
            if endpoint % self.transition_frame_stride == 0:
                if endpoint == 0:
                    left_head = head_frame.copy()
                    if wrist_current is not None:
                        left_wrist = wrist_current[local_index].copy()
                else:
                    if left_head is None:
                        raise RuntimeError("Missing previous stride endpoint for causal DreamDojo pair")
                    if not head_stream:
                        head_stream.append(left_head)
                        if wrist_current is not None:
                            if left_wrist is None:
                                raise RuntimeError("Missing previous wrist stride endpoint for causal DreamDojo pair")
                            wrist_stream.append(left_wrist)
                    head_stream.append(head_frame)
                    if wrist_current is not None:
                        wrist_stream.append(wrist_current[local_index])
                    endpoints.append(endpoint)
                    left_head = head_frame.copy()
                    if wrist_current is not None:
                        left_wrist = wrist_current[local_index].copy()
        self.previous_stride_head_frame = left_head
        self.previous_stride_wrist_frame = left_wrist
        if head_stream:
            new_head_latents = self._encode_dreamdojo_pairs(np.stack(head_stream))
            if wrist_current is None:
                new_latents = new_head_latents
            else:
                new_wrist_latents = self._encode_dreamdojo_pairs(np.stack(wrist_stream))
                new_latents = np.concatenate((new_head_latents, new_wrist_latents), axis=1)
        else:
            new_latents = np.empty((0, self.latent_dim), dtype=np.float32)
        if new_latents.size:
            # Match the cache's storage precision before the trainer/runtime
            # consumes float32 arrays.  The current 16x1000 cache is FP32.
            storage_dtype = np.dtype(self.latent_storage_dtype)
            new_latents = new_latents.astype(storage_dtype).astype(np.float32)
            if new_latents.shape[1:] != (self.latent_dim,) or not np.all(np.isfinite(new_latents)):
                raise ValueError(f"Invalid online LAM latents: {new_latents.shape}")
            self.latents.extend(new_latents)
            if self.use_memory_segments:
                if self.exec_start_idx is None:
                    raise RuntimeError("Missing exec_start_idx for typed memory")
                new_segments = classify_memory_segments(np.asarray(endpoints, dtype=np.int64), self.exec_start_idx)
                self.segment_ids.extend(int(value) for value in new_segments)
            self.total_transitions += len(new_latents)
        self.total_frames += len(head_current)
        overflow = max(0, len(self.latents) - self.memory_horizon)
        if overflow:
            del self.latents[:overflow]
            if self.use_memory_segments:
                del self.segment_ids[:overflow]
            self.dropped_transitions += overflow
        return {
            "add_buffer_finished": True,
            "added_frames": len(head_current),
            "added_transitions": len(new_latents),
            "memory_length": len(self.latents),
            "total_transitions": self.total_transitions,
            "dropped_transitions": self.dropped_transitions,
            "add_buffer_time_ms": (time.monotonic() - start) * 1000.0,
        }

    def infer(self, message: dict[str, Any]) -> dict[str, Any]:
        from openpi.training.cdlam_memory_dataset import drop_execution_lam_tail

        # The stride-4 encoder keeps its causal endpoint in
        # `previous_stride_head_frame`; the old adjacent-frame runtime field
        # `previous_lam_frame` no longer exists.  Frame count is the direct
        # contract for whether any buffer data has been accepted.
        if self.total_frames == 0:
            raise RuntimeError("infer called before add_buffer")
        prompt = message.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("infer requires a non-empty string prompt")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        prompt_tokens = self.prompt_tokenizer._tokenizer.encode(cleaned, add_bos=True)  # noqa: SLF001
        prompt_tokens += self.prompt_tokenizer._tokenizer.encode("\n")  # noqa: SLF001
        if len(prompt_tokens) > self.max_token_len:
            raise ValueError(
                f"Prompt needs {len(prompt_tokens)} tokens but the trained static limit is {self.max_token_len}; "
                "refusing to silently truncate"
            )
        if self.latents:
            memory = np.stack(self.latents, axis=0).astype(np.float32, copy=False)
            mask = np.ones(len(memory), dtype=np.bool_)
            segments = np.asarray(self.segment_ids, dtype=np.int32) if self.use_memory_segments else None
        else:
            memory = np.zeros((1, self.latent_dim), dtype=np.float32)
            mask = np.zeros(1, dtype=np.bool_)
            segments = np.zeros(1, dtype=np.int32) if self.use_memory_segments else None
        if segments is None:
            raise RuntimeError("Fixed execution-tail lag requires typed memory segments")
        stored_memory_length = int(mask.sum())
        memory, mask, segments = drop_execution_lam_tail(memory, mask, segments, self.memory_lag)
        lag_hidden_transitions = stored_memory_length - int(mask.sum())
        if len(memory) == 0:
            memory = np.zeros((1, self.latent_dim), dtype=np.float32)
            mask = np.zeros(1, dtype=np.bool_)
            segments = np.zeros(1, dtype=np.int32)
        left = self.memory_horizon - len(memory)
        if left < 0:
            raise RuntimeError("Memory history exceeded the configured rolling horizon")
        element = dict(message)
        element["memory_latents"] = np.pad(memory, ((left, 0), (0, 0)), constant_values=0)
        element["memory_mask"] = np.pad(mask, (left, 0), constant_values=False)
        if self.use_memory_segments:
            if len(self.segment_ids) != len(self.latents):
                raise RuntimeError("Online memory segments are not aligned with latent transitions")
            element["memory_segment_ids"] = np.pad(segments, (left, 0), constant_values=0)
        if self.use_demo_anchor:
            if self.demo_start_image is None or self.exec_start_idx is None:
                raise RuntimeError("Online demo anchor was not initialized")
            element["memory_demo_start_image"] = self.demo_start_image
            element["memory_demo_start_mask"] = np.bool_(self.exec_start_idx > 0)
        if self.use_execution_anchor:
            if self.execution_start_image is None or self.exec_start_idx is None:
                raise RuntimeError("Online execution anchor was not initialized")
            element["memory_execution_start_image"] = self.execution_start_image
            element["memory_execution_start_mask"] = np.bool_(True)
        noise = self.rng.standard_normal((self.action_horizon, self.action_dim)).astype(np.float32)
        output = self.policy.infer(element, noise=noise)
        self.inference_calls += 1
        output["memory_timing"] = {
            "memory_length": int(mask.sum()),
            "stored_memory_length": stored_memory_length,
            "memory_lag": self.memory_lag,
            "lag_hidden_transitions": lag_hidden_transitions,
            "total_frames": self.total_frames,
            "total_transitions": self.total_transitions,
            "dropped_transitions": self.dropped_transitions,
            "inference_calls": self.inference_calls,
            "prompt_tokens": len(prompt_tokens),
        }
        return output


class Server:
    def __init__(self, *, session_factory: Any, host: str, port: int, metadata: dict[str, Any]) -> None:
        self.session_factory = session_factory
        self.host = host
        self.port = port
        self.metadata = metadata
        self.gpu_lock = asyncio.Lock()

    async def handler(self, websocket: Any) -> None:
        import websockets

        from openpi_client import msgpack_numpy

        packer = msgpack_numpy.Packer()
        session = self.session_factory()
        await websocket.send(packer.pack(self.metadata))
        logging.info("connection opened: %s", websocket.remote_address)
        try:
            while True:
                message = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(message, dict):
                    raise TypeError("websocket payload must be a dictionary")
                async with self.gpu_lock:
                    if message.get("reset", False):
                        response = await asyncio.to_thread(session.reset, int(message.get("seed", 0)))
                    elif message.get("add_buffer", False):
                        response = await asyncio.to_thread(session.add_buffer, message)
                    else:
                        response = await asyncio.to_thread(session.infer, message)
                await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            logging.info("connection closed: %s", websocket.remote_address)
        except Exception:
            error = traceback.format_exc()
            logging.error("connection failed:\n%s", error)
            try:
                await websocket.send(packer.pack({"error": error}))
            finally:
                await websocket.close(code=1011, reason="policy server error")

    async def health(self, connection: Any, request: Any) -> Any:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def run(self) -> None:
        import websockets.asyncio.server

        async with websockets.asyncio.server.serve(
            self.handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_timeout=600,
            process_request=self.health,
        ) as server:
            logging.info("ready on ws://%s:%d", self.host, self.port)
            await server.serve_forever()


def main() -> None:
    args = parse_args()
    if args.num_inference_steps <= 0 or args.pair_batch_size <= 0:
        raise ValueError("num-inference-steps and pair-batch-size must be positive")
    if args.memory_lag != 10:
        raise ValueError(f"This checkpoint family was trained with fixed memory lag 10, got {args.memory_lag}")
    if args.config != DEFAULT_CONFIG:
        raise ValueError(f"This release only supports config {DEFAULT_CONFIG!r}, got {args.config!r}")
    checkpoint = args.checkpoint.expanduser().resolve()
    _checkpoint_is_committed(checkpoint)

    from openpi.training import config as training_config

    config = training_config.get_config(args.config)
    if not config.model.use_memory:
        raise ValueError(f"Config {args.config} is not memory-enabled")
    expected_model_values = {
        "action_horizon": 50,
        "memory_horizon": 1800,
        "memory_latent_dim": 64,
        "memory_prefix_order": "language_memory_vision",
        "memory_same_prefix_block": False,
        "memory_segment_embedding_after_projection": True,
    }
    model_mismatches = {
        key: (getattr(config.model, key), value)
        for key, value in expected_model_values.items()
        if getattr(config.model, key) != value
    }
    if model_mismatches:
        raise ValueError(f"Config is not compatible with this standalone runtime: {model_mismatches}")
    if config.policy_metadata.get("memory_random_drop_execution_tail") != "fixed_integer_10_per_sample":
        raise ValueError("Config metadata does not declare fixed execution-tail lag 10")
    if config.policy_metadata.get("memory_encoder_checkpoint_sha256") not in {
        identity[1] for identity in COMPATIBLE_ENCODER_IDENTITIES
    }:
        raise ValueError("Config metadata does not match the released DreamDojo LAM checkpoint")
    training_max_token_len = int(config.model.max_token_len)
    if args.max_token_len != training_max_token_len:
        raise ValueError(
            f"max-token-len must match the training value {training_max_token_len}, got {args.max_token_len}"
        )
    manifest, mean, std = _load_manifest(args.memory_manifest, config.model.memory_latent_dim)
    manifest_stride = int(manifest.get("transition_frame_stride", 1))
    transition_frame_stride = args.transition_frame_stride or manifest_stride
    if transition_frame_stride != manifest_stride or transition_frame_stride <= 0:
        raise ValueError(
            f"transition_frame_stride={transition_frame_stride} does not match manifest stride={manifest_stride}"
        )
    two_view_memory = manifest.get("image_keys") == ["image", "wrist_image"]
    if two_view_memory != (config.model.memory_latent_dim == 64):
        raise ValueError(
            "memory manifest/model view contract mismatch: "
            f"two_view={two_view_memory}, latent_dim={config.model.memory_latent_dim}"
        )
    from scripts.precompute_dreamdojo_memory import load_dreamdojo_encoder, resolve_checkpoint_identity, resolve_device

    memory_encoder_kind = "dreamdojo_lam400k"
    memory_encoder_checkpoint = args.dreamdojo_checkpoint
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            max_token_len=args.max_token_len,
            memory_latent_mean=tuple(float(value) for value in mean),
            memory_latent_std=tuple(float(value) for value in std),
        ),
    )
    encoder_identity = resolve_checkpoint_identity(
        memory_encoder_checkpoint,
        explicit_sha256=None,
        verify_sha256=args.verify_dreamdojo_sha256,
    )
    encoder_key = (encoder_identity.format, encoder_identity.sha256)
    if encoder_key not in COMPATIBLE_ENCODER_IDENTITIES:
        raise ValueError(f"Unsupported DreamDojo encoder identity: {encoder_key}")
    preflight = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "config": args.config,
        "memory_lag": args.memory_lag,
        "max_token_len": args.max_token_len,
        "memory_manifest": str(args.memory_manifest.resolve()),
        "memory_manifest_sha256": hashlib.sha256(args.memory_manifest.read_bytes()).hexdigest(),
        "memory_encoder_checkpoint_sha256": encoder_identity.sha256,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies import policy_config

    device = resolve_device(args.device, None)
    logging.info("loading JAX policy from %s", checkpoint)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"num_steps": args.num_inference_steps},
    )
    prompt_tokenizer = PaligemmaTokenizer(max_len=config.model.max_token_len)
    logging.info("loading %s memory encoder on %s", memory_encoder_kind, device)
    dreamdojo_model, encode_full, preprocess, encoder_identity = load_dreamdojo_encoder(
        args.dreamdojo_runtime_root,
        encoder_identity,
        device,
    )
    metadata = {
        "backend": "jax",
        "checkpoint": str(checkpoint),
        "config": args.config,
        "memory_horizon": config.model.memory_horizon,
        "memory_latent_dim": config.model.memory_latent_dim,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
        "num_inference_steps": args.num_inference_steps,
        "training_max_token_len": training_max_token_len,
        "runtime_max_token_len": config.model.max_token_len,
        "memory_manifest": str(args.memory_manifest.resolve()),
        "memory_manifest_sha256": hashlib.sha256(args.memory_manifest.read_bytes()).hexdigest(),
        "memory_encoder_kind": memory_encoder_kind,
        "memory_encoder_checkpoint_sha256": encoder_identity.sha256,
        "memory_lag": args.memory_lag,
        "memory_latent_storage_dtype": manifest["latent_dtype"],
        "rolling_memory": True,
        "memory_segment_embedding": config.model.memory_segment_embedding,
        "memory_same_prefix_block": config.model.memory_same_prefix_block,
        "memory_segment_embedding_after_projection": config.model.memory_segment_embedding_after_projection,
        "memory_demo_anchor": config.model.memory_demo_anchor,
        "memory_execution_anchor": config.model.memory_execution_anchor,
        "transition_frame_stride": transition_frame_stride,
        "two_view_memory": two_view_memory,
    }

    def session_factory() -> MemorySession:
        return MemorySession(
            policy=policy,
            dreamdojo_model=dreamdojo_model,
            encode_full=encode_full,
            preprocess=preprocess,
            device=device,
            memory_horizon=config.model.memory_horizon,
            latent_dim=config.model.memory_latent_dim,
            action_horizon=config.model.action_horizon,
            action_dim=config.model.action_dim,
            prompt_tokenizer=prompt_tokenizer,
            max_token_len=config.model.max_token_len,
            pair_batch_size=args.pair_batch_size,
            use_memory_segments=config.model.memory_segment_embedding,
            use_demo_anchor=config.model.memory_demo_anchor,
            use_execution_anchor=config.model.memory_execution_anchor,
            transition_frame_stride=transition_frame_stride,
            two_view_memory=two_view_memory,
            memory_lag=args.memory_lag,
            latent_storage_dtype=str(manifest["latent_dtype"]),
        )

    server = Server(session_factory=session_factory, host=args.host, port=args.port, metadata=metadata)
    asyncio.run(server.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    main()
