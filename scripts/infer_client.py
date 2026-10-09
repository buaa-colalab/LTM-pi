#!/usr/bin/env python3
"""Minimal client for one reset -> add_buffer -> infer round trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="ws://127.0.0.1:8200")
    parser.add_argument("--input", type=Path, required=True, help="NPZ containing images and wrist_images.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--exec-start-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional output NPZ path.")
    return parser.parse_args()


def _call(connection: Any, packer: Any, message: dict[str, Any]) -> dict[str, Any]:
    from openpi_client import msgpack_numpy

    connection.send(packer.pack(message))
    response = msgpack_numpy.unpackb(connection.recv())
    if not isinstance(response, dict):
        raise TypeError(f"Server returned {type(response).__name__}, expected dict")
    if response.get("error"):
        raise RuntimeError(response["error"])
    return response


def _state(data: Any) -> np.ndarray:
    if "state" in data:
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
    elif "joint_state" in data:
        joints = np.asarray(data["joint_state"], dtype=np.float32).reshape(-1)
        if joints.shape != (7,):
            raise ValueError(f"joint_state must have shape (7,), got {joints.shape}")
        state = np.zeros(8, dtype=np.float32)
        state[:7] = joints
    else:
        raise KeyError("Input NPZ must contain state[8] or joint_state[7]")
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise ValueError(f"state must be finite with shape (8,), got {state.shape}")
    return state


def main() -> None:
    args = parse_args()
    if args.exec_start_idx < 0:
        raise ValueError("--exec-start-idx must be non-negative")
    with np.load(args.input, allow_pickle=False) as data:
        images = np.asarray(data["images"])
        wrist_images = np.asarray(data["wrist_images"])
        state = _state(data)
    if images.ndim != 4 or images.shape[-1] != 3 or images.shape != wrist_images.shape:
        raise ValueError(
            f"images and wrist_images must share [T,H,W,3], got {images.shape} and {wrist_images.shape}"
        )
    if args.exec_start_idx >= len(images):
        raise ValueError("--exec-start-idx must identify a frame present in the initial buffer")

    import websockets.sync.client

    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()
    with websockets.sync.client.connect(
        args.server,
        compression=None,
        max_size=None,
        ping_timeout=600,
        open_timeout=120,
        close_timeout=30,
    ) as connection:
        metadata = msgpack_numpy.unpackb(connection.recv())
        reset = _call(connection, packer, {"reset": True, "seed": args.seed})
        added = _call(
            connection,
            packer,
            {
                "add_buffer": True,
                "images": images[:, None],
                "wrist_images": wrist_images[:, None],
                "exec_start_idx": args.exec_start_idx,
            },
        )
        output = _call(
            connection,
            packer,
            {
                "observation/image": images[-1],
                "observation/wrist_image": wrist_images[-1],
                "observation/state": state,
                "prompt": args.prompt,
            },
        )

    actions = np.asarray(output["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.all(np.isfinite(actions)):
        raise ValueError(f"Invalid actions returned by server: {actions.shape}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, actions=actions)
    summary = {
        "server": args.server,
        "metadata": metadata,
        "reset": reset,
        "add_buffer": added,
        "memory_timing": output.get("memory_timing"),
        "actions_shape": list(actions.shape),
        "output": str(args.output.resolve()) if args.output else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
