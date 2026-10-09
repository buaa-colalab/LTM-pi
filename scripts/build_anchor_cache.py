#!/usr/bin/env python3
"""Build mmap caches for RoboMME demo and execution-start visual anchors."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _episode_path(root: Path, data_path: str, episode_index: int, chunk_size: int) -> Path:
    return root / data_path.format(
        episode_index=episode_index,
        episode_chunk=episode_index // chunk_size,
    )


def _decode_image(value: dict, dataset_root: Path) -> np.ndarray:
    payload = value.get("bytes")
    if payload is not None:
        image = Image.open(io.BytesIO(payload))
    else:
        relative = value.get("path")
        if not relative:
            raise ValueError("Image row contains neither bytes nor path")
        image = Image.open(dataset_root / relative)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _video_path(
    dataset_root: Path,
    template: str,
    episode_index: int,
    chunk_size: int,
) -> Path:
    path = dataset_root / template.format(
        episode_index=episode_index,
        episode_chunk=episode_index // chunk_size,
        video_key="image",
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing head-camera episode video: {path}")
    return path


def _decode_video_anchors(path: Path, execution_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode frame zero and the execution-start frame from one episode video."""
    import av

    first = None
    execution = None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream in {path}, got {len(container.streams.video)}")
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == 0:
                first = frame.to_ndarray(format="rgb24")
            if index == execution_index:
                execution = frame.to_ndarray(format="rgb24")
                break
    if first is None or execution is None:
        raise ValueError(f"Video {path} does not contain execution frame {execution_index}")
    return first, execution


def main(
    dataset_root: Path,
    output_dir: Path,
) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    info = json.loads(info_path.read_text())
    total_episodes = int(info["total_episodes"])
    data_path = str(info["data_path"])
    chunk_size = int(info["chunks_size"])
    image_shape = tuple(int(value) for value in info["features"]["image"]["shape"])
    image_is_video = info["features"]["image"].get("dtype") == "video"
    video_path_template = info.get("video_path")
    if image_is_video and not isinstance(video_path_template, str):
        raise ValueError("Video-backed dataset is missing its video_path template")
    if len(image_shape) != 3 or image_shape[-1] != 3:
        raise ValueError(f"Expected RGB head images, got shape {image_shape}")

    episode_paths = [
        _episode_path(dataset_root, data_path, episode_index, chunk_size) for episode_index in range(total_episodes)
    ]
    missing = [str(path) for path in episode_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} episode parquet files; first={missing[0]}")

    exec_start_idx = np.empty(total_episodes, dtype=np.int32)
    for episode_index, path in enumerate(episode_paths):
        column = pq.read_table(path, columns=["exec_start_idx"])["exec_start_idx"]
        values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.int64)
        if not len(values) or np.any(values != values[0]) or values[0] < 0 or values[0] >= len(values):
            raise ValueError(f"Invalid exec_start_idx in episode {episode_index}: {values[:4]}")
        exec_start_idx[episode_index] = int(values[0])

    demo_episodes = int((exec_start_idx > 0).sum())
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=output_dir))
    try:
        images_path = temporary_dir / "images.npy"
        execution_images_path = temporary_dir / "execution_images.npy"
        mapping_path = temporary_dir / "episode_to_anchor.npy"
        exec_path = temporary_dir / "exec_start_idx.npy"
        images = np.lib.format.open_memmap(
            images_path,
            mode="w+",
            dtype=np.uint8,
            shape=(demo_episodes, *image_shape),
        )
        execution_images = np.lib.format.open_memmap(
            execution_images_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total_episodes, *image_shape),
        )
        episode_to_anchor = np.full(total_episodes, -1, dtype=np.int32)
        anchor_index = 0
        for episode_index, path in enumerate(episode_paths):
            columns = ["frame_index", "is_demo"] if image_is_video else ["image", "frame_index", "is_demo"]
            table = pq.read_table(path, columns=columns)
            execution_index = int(exec_start_idx[episode_index])
            if int(table["frame_index"][execution_index].as_py()) != execution_index:
                raise ValueError(f"Episode {episode_index} execution anchor is not frame {execution_index}")
            if bool(table["is_demo"][execution_index].as_py()):
                raise ValueError(f"Episode {episode_index} execution anchor is incorrectly marked as demo")
            if image_is_video:
                image, execution_image = _decode_video_anchors(
                    _video_path(dataset_root, video_path_template, episode_index, chunk_size),
                    execution_index,
                )
            else:
                image = _decode_image(table["image"][0].as_py(), dataset_root)
                execution_image = _decode_image(table["image"][execution_index].as_py(), dataset_root)
            if execution_image.shape != image_shape:
                raise ValueError(
                    f"Episode {episode_index} execution anchor shape is {execution_image.shape}, expected {image_shape}"
                )
            execution_images[episode_index] = execution_image
            if execution_index > 0:
                if int(table["frame_index"][0].as_py()) != 0 or not bool(table["is_demo"][0].as_py()):
                    raise ValueError(f"Episode {episode_index} does not start with a demo frame")
                if image.shape != image_shape:
                    raise ValueError(f"Episode {episode_index} anchor shape is {image.shape}, expected {image_shape}")
                images[anchor_index] = image
                episode_to_anchor[episode_index] = anchor_index
                anchor_index += 1
        images.flush()
        execution_images.flush()
        del images
        del execution_images
        if anchor_index != demo_episodes:
            raise RuntimeError(f"Wrote {anchor_index} anchors, expected {demo_episodes}")
        np.save(mapping_path, episode_to_anchor, allow_pickle=False)
        np.save(exec_path, exec_start_idx, allow_pickle=False)

        info_sha256 = _sha256(info_path)
        episodes_sha256 = _sha256(episodes_path)
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "definition": "episode_frame_0_demo_and_episode_frame_exec_start_execution",
            "source_image_key": "image",
            "source_media": "video" if image_is_video else "image",
            "dataset_root": str(dataset_root),
            "dataset_info_sha256": info_sha256,
            "dataset_episodes_sha256": episodes_sha256,
            "dataset_fingerprint": hashlib.sha256(f"{info_sha256}:{episodes_sha256}".encode()).hexdigest(),
            "total_episodes": total_episodes,
            "demo_episodes": demo_episodes,
            "image_shape": list(image_shape),
            "image_dtype": "uint8",
            "images_file": "images.npy",
            "images_sha256": _sha256(images_path),
            "execution_images_file": "execution_images.npy",
            "execution_images_sha256": _sha256(execution_images_path),
            "episode_to_anchor_file": "episode_to_anchor.npy",
            "episode_to_anchor_sha256": _sha256(mapping_path),
            "exec_start_idx_file": "exec_start_idx.npy",
            "exec_start_idx_sha256": _sha256(exec_path),
        }
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for path in temporary_dir.iterdir():
            os.replace(path, output_dir / path.name)
        temporary_dir.rmdir()
    except BaseException:
        for path in temporary_dir.glob("*"):
            path.unlink(missing_ok=True)
        temporary_dir.rmdir()
        output_dir.rmdir()
        raise
    print(f"Wrote {demo_episodes} demo anchors and {total_episodes} execution anchors to {output_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
