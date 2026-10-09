# Data contract

This document contains the detailed data requirements used by validation and
training. Most users only need the directory layout in the main README and can
run `bin/validate.sh` to check these rules automatically.

## Dataset views

The release uses one LeRobot v2.1 dataset, `video`. Its parquet rows contain
metadata and state/action fields, while `image` and `wrist_image` are stored as
H.264 streams. The same dataset is used for cache generation, norm statistics,
and training.

Cache and anchor manifests use the `meta/episodes.jsonl` digest as the
storage-independent row-alignment identity. This also permits an existing
artifact generated from an image-backed copy of the same episodes to be used
with the video-only release without rewriting its source provenance.

Expected release inventory:

| Field | Value |
|---|---:|
| Episodes | 16,000 |
| Frames | 7,636,924 |
| Instruction tasks | 134 |
| Maximum episode length | 1,800 frames |
| Video streams | 32,000 |
| Video views | `image`, `wrist_image` |
| FPS | 10 |
| Codec / GOP | H.264 / 10 |

Each parquet row contains:

- `state`: float32 `[8]`
- `actions`: float32 `[8]`
- `exec_start_idx`: execution start within the episode
- `is_demo`: demonstration marker
- consecutive `frame_index`, `episode_index`, and `task_index`

The machine-readable version is `configs/data_contract.json`.

## DreamDojo memory

The two 32-D view features are concatenated into one 64-D FP32 row. Transition
alignment is:

```text
row_0 = zero and invalid
row_i = DreamDojo-LAM-400k(frame_(i-1), frame_i), i >= 1
```

The cache uses stride 1 and stores
`concat(image_z_mu[32], wrist_image_z_mu[32])`. The released DreamDojo LAM
cache was generated with an inference-only export whose tensors are identical to
the official `LAM_400k.ckpt`. The runtime accepts both the legacy export and the
official DreamDojo Lightning checkpoint. The expected SHA-256 of the official
checkpoint is:

```text
d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f
```

Preprocessing is a 4:3 center crop followed by `480×640` and `240×320`
resizes. The runtime modules are vendored from NVIDIA DreamDojo commit
`02f119b759d5c7f84a399fdeea3c6e82e7ed6cff`.

## Segment and lag semantics

| ID | Meaning |
|---:|---|
| 0 | padding |
| 1 | demonstration transition |
| 2 | demo-to-execution boundary |
| 3 | execution transition |

The fixed lag removes only the newest 10 segment-3 tokens. Demonstration and
boundary tokens are retained. Memory remains left padded so the newest visible
transition is aligned to the right edge of the memory window.

## Checkpoint structure

A committed checkpoint must contain:

```text
_CHECKPOINT_METADATA
params/_METADATA
params/manifest.ocdbt
assets/**/norm_stats.json
```

There must be exactly one `norm_stats.json`. Validate a checkpoint with:

```bash
bin/validate.sh --checkpoint /path/to/checkpoint/STEP
```
