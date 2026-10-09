# LTM-Pi for RoboMME

English | [简体中文](README_zh.md)

PI0.5 training and online inference for long-horizon RoboMME manipulation tasks. The
default experiment uses the RoboMME-10x dataset with 10 FPS H.264 video, dual-view
memory produced by DreamDojo LAM 400k, and a fixed lag that hides the latest 10
execution memory tokens during both training and inference.

This repository contains the source code required for training, cache generation,
normalization-statistics computation, and WebSocket inference. The minimal DreamDojo
LAM runtime is vendored under `third_party/dreamdojo/`; no additional DreamDojo or
CD-LAM repository clone is required. Datasets and model weights are not distributed
with the source code.

## Environment

Linux, Python 3.11, and CUDA 12 are recommended. The full training configuration uses
8×H100 GPUs. Cache generation and inference can run on a single NVIDIA GPU with BF16
support.

```bash
git clone https://github.com/buaa-colalab/LTM-pi.git ltm_pi_robomme
cd ltm_pi_robomme
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bin/setup.sh
```

The final command creates the sibling `../ltm_pi_robomme-resources` directory,
resolves its absolute path, and writes that path directly to `ROBOMME_RESOURCE_ROOT`
in a new `.env`. All other paths are derived from that resource root. Edit the value
or override individual variables only when your directory layout differs. An existing
`.env` is left unchanged. You may pass a custom resource directory as the first
argument to `bin/setup.sh`.

`.env` is machine-local and intentionally excluded from Git. Do not commit it; the
repository contains only the portable `configs/paths.env.example` template. Before
publishing, run `make release-check` to reject tracked local paths, Hugging Face
tokens, proxy credentials, and accidentally tracked environment files.

## Data and checkpoints

Download the external weights and datasets from the following locations:

| Resource | Download link | Default destination |
| --- | --- | --- |
| PI0.5 base parameters | [OpenPI model checkpoints](https://github.com/Physical-Intelligence/openpi#model-checkpoints) (`gs://openpi-assets/checkpoints/pi05_base/params`) | `ltm_pi_robomme-resources/models/pi05/params/` |
| DreamDojo LAM 400k weights | [LAM_400k.ckpt](https://huggingface.co/nvidia/DreamDojo/resolve/main/LAM_400k.ckpt?download=true) | `ltm_pi_robomme-resources/models/dreamdojo/` |
| RoboMME-10x video dataset | [maxenceSUN/RoboMME-10x](https://huggingface.co/datasets/maxenceSUN/RoboMME-10x) | `ltm_pi_robomme-resources/datasets/video/` |
| Precomputed DreamDojo transition cache | [maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache](https://huggingface.co/datasets/maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache) | `ltm_pi_robomme-resources/artifacts/memory/` |

Arrange the downloaded files as follows:

```text
ltm_pi_robomme-resources/
├── artifacts/
│   └── memory/          # optional precomputed DreamDojo transition cache
├── datasets/
│   └── video/            # H.264 RoboMME-10x; used for cache, norm stats, and training
└── models/
    ├── pi05/params/
    └── dreamdojo/LAM_400k.ckpt
```

The SHA-256 checksum of the DreamDojo checkpoint must be:

```text
d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f
```

### Download the precomputed memory cache (recommended)

The public cache was generated from the RoboMME-10x 16-task × 1,000-episode
corpus with the DreamDojo checkpoint above. After `bin/setup.sh` has created
`.env`, download it directly into the configured cache directory:

```bash
set -a
source .env
set +a
hf download maxenceSUN/RoboMME-10x_Dreamdojo_transition_cache \
  --repo-type dataset \
  --local-dir "${ROBOMME_MEMORY_CACHE}"
```

The dataset is public, so authentication is not required. It provides
`latents.npy`, `frame_index.npz`, and `manifest.json`, and lets you skip the
GPU-intensive `bin/prepare.sh memory` stage. It does not contain the anchor cache
or normalization statistics; prepare those locally and then validate everything:

```bash
bin/prepare.sh anchors
bin/prepare.sh norm-stats
bin/validate.sh
```

This cache is reusable only with the matching RoboMME corpus, transition contract,
and DreamDojo encoder. Recompute the memory cache if any of those inputs change.

## Prepare the cache and normalization statistics

If you downloaded the precomputed memory cache above, do **not** run
`bin/prepare.sh all` or `bin/prepare.sh memory`. Run only the `anchors` and
`norm-stats` stages shown above, followed by `bin/validate.sh`. The commands in
the remainder of this section are for generating the memory cache from scratch.

The following command generates the dual anchors, the head+wrist DreamDojo memory
cache, and the state/action normalization statistics required for training. On an
8-GPU machine, set `CACHE_GPUS` for cache generation as shown below. If it is unset,
the process uses the single GPU selected by `DREAMDOJO_DEVICE` (default: `cuda:0`).

```bash
CACHE_GPUS=0,1,2,3,4,5,6,7 \
CACHE_PARALLEL_VIEWS=1 \
PAIR_BATCH_SIZE=128 \
PREPROCESS_CHUNK_SIZE=32 \
CACHE_PREFETCH_EPISODES=2 \
bin/prepare.sh all
```

The stages can also be run separately for easier checkpointing and inspection:

```bash
bin/prepare.sh anchors
CACHE_GPUS=0,1,2,3,4,5,6,7 \
CACHE_PARALLEL_VIEWS=1 \
PAIR_BATCH_SIZE=128 \
PREPROCESS_CHUNK_SIZE=32 \
CACHE_PREFETCH_EPISODES=2 \
bin/prepare.sh memory
bin/prepare.sh norm-stats
```

The `memory` stage decodes frames from the H.264 `image` and `wrist_image` streams in
the video dataset, extracts a 32-D DreamDojo posterior mean from each view, and
concatenates them into a 64-D FP32 cache. The `norm-stats` stage reads only the
state/action columns from the same dataset's Parquet files and does not decode video.

Validate the generated artifacts:

```bash
bin/validate.sh
```

Cache alignment, field, and manifest requirements are documented in
[docs/data-contract.md](docs/data-contract.md). The DreamDojo model architecture and
version are pinned in `configs/dreamdojo_lam400k.json`.

## Training

Run the read-only preflight check, then launch training:

```bash
bin/train.sh preflight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bin/train.sh launch
bin/train.sh status
```

Default configuration: PI0.5, action horizon 50, fixed lag 10, global batch size 128,
and 120k training steps. Checkpoints and logs are written to `checkpoints/`, `runs/`,
and `artifacts/` under the repository by default; all three directories are ignored by
Git. W&B runs offline by default. Add `WANDB_MODE=online` to `.env` to enable online
logging.

## Inference

First verify that the training checkpoint matches the DreamDojo cache and encoder:

```bash
bin/serve.sh --checkpoint /path/to/checkpoint/10000 --preflight-only
```

Start a single-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 bin/serve.sh \
  --checkpoint /path/to/checkpoint/10000 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8200
```

The online client sends camera frames and state directly over a single WebSocket
connection. Initialize the history with:

```python
{
    "add_buffer": True,
    "images": head_frames[:, None],          # [T, 1, H, W, 3] uint8
    "wrist_images": wrist_frames[:, None],  # [T, 1, H, W, 3] uint8
    "exec_start_idx": 120,
}
```

Then send the current observation for inference:

```python
{
    "observation/image": head_rgb,          # [H, W, 3] uint8
    "observation/wrist_image": wrist_rgb,  # [H, W, 3] uint8
    "observation/state": state,             # [8] float32
    "prompt": "task instruction",
}
```

The server returns a `[50, 8]` action chunk directly. History continues to accumulate
within the connection; a new connection or `reset` clears it. See
[docs/inference-protocol.md](docs/inference-protocol.md) for protocol details.

## RoboMME batch evaluation

The batch evaluator uses the RoboMME simulator directly; it does not read or write
NPZ inputs. On a minimal Ubuntu host, install the OpenCV runtime library first, then
install the external benchmark in its own environment:

```bash
sudo apt-get install -y libgl1
git clone https://github.com/RoboMME/robomme_benchmark.git /path/to/robomme_benchmark
cd /path/to/robomme_benchmark
uv sync
uv pip install -e .
```

Add the benchmark checkout and the checkpoint to this repository's `.env`:

```bash
ROBOMME_BENCHMARK_ROOT=/path/to/robomme_benchmark
ROBOMME_EVAL_CHECKPOINT=/path/to/checkpoint/STEP
```

Then validate and launch the evaluation:

```bash
bin/evaluate.sh preflight
bin/evaluate.sh launch
bin/evaluate.sh status
bin/evaluate.sh attach
```

`preflight` validates the release checkpoint, all 16×50 test episodes, Python
dependencies, and GPU rendering. `launch` also requires the selected GPUs and ports
to be free, starts one policy server per GPU, and runs one smoke rollout before
creating the full 800-job queue. It never reuses an existing result directory.
Vulkan uses the host loader's normal device discovery; no distribution-specific ICD
or library path is built in. Set `VK_ICD_FILENAMES` or
`SAPIEN_VULKAN_LIBRARY_PATH` only if the host requires an explicit override.

The default topology uses GPU 0 and 12 simulator workers. Override it in `.env`, for
example with `ROBOMME_EVAL_GPUS=0,1,2,3` and
`ROBOMME_EVAL_WORKERS_PER_GPU=12`. Workers replan every 10 actions and use a SQLite
WAL queue with heartbeats, stale-job recovery, and up to three attempts. Failed
rollouts are saved under `evaluations/<run>/videos/`; live and final metrics, success
rate, and ETA are written to `evaluations/<run>/summary.json`. W&B is disabled by
default. `bin/evaluate.sh status` prints the queue summary, tmux panes, and GPU use.

## Checks

```bash
make check
bin/validate.sh --code-only
bin/validate.sh --checkpoint /path/to/checkpoint/STEP
```

## License and citation

This repository contains modified OpenPI code and the minimal DreamDojo runtime
modules. See `LICENSE`, `LICENSE_GEMMA.txt`, `third_party/dreamdojo/LICENSE`, and
`NOTICE.md` for license and third-party notices. Citation information is provided in
`CITATION.cff`.
