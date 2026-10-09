#!/usr/bin/env bash
set -euo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1

readonly BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n ${ROBOMME_ENV_FILE:-} && ! -f ${ROBOMME_ENV_FILE} ]]; then
  printf 'error: ROBOMME_ENV_FILE does not exist: %s\n' "${ROBOMME_ENV_FILE}" >&2
  exit 2
fi
if [[ -f ${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env} ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env}"
  set +a
fi

readonly PY=${PYTHON_BIN:-$(command -v python3)}
readonly MODE=${1:-all}
readonly CHECKPOINT_SHA=d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f
readonly ASSET_ID=${ROBOMME_ASSET_ID:-lerobot/robomme_16x1000}
readonly HEAD_CACHE=${ROBOMME_HEAD_CACHE:-${ROBOMME_MEMORY_CACHE:-}.head32}
readonly WRIST_CACHE=${ROBOMME_WRIST_CACHE:-${ROBOMME_MEMORY_CACHE:-}.wrist32}
readonly CACHE_GPUS=${CACHE_GPUS:-}

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
required() { [[ -n ${2:-} ]] || die "missing $1; copy configs/paths.env.example to .env"; }

check_python() {
  required PYTHON_BIN "${PY}"
  [[ -x ${PY} ]] || die "Python is not executable: ${PY}"
}

check_video_dataset() {
  required ROBOMME_VIDEO_DATASET "${ROBOMME_VIDEO_DATASET:-}"
  [[ -f ${ROBOMME_VIDEO_DATASET}/_READY ]] || die 'video dataset is not ready'
  [[ -f ${ROBOMME_VIDEO_DATASET}/meta/info.json ]] || die 'video dataset metadata is missing'
}

check_dreamdojo() {
  required DREAMDOJO_CHECKPOINT "${DREAMDOJO_CHECKPOINT:-}"
  [[ -f ${DREAMDOJO_CHECKPOINT} ]] || die 'DreamDojo LAM checkpoint is missing'
  [[ $(sha256sum "${DREAMDOJO_CHECKPOINT}" | awk '{print $1}') == ${CHECKPOINT_SHA} ]] || \
    die 'DreamDojo LAM checkpoint SHA-256 mismatch'
}

build_anchors() {
  check_video_dataset
  required ROBOMME_ANCHOR_CACHE "${ROBOMME_ANCHOR_CACHE:-}"
  [[ ! -e ${ROBOMME_ANCHOR_CACHE} ]] || die "anchor output already exists: ${ROBOMME_ANCHOR_CACHE}"
  "${PY}" "${BUNDLE_ROOT}/scripts/build_anchor_cache.py" \
    --dataset-root "${ROBOMME_VIDEO_DATASET}" \
    --output-dir "${ROBOMME_ANCHOR_CACHE}"
}

build_one_cache() {
  local image_key=$1 output=$2
  local -a common_args=(
    "${BUNDLE_ROOT}/scripts/precompute_dreamdojo_memory.py" extract
    --dataset-root "${ROBOMME_VIDEO_DATASET}"
    --output-root "${output}"
    --dreamdojo-runtime-root "${DREAMDOJO_RUNTIME_ROOT:-${BUNDLE_ROOT}/third_party/dreamdojo/runtime}"
    --checkpoint "${DREAMDOJO_CHECKPOINT}"
    --checkpoint-sha256 "${CHECKPOINT_SHA}"
    --image-key "${image_key}"
    --storage-dtype float32
    --transition-frame-stride 1
    --pair-batch-size "${PAIR_BATCH_SIZE:-32}"
    --preprocess-chunk-size "${PREPROCESS_CHUNK_SIZE:-64}"
    --prefetch-episodes "${CACHE_PREFETCH_EPISODES:-0}"
    --sequence-window-crop-min-scale 1.0
    --sequence-window-crop-probability 0.0
    --repair
  )

  if [[ -z ${CACHE_GPUS} ]]; then
    "${PY}" "${common_args[@]}" --device "${DREAMDOJO_DEVICE:-cuda:0}"
  else
    local -a gpu_ids pids
    local gpu rank world_size failed=0 seen=,
    IFS=',' read -r -a gpu_ids <<<"${CACHE_GPUS}"
    world_size=${#gpu_ids[@]}
    (( world_size > 0 )) || die 'CACHE_GPUS must contain at least one GPU index'
    for gpu in "${gpu_ids[@]}"; do
      [[ ${gpu} =~ ^[0-9]+$ ]] || die "invalid GPU index in CACHE_GPUS: ${gpu}"
      [[ ${seen} != *",${gpu},"* ]] || die "duplicate GPU index in CACHE_GPUS: ${gpu}"
      seen+="${gpu},"
    done
    printf 'extracting image_key=%s with %d ranks on physical GPUs=%s\n' \
      "${image_key}" "${world_size}" "${CACHE_GPUS}"
    for rank in "${!gpu_ids[@]}"; do
      gpu=${gpu_ids[rank]}
      CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${common_args[@]}" \
        --device cuda:0 --rank "${rank}" --world-size "${world_size}" &
      pids+=("$!")
    done
    for rank in "${!pids[@]}"; do
      if ! wait "${pids[rank]}"; then
        printf 'error: cache rank %d on physical GPU %s failed\n' \
          "${rank}" "${gpu_ids[rank]}" >&2
        failed=1
      fi
    done
    (( failed == 0 )) || die "one or more ${image_key} cache ranks failed"
  fi
  "${PY}" "${BUNDLE_ROOT}/scripts/precompute_dreamdojo_memory.py" finalize \
    --dataset-root "${ROBOMME_VIDEO_DATASET}" \
    --output-root "${output}" \
    --image-key "${image_key}" \
    --checkpoint-sha256 "${CHECKPOINT_SHA}"
  "${PY}" "${BUNDLE_ROOT}/scripts/precompute_dreamdojo_memory.py" validate \
    --dataset-root "${ROBOMME_VIDEO_DATASET}" \
    --output-root "${output}" \
    --image-key "${image_key}" \
    --checkpoint-sha256 "${CHECKPOINT_SHA}" \
    --verify-cache-hashes
}

build_memory() {
  check_video_dataset
  check_dreamdojo
  required ROBOMME_MEMORY_CACHE "${ROBOMME_MEMORY_CACHE:-}"
  [[ ! -e ${ROBOMME_MEMORY_CACHE} ]] || die "combined memory output already exists: ${ROBOMME_MEMORY_CACHE}"
  if [[ ${CACHE_PARALLEL_VIEWS:-0} == 1 ]]; then
    local head_pid wrist_pid failed=0
    printf 'extracting head and wrist caches concurrently\n'
    build_one_cache image "${HEAD_CACHE}" &
    head_pid=$!
    build_one_cache wrist_image "${WRIST_CACHE}" &
    wrist_pid=$!
    if ! wait "${head_pid}"; then
      printf 'error: head cache extraction failed\n' >&2
      failed=1
    fi
    if ! wait "${wrist_pid}"; then
      printf 'error: wrist cache extraction failed\n' >&2
      failed=1
    fi
    (( failed == 0 )) || die 'one or more camera cache extractions failed'
  else
    build_one_cache image "${HEAD_CACHE}"
    build_one_cache wrist_image "${WRIST_CACHE}"
  fi
  PYTHONPATH="${BUNDLE_ROOT}" "${PY}" "${BUNDLE_ROOT}/scripts/combine_dreamdojo_memory_caches.py" \
    --external-cache "${HEAD_CACHE}" \
    --wrist-cache "${WRIST_CACHE}" \
    --output-root "${ROBOMME_MEMORY_CACHE}"
}

compute_norm_stats() {
  check_video_dataset
  required ROBOMME_ASSETS_ROOT "${ROBOMME_ASSETS_ROOT:-}"
  export ROBOMME_ASSET_ID=${ASSET_ID}
  export OPENPI_VIDEO_BACKEND=pyav
  export OPENPI_NORM_STATE_ACTION_ONLY=1
  export OPENPI_NORM_NUM_WORKERS=${OPENPI_NORM_NUM_WORKERS:-8}
  PYTHONPATH="${BUNDLE_ROOT}/src:${BUNDLE_ROOT}" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
    "${PY}" "${BUNDLE_ROOT}/scripts/compute_norm_stats.py" \
    --config-name pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default \
    --output-dir "${ROBOMME_ASSETS_ROOT}/${ASSET_ID}"
}

check_python
case "${MODE}" in
  anchors) build_anchors ;;
  memory) build_memory ;;
  norm-stats) compute_norm_stats ;;
  all) build_anchors; build_memory; compute_norm_stats ;;
  *) die "usage: $0 {anchors|memory|norm-stats|all}" ;;
esac
