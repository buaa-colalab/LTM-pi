#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=${ROBOMME_ENV_FILE:-${ROOT}/.env}
if [[ -f ${ENV_FILE} ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PY=${PYTHON_BIN:-python3}
CONFIG=pi05_robomme_16x1000_dreamdojo_lmv_causal_fixedlag10_h50_default
EXP=${ROBOMME_EXP:-ltm_pi_robomme_fixedlag10_v1}
CHECKPOINT_BASE=${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}
RUN_DIR=${RUN_BASE_DIR:-${ROOT}/runs}/${EXP}
SESSION=${ROBOMME_SESSION:-ltm-pi-robomme-video-dreamdojo-fixedlag10-h50-b128-120k-v1}
ASSET_ID=${ROBOMME_ASSET_ID:-lerobot/robomme_16x1000}
GPU_IDS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SAVE_STEPS=(10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000)

die() { echo "error: $*" >&2; exit 2; }

preflight() {
  command -v "${PY}" >/dev/null || die "python not found: ${PY}"
  [[ -d ${ROBOMME_VIDEO_DATASET:-} ]] || die "dataset not found: ${ROBOMME_VIDEO_DATASET:-unset}"
  [[ -f ${ROBOMME_MEMORY_CACHE:-}/manifest.json ]] || die "memory cache not found: ${ROBOMME_MEMORY_CACHE:-unset}"
  [[ -f ${ROBOMME_ANCHOR_CACHE:-}/manifest.json ]] || die "anchor cache not found: ${ROBOMME_ANCHOR_CACHE:-unset}"
  [[ -f ${ROBOMME_ASSETS_ROOT:-}/${ASSET_ID}/norm_stats.json ]] || die "norm stats not found"
  [[ -d ${PI05_BASE_PARAMS:-} ]] || die "base params not found: ${PI05_BASE_PARAMS:-unset}"
}

read_normalization() {
  "${PY}" - "${ROBOMME_MEMORY_CACHE}/manifest.json" <<'PY'
import json, pathlib, sys
stats = json.loads(pathlib.Path(sys.argv[1]).read_text())["latent_normalization"]
print(" ".join(map(str, stats["mean"])))
print(" ".join(map(str, stats["std"])))
PY
}

run() {
  preflight
  mapfile -t stats < <(read_normalization)
  read -r -a mean <<<"${stats[0]}"
  read -r -a std <<<"${stats[1]}"

  mkdir -p "${RUN_DIR}"
  cd "${ROOT}"
  ulimit -n 65536
  exec > >(tee -a "${RUN_DIR}/train.log") 2>&1
  exec env \
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    HF_HOME=${HF_CACHE:-${ROOT}/.cache/huggingface} \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    WANDB_MODE=${WANDB_MODE:-offline} PYTHONPATH=${ROOT}/src \
    "${PY}" scripts/train.py "${CONFIG}" \
      --exp-name "${EXP}" --checkpoint-base-dir "${CHECKPOINT_BASE}" \
      --weight-loader.params-path "${PI05_BASE_PARAMS}" \
      --batch-size 128 --gradient-accumulation-steps 1 --fsdp-devices 8 \
      --num-train-steps 120000 --num-workers 32 --prefetch-factor 2 --no-data-loader-in-order \
      --memory-bucket-size-multiplier 50 --memory-pad-to-multiple 512 \
      --lr-schedule.warmup-steps 1000 --lr-schedule.peak-lr 2.5e-5 \
      --lr-schedule.decay-steps 120000 --lr-schedule.decay-lr 2.5e-6 \
      --model.memory-frames-per-token 1 --model.memory-pooling-mode concatenate \
      --model.memory-projector-hidden-dim 512 --model.memory-token-dropout-rate 0.0 \
      --model.memory-segment-embedding-after-projection \
      --model.memory-segment-vocab-size 6 \
      --model.memory-demo-anchor-type-id 4 --model.memory-execution-anchor-type-id 5 \
      --model.memory-demo-lam-boundary-token --model.memory-demo-lam-boundary-type-id 2 \
      --model.memory-latent-mean "${mean[@]}" --model.memory-latent-std "${std[@]}" \
      --seed 0 --log-interval 20 --wandb-enabled --save-steps "${SAVE_STEPS[@]}"
}

launch() {
  local command
  printf -v command '%q run' "${ROOT}/bin/train.sh"
  tmux new-session -d -s "${SESSION}" -n train "${command}"
  echo "started tmux=${SESSION} exp=${EXP}"
}

status() {
  tmux list-panes -t "${SESSION}" -F 'session=#{session_name} dead=#{pane_dead} pid=#{pane_pid}' 2>/dev/null || true
  [[ ! -f ${RUN_DIR}/train.log ]] || tail -n 80 "${RUN_DIR}/train.log"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

case "${1:-launch}" in
  preflight) preflight ;;
  launch) launch ;;
  run) run ;;
  status) status ;;
  *) die "usage: $0 {preflight|launch|run|status}" ;;
esac
