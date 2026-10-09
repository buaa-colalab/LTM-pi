#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly ENV_FILE="${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env}"
if [[ -n ${ROBOMME_ENV_FILE:-} && ! -f ${ROBOMME_ENV_FILE} ]]; then
  printf 'error: ROBOMME_ENV_FILE does not exist: %s\n' "${ROBOMME_ENV_FILE}" >&2
  exit 2
fi
if [[ -f ${ENV_FILE} ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

readonly RELEASE_PY="${PYTHON_BIN:-$(command -v python3)}"
readonly CHECKPOINT="${ROBOMME_EVAL_CHECKPOINT:-}"
readonly BENCHMARK_ROOT="${ROBOMME_BENCHMARK_ROOT:-}"
readonly BENCHMARK_PY_DEFAULT="${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/.venv/bin/python}"
if [[ -n ${ROBOMME_EVAL_PYTHON:-} ]]; then
  readonly EVAL_PY="${ROBOMME_EVAL_PYTHON}"
elif [[ -n ${BENCHMARK_PY_DEFAULT} && -x ${BENCHMARK_PY_DEFAULT} ]]; then
  readonly EVAL_PY="${BENCHMARK_PY_DEFAULT}"
else
  readonly EVAL_PY="${RELEASE_PY}"
fi

readonly RUN_NAME="${ROBOMME_EVAL_NAME:-robomme-eval}"
readonly GPU_CSV="${ROBOMME_EVAL_GPUS:-0}"
readonly WORKERS_PER_GPU="${ROBOMME_EVAL_WORKERS_PER_GPU:-12}"
readonly BASE_PORT="${ROBOMME_EVAL_BASE_PORT:-8200}"
readonly RESULTS_ROOT="${ROBOMME_EVAL_RESULTS_ROOT:-${BUNDLE_ROOT}/evaluations}"
readonly RESULTS_DIR="${RESULTS_ROOT}/${RUN_NAME}"
readonly MAX_ATTEMPTS="${ROBOMME_EVAL_MAX_ATTEMPTS:-3}"
readonly MAX_GPU_USED_MIB="${ROBOMME_EVAL_MAX_GPU_USED_MIB:-1024}"
readonly SESSION_DEFAULT="robomme-eval-$(printf '%s' "${RUN_NAME}" | tr -cs '[:alnum:]_.-' '-')"
readonly SESSION="${ROBOMME_EVAL_SESSION:-${SESSION_DEFAULT}}"
readonly DB="${RESULTS_DIR}/jobs.sqlite3"
readonly SMOKE_DIR="${RESULTS_DIR}/smoke"
readonly SMOKE_DB="${SMOKE_DIR}/jobs.sqlite3"
readonly EVAL_SCRIPT="${BUNDLE_ROOT}/scripts/eval_robomme.py"
readonly SELF="$(readlink -f -- "$0")"
readonly PYTHONPATH_VALUE="${BUNDLE_ROOT}:${BUNDLE_ROOT}/src${BENCHMARK_ROOT:+:${BENCHMARK_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
IFS=',' read -r -a GPU_IDS <<<"${GPU_CSV}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage: bin/evaluate.sh {preflight|launch|run|status|attach|finalize}

  preflight  Validate the checkpoint, benchmark environment, and GPU renderer.
  launch     Start policy servers, a smoke rollout, then the 800-job evaluation.
  run        Internal tmux entry point used by launch.
  status     Show tmux panes, evaluation summary, and GPU utilization.
  attach     Attach to the evaluation tmux session.
  finalize   Internal completion watcher used by launch.

Required environment variables:
  ROBOMME_EVAL_CHECKPOINT   Training checkpoint directory.
  ROBOMME_BENCHMARK_ROOT    RoboMME benchmark checkout with its environment.
EOF
}

is_positive_integer() {
  [[ $1 =~ ^[1-9][0-9]*$ ]]
}

validate_gpu_list() {
  local gpu seen=,
  (( ${#GPU_IDS[@]} > 0 )) || die "ROBOMME_EVAL_GPUS is empty"
  for gpu in "${GPU_IDS[@]}"; do
    [[ ${gpu} =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
    [[ ${seen} != *",${gpu},"* ]] || die "duplicate GPU id: ${gpu}"
    seen+="${gpu},"
  done
}

validate_settings() {
  [[ -n ${CHECKPOINT} ]] || die "set ROBOMME_EVAL_CHECKPOINT"
  [[ -d ${CHECKPOINT} ]] || die "checkpoint directory does not exist: ${CHECKPOINT}"
  [[ -n ${BENCHMARK_ROOT} ]] || die "set ROBOMME_BENCHMARK_ROOT"
  [[ -d ${BENCHMARK_ROOT} ]] || die "benchmark checkout does not exist: ${BENCHMARK_ROOT}"
  [[ -x ${RELEASE_PY} ]] || die "release Python is not executable: ${RELEASE_PY}"
  [[ -x ${EVAL_PY} ]] || die "evaluation Python is not executable: ${EVAL_PY}"
  [[ -f ${EVAL_SCRIPT} ]] || die "evaluator is missing: ${EVAL_SCRIPT}"
  [[ ${RUN_NAME} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "ROBOMME_EVAL_NAME may contain only letters, digits, dot, underscore, and dash"
  is_positive_integer "${WORKERS_PER_GPU}" || die "ROBOMME_EVAL_WORKERS_PER_GPU must be positive"
  is_positive_integer "${BASE_PORT}" || die "ROBOMME_EVAL_BASE_PORT must be positive"
  (( BASE_PORT + ${#GPU_IDS[@]} - 1 <= 65535 )) || die "evaluation port range exceeds 65535"
  is_positive_integer "${MAX_ATTEMPTS}" || die "ROBOMME_EVAL_MAX_ATTEMPTS must be positive"
  [[ ${MAX_GPU_USED_MIB} =~ ^[0-9]+$ ]] || die "ROBOMME_EVAL_MAX_GPU_USED_MIB must be non-negative"
  validate_gpu_list
}

eval_python() {
  env \
    PYTHONPATH="${PYTHONPATH_VALUE}" \
    PYTHONDONTWRITEBYTECODE=1 \
    "$EVAL_PY" "$EVAL_SCRIPT" "$@"
}

compile_evaluator() {
  "$EVAL_PY" - "$EVAL_SCRIPT" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
compile(path.read_bytes(), str(path), "exec")
PY
}

common_preflight() {
  validate_settings
  compile_evaluator
  "${BUNDLE_ROOT}/bin/serve.sh" --checkpoint "${CHECKPOINT}" --preflight-only
  eval_python preflight
}

gpu_render_preflight() {
  local gpu=${GPU_IDS[0]}
  local -a render_env=(
    PYTHONPATH="${PYTHONPATH_VALUE}"
    PYTHONDONTWRITEBYTECODE=1
    CUDA_VISIBLE_DEVICES="${gpu}"
    SAPIEN_RENDER_DEVICE=cuda:0
    ROBOMME_RENDER_BACKEND=cuda:0
  )
  [[ -z ${VK_ICD_FILENAMES:-} ]] || render_env+=(VK_ICD_FILENAMES="${VK_ICD_FILENAMES}")
  [[ -z ${SAPIEN_VULKAN_LIBRARY_PATH:-} ]] || \
    render_env+=(SAPIEN_VULKAN_LIBRARY_PATH="${SAPIEN_VULKAN_LIBRARY_PATH}")
  env "${render_env[@]}" "$EVAL_PY" "$EVAL_SCRIPT" preflight --gpu-smoke
}

optional_vulkan_exports() {
  local name value
  for name in VK_ICD_FILENAMES SAPIEN_VULKAN_LIBRARY_PATH; do
    value=${!name:-}
    if [[ -n ${value} ]]; then
      printf ' export %s=' "${name}"
      printf '%q' "${value}"
      printf ';'
    fi
  done
}

require_free_gpus() {
  local gpu used
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  for gpu in "${GPU_IDS[@]}"; do
    used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ ${used} =~ ^[0-9]+$ ]] || die "could not read GPU ${gpu} memory usage"
    (( used <= MAX_GPU_USED_MIB )) || die "GPU ${gpu} is busy (${used} MiB used)"
  done
}

require_free_ports() {
  local slot port
  for slot in "${!GPU_IDS[@]}"; do
    port=$((BASE_PORT + slot))
    "$RELEASE_PY" - "$port" <<'PY' || die "port ${port} is already in use"
import socket
import sys

sock = socket.socket()
try:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
finally:
    sock.close()
PY
  done
}

db_scalar() {
  "$EVAL_PY" -c \
    'import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute(sys.argv[2]).fetchone()[0])' \
    "$1" "$2"
}

wait_for_servers() {
  local deadline=$((SECONDS + 3600)) slot port all_ready
  while (( SECONDS < deadline )); do
    all_ready=1
    for slot in "${!GPU_IDS[@]}"; do
      port=$((BASE_PORT + slot))
      curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/healthz" >/dev/null || all_ready=0
    done
    (( all_ready == 1 )) && return 0
    sleep 5
  done
  die "policy servers did not become healthy within one hour"
}

start_servers() {
  local slot gpu port log command
  for slot in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[slot]}
    port=$((BASE_PORT + slot))
    log="${RESULTS_DIR}/logs/server_gpu${gpu}.log"
    command="cd '${BUNDLE_ROOT}' && export CUDA_VISIBLE_DEVICES='${gpu}' XLA_PYTHON_CLIENT_PREALLOCATE=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1; exec '${BUNDLE_ROOT}/bin/serve.sh' --checkpoint '${CHECKPOINT}' --device cuda:0 --host 0.0.0.0 --port '${port}' --num-inference-steps 10 --pair-batch-size 4 >>'${log}' 2>&1"
    tmux new-window -d -t "${SESSION}" -n "server${gpu}" "${command}"
  done
}

wait_for_one_terminal_job() {
  local deadline=$((SECONDS + 1800)) terminal=0
  while (( SECONDS < deadline )); do
    terminal=$(db_scalar "${SMOKE_DB}" "SELECT COUNT(*) FROM jobs WHERE status IN ('success','failure','error')")
    (( terminal >= 1 )) && return 0
    sleep 5
  done
  die "smoke rollout did not finish within 30 minutes"
}

run_smoke_rollout() {
  local gpu=${GPU_IDS[0]} port=${BASE_PORT} command errors vulkan_exports
  mkdir -p "${SMOKE_DIR}/logs"
  eval_python init --db "${SMOKE_DB}" --run-name "${RUN_NAME}-smoke" --max-attempts 1
  vulkan_exports=$(optional_vulkan_exports)
  command="cd '${BENCHMARK_ROOT}' && unset LIBGL_ALWAYS_SOFTWARE; export PYTHONPATH='${PYTHONPATH_VALUE}' PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='${gpu}' SAPIEN_RENDER_DEVICE=cuda:0 ROBOMME_RENDER_BACKEND=cuda:0 PYTHONUNBUFFERED=1;${vulkan_exports} exec '${EVAL_PY}' '${EVAL_SCRIPT}' worker --db '${SMOKE_DB}' --results-dir '${SMOKE_DIR}' --worker-id smoke --gpu-id '${gpu}' --server-host 127.0.0.1 --server-port '${port}' --max-steps 1300 --replan-steps 10 --model-seed 42 --max-attempts 1 --stale-seconds 1800 --save-video failures --video-stride 2 >>'${SMOKE_DIR}/logs/worker_smoke.log' 2>&1"
  tmux new-window -d -t "${SESSION}" -n smoke "${command}"
  wait_for_one_terminal_job
  errors=$(db_scalar "${SMOKE_DB}" "SELECT COUNT(*) FROM jobs WHERE status='error'")
  tmux kill-window -t "${SESSION}:smoke" 2>/dev/null || true
  (( errors == 0 )) || die "smoke rollout ended with an evaluation error"
  printf '%s GPU-rendered smoke rollout passed\n' "$(date -Is)"
}

start_workers_monitor_and_finalizer() {
  local slot gpu port worker_slot worker_id log command vulkan_exports
  eval_python init --db "${DB}" --run-name "${RUN_NAME}" --max-attempts "${MAX_ATTEMPTS}"
  vulkan_exports=$(optional_vulkan_exports)

  for slot in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[slot]}
    port=$((BASE_PORT + slot))
    for ((worker_slot=0; worker_slot<WORKERS_PER_GPU; worker_slot++)); do
      worker_id="g${gpu}w${worker_slot}"
      log="${RESULTS_DIR}/logs/worker_${worker_id}.log"
      command="cd '${BENCHMARK_ROOT}' && unset LIBGL_ALWAYS_SOFTWARE; export PYTHONPATH='${PYTHONPATH_VALUE}' PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='${gpu}' SAPIEN_RENDER_DEVICE=cuda:0 ROBOMME_RENDER_BACKEND=cuda:0 PYTHONUNBUFFERED=1;${vulkan_exports} while true; do '${EVAL_PY}' '${EVAL_SCRIPT}' worker --db '${DB}' --results-dir '${RESULTS_DIR}' --worker-id '${worker_id}' --gpu-id '${gpu}' --server-host 127.0.0.1 --server-port '${port}' --max-steps 1300 --replan-steps 10 --model-seed 42 --max-attempts '${MAX_ATTEMPTS}' --stale-seconds 1800 --save-video failures --video-stride 2 >>'${log}' 2>&1; rc=\$?; (( rc == 0 )) && break; printf 'worker_exit=%s\\n' \"\$rc\" >>'${log}'; sleep 10; done"
      tmux new-window -d -t "${SESSION}" -n "${worker_id}" "${command}"
      sleep 1
    done
  done

  command="cd '${BENCHMARK_ROOT}' && export PYTHONPATH='${PYTHONPATH_VALUE}' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1; exec '${EVAL_PY}' '${EVAL_SCRIPT}' monitor --db '${DB}' --results-dir '${RESULTS_DIR}' --run-name '${RUN_NAME}' --interval 15 --no-wandb >>'${RESULTS_DIR}/logs/summary.log' 2>&1"
  tmux new-window -d -t "${SESSION}" -n summary "${command}"

  command="export ROBOMME_ENV_FILE='${ENV_FILE}' ROBOMME_EVAL_CHECKPOINT='${CHECKPOINT}' ROBOMME_BENCHMARK_ROOT='${BENCHMARK_ROOT}' ROBOMME_EVAL_PYTHON='${EVAL_PY}' ROBOMME_EVAL_NAME='${RUN_NAME}' ROBOMME_EVAL_GPUS='${GPU_CSV}' ROBOMME_EVAL_WORKERS_PER_GPU='${WORKERS_PER_GPU}' ROBOMME_EVAL_BASE_PORT='${BASE_PORT}' ROBOMME_EVAL_RESULTS_ROOT='${RESULTS_ROOT}' ROBOMME_EVAL_MAX_ATTEMPTS='${MAX_ATTEMPTS}' ROBOMME_EVAL_MAX_GPU_USED_MIB='${MAX_GPU_USED_MIB}' ROBOMME_EVAL_SESSION='${SESSION}'; exec '${SELF}' finalize >>'${RESULTS_DIR}/logs/finalizer.log' 2>&1"
  tmux new-window -d -t "${SESSION}" -n finalizer "${command}"
  tmux select-window -t "${SESSION}:summary"
}

run() {
  common_preflight
  mkdir -p "${RESULTS_DIR}/logs"
  exec >>"${RESULTS_DIR}/logs/launcher.log" 2>&1
  printf '%s starting RoboMME evaluation\n' "$(date -Is)"
  printf 'tasks=16 episodes_per_task=50 expected_jobs=800\n'
  printf 'checkpoint=%s\n' "${CHECKPOINT}"
  printf 'benchmark_root=%s\n' "${BENCHMARK_ROOT}"
  printf 'gpus=%s workers_per_gpu=%s replan_steps=10\n' "${GPU_CSV}" "${WORKERS_PER_GPU}"
  start_servers
  wait_for_servers
  printf '%s all policy servers healthy\n' "$(date -Is)"
  run_smoke_rollout
  start_workers_monitor_and_finalizer
  printf '%s full 800-job evaluation started\n' "$(date -Is)"
}

launch() {
  common_preflight
  require_free_gpus
  require_free_ports
  gpu_render_preflight
  [[ ! -e ${RESULTS_DIR} ]] || die "result directory already exists: ${RESULTS_DIR}"
  ! tmux has-session -t "${SESSION}" 2>/dev/null || die "tmux session already exists: ${SESSION}"
  mkdir -p "${RESULTS_DIR}/logs"
  local command vulkan_exports
  vulkan_exports=$(optional_vulkan_exports)
  command="export ROBOMME_ENV_FILE='${ENV_FILE}' ROBOMME_EVAL_CHECKPOINT='${CHECKPOINT}' ROBOMME_BENCHMARK_ROOT='${BENCHMARK_ROOT}' ROBOMME_EVAL_PYTHON='${EVAL_PY}' ROBOMME_EVAL_NAME='${RUN_NAME}' ROBOMME_EVAL_GPUS='${GPU_CSV}' ROBOMME_EVAL_WORKERS_PER_GPU='${WORKERS_PER_GPU}' ROBOMME_EVAL_BASE_PORT='${BASE_PORT}' ROBOMME_EVAL_RESULTS_ROOT='${RESULTS_ROOT}' ROBOMME_EVAL_MAX_ATTEMPTS='${MAX_ATTEMPTS}' ROBOMME_EVAL_MAX_GPU_USED_MIB='${MAX_GPU_USED_MIB}' ROBOMME_EVAL_SESSION='${SESSION}';${vulkan_exports} exec '${SELF}' run"
  tmux new-session -d -s "${SESSION}" -n bootstrap "${command}"
  tmux set-option -t "${SESSION}" remain-on-exit on
  tmux set-option -t "${SESSION}" history-limit 100000
  printf 'started: %s\nattach:  bin/evaluate.sh attach\nresults: %s\n' "${SESSION}" "${RESULTS_DIR}"
}

status() {
  validate_settings
  tmux list-panes -s -t "${SESSION}" \
    -F '#{session_name}:#{window_name} dead=#{pane_dead} command=#{pane_current_command}' 2>/dev/null || true
  if [[ -s ${RESULTS_DIR}/logs/launcher.log ]]; then
    tail -n 30 "${RESULTS_DIR}/logs/launcher.log"
  fi
  if [[ -s ${DB} ]]; then
    eval_python monitor --db "${DB}" --results-dir "${RESULTS_DIR}" --once --no-wandb
  elif [[ -s ${SMOKE_DB} ]]; then
    eval_python monitor --db "${SMOKE_DB}" --results-dir "${SMOKE_DIR}" --once --no-wandb
  else
    printf 'no evaluation database at %s\n' "${DB}"
  fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  fi
}

finalize() {
  validate_settings
  [[ -s ${DB} ]] || die "evaluation database does not exist: ${DB}"
  local done_count window
  while true; do
    done_count=$(db_scalar "${DB}" "SELECT COUNT(*) FROM jobs WHERE status IN ('success','failure','error')")
    (( done_count == 800 )) && break
    sleep 30
  done
  sleep 30
  for window in $(tmux list-windows -t "${SESSION}" -F '#{window_name}' 2>/dev/null); do
    case "${window}" in
      server*|g*w*|summary) tmux kill-window -t "${SESSION}:${window}" 2>/dev/null || true ;;
    esac
  done
  eval_python monitor --db "${DB}" --results-dir "${RESULTS_DIR}" --once --no-wandb
  printf 'evaluation complete: %s\n' "${RESULTS_DIR}"
}

case "${1:-}" in
  preflight) common_preflight; gpu_render_preflight ;;
  launch) launch ;;
  run) run ;;
  status) status ;;
  attach) exec tmux attach -t "${SESSION}:summary" ;;
  finalize) finalize ;;
  -h|--help|'') usage ;;
  *) usage >&2; exit 2 ;;
esac
