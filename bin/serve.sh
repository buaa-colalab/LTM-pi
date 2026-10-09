#!/usr/bin/env bash
set -euo pipefail
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

if [[ $# -eq 0 ]]; then
  exec "${PY}" "${BUNDLE_ROOT}/scripts/serve.py" --help
fi

export PYTHONPATH="${BUNDLE_ROOT}/src:${BUNDLE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
args=("$@")
has_manifest=false
has_dreamdojo_runtime=false
has_dreamdojo_checkpoint=false
for arg in "$@"; do
  [[ ${arg} == --memory-manifest ]] && has_manifest=true
  [[ ${arg} == --dreamdojo-runtime-root ]] && has_dreamdojo_runtime=true
  [[ ${arg} == --dreamdojo-checkpoint ]] && has_dreamdojo_checkpoint=true
done
if [[ ${has_manifest} == false && -n ${ROBOMME_MEMORY_CACHE:-} ]]; then
  args+=(--memory-manifest "${ROBOMME_MEMORY_CACHE}/manifest.json")
fi
if [[ ${has_dreamdojo_runtime} == false && -n ${DREAMDOJO_RUNTIME_ROOT:-} ]]; then
  args+=(--dreamdojo-runtime-root "${DREAMDOJO_RUNTIME_ROOT}")
fi
if [[ ${has_dreamdojo_checkpoint} == false && -n ${DREAMDOJO_CHECKPOINT:-} ]]; then
  args+=(--dreamdojo-checkpoint "${DREAMDOJO_CHECKPOINT}")
fi
exec "${PY}" "${BUNDLE_ROOT}/scripts/serve.py" "${args[@]}"
