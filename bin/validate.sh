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
MODE=full
CHECKPOINT=
case "${1:-}" in
  "") ;;
  --code-only) MODE=code ;;
  --checkpoint)
    [[ $# -eq 2 ]] || { printf 'usage: %s [--code-only | --checkpoint PATH]\n' "$0" >&2; exit 2; }
    CHECKPOINT=$2
    ;;
  *) printf 'usage: %s [--code-only | --checkpoint PATH]\n' "$0" >&2; exit 2 ;;
esac

[[ -x ${PY} ]] || { printf 'error: Python is not executable: %s\n' "${PY}" >&2; exit 2; }

"${BUNDLE_ROOT}/bin/release-check.sh"

if [[ -f ${BUNDLE_ROOT}/MANIFEST.sha256 ]]; then
  (cd "${BUNDLE_ROOT}" && sha256sum -c MANIFEST.sha256)
fi

bash -n "${BUNDLE_ROOT}"/bin/*.sh

readonly PYCACHE_DIR="$(mktemp -d)"
trap 'rm -rf -- "${PYCACHE_DIR}"' EXIT
mapfile -d '' python_files < <(
  find "${BUNDLE_ROOT}/src" "${BUNDLE_ROOT}/scripts" "${BUNDLE_ROOT}/third_party" -type f -name '*.py' -print0
)
PYTHONPYCACHEPREFIX="${PYCACHE_DIR}" "${PY}" -m py_compile "${python_files[@]}"

"${BUNDLE_ROOT}/bin/serve.sh" --help >/dev/null
"${BUNDLE_ROOT}/bin/client.sh" --help >/dev/null

if [[ ${MODE} == code ]]; then
  PYTHONPATH="${BUNDLE_ROOT}/src" "${PY}" -m unittest discover -s "${BUNDLE_ROOT}/tests" -v
  printf 'standalone code validation: ok\n'
  exit 0
fi

"${BUNDLE_ROOT}/bin/train.sh" preflight

if [[ -n ${CHECKPOINT} ]]; then
  "${BUNDLE_ROOT}/bin/serve.sh" \
    --checkpoint "${CHECKPOINT}" \
    --memory-manifest "${ROBOMME_MEMORY_CACHE}/manifest.json" \
    --dreamdojo-runtime-root "${DREAMDOJO_RUNTIME_ROOT:-${BUNDLE_ROOT}/third_party/dreamdojo/runtime}" \
    --dreamdojo-checkpoint "${DREAMDOJO_CHECKPOINT}" \
    --preflight-only
else
  printf 'checkpoint preflight: skipped (pass --checkpoint PATH to enable it)\n'
fi

printf 'standalone validation: ok\n'
