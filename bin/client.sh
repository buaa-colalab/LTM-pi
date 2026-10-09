#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

readonly BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f ${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env} ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env}"
  set +a
fi
readonly PY=${PYTHON_BIN:-$(command -v python3)}
export PYTHONPATH="${BUNDLE_ROOT}/src:${BUNDLE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PY}" "${BUNDLE_ROOT}/scripts/infer_client.py" "$@"
