#!/usr/bin/env bash
set -euo pipefail

readonly BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly ENV_FILE="${ROBOMME_ENV_FILE:-${BUNDLE_ROOT}/.env}"
readonly ENV_TEMPLATE="${BUNDLE_ROOT}/configs/paths.env.example"
readonly RESOURCE_INPUT="${1:-${BUNDLE_ROOT}/../ltm_pi_robomme-resources}"

if [[ -e "${ENV_FILE}" ]]; then
  printf 'Keeping existing configuration: %s\n' "${ENV_FILE}"
  exit 0
fi

mkdir -p "${RESOURCE_INPUT}"
readonly RESOURCE_ROOT="$(cd "${RESOURCE_INPUT}" && pwd -P)"
temp_env="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
trap 'rm -f "${temp_env}"' EXIT

replacement_count=0
while IFS= read -r line || [[ -n "${line}" ]]; do
  if [[ "${line}" == ROBOMME_RESOURCE_ROOT=* ]]; then
    printf 'ROBOMME_RESOURCE_ROOT=%q\n' "${RESOURCE_ROOT}"
    replacement_count=$((replacement_count + 1))
  else
    printf '%s\n' "${line}"
  fi
done < "${ENV_TEMPLATE}" > "${temp_env}"

if [[ "${replacement_count}" -ne 1 ]]; then
  printf 'Expected exactly one ROBOMME_RESOURCE_ROOT entry in %s; found %d.\n' \
    "${ENV_TEMPLATE}" "${replacement_count}" >&2
  exit 1
fi

chmod 600 "${temp_env}"
mv -n "${temp_env}" "${ENV_FILE}"
if [[ -e "${temp_env}" ]]; then
  printf 'Keeping configuration created concurrently: %s\n' "${ENV_FILE}"
  exit 0
fi
trap - EXIT

printf 'Created resource directory: %s\n' "${RESOURCE_ROOT}"
printf 'Created configuration: %s\n' "${ENV_FILE}"
