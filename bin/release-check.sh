#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

die() {
  printf 'release check failed: %s\n' "$*" >&2
  exit 1
}

if git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git -C "${ROOT}" ls-files --error-unmatch .env >/dev/null 2>&1; then
    die '.env is tracked; remove it from the Git index before publishing'
  fi
  files=()
  while IFS= read -r -d '' file; do
    files+=("${file}")
  done < <(git -C "${ROOT}" ls-files -z)
else
  files=()
  while IFS= read -r -d '' file; do
    files+=("${file}")
  done < <(
    cd "${ROOT}"
    find . \
      \( -name '__pycache__' -o -name '.pytest_cache' -o \
         -path './.git' -o -path './.venv' -o -path './.cache' -o \
         -path './artifacts' -o -path './checkpoints' -o -path './evaluations' -o \
         -path './runs' -o -path './wandb' -o -path './.ruff_cache' \) -prune -o \
      -type f ! -name '.env' ! -name '.env.*' -print0
  )
fi

(( ${#files[@]} > 0 )) || die 'no release files found'

bad_env=()
for file in "${files[@]}"; do
  relative=${file#./}
  case "/${relative}" in
    */.env|*/.env.*)
      [[ ${relative} == *.env.example ]] || bad_env+=("${relative}")
      ;;
  esac
done
(( ${#bad_env[@]} == 0 )) || die "machine-local environment file is publishable: ${bad_env[*]}"

# Exclude this checker because it necessarily contains the signatures it detects.
scan_files=()
for file in "${files[@]}"; do
  relative=${file#./}
  [[ ${relative} == bin/release-check.sh ]] || scan_files+=("${ROOT}/${relative}")
done

readonly MACHINE_PATH_PATTERN="(^|[=\"'[:space:]])/(share/project|Users|home|root|mnt|workspace|data|etc|lib|lib32|lib64|opt|usr/local|dev/shm|tmp)/"
readonly SECRET_PATTERN='hf_[A-Za-z0-9]{20,}|BAAIproxy[A-Za-z0-9]*|https?://[^[:space:]/]+:[^[:space:]@]+@'

if LC_ALL=C grep -I -n -E "${MACHINE_PATH_PATTERN}" "${scan_files[@]}"; then
  die 'release files contain a machine-local or distribution-specific absolute path'
fi
if LC_ALL=C grep -I -n -E "${SECRET_PATTERN}" "${scan_files[@]}"; then
  die 'release files contain a token or credential signature'
fi

printf 'release check: ok (%d files)\n' "${#files[@]}"
