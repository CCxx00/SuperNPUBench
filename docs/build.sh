#!/usr/bin/env bash
set -euo pipefail

docs_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$docs_root/.." && pwd)
command=${1:-build}

case "$command" in
  build)
    (cd "$repo_root" && python3 -m mkdocs build --strict --config-file docs/mkdocs.yml)
    python3 "$docs_root/scripts/verify_golden_manual.py" --site "$repo_root/site"
    ;;
  serve)
    cd "$repo_root"
    exec python3 -m mkdocs serve --config-file docs/mkdocs.yml
    ;;
  verify)
    python3 "$docs_root/scripts/verify_golden_manual.py" --site "$repo_root/site"
    ;;
  *)
    printf 'usage: %s [build|serve|verify]\n' "$0" >&2
    exit 2
    ;;
esac
