#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <dataset.csv>" >&2
    exit 1
fi

# Resolve the dataset before changing directories; relative paths use the caller's cwd.
DATASET="$(realpath -e -- "$1")"
if [[ ! -f "$DATASET" ]]; then
    echo "Dataset is not a regular file: $1" >&2
    exit 1
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "$SCRIPT_DIR/../src" && pwd)"
if [[ ! -x "$SRC_DIR/futurism" ]]; then
    echo "Build first with scripts/build.sh" >&2
    exit 1
fi
cd -- "$SRC_DIR"
exec ./futurism "$DATASET"
