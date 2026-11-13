#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "$SCRIPT_DIR/../src" && pwd)"
cd -- "$SRC_DIR"

python3 setup.py build
"${CC:-gcc}" -std=c11 read_input.c -I. -L. -lfuturism \
    '-Wl,-rpath,$ORIGIN' -o futurism
echo "Built src/libfuturism.so and src/futurism"
