#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ_FILE="$ROOT_DIR/requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
    python -m venv "$VENV_DIR"
fi

if [[ -z "${VIRTUAL_ENV:-}" || "${VIRTUAL_ENV:-}" != "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"

echo "Virtual environment ready at $VENV_DIR"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    exec "${SHELL:-/bin/bash}"
fi