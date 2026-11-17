#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec .local-play/venv/bin/python engine/showdown/run_public_evaluation.py "$@"
