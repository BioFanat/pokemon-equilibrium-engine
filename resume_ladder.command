#!/bin/bash
# Keep this Terminal window open until the batch finishes.
set -euo pipefail
cd "$(dirname "$0")"
exec .local-play/venv/bin/python engine/showdown/resume_public_ladder.py "$@"
