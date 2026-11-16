#!/bin/bash
# Run from Terminal or double-click on macOS. Runtime stays inside this project.
set -euo pipefail
cd "$(dirname "$0")"
for candidate in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" engine/showdown/local_play.py play "$@"
  fi
done
echo "Python 3.12 is required. Install it, then rerun play_bot.command." >&2
exit 1
