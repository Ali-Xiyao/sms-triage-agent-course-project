#!/bin/zsh
set -e
cd "${0:A:h}"
if [[ -x .venv/bin/python ]]; then
  .venv/bin/python src/app.py
else
  python3 src/app.py
fi
