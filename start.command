#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv || exit 1
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
if [[ ! -f artifacts/intent.pt ]]; then
  .venv/bin/python train.py || exit 1
fi
.venv/bin/python app.py
