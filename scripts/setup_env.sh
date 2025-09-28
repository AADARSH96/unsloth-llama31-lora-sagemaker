#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip -q install -U pip
pip -q install -r requirements.txt
echo "Environment ready"
