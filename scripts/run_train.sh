#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m src.train --config configs/train.yaml
python -m src.package_adapters --out artifacts/llama31-lora-adapters.tar.gz
# Upload example:
# python -m src.upload_s3 --file artifacts/llama31-lora-adapters.tar.gz --bucket YOUR_BUCKET --key models/llama31-lora-adapters.tar.gz
echo "Training and packaging complete"
