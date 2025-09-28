.PHONY: venv install train package upload

venv:
	python3 -m venv .venv

install: venv
	. .venv/bin/activate && pip -q install -U pip && pip -q install -r requirements.txt

train:
	. .venv/bin/activate && python -m src.train --config configs/train.yaml

package:
	. .venv/bin/activate && python -m src.package_adapters --out artifacts/llama31-lora-adapters.tar.gz

upload:
	. .venv/bin/activate && python -m src.upload_s3 --file artifacts/llama31-lora-adapters.tar.gz --bucket YOUR_BUCKET --key models/llama31-lora-adapters.tar.gz
