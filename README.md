# Unsloth Llama 3.1 8B LoRA

Fine tune Llama 3.1 8B in 4 bit using Unsloth and save LoRA adapters. This repo is designed to run on AWS SageMaker and also works on any single GPU machine that supports bitsandbytes.

## Why use Unsloth
- Efficient 4 bit loading so an 8B model fits on a single 24 GB GPU.
- Simple LoRA setup that uses PEFT under the hood.
- Gradient checkpointing to keep VRAM under control.
- Small artifacts. You ship adapters instead of full model weights.

## Why LoRA for this project
- You only train a small set of adapter weights. This cuts compute, time, and storage.
- You can keep one base model and swap multiple domain adapters.
- Adapters are easy to manage and version.

## Why SageMaker for fine tuning
- Managed GPU instances with current CUDA and drivers.
- Scale up or down without rewriting your training code.
- First class S3 integration for datasets and artifacts.
- Training logs and metrics in CloudWatch improve reproducibility and debugging.
- IAM and VPC features support stricter enterprise environments.

---

## Project structure

```
unsloth-llama31-lora/
├─ README.md                  - This guide
├─ LICENSE
├─ .gitignore                 - Standard ignores for Python, venv, outputs
├─ pyproject.toml             - Code style config for black and isort
├─ requirements.txt           - Python dependencies
├─ Makefile                   - Common tasks
├─ configs/
│  └─ train.yaml              - Main training config that you can edit
├─ src/
│  ├─ __init__.py
│  ├─ data.py                 - Dataset loading helper
│  ├─ format.py               - Prompt builders and formatting function
│  ├─ train.py                - Unsloth fine tuning entry point
│  ├─ package_adapters.py     - Create a tar.gz from the saved adapters
│  └─ upload_s3.py            - Upload any file to S3 with a progress bar
├─ scripts/
│  ├─ setup_env.sh            - Create venv and install requirements
│  └─ run_train.sh            - Run training and package adapters
├─ notebooks/
│  └─ exploration.ipynb       - Optional space for your EDA and tests
├─ outputs/                   - Training outputs land here
│  └─ run1/
│     └─ adapters/            - Saved LoRA adapters and tokenizer
└─ artifacts/
   └─ llama31-lora-adapters.tar.gz  - Packaged adapters ready for upload
```

### What lives where

- `configs/train.yaml`  
  Main place to control the run. Change dataset repo and split. Adjust sequence length, LoRA ranks, batch size, epochs, and learning rate.

- `src/format.py`  
  Converts dataset rows into a consistent instruction format. This keeps training stable across datasets.

- `src/data.py`  
  Thin wrapper around Hugging Face datasets so you can switch repos without changing training code.

- `src/train.py`  
  The training pipeline:
  1. Loads config and dataset.
  2. Loads Unsloth 4 bit base model and attaches LoRA.
  3. Runs TRL SFTTrainer.
  4. Saves adapters and tokenizer to `outputs/run1/adapters`.

- `src/package_adapters.py`  
  Creates `artifacts/llama31-lora-adapters.tar.gz` from the saved adapters folder. Use this artifact for deployment.

- `src/upload_s3.py`  
  Uploads any file to S3 with a simple progress bar. Helpful when moving artifacts to the cloud.

- `scripts/setup_env.sh` and `scripts/run_train.sh`  
  Make it easy to reproduce the steps. Edit the scripts if your environment paths differ.

---

## Quick start on SageMaker Studio or Notebook Instances

```bash
bash scripts/setup_env.sh
bash scripts/run_train.sh
```

What happens
- The script creates a virtual environment and installs dependencies.
- Training runs with defaults from `configs/train.yaml`.
- Adapters and tokenizer are saved under `outputs/run1/adapters`.
- A tar.gz is produced under `artifacts/`.

To upload the artifact to S3:
```bash
python -m src.upload_s3 \
  --file artifacts/llama31-lora-adapters.tar.gz \
  --bucket YOUR_BUCKET \
  --key models/llama31-lora-adapters.tar.gz
```

---

## Configuration details

`configs/train.yaml` controls both model and trainer settings.

```yaml
seed: 42

dataset:
  name: iamtarun/python_code_instructions_18k_alpaca
  split: train[:2000]

model:
  base_id: unsloth/Meta-Llama-3.1-8B-bnb-4bit
  max_seq_len: 2048
  lora:
    r: 16
    alpha: 16
    dropout: 0
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

train:
  batch_size: 2
  grad_accum: 4
  epochs: 1
  lr: 2e-4
  log_steps: 10
  save_steps: 200
  out_dir: outputs/run1
```

Tips
- Increase `epochs` to 3 or more once a smoke test passes.
- If you OOM, reduce `batch_size` to 1 or increase `grad_accum`.
- Keep `max_seq_len` at 2048 until you validate stability.
- The LoRA rank `r` controls adapter capacity. Start at 16 and tune up or down based on your task.

---

## How training works in this repo

- The dataset is turned into prompts that look like this:

```
### Instruction:
Explain binary search

### Response:
Binary search repeatedly halves the search space...
```

- The base model is `unsloth/Meta-Llama-3.1-8B-bnb-4bit`. Unsloth loads it in 4 bit to reduce memory.
- LoRA adapters are attached to the attention and MLP projections.
- TRL SFTTrainer performs supervised fine tuning with your formatted prompts.
- Only the adapter weights are saved, not full model weights.

---

## Outputs

- `outputs/run1/adapters` - PEFT adapter files and tokenizer.
- `artifacts/llama31-lora-adapters.tar.gz` - a single archive with the adapters folder.
- Training logs are printed to stdout. You can wire in wandb or other loggers if you prefer.

---
