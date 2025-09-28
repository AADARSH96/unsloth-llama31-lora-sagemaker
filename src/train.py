"""
Training entry point for Unsloth LoRA SFT on Llama 3.1 8B.

This script loads a dataset, formats it into instruction prompts,
loads a 4 bit base model via Unsloth, adds LoRA adapters, and trains
with TRL SFTTrainer. Adapters and tokenizer are saved at the end.
"""

import argparse
import os
import random
import yaml
import torch
from typing import Any, Dict
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from .data import load_ds
from .format import formatting_func


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg_path: str) -> None:
    """
    Main training pipeline.

    Steps:
      1. Load config
      2. Load dataset and preview
      3. Load Unsloth base model and attach LoRA
      4. Train with TRL SFTTrainer
      5. Save adapters and tokenizer

    Args:
        cfg_path: Path to a YAML config file.
    """
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))

    ds = load_ds(cfg["dataset"]["name"], cfg["dataset"]["split"])
    print("Dataset cols:", ds.column_names)
    print("Sample 0:", ds[0])

    torch.cuda.empty_cache()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["base_id"],
        max_seq_length=int(cfg["model"]["max_seq_len"]),
        dtype=None,
        load_in_4bit=True,
    )

    # padding for causal LM
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora = cfg["model"]["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["r"]),
        target_modules=list(lora["target_modules"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    train_cfg = cfg["train"]
    args = TrainingArguments(
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        gradient_accumulation_steps=int(train_cfg["grad_accum"]),
        num_train_epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["lr"]),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=int(train_cfg["log_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        output_dir=str(train_cfg["out_dir"]),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        formatting_func=formatting_func,
        max_seq_length=int(cfg["model"]["max_seq_len"]),
        packing=False,
        args=args,
    )

    result = trainer.train()
    print("Train result:", result)

    adapters_dir = os.path.join(str(train_cfg["out_dir"]), "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    trainer.model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)
    print("Saved adapters to:", adapters_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unsloth LoRA SFT trainer")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
