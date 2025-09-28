"""
Prompt formatting utilities for supervised fine tuning.

This keeps the prompt style consistent so that the model sees uniform patterns.
"""

from typing import Dict, List, Any


def build_prompt(instruction: str, response: str, input_text: str | None = None) -> str:
    """
    Build a single training prompt string in a simple instruction format.

    Args:
        instruction: The task or question.
        response: The target answer to learn.
        input_text: Optional additional context or input for the task.

    Returns:
        A single string with Instruction, optional Input, and Response sections.
    """
    if input_text and input_text.strip():
        return (
            "### Instruction:\n"
            + instruction.strip()
            + "\n\n### Input:\n"
            + input_text.strip()
            + "\n\n### Response:\n"
            + response.strip()
        )
    return "### Instruction:\n" + instruction.strip() + "\n\n### Response:\n" + response.strip()


def formatting_func(batch: Dict[str, List[Any]]) -> List[str]:
    """
    Convert a batch from a dataset into a list of formatted prompts.

    Supports two common schemas:
    1. Alpaca style with fields: instruction, input, output
    2. Plain text style with field: text

    Args:
        batch: A dict of column name to list of values for a batch.

    Returns:
        List of prompt strings, one per row in the batch.

    Raises:
        ValueError if the dataset schema is not recognized.
    """
    texts: List[str] = []
    keys = set(batch.keys())
    if {"instruction", "output"}.issubset(keys):
        ins = batch["instruction"]
        outs = batch["output"]
        inps = batch["input"] if "input" in keys else ["" for _ in range(len(ins))]
        for i, o, inp in zip(ins, outs, inps):
            texts.append(build_prompt(i, o, inp))
        return texts
    if "text" in keys:
        return batch["text"]
    raise ValueError(f"Unsupported dataset schema. Keys: {sorted(keys)}")
