"""
Dataset utilities.
"""

from datasets import load_dataset


def load_ds(name: str, split: str):
    """
    Load a dataset from Hugging Face Hub.

    Args:
        name: Dataset repository id on the Hub.
        split: Dataset split spec, for example "train" or "train[:2000]".

    Returns:
        A datasets.Dataset instance.
    """
    ds = load_dataset(name, split=split)
    return ds
