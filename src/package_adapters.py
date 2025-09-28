"""
Package saved LoRA adapters into a .tar.gz archive.

This is useful when you want to upload the adapters as a single artifact.
"""

import argparse
import os
import tarfile


def tar_dir(src_dir: str, out_path: str) -> None:
    """Create a gzipped tar archive from a directory."""
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(src_dir, arcname=os.path.basename(src_dir))


def main(out_path: str, adapters_dir: str) -> None:
    """Package the adapters directory into a tarball."""
    if not os.path.isdir(adapters_dir):
        raise FileNotFoundError(f"Adapters dir not found: {adapters_dir}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tar_dir(adapters_dir, out_path)
    print("Created:", out_path, f"{os.path.getsize(out_path)/(1024**2):.2f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tar up LoRA adapters directory")
    p.add_argument("--out", type=str, default="artifacts/llama31-lora-adapters.tar.gz", help="Output tar.gz path")
    p.add_argument("--dir", type=str, default="outputs/run1/adapters", help="Adapters directory to package")
    a = p.parse_args()
    main(a.out, a.dir)
