"""
Upload a file to Amazon S3 with a simple progress bar.

Usage example:
    python -m src.upload_s3 --file artifacts/llama31-lora-adapters.tar.gz --bucket my-bucket --key models/llama31-lora-adapters.tar.gz
"""

import argparse
import os
import sys
import boto3
from boto3.s3.transfer import TransferConfig


class Progress:
    """Simple progress callback for multipart uploads."""
    def __init__(self, path: str):
        self.path = path
        self.total = float(os.path.getsize(path))
        self.seen = 0

    def __call__(self, b: int) -> None:
        self.seen += b
        pct = self.seen / self.total * 100
        sys.stdout.write(
            f"\\rUploading {self.seen/(1024**2):.2f}/{self.total/(1024**2):.2f} MB ({pct:.1f}%)"
        )
        sys.stdout.flush()


def main(file_path: str, bucket: str, key: str) -> None:
    """Upload a local file to S3."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    s3 = boto3.client("s3")
    cfg = TransferConfig(
        multipart_threshold=25 * 1024 * 1024,
        multipart_chunksize=25 * 1024 * 1024,
        max_concurrency=10,
        use_threads=True,
    )
    s3.upload_file(file_path, bucket, key, Config=cfg, Callback=Progress(file_path))
    print()
    print("Uploaded:", f"s3://{bucket}/{key}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Upload a file to S3")
    p.add_argument("--file", required=True, help="Local file path")
    p.add_argument("--bucket", required=True, help="S3 bucket name")
    p.add_argument("--key", required=True, help="S3 object key")
    args = p.parse_args()
    main(args.file, args.bucket, args.key)
