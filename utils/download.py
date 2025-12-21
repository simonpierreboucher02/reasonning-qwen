"""
File downloading utilities.

This module provides utilities for downloading model weights and tokenizers
from remote sources with progress tracking and fallback URLs.
"""

import sys
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


def download_file(
    url: str, out_dir: str = ".", backup_url: Optional[str] = None
) -> Path:
    """Download a file with progress tracking and backup URL support.

    Args:
        url: Primary URL to download from
        out_dir: Output directory for downloaded file
        backup_url: Optional backup URL if primary fails

    Returns:
        Path to the downloaded file

    Raises:
        RuntimeError: If download fails from both primary and backup URLs

    Example:
        >>> path = download_file(
        ...     "https://example.com/model.pth",
        ...     out_dir="./models",
        ...     backup_url="https://backup.com/model.pth"
        ... )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    dest = out_dir / filename

    def try_download(u: str) -> bool:
        """Attempt to download from a specific URL."""
        try:
            with requests.get(u, stream=True, timeout=30) as r:
                r.raise_for_status()
                size_remote = int(r.headers.get("Content-Length", 0))

                # Skip download if already complete
                if dest.exists() and size_remote and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True

                # Download in 1 MiB chunks with progress display
                block_size = 1024 * 1024  # 1 MiB
                downloaded = 0

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Display progress
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            downloaded_mb = downloaded // (1024 * 1024)
                            total_mb = size_remote // (1024 * 1024)
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% ({downloaded_mb} MiB / {total_mb} MiB)"
                            )
                            sys.stdout.flush()

                if size_remote:
                    sys.stdout.write("\n")

                return True

        except requests.RequestException as e:
            print(f"\nDownload failed: {e}")
            return False

    # Try primary URL first
    if try_download(url):
        return dest

    # Try backup URL if provided
    if backup_url:
        print(f"Primary URL failed. Trying backup URL...")
        if try_download(backup_url):
            return dest

    raise RuntimeError(f"Failed to download {filename} from both mirrors.")


def download_qwen3_model(
    model_type: str = "base",
    size: str = "0.6B",
    tokenizer_only: bool = False,
    out_dir: str = ".",
) -> dict:
    """Download Qwen3 model weights and tokenizer.

    Args:
        model_type: Model type ("base" or "reasoning")
        size: Model size ("0.6B", "1.7B", "4B", "8B", "14B", "32B")
        tokenizer_only: If True, only download tokenizer
        out_dir: Output directory for downloaded files

    Returns:
        Dictionary with paths to downloaded files

    Example:
        >>> paths = download_qwen3_model(model_type="base", size="0.6B")
        >>> print(paths["model"], paths["tokenizer"])
    """
    if model_type not in ["base", "reasoning"]:
        raise ValueError("model_type must be 'base' or 'reasoning'")

    # Configure URLs
    repo = "rasbt/qwen3-from-scratch"
    hf_base = f"https://huggingface.co/{repo}/resolve/main"
    backup_base = "https://f001.backblazeb2.com/file/reasoning-from-scratch"

    # File names
    if size == "0.6B":
        model_file = f"qwen3-{size}-{model_type}.pth"
        tokenizer_file = f"tokenizer-{model_type}.json"
        backup_path = f"{backup_base}/qwen3-0.6B"
    else:
        raise ValueError(f"Size {size} not yet supported. Only 0.6B is available.")

    paths = {}

    # Download tokenizer
    tokenizer_url = f"{hf_base}/{tokenizer_file}"
    tokenizer_backup = f"{backup_path}/{tokenizer_file}"
    paths["tokenizer"] = download_file(
        tokenizer_url, out_dir=out_dir, backup_url=tokenizer_backup
    )

    # Download model if requested
    if not tokenizer_only:
        model_url = f"{hf_base}/{model_file}"
        model_backup = f"{backup_path}/{model_file}"
        paths["model"] = download_file(
            model_url, out_dir=out_dir, backup_url=model_backup
        )

    return paths
