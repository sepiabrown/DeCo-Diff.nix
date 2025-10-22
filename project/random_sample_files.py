#!/usr/bin/env python3
"""
Random File Sampler
Randomly selects files from a given folder and copies them to a new folder.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List


def get_all_files(folder: Path, recursive: bool = False) -> List[Path]:
    """
    Get all files from the specified folder.

    Args:
        folder: Source folder path
        recursive: Whether to search recursively in subdirectories

    Returns:
        List of file paths
    """
    if recursive:
        return [f for f in folder.rglob('*') if f.is_file()]
    else:
        return [f for f in folder.iterdir() if f.is_file()]


def random_sample_files(
    source_folder: Path,
    num_samples: int,
    output_suffix: str = "_sampled",
    recursive: bool = False,
    seed: int = None
) -> Path:
    """
    Randomly sample files from source folder and copy to new folder.

    Args:
        source_folder: Source folder containing files
        num_samples: Number of files to randomly sample
        output_suffix: Suffix for the output folder name
        recursive: Whether to search recursively in subdirectories
        seed: Random seed for reproducibility

    Returns:
        Path to the created output folder
    """
    # Validate source folder
    if not source_folder.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")

    if not source_folder.is_dir():
        raise ValueError(f"Source path is not a directory: {source_folder}")

    # Get all files
    all_files = get_all_files(source_folder, recursive)

    if not all_files:
        raise ValueError(f"No files found in {source_folder}")

    # Validate number of samples
    if num_samples <= 0:
        raise ValueError(f"Number of samples must be positive, got {num_samples}")

    if num_samples > len(all_files):
        print(f"Warning: Requested {num_samples} samples but only {len(all_files)} files available.")
        print(f"Sampling all {len(all_files)} files.")
        num_samples = len(all_files)

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Random sample
    sampled_files = random.sample(all_files, num_samples)

    # Create output folder next to source folder
    output_folder = source_folder.parent / f"{source_folder.name}{output_suffix}"
    output_folder.mkdir(exist_ok=True)

    # Copy files
    print(f"Sampling {num_samples} files from {source_folder}")
    print(f"Output folder: {output_folder}")

    for i, file_path in enumerate(sampled_files, 1):
        # Preserve relative structure if recursive
        if recursive:
            rel_path = file_path.relative_to(source_folder)
            dest_path = output_folder / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest_path = output_folder / file_path.name

        shutil.copy2(file_path, dest_path)
        print(f"[{i}/{num_samples}] Copied: {file_path.name}")

    print(f"\nSuccessfully copied {num_samples} files to {output_folder}")
    return output_folder


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample files from a folder and copy to a new folder"
    )
    parser.add_argument(
        "source_folder",
        type=Path,
        help="Source folder containing files to sample from"
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of files to randomly sample"
    )
    parser.add_argument(
        "-s", "--suffix",
        type=str,
        default="_sampled",
        help="Suffix for output folder name (default: '_sampled')"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search recursively in subdirectories"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    try:
        random_sample_files(
            source_folder=args.source_folder,
            num_samples=args.num_samples,
            output_suffix=args.suffix,
            recursive=args.recursive,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
