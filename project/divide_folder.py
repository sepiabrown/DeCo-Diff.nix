#!/usr/bin/env python3
"""
Script to divide files from a folder into equal-sized subfolders.

Usage:
    python divide_folder.py <source_folder> [num_parts]

Examples:
    python divide_folder.py annotations 4
    python divide_folder.py datasets 8
"""

import os
import sys
import shutil
import math
from pathlib import Path
from typing import List, Tuple


def get_files_in_folder(folder_path: str) -> List[str]:
    """Get all files in the folder (excluding subdirectories)."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            files.append(item)
    
    return sorted(files)


def divide_files(files: List[str], num_parts: int) -> List[List[str]]:
    """Divide files into equal-sized parts."""
    if num_parts <= 0:
        raise ValueError("Number of parts must be positive")
    
    if num_parts > len(files):
        print(f"Warning: Number of parts ({num_parts}) is greater than number of files ({len(files)})")
        num_parts = len(files)
    
    # Calculate files per part
    files_per_part = math.ceil(len(files) / num_parts)
    
    # Divide files into parts
    parts = []
    for i in range(0, len(files), files_per_part):
        part = files[i:i + files_per_part]
        parts.append(part)
    
    return parts


def create_subfolders(base_folder: str, folder_name: str, num_parts: int) -> List[str]:
    """Create subfolders with names like <folder_name>_1, <folder_name>_2, etc."""
    subfolder_paths = []
    
    for i in range(1, num_parts + 1):
        subfolder_name = f"{folder_name}_{i}"
        subfolder_path = os.path.join(base_folder, subfolder_name)
        
        # Create subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        subfolder_paths.append(subfolder_path)
        
        print(f"Created subfolder: {subfolder_path}")
    
    return subfolder_paths


def copy_files_to_subfolders(source_folder: str, files_parts: List[List[str]], subfolder_paths: List[str]):
    """Copy files to their respective subfolders."""
    total_files = sum(len(part) for part in files_parts)
    copied_files = 0
    
    for part_idx, (files_part, subfolder_path) in enumerate(zip(files_parts, subfolder_paths)):
        print(f"\nCopying files to {os.path.basename(subfolder_path)}:")
        
        for file_name in files_part:
            source_path = os.path.join(source_folder, file_name)
            dest_path = os.path.join(subfolder_path, file_name)
            
            try:
                shutil.copy2(source_path, dest_path)
                print(f"  ✓ {file_name}")
                copied_files += 1
            except Exception as e:
                print(f"  ✗ {file_name} - Error: {e}")
    
    print(f"\nTotal files copied: {copied_files}/{total_files}")


def main():
    """Main function to divide folder contents."""
    if len(sys.argv) < 2:
        print("Usage: python divide_folder.py <source_folder> [num_parts]")
        print("Examples:")
        print("  python divide_folder.py annotations 4")
        print("  python divide_folder.py datasets 8")
        sys.exit(1)
    
    source_folder = sys.argv[1]
    num_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    try:
        print(f"Dividing folder: {source_folder}")
        print(f"Number of parts: {num_parts}")
        print("=" * 50)
        
        # Get all files in the source folder
        files = get_files_in_folder(source_folder)
        print(f"Found {len(files)} files in {source_folder}")
        
        if len(files) == 0:
            print("No files found in the source folder.")
            sys.exit(1)
        
        # Divide files into parts
        files_parts = divide_files(files, num_parts)
        
        # Show division plan
        print(f"\nDivision plan:")
        for i, part in enumerate(files_parts, 1):
            print(f"  Part {i}: {len(part)} files")
        
        # Get base folder and folder name
        base_folder = os.path.dirname(source_folder) or "."
        folder_name = os.path.basename(source_folder)
        
        # Create subfolders
        print(f"\nCreating subfolders in: {base_folder}")
        subfolder_paths = create_subfolders(base_folder, folder_name, num_parts)
        
        # Copy files to subfolders
        print(f"\nCopying files...")
        copy_files_to_subfolders(source_folder, files_parts, subfolder_paths)
        
        print(f"\n" + "=" * 50)
        print(f"Division completed successfully!")
        print(f"Files divided into {num_parts} subfolders:")
        for i, subfolder_path in enumerate(subfolder_paths, 1):
            file_count = len(os.listdir(subfolder_path))
            print(f"  {os.path.basename(subfolder_path)}: {file_count} files")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 