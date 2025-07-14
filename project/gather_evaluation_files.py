import os
import glob
import argparse
import re
import shutil
from pathlib import Path

def find_evaluation_files(base_path, pattern):
    """
    Find all evaluation JSON files under folders matching the pattern with timestamps.
    
    Args:
        base_path: Base directory to search in (e.g., './results')
        pattern: Pattern to match folder names (e.g., 'test')
    
    Returns:
        list: List of paths to evaluation JSON files
    """
    evaluation_files = []
    
    # Convert to Path object for easier handling
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return evaluation_files
    
    # Find all directories that match the pattern with timestamp
    # Pattern: {pattern}_YYYYMMDD_HHMMSS
    timestamp_pattern = re.compile(rf"{re.escape(pattern)}_\d{{6}}_\d{{6}}")
    
    # Search for matching directories
    matching_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and timestamp_pattern.match(item.name):
            matching_dirs.append(item)
    
    if not matching_dirs:
        print(f"No directories found matching pattern '{pattern}_YYYYMMDD_HHMMSS' in {base_path}")
        return evaluation_files
    
    print(f"Found {len(matching_dirs)} matching directories:")
    for dir_path in matching_dirs:
        print(f"  - {dir_path}")
    
    # Search for evaluation_results directories and evaluation JSON files
    for dir_path in matching_dirs:
        evaluation_results_dir = dir_path / "evaluation_results"
        
        if not evaluation_results_dir.exists():
            print(f"Warning: No 'evaluation_results' directory found in {dir_path}")
            continue
        
        # Find all evaluation JSON files
        eval_files = list(evaluation_results_dir.glob("*__evaluation.json"))
        
        if eval_files:
            print(f"  Found {len(eval_files)} evaluation files in {dir_path.name}")
            evaluation_files.extend(eval_files)
        else:
            print(f"  No evaluation files found in {dir_path.name}/evaluation_results")
    
    return evaluation_files

def copy_files_to_directory(file_paths, output_dir):
    """
    Copy all evaluation files to the specified output directory.
    
    Args:
        file_paths: List of file paths to copy
        output_dir: Directory to copy files to
    
    Returns:
        int: Number of files successfully copied
    """
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    for file_path in file_paths:
        try:
            # Copy file to output directory
            shutil.copy2(file_path, output_path)
            print(f"  Copied: {file_path.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {file_path.name}: {e}")
    
    return copied_count

def main():
    parser = argparse.ArgumentParser(description="Gather evaluation JSON files from timestamped folders")
    parser.add_argument("base_path", help="Base directory to search in (e.g., './results')")
    parser.add_argument("pattern", help="Pattern to match folder names (e.g., 'test')")
    parser.add_argument("--output", "-o", help="Output directory to copy evaluation files to (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"Searching for evaluation files...")
    print(f"Base path: {args.base_path}")
    print(f"Pattern: {args.pattern}")
    print("-" * 50)
    
    evaluation_files = find_evaluation_files(args.base_path, args.pattern)
    
    if evaluation_files:
        print(f"\nFound {len(evaluation_files)} evaluation files:")
        for i, file_path in enumerate(evaluation_files, 1):
            if args.verbose:
                print(f"  {i:3d}. {file_path}")
            else:
                print(f"  {i:3d}. {file_path.name}")
        
        # Copy files to output directory if specified
        if args.output:
            print(f"\nCopying files to: {args.output}")
            copied_count = copy_files_to_directory(evaluation_files, args.output)
            print(f"\nSuccessfully copied {copied_count} out of {len(evaluation_files)} files to {args.output}")
        
        return evaluation_files
    else:
        print("No evaluation files found.")
        return []

if __name__ == "__main__":
    main() 