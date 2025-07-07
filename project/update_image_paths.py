#!/usr/bin/env python3
"""
Script to update image_path field in JSON annotation files by changing folder names.
Example: Change 'PCB' to 'PCB_bak' in image paths.
"""

import json
import os
import argparse
from pathlib import Path
import glob


def update_image_paths_in_json(json_file_path, old_folder_name, new_folder_name, dry_run=False):
    """
    Update image_path field in a JSON annotation file.
    
    Args:
        json_file_path (str): Path to the JSON file
        old_folder_name (str): Old folder name to replace
        new_folder_name (str): New folder name to use
        dry_run (bool): If True, only print what would be changed without saving
    
    Returns:
        bool: True if changes were made, False otherwise
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if image_path field exists
        if 'image_path' not in data:
            print(f"Warning: No 'image_path' field found in {json_file_path}")
            return False
        
        old_path = data['image_path']
        
        # Replace the folder name in the path
        if old_folder_name in old_path:
            new_path = old_path.replace(old_folder_name, new_folder_name)
            
            if dry_run:
                print(f"Would change: {old_path}")
                print(f"To:          {new_path}")
                print("-" * 80)
            else:
                data['image_path'] = new_path
                
                # Write back to file
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"Updated: {json_file_path}")
                print(f"  {old_path} -> {new_path}")
                return True
        else:
            if dry_run:
                print(f"No '{old_folder_name}' found in path: {old_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return False


def process_directory(directory_path, old_folder_name, new_folder_name, pattern="*.json", dry_run=False):
    """
    Process all JSON files in a directory and its subdirectories.
    
    Args:
        directory_path (str): Directory to search for JSON files
        old_folder_name (str): Old folder name to replace
        new_folder_name (str): New folder name to use
        pattern (str): File pattern to match (default: "*.json")
        dry_run (bool): If True, only print what would be changed without saving
    
    Returns:
        tuple: (total_files, changed_files)
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist")
        return 0, 0
    
    # Find all JSON files
    json_files = list(directory.rglob(pattern))
    
    if not json_files:
        print(f"No {pattern} files found in {directory_path}")
        return 0, 0
    
    print(f"Found {len(json_files)} {pattern} files to process")
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
    print("=" * 80)
    
    total_files = len(json_files)
    changed_files = 0
    
    for json_file in json_files:
        if update_image_paths_in_json(str(json_file), old_folder_name, new_folder_name, dry_run):
            changed_files += 1
    
    return total_files, changed_files


def main():
    parser = argparse.ArgumentParser(
        description="Update image_path field in JSON annotation files by changing folder names"
    )
    parser.add_argument(
        "directory",
        help="Directory containing JSON files to process"
    )
    parser.add_argument(
        "old_folder",
        help="Old folder name to replace in image paths"
    )
    parser.add_argument(
        "new_folder",
        help="New folder name to use in image paths"
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File pattern to match (default: '*.json')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually modifying files"
    )
    parser.add_argument(
        "--single-file",
        help="Process only a single JSON file instead of a directory"
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process single file
        if not os.path.exists(args.single_file):
            print(f"Error: File {args.single_file} does not exist")
            return
        
        print(f"Processing single file: {args.single_file}")
        if args.dry_run:
            print("DRY RUN MODE - No files will be modified")
        print("=" * 80)
        
        update_image_paths_in_json(args.single_file, args.old_folder, args.new_folder, args.dry_run)
    else:
        # Process directory
        total_files, changed_files = process_directory(
            args.directory, 
            args.old_folder, 
            args.new_folder, 
            args.pattern, 
            args.dry_run
        )
        
        print("=" * 80)
        print(f"Summary:")
        print(f"  Total files processed: {total_files}")
        print(f"  Files changed: {changed_files}")
        
        if args.dry_run and changed_files > 0:
            print(f"\nTo apply these changes, run without --dry-run flag")


if __name__ == "__main__":
    main() 