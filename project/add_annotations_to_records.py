#!/usr/bin/env python3
"""
Script to add annotation files to filtered_records_all_evaluation_records.json

This script reads annotation files from a folder and adds them to the main JSON file
with empty patch_coords arrays, similar to the last 2 items in the records.
"""

import json
import os
import argparse
from pathlib import Path
import glob


def read_json_file(file_path):
    """Read and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def write_json_file(file_path, data):
    """Write data to a JSON file with proper formatting."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")


def extract_image_path_from_annotation(annotation_file_path):
    """Extract image_path from an annotation file."""
    try:
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        image_path = annotation_data.get('image_path')
        if not image_path:
            print(f"Warning: No image_path found in {annotation_file_path}")
            return None
        
        return image_path
    except Exception as e:
        print(f"Error reading annotation file {annotation_file_path}: {e}")
        return None


def add_annotations_to_records(main_json_path, annotations_folder, output_json_path=None):
    """
    Add annotation files to the main JSON records.
    
    Args:
        main_json_path: Path to the main JSON file (filtered_records_all_evaluation_records.json)
        annotations_folder: Path to the folder containing annotation files
        output_json_path: Path for the output JSON file (if None, overwrites the input)
    """
    
    # Read the main JSON file
    print(f"Reading main JSON file: {main_json_path}")
    main_data = read_json_file(main_json_path)
    if main_data is None:
        return False
    
    if 'records' not in main_data:
        print("Error: Main JSON file does not contain 'records' key")
        return False
    
    # Find all annotation files in the folder
    print(f"Scanning for annotation files in: {annotations_folder}")
    annotation_pattern = os.path.join(annotations_folder, "*__annotations.json")
    annotation_files = glob.glob(annotation_pattern)
    
    if not annotation_files:
        print(f"No annotation files found in {annotations_folder}")
        print(f"Expected pattern: *__annotations.json")
        return False
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # Track existing image paths to avoid duplicates
    existing_image_paths = set()
    for record in main_data['records']:
        if 'image_path' in record:
            existing_image_paths.add(record['image_path'])
    
    print(f"Found {len(existing_image_paths)} existing image paths in main JSON")
    
    # Process each annotation file
    added_count = 0
    skipped_count = 0
    
    for annotation_file in annotation_files:
        print(f"Processing: {os.path.basename(annotation_file)}")
        
        # Extract image_path from annotation
        image_path = extract_image_path_from_annotation(annotation_file)
        if not image_path:
            skipped_count += 1
            continue
        
        # Check if this image path already exists
        if image_path in existing_image_paths:
            print(f"  Skipping {os.path.basename(image_path)} - already exists in main JSON")
            skipped_count += 1
            continue
        
        # Create new record entry
        new_record = {
            "image_path": image_path,
            "image_path_original": image_path.replace("\\", "__").replace(":", ""),
            "patch_coords": [],  # Empty array for equal-spaced cropping
            "object": "pcb",
            "split": "train",
            "label": "normal",
            "mask_path": "",
            "category": "good"
        }
        
        # Add to records
        main_data['records'].append(new_record)
        existing_image_paths.add(image_path)
        added_count += 1
        
        print(f"  Added: {os.path.basename(image_path)}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  - Annotation files processed: {len(annotation_files)}")
    print(f"  - New records added: {added_count}")
    print(f"  - Records skipped: {skipped_count}")
    print(f"  - Total records in output: {len(main_data['records'])}")
    
    # Write output
    output_path = output_json_path if output_json_path else main_json_path
    print(f"\nWriting output to: {output_path}")
    write_json_file(output_path, main_data)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Add annotation files to filtered_records_all_evaluation_records.json"
    )
    parser.add_argument(
        "main_json",
        help="Path to the main JSON file (filtered_records_all_evaluation_records.json)"
    )
    parser.add_argument(
        "annotations_folder",
        help="Path to the folder containing annotation files (*__annotations.json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (if not specified, overwrites the input file)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without writing the file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.main_json):
        print(f"Error: Main JSON file not found: {args.main_json}")
        return 1
    
    if not os.path.exists(args.annotations_folder):
        print(f"Error: Annotations folder not found: {args.annotations_folder}")
        return 1
    
    if not os.path.isdir(args.annotations_folder):
        print(f"Error: Annotations path is not a directory: {args.annotations_folder}")
        return 1
    
    print("=" * 60)
    print("Annotation Adder Script")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()
    
    # Process the files
    success = add_annotations_to_records(
        args.main_json,
        args.annotations_folder,
        args.output if not args.dry_run else None
    )
    
    if success:
        print("\n✅ Script completed successfully!")
        return 0
    else:
        print("\n❌ Script failed!")
        return 1


if __name__ == "__main__":
    exit(main())
