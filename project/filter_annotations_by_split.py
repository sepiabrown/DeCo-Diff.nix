#!/usr/bin/env python3
"""
Filter annotation files based on JSON records and split criteria.

This script filters annotation files in a given folder to only include those that correspond
to images in the JSON records that match the specified split.

Usage:
    python filter_annotations_by_split.py <json_file> <annotations_folder> <split>

Arguments:
    json_file: Path to JSON file with records (e.g., filtered_records_all_evaluation_records.json)
    annotations_folder: Path to folder containing annotation files
    split: Split to filter by (e.g., "train", "test", "val")

Example:
    python filter_annotations_by_split.py filtered_records_all_evaluation_records.json annotations_lsg_128_head train
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Set, List, Dict, Any

def load_json_records(json_file: str) -> Dict[str, Any]:
    """
    Load JSON file containing evaluation records.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        Dictionary containing the JSON data
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file {json_file}: {e}", e.doc, e.pos)

def extract_image_paths_by_split(records_data: Dict[str, Any], target_split: str) -> Set[str]:
    """
    Extract image paths from records that match the target split.
    
    Args:
        records_data: Dictionary containing records data
        target_split: Split to filter by (e.g., "train", "test")
        
    Returns:
        Set of image paths that match the target split
    """
    matching_image_paths = set()
    
    if 'records' not in records_data:
        print(f"⚠️  Warning: No 'records' key found in JSON data")
        return matching_image_paths
    
    records = records_data['records']
    print(f"📊 Found {len(records)} records in JSON file")
    
    for i, record in enumerate(records):
        # Check if record has the required fields
        if not isinstance(record, dict):
            print(f"⚠️  Warning: Record {i} is not a dictionary, skipping")
            continue
            
        # Extract split and image path
        record_split = record.get('split', '')
        image_path = record.get('image_path', '')
        
        if not image_path:
            print(f"⚠️  Warning: Record {i} has no image_path, skipping")
            continue
            
        # Check if split matches target
        if record_split == target_split:
            matching_image_paths.add(image_path)
            print(f"✅ Record {i}: {os.path.basename(image_path)} (split: {record_split})")
        else:
            print(f"⏭️  Record {i}: {os.path.basename(image_path)} (split: {record_split}, not matching)")
    
    print(f"\n📊 Summary: Found {len(matching_image_paths)} images matching split '{target_split}'")
    return matching_image_paths

def find_matching_annotation_files(annotations_folder: str, matching_image_paths: Set[str]) -> List[str]:
    """
    Find annotation files that correspond to the matching image paths.
    
    Args:
        annotations_folder: Path to folder containing annotation files
        matching_image_paths: Set of image paths to match
        
    Returns:
        List of paths to matching annotation files
    """
    if not os.path.exists(annotations_folder):
        raise FileNotFoundError(f"Annotations folder not found: {annotations_folder}")
    
    if not os.path.isdir(annotations_folder):
        raise NotADirectoryError(f"Path is not a directory: {annotations_folder}")
    
    matching_annotation_files = []
    
    # Get all JSON files in the annotations folder
    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    print(f"📁 Found {len(annotation_files)} annotation files in folder")
    
    # Check each annotation file
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotations_folder, annotation_file)
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            # Extract image path from annotation
            annotation_image_path = annotation_data.get('image_path', '')
            
            if not annotation_image_path:
                print(f"⚠️  Warning: {annotation_file} has no image_path, skipping")
                continue
            
            # Check if this annotation corresponds to a matching image
            if annotation_image_path in matching_image_paths:
                matching_annotation_files.append(annotation_path)
                print(f"✅ Match found: {annotation_file} -> {os.path.basename(annotation_image_path)}")
            else:
                print(f"⏭️  No match: {annotation_file} -> {os.path.basename(annotation_image_path)}")
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not read {annotation_file}: {e}")
            continue
    
    print(f"\n📊 Summary: Found {len(matching_annotation_files)} matching annotation files")
    return matching_annotation_files

def save_filtered_annotations_list(matching_files: List[str], output_file: str, target_split: str):
    """
    Save the list of matching annotation files to a text file.
    
    Args:
        matching_files: List of paths to matching annotation files
        output_file: Path to output file
        target_split: Split that was filtered by
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Filtered annotation files for split: {target_split}\n")
            f.write(f"# Total files: {len(matching_files)}\n")
            f.write(f"# Generated on: {__import__('datetime').datetime.now().isoformat()}\n\n")
            
            for file_path in matching_files:
                f.write(f"{file_path}\n")
        
        print(f"💾 Saved filtered annotations list to: {output_file}")
        
    except IOError as e:
        print(f"⚠️  Warning: Could not save output file: {e}")

def create_filtered_annotations_folder(matching_files: List[str], output_folder: str, target_split: str):
    """
    Create a new folder with only the matching annotation files.
    
    Args:
        matching_files: List of paths to matching annotation files
        output_folder: Path to output folder
        target_split: Split that was filtered by
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"📁 Created output folder: {output_folder}")
        
        copied_count = 0
        for file_path in matching_files:
            try:
                # Get just the filename
                filename = os.path.basename(file_path)
                dest_path = os.path.join(output_folder, filename)
                
                # Copy the file
                import shutil
                shutil.copy2(file_path, dest_path)
                copied_count += 1
                print(f"📋 Copied: {filename}")
                
            except (IOError, OSError) as e:
                print(f"⚠️  Warning: Could not copy {os.path.basename(file_path)}: {e}")
        
        print(f"✅ Successfully copied {copied_count}/{len(matching_files)} annotation files")
        
    except OSError as e:
        print(f"❌ Error creating output folder: {e}")

def main():
    """Main function to run the annotation filtering script."""
    parser = argparse.ArgumentParser(
        description="Filter annotation files based on JSON records and split criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter annotations for train split
  python filter_annotations_by_split.py records.json annotations_folder train
  
  # Filter annotations for test split
  python filter_annotations_by_split.py records.json annotations_folder test
  
  # Filter annotations for validation split
  python filter_annotations_by_split.py records.json annotations_folder val
        """
    )
    
    parser.add_argument("json_file", help="Path to JSON file with evaluation records")
    parser.add_argument("annotations_folder", help="Path to folder containing annotation files")
    parser.add_argument("split", help="Split to filter by (e.g., train, test, val)")
    parser.add_argument("--output-list", help="Path to save list of matching files (optional)")
    parser.add_argument("--output-folder", help="Path to create filtered annotations folder (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set up logging level
    if not args.verbose:
        # Suppress some output if not verbose
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    
    print(f"🔍 Annotation Filtering Script")
    print(f"================================")
    print(f"JSON file: {args.json_file}")
    print(f"Annotations folder: {args.annotations_folder}")
    print(f"Target split: {args.split}")
    print()
    
    try:
        # Step 1: Load JSON records
        print("📖 Loading JSON records...")
        records_data = load_json_records(args.json_file)
        print(f"✅ Loaded JSON file successfully")
        
        # Step 2: Extract image paths by split
        print(f"\n🔍 Filtering records by split '{args.split}'...")
        matching_image_paths = extract_image_paths_by_split(records_data, args.split)
        
        if not matching_image_paths:
            print(f"❌ No images found for split '{args.split}'")
            return
        
        # Step 3: Find matching annotation files
        print(f"\n🔍 Finding matching annotation files...")
        matching_annotation_files = find_matching_annotation_files(args.annotations_folder, matching_image_paths)
        
        if not matching_annotation_files:
            print(f"❌ No matching annotation files found")
            return
        
        # Step 4: Generate output
        print(f"\n💾 Generating output...")
        
        # Save list if requested
        if args.output_list:
            save_filtered_annotations_list(matching_annotation_files, args.output_list, args.split)
        
        # Create filtered folder if requested
        if args.output_folder:
            create_filtered_annotations_folder(matching_annotation_files, args.output_folder, args.split)
        
        # If no output options specified, show summary
        if not args.output_list and not args.output_folder:
            print(f"\n📊 Final Summary:")
            print(f"   Target split: {args.split}")
            print(f"   Matching images: {len(matching_image_paths)}")
            print(f"   Matching annotation files: {len(matching_annotation_files)}")
            print(f"   Annotations folder: {args.annotations_folder}")
            print(f"\n💡 Use --output-list or --output-folder to save results")
        
        print(f"\n✅ Annotation filtering completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
