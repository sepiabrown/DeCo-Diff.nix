#!/usr/bin/env python3
"""
Script to create annotation JSON files for images in the original path.
If the image is normal (non-defective), the defective_patches array is set to [].

Features:
- Process all images in a directory or a single image
- Optional random sampling for creating smaller datasets
- Configurable random seed for reproducible or truly random selection
- Configurable grid size and image extensions
- Mark images as defective or normal
"""

import os
import json
import argparse
import random
from pathlib import Path
from glob import glob
from typing import Optional, List
from utils import path_to_safe_filename


def create_annotation(image_path: str, output_dir: str, grid_size: int = 128, is_defective: bool = False) -> str:
    """
    Create an annotation JSON file for a image with defective_patches.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save the annotation JSON
        grid_size: Size of the grid patches (default: 128)
        is_defective: Whether the image is defective (default: False)
    
    Returns:
        Path to the created annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create annotation data
    annotation = {
        "image_path": os.path.abspath(image_path),
        "defective_patches": [[0, 0]] if is_defective else [],  # [0,0] for defective
        "grid_size": grid_size
    }
    
    # Create filename for the annotation
    safe_name = path_to_safe_filename(image_path)
    annotation_filename = f"{safe_name}__annotations.json"
    annotation_path = os.path.join(output_dir, annotation_filename)
    
    # Save the annotation
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    return annotation_path


def process_directory(input_dir: str, output_dir: str, grid_size: int = 128, 
                     image_extensions: Optional[List[str]] = None, is_defective: bool = False, 
                     random_sample: Optional[int] = None, random_seed: Optional[int] = None) -> List[str]:
    """
    Process all images in a directory and create annotation files.
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save annotation files
        grid_size: Size of the grid patches
        image_extensions: List of image file extensions to process
        is_defective: Whether the images are defective (default: False)
        random_sample: Number of files to randomly sample (default: None, process all)
        random_seed: Random seed for reproducible sampling (default: None, truly random)
    
    Returns:
        List of created annotation file paths
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Convert to absolute path and print
    output_dir_abs = os.path.abspath(output_dir)
    print(f"Output directory (absolute): {output_dir_abs}")
    
    created_files = []
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Apply random sampling if specified
    if random_sample is not None:
        if random_sample > len(image_files):
            print(f"Warning: Requested {random_sample} files but only {len(image_files)} found. Processing all files.")
            random_sample = len(image_files)
        else:
            # Set random seed if specified, otherwise use truly random selection
            if random_seed is not None:
                random.seed(random_seed)
                print(f"Using fixed random seed: {random_seed} (reproducible selection)")
            else:
                print("Using truly random selection (different each time)")
            
            original_count = len(image_files)
            image_files = random.sample(image_files, random_sample)
            print(f"Randomly selected {len(image_files)} files from {original_count} total images")
    
    # Create annotations for each image
    for image_path in image_files:
        try:
            annotation_path = create_annotation(image_path, output_dir, grid_size, is_defective)
            created_files.append(annotation_path)
            print(f"Created annotation: {os.path.basename(annotation_path)}")
        except Exception as e:
            print(f"Error creating annotation for {image_path}: {e}")
    
    return created_files


def main():
    parser = argparse.ArgumentParser(description="Create annotation JSON files for images in the original path")
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Directory containing images to annotate")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Directory to save annotation JSON files")
    parser.add_argument("--grid-size", "-g", type=int, default=128,
                       help="Grid size for patches (default: 128)")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
                       help="Image file extensions to process")
    parser.add_argument("--single-image", "-s",
                       help="Process a single image instead of a directory")
    parser.add_argument("--is-defective", action="store_true", default=False,
                       help="Mark images as defective (default: False)")
    parser.add_argument("--random-sample", "-r", type=int, metavar="N",
                       help="Randomly sample N files from the input directory (default: process all files). "
                            "Useful for creating smaller datasets for testing or when you want a subset of annotations.")
    parser.add_argument("--random-seed", type=int, metavar="SEED",
                       help="Random seed for reproducible sampling (default: None, truly random each time)")
    
    args = parser.parse_args()
    
    if args.single_image:
        # Process a single image
        if not os.path.exists(args.single_image):
            print(f"Error: Image file {args.single_image} not found")
            return
        
        # Print absolute output directory path
        output_dir_abs = os.path.abspath(args.output_dir)
        print(f"Output directory (absolute): {output_dir_abs}")
        
        try:
            annotation_path = create_annotation(args.single_image, args.output_dir, args.grid_size, args.is_defective)
            print(f"Created annotation: {annotation_path}")
        except Exception as e:
            print(f"Error creating annotation: {e}")
    else:
        # Process a directory
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            return
        
        created_files = process_directory(args.input_dir, args.output_dir, 
                                        args.grid_size, args.extensions, args.is_defective,
                                        args.random_sample, args.random_seed)
        print(f"\nCreated {len(created_files)} annotation files in {args.output_dir}")


if __name__ == "__main__":
    main() 