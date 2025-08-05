#!/usr/bin/env python3
"""
Script to create annotation JSON files for defective images.
This creates JSON files with defective_patches arrays for images that have defects.
The script parses filenames to extract original image paths and patch coordinates.
"""

import os
import json
import argparse
import re
from pathlib import Path
from glob import glob
from typing import Optional, List, Dict, Set, Tuple
from utils import path_to_safe_filename


def parse_defective_filename(filename: str) -> Tuple[str, int, int]:
    """
    Parse a defective image filename to extract the original image path and patch coordinates.
    
    Expected filename format:
    C__Users__Public__Documents__deco-diff__datasets__PCB__Huang__PCB_DATASET__PCB_SELECTED_GRAY__n10_t2_l100-120_d-10_c0.2_s3.0__images__05_defect__png_x0_y128__anomaly.png
    
    Args:
        filename: The filename to parse
        
    Returns:
        Tuple of (original_image_path, patch_x, patch_y)
    """
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by double underscores to find the patch coordinates
    parts = name_without_ext.split('__')
    
    # Find the part that contains coordinates (format: x{number}_y{number} or png_x{number}_y{number})
    coord_part = None
    for part in parts:
        if re.match(r'.*x\d+_y\d+', part):
            coord_part = part
            break
    
    if coord_part is None:
        raise ValueError(f"Could not find coordinates in filename: {filename}")
    
    # Extract coordinates (handle cases like 'png_x0_y128' or 'x0_y128')
    coord_match = re.search(r'x(\d+)_y(\d+)', coord_part)
    if coord_match is None:
        raise ValueError(f"Could not parse coordinates from: {coord_part}")
    
    patch_x = int(coord_match.group(1))
    patch_y = int(coord_match.group(2))
    
    # Reconstruct the original image path
    # Find the part before the coordinates
    coord_index = parts.index(coord_part)
    original_parts = parts[:coord_index]
    
    # Reconstruct the original path by joining with double underscores
    original_path = '__'.join(original_parts)
    
    # Convert back to proper path format
    # Replace double underscores with path separators
    original_path = original_path.replace('__', os.sep)
    
    # Handle Windows drive letter format
    if original_path.startswith('C__'):
        original_path = 'C:' + original_path[3:]
    elif original_path.startswith('D__'):
        original_path = 'D:' + original_path[3:]
    elif original_path.startswith('E__'):
        original_path = 'E:' + original_path[3:]
    # Add more drive letters as needed
    
    # Ensure the path starts with a proper drive letter format
    if not original_path.startswith('C:') and not original_path.startswith('D:') and not original_path.startswith('E:'):
        # If it doesn't start with a drive letter, it might be a relative path
        # In this case, we need to make it absolute
        if original_path.startswith('C\\'):
            original_path = 'C:' + original_path[1:]
        elif original_path.startswith('D\\'):
            original_path = 'D:' + original_path[1:]
        elif original_path.startswith('E\\'):
            original_path = 'E:' + original_path[1:]
    
    # Fix the path - remove any duplicate parts
    # The path should start with the drive letter, not the full current directory
    if original_path.startswith('C:') and 'C:' in original_path[2:]:
        # Remove the duplicate C: part
        original_path = original_path[:2] + original_path[original_path[2:].find('C:') + 2:]
    
    # Also fix the case where the path starts with the current directory
    if original_path.startswith('C:\\Users\\Public\\Documents\\deco-diff\\C:'):
        original_path = original_path[original_path.find('C:') + 2:]
    
    # Add the file extension back
    # The extension is embedded in the coord_part like "png_x0_y128"
    original_ext = None
    if coord_part.startswith('png_') or coord_part.startswith('jpg_') or coord_part.startswith('jpeg_') or coord_part.startswith('bmp_') or coord_part.startswith('tiff_') or coord_part.startswith('tif_'):
        # Extract the extension from the coord_part
        original_ext = coord_part.split('_')[0]
    
    if original_ext:
        # Add the extension if it's not already there
        if not original_path.endswith(f'.{original_ext}'):
            original_path += f'.{original_ext}'
    
    return original_path, patch_x, patch_y


def create_normal_annotation(image_path: str, output_dir: str, grid_size: int = 128) -> str:
    """
    Create an annotation JSON file for a normal image with empty defective_patches.
    
    Args:
        image_path: Path to the original image file
        output_dir: Directory to save the annotation JSON
        grid_size: Size of the grid patches (default: 128)
    
    Returns:
        Path to the created annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create annotation data
    annotation = {
        "image_path": os.path.abspath(image_path),
        "defective_patches": [],  # Empty array for normal images
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


def create_defective_annotation(image_path: str, output_dir: str, patch_coords: List[Tuple[int, int]], 
                              grid_size: int = 128) -> str:
    """
    Create an annotation JSON file for a defective image with defective patches.
    
    Args:
        image_path: Path to the original image file
        output_dir: Directory to save the annotation JSON
        patch_coords: List of (x, y) coordinates of defective patches
        grid_size: Size of the grid patches (default: 128)
    
    Returns:
        Path to the created annotation file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create annotation data
    annotation = {
        "image_path": os.path.abspath(image_path),
        "defective_patches": [(0, 0)],  # List of defective patch coordinates
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


def process_input_directory(input_dir: str, output_dir: str, grid_size: int = 128,
                          image_extensions: Optional[List[str]] = None) -> Set[str]:
    """
    Process all images in input directory and create normal annotation files.
    
    Args:
        input_dir: Directory containing all original images
        output_dir: Directory to save annotation files
        grid_size: Size of the grid patches
        image_extensions: List of image file extensions to process
    
    Returns:
        Set of processed image paths
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    processed_images = set()
    
    # Create normal annotations for each image
    for image_path in image_files:
        try:
            annotation_path = create_normal_annotation(image_path, output_dir, grid_size)
            processed_images.add(image_path)
            print(f"Created normal annotation: {os.path.basename(annotation_path)}")
        except Exception as e:
            print(f"Error creating normal annotation for {image_path}: {e}")
    
    return processed_images


def process_defective_directory(defective_dir: str, output_dir: str, grid_size: int = 128,
                              image_extensions: Optional[List[str]] = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    Process all defective images in a directory and create annotation files.
    
    Args:
        defective_dir: Directory containing defective images
        output_dir: Directory to save annotation files
        grid_size: Size of the grid patches
        image_extensions: List of image file extensions to process
    
    Returns:
        Dictionary mapping original image paths to lists of defective patch coordinates
    """
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    # Dictionary to store defective patches for each original image
    image_defects: Dict[str, Set[Tuple[int, int]]] = {}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(defective_dir, f"*{ext}")))
        image_files.extend(glob(os.path.join(defective_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} defective images in {defective_dir}")
    
    # Process each defective image
    for defective_image_path in image_files:
        try:
            filename = os.path.basename(defective_image_path)
            original_path, patch_x, patch_y = parse_defective_filename(filename)
            
            # Add to the set of defective patches for this image
            if original_path not in image_defects:
                image_defects[original_path] = set()
            image_defects[original_path].add((patch_x, patch_y))
            
            print(f"Parsed: {filename} -> {original_path} at ({patch_x}, {patch_y})")
            
        except Exception as e:
            print(f"Error parsing filename {defective_image_path}: {e}")
    
    # Create annotation files for each original image
    created_files = []
    for original_path, patch_coords_set in image_defects.items():
        try:
            # Convert set to sorted list for consistent output
            patch_coords_list = sorted(list(patch_coords_set))
            
            annotation_path = create_defective_annotation(
                original_path, output_dir, patch_coords_list, grid_size
            )
            created_files.append(annotation_path)
            print(f"Created defective annotation for {original_path}: {len(patch_coords_list)} defective patches")
            
        except Exception as e:
            print(f"Error creating defective annotation for {original_path}: {e}")
    
    return {path: sorted(list(coords)) for path, coords in image_defects.items()}


def main():
    parser = argparse.ArgumentParser(description="Create annotation JSON files for defective images")
    parser.add_argument("--input-dir", "-i", required=True,
                       help="Directory containing all original images (for normal annotations)")
    parser.add_argument("--defective-dir", "-d", required=True,
                       help="Directory containing defective images")
    parser.add_argument("--output-dir", "-o", required=True,
                       help="Directory to save annotation JSON files")
    parser.add_argument("--grid-size", "-g", type=int, default=128,
                       help="Grid size for patches (default: 128)")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
                       help="Image file extensions to process")
    parser.add_argument("--single-image", "-s",
                       help="Process a single defective image instead of a directory")
    
    args = parser.parse_args()
    
    # Print absolute output directory path
    output_dir_abs = os.path.abspath(args.output_dir)
    print(f"Output directory (absolute): {output_dir_abs}")
    
    if args.single_image:
        # Process a single defective image
        if not os.path.exists(args.single_image):
            print(f"Error: Defective image file {args.single_image} not found")
            return
        
        try:
            filename = os.path.basename(args.single_image)
            original_path, patch_x, patch_y = parse_defective_filename(filename)
            
            annotation_path = create_defective_annotation(
                original_path, args.output_dir, [(patch_x, patch_y)], args.grid_size
            )
            print(f"Created annotation: {annotation_path}")
            print(f"Original image: {original_path}")
            print(f"Defective patch: ({patch_x}, {patch_y})")
            
        except Exception as e:
            print(f"Error creating annotation: {e}")
    else:
        # Check if directories exist
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            return
        
        if not os.path.exists(args.defective_dir):
            print(f"Error: Defective directory {args.defective_dir} not found")
            return
        
        # Step 1: Create normal annotations for all images in input directory
        print(f"\nStep 1: Creating normal annotations for all images...")
        processed_images = process_input_directory(
            args.input_dir, args.output_dir, args.grid_size, args.extensions
        )
        
        # Step 2: Process defective images and update annotations
        print(f"\nStep 2: Processing defective images and updating annotations...")
        image_defects = process_defective_directory(
            args.defective_dir, args.output_dir, args.grid_size, args.extensions
        )
        
        print(f"\nSummary:")
        print(f"Created normal annotations for {len(processed_images)} images")
        print(f"Updated {len(image_defects)} images with defective patches")
        total_patches = sum(len(patches) for patches in image_defects.values())
        print(f"Total defective patches: {total_patches}")
        print(f"Created annotation files in {args.output_dir}")


if __name__ == "__main__":
    main() 