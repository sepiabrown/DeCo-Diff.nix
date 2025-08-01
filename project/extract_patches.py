#!/usr/bin/env python3
"""
Extract patch images from false positives listed in detailed_confusion_matrix.json
"""

import json
import os
import cv2
import numpy as np
from PIL import Image
import argparse

def extract_patches_from_json(json_file_path, output_dir="./extracted_patches"):
    """
    Extract patch images from false positives in the JSON file
    
    Args:
        json_file_path: Path to the detailed_confusion_matrix.json file
        output_dir: Directory to save extracted patches
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get false positives
    false_positives = data.get('false_positives', [])
    
    if not false_positives:
        print("No false positives found in the JSON file.")
        return
    
    print(f"Found {len(false_positives)} false positives to extract.")
    
    # Track unique images to avoid loading the same image multiple times
    image_cache = {}
    
    for i, fp in enumerate(false_positives):
        image_path = fp['image_path']
        pixel_coords = fp['pixel_coordinates']  # [x1, y1, x2, y2]
        patch_size = fp.get('patch_size', 128)
        grid_position = fp.get('grid_position', [0, 0])
        
        print(f"Processing false positive {i+1}/{len(false_positives)}:")
        print(f"  Image: {image_path}")
        print(f"  Pixel coordinates: {pixel_coords}")
        print(f"  Grid position: {grid_position}")
        print(f"  Patch size: {patch_size}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found: {image_path}")
            continue
        
        # Load image (use cache if already loaded)
        if image_path not in image_cache:
            try:
                # Try loading with PIL first
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                image_cache[image_path] = image
                print(f"  Loaded image with shape: {image.shape}")
            except Exception as e:
                print(f"  ERROR loading image: {e}")
                continue
        else:
            image = image_cache[image_path]
        
        # Extract patch coordinates
        x1, y1, x2, y2 = pixel_coords
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Extract the patch
        patch = image[y1:y2, x1:x2]
        
        if patch.size == 0:
            print(f"  WARNING: Empty patch extracted")
            continue
        
        # Create filename
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        grid_x, grid_y = grid_position
        
        # Create descriptive filename
        filename = f"fp_{i+1:03d}_{image_name}_grid_{grid_x}_{grid_y}_coords_{x1}_{y1}_{x2}_{y2}.png"
        
        # Save patch
        patch_path = os.path.join(output_dir, filename)
        
        try:
            # Save with PIL to handle different image formats
            patch_pil = Image.fromarray(patch)
            patch_pil.save(patch_path)
            print(f"  Saved patch to: {patch_path}")
            print(f"  Patch shape: {patch.shape}")
        except Exception as e:
            print(f"  ERROR saving patch: {e}")
    
    print(f"\nExtraction complete! Patches saved to: {output_dir}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total false positives: {len(false_positives)}")
    print(f"  Unique images processed: {len(image_cache)}")
    print(f"  Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract patch images from false positives in JSON file")
    parser.add_argument("json_file", help="Path to detailed_confusion_matrix.json file")
    parser.add_argument("--output-dir", default="./extracted_patches", 
                       help="Output directory for extracted patches (default: ./extracted_patches)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"ERROR: JSON file not found: {args.json_file}")
        return
    
    extract_patches_from_json(args.json_file, args.output_dir)

if __name__ == "__main__":
    main() 