#!/usr/bin/env python3
"""
Process raw data files to generate evaluation results.

This script takes the intermediate raw data files saved by evaluation_DeCo_Diff_raw.py
and processes them to generate the same results as evaluation_DeCo_Diff2.py.
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any
from collections import OrderedDict
from tqdm import tqdm
import glob
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt

# Import necessary functions from evaluation_DeCo_Diff2.py
from evaluation_DeCo_Diff2 import (
    make_record, _to_numpy, _binary_mask,
    make_excel, draw_patch_rectangles_on_image
)

# Import utility functions
from utils import (
    path_to_safe_filename,
    safe_filename_to_path,
    _binary_mask_exclude_boundary3,
    load_original_images,
    load_ground_truth_map
)

# Type definitions
Kinded = Tuple[str, Any]  # (kind, value)
Record = OrderedDict[str, Kinded]


def _get_patch_base_key_from_filename(file_path: str) -> str:
    """
    Return a stable base key for a minimal_diff artifact by stripping the
    suffix after "__minimal_diff" regardless of the specific type or extension.

    Examples:
      - "...__minimal_diff_encodedrecon.npy" -> "...__minimal_diff"
      - "...__minimal_diff_coords.npy"      -> "...__minimal_diff"

    The returned key matches for all artifacts (encodedrecon/latent/anomaly/coords)
    that belong to the same patch.
    """
    name = os.path.basename(file_path)
    stem, _ = os.path.splitext(name)
    # Split once at the marker and use the left side (plus the marker itself)
    parts = stem.split("__minimal_diff", 1)
    if len(parts) == 0:
        return stem
    # Re-attach the marker so keys are unambiguous and consistent
    return parts[0] + "__minimal_diff"


def create_anomaly_overlay(original_image: np.ndarray, anomaly_map: np.ndarray, is_binary: bool = True) -> np.ndarray:
    """
    Create an overlay of anomaly map on original image.
    
    Args:
        original_image: Original image array
        anomaly_map: Anomaly map array
        is_binary: Whether the anomaly map is binary (True) or continuous (False)
        
    Returns:
        Overlay image array
    """
    if is_binary:
        # Binary overlay using pure red
        overlay = original_image.copy()
        # For binary maps, use the values directly (they're already 0 or 1)
        mask = anomaly_map.astype(bool)
        # Create pure red overlay for anomaly regions
        h, w = anomaly_map.shape
        anomaly_colored = np.zeros((h, w, 3), dtype=np.uint8)
        # Set red channel for anomaly regions (pure red)
        anomaly_colored[mask, 0] = 255  # Red channel (RGB format)
        # Apply the red anomaly regions to the overlay
        overlay[mask] = anomaly_colored[mask]
    else:
        # Continuous overlay using HOT colormap
        # Convert [0, 1] range to [0, 255] range for colormap
        anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
        # Apply HOT colormap to the map (returns BGR)
        anomaly_colored_bgr = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_HOT)
        # Convert BGR to RGB for proper overlay
        anomaly_colored = cv2.cvtColor(anomaly_colored_bgr, cv2.COLOR_BGR2RGB)
        # Create overlay using alpha blending
        overlay = cv2.addWeighted(original_image, 0.2, anomaly_colored, 0.8, 0)
    
    return overlay

def create_anomaly_map_image(anomaly_map: np.ndarray, is_binary: bool = True) -> np.ndarray:
    """
    Create an image from anomaly map data.
    
    Args:
        anomaly_map: Anomaly map array
        is_binary: Whether the anomaly map is binary (True) or continuous (False)
        
    Returns:
        Image array ready for saving
    """
    if is_binary:
        # Binary maps: convert 0/1 to 0/255 for proper display
        map_img = (anomaly_map * 255).astype(np.uint8)
        # Convert to RGB by stacking the same channel 3 times
        if len(map_img.shape) == 2:
            map_img = np.stack([map_img] * 3, axis=-1)
    else:
        # Continuous maps: use cv2.applyColorMap with COLORMAP_HOT
        # Convert [0, 1] range to [0, 255] range for colormap
        anomaly_map_uint8 = (anomaly_map * 255).astype(np.uint8)
        # Apply HOT colormap to the map (returns BGR)
        anomaly_colored_bgr = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_HOT)
        # Convert BGR to RGB for proper display
        map_img = cv2.cvtColor(anomaly_colored_bgr, cv2.COLOR_BGR2RGB)
    
    return map_img

def save_side_by_side_image(original_image: np.ndarray, processed_image: np.ndarray, 
                           output_path: str) -> None:
    """
    Save a side-by-side image with original on left and processed on right.
    
    Args:
        original_image: Original image array
        processed_image: Processed image array
        output_path: Path to save the image
    """
    # Ensure both images have the same number of dimensions
    if len(original_image.shape) != len(processed_image.shape):
        # If processed_image is 2D and original_image is 3D, convert processed_image to 3D
        if len(original_image.shape) == 3 and len(processed_image.shape) == 2:
            # Convert 2D to 3D by repeating the channel
            processed_image = np.stack([processed_image] * 3, axis=-1)
        elif len(original_image.shape) == 2 and len(processed_image.shape) == 3:
            # Convert 3D to 2D by taking the first channel
            processed_image = processed_image[:, :, 0]
    
    
    side_by_side_image = np.hstack([original_image, processed_image])
    PILImage.fromarray(side_by_side_image).save(output_path)

def save_image_results_from_raw_data(
    records: List[Record],
    output_dir: str,
    ground_truth_map: Dict[str, Set[Tuple[int, int]]] = None,
    original_images: Dict[str, np.ndarray] = None,
    enable_save_optional_image_results: bool = False,
    enable_save_whole_image_results: bool = False,
    patch_size: int = 128
) -> None:
    """
    Save image results from raw data records.
    This function creates marked images and anomaly maps similar to the original evaluation.
    """
    # Create output directories
    marked_images_dir = os.path.join(output_dir, "marked_images")
    os.makedirs(marked_images_dir, exist_ok=True)
    
    # Create status-based subfolders
    status_folders = {}
    for status in ['TP', 'FN', 'FP', 'TN']:
        status_dir = os.path.join(marked_images_dir, status)
        os.makedirs(status_dir, exist_ok=True)
        status_folders[status] = status_dir
    
    # Create image-level directory
    image_level_dir = os.path.join(marked_images_dir, "image_level")
    os.makedirs(image_level_dir, exist_ok=True)
    
    # Create anomaly maps directory
    anomaly_maps_dir = os.path.join(output_dir, "anomaly_maps")
    os.makedirs(anomaly_maps_dir, exist_ok=True)
    
    # Process each patch individually for patch-level images
    for record in tqdm(records, desc="Saving patch-level images"):
        # Get patch information
        patch_x, patch_y = record["patch_coords"][1]
        status = record["status"][1]
        image_path = record["image_path"][1]

        # Create unique name for this patch
        if "image_path_original" in record:
            file_info_original = record["image_path_original"][1]
        else:
            file_info_original = record["image_path"][1]

        patch_name = f"{file_info_original}_x{patch_x}_y{patch_y}"
        
        # Check if filename contains "108826_198" for debug and visualization
        should_debug = "108826_198" in patch_name
        should_visualize = "108826_198" in patch_name

        # Save additional patch-level images if enabled
        if enable_save_optional_image_results:
            
            original_image = original_images[image_path]
            # Calculate the actual patch dimensions to extract
            h, w = original_image.shape[:2]
            patch_height = min(patch_size, h - patch_y)
            patch_width = min(patch_size, w - patch_x)
            original_patch = original_image[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width]
            
            # Save binary anomaly map
            binary_map = record["anomaly_map_arithmetic_binary"][1]
            # Ensure binary_map is 2D (it's already cropped to the correct size)
            if len(binary_map.shape) > 2:
                binary_map = binary_map.squeeze()
            
            # Fix: Handle 1D arrays by reshaping to match the original patch dimensions
            if len(binary_map.shape) == 1:
                # Get the original patch dimensions from the record
                patch_x, patch_y = record["patch_coords"][1]
                original_image = original_images[image_path]
                h, w = original_image.shape[:2]
                actual_patch_height = min(patch_size, h - patch_y)
                actual_patch_width = min(patch_size, w - patch_x)
                
                # Reshape the 1D array to match the actual patch dimensions
                if binary_map.shape[0] == actual_patch_height * actual_patch_width:
                    binary_map = binary_map.reshape(actual_patch_height, actual_patch_width)
                else:
                    # If the size doesn't match, create a properly sized array
                    # Create a zero array of the correct size
                    binary_map = np.zeros((actual_patch_height, actual_patch_width), dtype=binary_map.dtype)
            
            # Debug messages for files containing "108826_198"
            if should_debug:
                print(f"Processing patch: {patch_name}")
                print(f"  Status: {status}")
                print(f"  Patch coordinates: ({patch_x}, {patch_y})")
                print(f"  Original patch shape: {original_patch.shape}")
                print(f"  Binary map shape: {binary_map.shape}")
                print(f"  Binary map range: [{binary_map.min()}, {binary_map.max()}]")
                print(f"  Binary map sum: {np.sum(binary_map)}")
            
            # Create binary map image
            binary_image = create_anomaly_map_image(binary_map, is_binary=True)
            
            # Visualization for files containing "108826_198"
            if should_visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original patch
                    axes[0].imshow(original_patch)
                    axes[0].set_title(f'Original Patch\nShape: {original_patch.shape}')
                    axes[0].axis('off')
                    
                    # Binary map
                    axes[1].imshow(binary_map, cmap='gray')
                    axes[1].set_title(f'Binary Anomaly Map\nSum: {np.sum(binary_map)}')
                    axes[0].axis('off')
                    
                    # Side-by-side image
                    side_by_side = np.hstack([original_patch, binary_image])
                    axes[2].imshow(side_by_side)
                    axes[2].set_title(f'Side-by-Side\nOriginal | Binary Map')
                    axes[2].axis('off')
                    
                    plt.suptitle(f'Patch: {patch_name}')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                except ImportError:
                    print("Warning: matplotlib not available for visualization")
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            save_side_by_side_image(original_patch, binary_image, 
                                  os.path.join(status_folders[status], f"{patch_name}__binary.png"))
            
            # Save continuous anomaly map
            anomaly_map = record["anomaly_map_arithmetic"][1]
            # Ensure anomaly_map is 2D and crop to match original patch size
            if len(anomaly_map.shape) > 2:
                anomaly_map = anomaly_map.squeeze()
            patch_height, patch_width = original_patch.shape[:2]
            anomaly_map = anomaly_map[:patch_height, :patch_width]
            
            # Debug messages for continuous anomaly map
            if should_debug:
                print(f"  Continuous anomaly map shape: {anomaly_map.shape}")
                print(f"  Continuous anomaly map range: [{anomaly_map.min():.3f}, {anomaly_map.max():.3f}]")
                print(f"  Continuous anomaly map mean: {anomaly_map.mean():.3f}")
                print(f"  Continuous anomaly map std: {anomaly_map.std():.3f}")
            
            # Create continuous anomaly map image
            anomaly_image = create_anomaly_map_image(anomaly_map, is_binary=False)
            
            # Visualization for continuous anomaly map
            if should_visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original patch
                    axes[0].imshow(original_patch)
                    axes[0].set_title(f'Original Patch\nShape: {original_patch.shape}')
                    axes[0].axis('off')
                    
                    # Continuous anomaly map
                    axes[1].imshow(anomaly_map, cmap='hot')
                    axes[1].set_title(f'Continuous Anomaly Map\nRange: [{anomaly_map.min():.3f}, {anomaly_map.max():.3f}]')
                    axes[1].axis('off')
                    
                    # Side-by-side image
                    side_by_side = np.hstack([original_patch, anomaly_image])
                    axes[2].imshow(side_by_side)
                    axes[2].set_title(f'Side-by-Side\nOriginal | Continuous Map')
                    axes[2].axis('off')
                    
                    plt.suptitle(f'Patch: {patch_name} (Continuous)')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                except ImportError:
                    print("Warning: matplotlib not available for visualization")
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            save_side_by_side_image(original_patch, anomaly_image, 
                                  os.path.join(status_folders[status], f"{patch_name}__anomaly.png"))
            
            # Save latent map
            latent_map = record["encoded"][1]  # This is the latent data
            # Ensure latent_map is 2D
            if len(latent_map.shape) > 2:
                latent_map = latent_map.squeeze()
            # Crop latent_map to match the original patch size
            patch_height, patch_width = original_patch.shape[:2]
            latent_map = latent_map[:patch_height, :patch_width]
            # Convert to RGB
            if len(latent_map.shape) == 2:
                latent_image = np.stack([latent_map] * 3, axis=-1)
            else:
                latent_image = latent_map
            # Normalize latent to 0-255 range
            latent_image = ((latent_image - latent_image.min()) / (latent_image.max() - latent_image.min()) * 255).astype(np.uint8)
            
            # Create side-by-side: original patch on left, latent map on right
            side_by_side_latent = np.hstack([original_patch, latent_image])
            latent_path = os.path.join(status_folders[status], f"{patch_name}__latent.png")
            PILImage.fromarray(side_by_side_latent).save(latent_path)
            
            # Create overlay versions (anomaly map overlaid on original patch)
            # Binary overlay
            binary_overlay = create_anomaly_overlay(original_patch, binary_map, is_binary=True)
            
            # Debug messages for binary overlay
            if should_debug:
                print(f"  Binary overlay shape: {binary_overlay.shape}")
                print(f"  Binary overlay range: [{binary_overlay.min()}, {binary_overlay.max()}]")
            
            # Visualization for binary overlay
            if should_visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Original patch
                    axes[0].imshow(original_patch)
                    axes[0].set_title(f'Original Patch')
                    axes[0].axis('off')
                    
                    # Binary overlay
                    axes[1].imshow(binary_overlay)
                    axes[1].set_title(f'Binary Overlay\nRed = Anomaly')
                    axes[1].axis('off')
                    
                    plt.suptitle(f'Patch: {patch_name} (Binary Overlay)')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                except ImportError:
                    print("Warning: matplotlib not available for visualization")
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            save_side_by_side_image(original_patch, binary_overlay, 
                                  os.path.join(status_folders[status], f"{patch_name}__ao_binary.png"))
            
            # Continuous overlay
            continuous_overlay = create_anomaly_overlay(original_patch, anomaly_map, is_binary=False)
            
            # Debug messages for continuous overlay
            if should_debug:
                print(f"  Continuous overlay shape: {continuous_overlay.shape}")
                print(f"  Continuous overlay range: [{continuous_overlay.min()}, {continuous_overlay.max()}]")
            
            # Visualization for continuous overlay
            if should_visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Original patch
                    axes[0].imshow(original_patch)
                    axes[0].set_title(f'Original Patch')
                    axes[0].axis('off')
                    
                    # Continuous overlay
                    axes[1].imshow(continuous_overlay)
                    axes[1].set_title(f'Continuous Overlay\nHOT colormap')
                    axes[1].axis('off')
                    
                    plt.suptitle(f'Patch: {patch_name} (Continuous Overlay)')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                except ImportError:
                    print("Warning: matplotlib not available for visualization")
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            save_side_by_side_image(original_patch, continuous_overlay, 
                                  os.path.join(status_folders[status], f"{patch_name}__ao_anomaly.png"))
    
    if enable_save_whole_image_results:
        ## Group records by image path for image-level processing
        image_records = {}
        for record in records:
            image_path = record["image_path"][1]
            if image_path not in image_records:
                image_records[image_path] = []
            image_records[image_path].append(record)
        # Process image-level images (full image reconstructions)
        for _, (image_path, image_record_list) in enumerate(tqdm(image_records.items(), desc="Saving image-level images")):        
            # Build predicted defective set and ground truth defective set
            predicted_defective_set = set()

            # Get ground truth defective set for this image
            ground_truth_defective = ground_truth_map.get(image_path, set()) if ground_truth_map else set()
            for record in image_record_list:
                patch_x, patch_y = record["patch_coords"][1]
                anomaly_pixels = record["anomaly_pixels"][1]
                status = record["status"][1]

                # Calculate grid coordinates
                grid_row = patch_y // patch_size
                grid_col = patch_x // patch_size

                # Add to predicted defective set if predicted defective (use the same logic as in record creation)
                if record["is_predicted_defective"][1]:
                    predicted_defective_set.add((grid_row, grid_col))

            # Calculate overlapping regions
            overlapping = predicted_defective_set.intersection(ground_truth_defective)

            # Create image-level reconstruction (full image, not patch-level)
            # Get image dimensions from the original image
            image_path = image_record_list[0]["image_path"][1]
            if original_images and image_path in original_images:
                orig_image = original_images[image_path]
                max_y, max_x = orig_image.shape[:2]  # Height, Width
            else:
                # Fallback: estimate from patch coordinates
                max_x = max(record["patch_coords"][1][0] for record in image_record_list) + patch_size
                max_y = max(record["patch_coords"][1][1] for record in image_record_list) + patch_size

            # Create full image reconstruction
            full_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

            # Fill the full image with patch data
            for record in image_record_list:
                patch_x, patch_y = record["patch_coords"][1]
                patch_data = record["orig"][1]

                # Convert patch data to RGB
                if len(patch_data.shape) == 2:
                    patch_rgb = np.stack([patch_data] * 3, axis=-1)
                else:
                    patch_rgb = patch_data

                # Convert to uint8
                patch_rgb = (patch_rgb * 255).astype(np.uint8)

                # Ensure patch fits in the full image
                patch_height = min(patch_size, max_y - patch_y)
                patch_width = min(patch_size, max_x - patch_x)
                if patch_height > 0 and patch_width > 0:
                    # Ensure patch_rgb has the right shape
                    if patch_rgb.shape[-1] == 1:
                        patch_rgb = np.repeat(patch_rgb, 3, axis=-1)

                    full_image[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = \
                        patch_rgb[:patch_height, :patch_width]

            # Create marked full image for image_level
            marked_full_img = draw_patch_rectangles_on_image(
                full_image, predicted_defective_set, ground_truth_defective, overlapping, 
                patch_size=patch_size, grid_thickness=1
            )

            # Create base name without patch coordinates for image_level
            if image_record_list:
                if "image_path_original" in image_record_list[0]:
                    file_info_original = image_record_list[0]["image_path_original"][1]
                else:
                    file_info_original = image_record_list[0]["image_path"][1]
                base_name = f"{file_info_original}"
            else:
                base_name = f"unknown_image"

            # Save in image_level directory
            image_level_path = os.path.join(image_level_dir, f"{base_name}.png")
            PILImage.fromarray(marked_full_img).save(image_level_path)

            # Create and save anomaly maps
            # Create anomaly maps for the full image
            anomaly_maps = {
                'arithmetic': np.zeros((max_y, max_x), dtype=np.float32),
                'arithmetic_binary': np.zeros((max_y, max_x), dtype=np.float32),
            }

            # Fill anomaly maps with patch data
            for record in image_record_list:
                patch_x, patch_y = record["patch_coords"][1]
                patch_arithmetic = record["anomaly_map_arithmetic"][1]
                patch_arithmetic_binary = record["anomaly_map_arithmetic_binary"][1]

                # Ensure patch data is 2D by squeezing extra dimensions
                if len(patch_arithmetic.shape) > 2:
                    patch_arithmetic = patch_arithmetic.squeeze()
                if len(patch_arithmetic_binary.shape) > 2:
                    patch_arithmetic_binary = patch_arithmetic_binary.squeeze()

                # Ensure patch data fits in the map
                patch_height = min(patch_size, max_y - patch_y)
                patch_width = min(patch_size, max_x - patch_x)

                if patch_height > 0 and patch_width > 0:
                    # Ensure patch data is cropped to match the available space
                    # The patch data in records is always full size (128x128), so we need to crop it
                    patch_arithmetic_cropped = patch_arithmetic[:patch_height, :patch_width]

                    # Handle 1D binary arrays by reshaping them to 2D
                    if len(patch_arithmetic_binary.shape) == 1:
                        # If it's 1D, reshape it to match the expected dimensions
                        if patch_arithmetic_binary.shape[0] == patch_height * patch_width:
                            patch_arithmetic_binary = patch_arithmetic_binary.reshape(patch_height, patch_width)
                        else:
                            # If the size doesn't match, create a zero array of the correct size
                            patch_arithmetic_binary = np.zeros((patch_height, patch_width), dtype=patch_arithmetic_binary.dtype)

                    patch_arithmetic_binary_cropped = patch_arithmetic_binary[:patch_height, :patch_width]

                    anomaly_maps['arithmetic'][patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = \
                        patch_arithmetic_cropped
                    anomaly_maps['arithmetic_binary'][patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = \
                        patch_arithmetic_binary_cropped

            # Save anomaly maps
            for map_name, map_data in anomaly_maps.items():
                # Create overlay images (anomaly map overlaid on original image)
                # Try to load the original image from the file path
                original_image = None
                try:
                    # Extract the original image path from the filename
                    # The file_path in the record contains the original path
                    original_image_path = image_path
                    if os.path.exists(original_image_path):
                        original_image = np.array(PILImage.open(original_image_path).convert('RGB'))
                        # print(f"Loaded original image: {original_image_path}")
                    else:
                        # print(f"Original image not found: {original_image_path}, using reconstructed image")
                        original_image = full_image.copy()
                except Exception as e:
                    # print(f"Error loading original image {image_path}: {e}, using reconstructed image")
                    original_image = full_image.copy()
                # Create anomaly map image
                is_binary = map_name.endswith('_binary')
                map_img = create_anomaly_map_image(map_data, is_binary=is_binary)

                # Save side-by-side image
                save_side_by_side_image(original_image, map_img, 
                                      os.path.join(anomaly_maps_dir, f"{base_name}__{map_name}.png"))
                # Create overlay
                is_binary = map_name.endswith('_binary')
                overlay = create_anomaly_overlay(original_image, map_data, is_binary=is_binary)

                # Save side-by-side overlay image
                save_side_by_side_image(original_image, overlay, 
                                      os.path.join(anomaly_maps_dir, f"{base_name}__ao_{map_name}.png"))

                # Create marked overlay (overlay with patch rectangles)
                marked_overlay = draw_patch_rectangles_on_image(
                    overlay, predicted_defective_set, ground_truth_defective, overlapping, 
                    patch_size=patch_size, grid_thickness=1
                )

                # Create side-by-side image: original image on left, marked overlay on right
                side_by_side_marked_overlay = np.hstack([original_image, marked_overlay])

                # Save side-by-side marked overlay image
                marked_overlay_path = os.path.join(anomaly_maps_dir, f"{base_name}__mo_{map_name}.png")
                PILImage.fromarray(side_by_side_marked_overlay).save(marked_overlay_path)

        print(f"Image results saved to: {output_dir}")

def save_json_results_from_raw_data(
    records: List[Record],
    output_dir: str,
    patch_size: int = 128
) -> None:
    """
    Save JSON results from raw data records.
    This function creates evaluation JSON files similar to evaluation_DeCo_Diff2.py.
    """
    # Group records by image path
    image_records = {}
    for record in records:
        image_path = record["image_path"][1]
        if image_path not in image_records:
            image_records[image_path] = []
        image_records[image_path].append(record)
    
    print(f"Saving JSON results for {len(image_records)} images...")
    
    for image_path, image_record_list in tqdm(image_records.items(), desc="Saving JSON results"):
        # Create patch analysis for this image
        patch_analysis = []
        
        for record in image_record_list:
            patch_x, patch_y = record["patch_coords"][1]
            grid_row = patch_y // patch_size
            grid_col = patch_x // patch_size
            
            patch_analysis.append({
                "grid_row": grid_row,
                "grid_col": grid_col,
                "anomaly_max": int(record["anomaly_max"][1]),  # Convert to int
                "anomaly_pixels": int(record["anomaly_pixels"][1]),  # Convert to int
                #"anomaly_pixels_raw": int(record["anomaly_pixels_raw"][1]),  # Convert to int
                "status": record["status"][1]
            })
        
        # Create evaluation result
        safe_name = path_to_safe_filename(image_path)
        result_filename = f"{safe_name}__evaluation.json"
        result_path = os.path.join(output_dir, result_filename)
        
        evaluation_result = {
            "image_path": image_path,
            "patch_analysis": patch_analysis,
            "grid_size": patch_size
        }
        
        with open(result_path, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
    
    print(f"JSON results saved to: {output_dir}")

def create_confusion_matrix_from_records(
    records: List[Record],
    output_dir: str,
    annotation_dir: str = None,
    patch_size: int = 128
) -> None:
    """
    Create confusion matrix visualization from records.
    This function creates a confusion matrix plot similar to evaluation_DeCo_Diff2.py.
    """
    if not records:
        print("No records provided for confusion matrix generation")
        return
    
    # Initialize confusion matrix counters
    all_TP = all_FP = all_FN = all_TN = 0
    
    # Group records by image path
    image_records = {}
    for record in records:
        image_path = record["image_path"][1]
        if image_path not in image_records:
            image_records[image_path] = []
        image_records[image_path].append(record)
    
    print(f"Creating confusion matrix for {len(image_records)} images...")
    
    for image_path, image_record_list in tqdm(image_records.items(), desc="Processing confusion matrix"):
        # Get predicted defective patches
        predicted = set()
        for record in image_record_list:
            patch_x, patch_y = record["patch_coords"][1]
            grid_row = patch_y // patch_size
            grid_col = patch_x // patch_size
            status = record["status"][1]
            
            # Add to predicted if TP or FP (predicted as defective)
            if status in ["TP", "FP"]:
                predicted.add((grid_row, grid_col))
        
        # Get ground truth defective patches
        gt = set()
        if annotation_dir:
            annotation_filename = f"{os.path.basename(image_path).replace('.png', '')}__annotations.json"
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            if os.path.exists(annotation_path):
                try:
                    with open(annotation_path, 'r') as f:
                        annotation = json.load(f)
                        gt = set(tuple(x) for x in annotation.get("defective_patches", []))
                except Exception as e:
                    print(f"Warning: Error reading annotation file {annotation_path}: {e}")
        
        # Calculate grid dimensions
        try:
            img = PILImage.open(image_path)
            h, w = img.height, img.width
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}")
            # Estimate grid size from patch coordinates
            max_x = max(record["patch_coords"][1][0] for record in image_record_list) + patch_size
            max_y = max(record["patch_coords"][1][1] for record in image_record_list) + patch_size
            h, w = max_y, max_x
        
        n_rows = (h + patch_size - 1) // patch_size  # Ceiling division
        n_cols = (w + patch_size - 1) // patch_size   # Ceiling division
        all_cells = set((r, c) for r in range(n_rows) for c in range(n_cols))
        
        # Count confusion matrix elements
        for cell in all_cells:
            pred = cell in predicted
            truth = cell in gt
            if pred and truth:
                all_TP += 1
            elif pred and not truth:
                all_FP += 1
            elif not pred and truth:
                all_FN += 1
            else:
                all_TN += 1
    
    total = all_TP + all_FP + all_FN + all_TN
    accuracy = (all_TP + all_TN) / total if total > 0 else 0
    
    # Calculate additional metrics
    precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0
    recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Confusion Matrix (patch-level):")
    print(f"TP: {all_TP}, FP: {all_FP}, FN: {all_FN}, TN: {all_TN}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Create confusion matrix visualization
    cm = np.array([[all_TP, all_FN], [all_FP, all_TN]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Patch-level)', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Set labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Defective', 'Normal'], fontsize=12)
    plt.yticks(tick_marks, ['Defective', 'Normal'], fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1_score:.4f}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save the confusion matrix plot
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    
    # Save detailed results to file
    result = {
        "TP": all_TP,
        "FP": all_FP,
        "FN": all_FN,
        "TN": all_TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "total_patches": total
    }
    with open(os.path.join(output_dir, "confusion_matrix.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Confusion matrix results saved to: {os.path.join(output_dir, 'confusion_matrix.json')}")

def parse_filename_to_info(filename: str) -> Dict[str, Any]:
    """
    Parse the filename to extract image path and patch coordinates.
    
    Expected format: <file_info>__<patch_info>__minimal_diff_<type>.npy
    where <patch_info> is x<x1>_y<y1>_x<x2>_y<y2>_x<x3>_y<y3>_x<x4>_y<y4>
    """
    # Remove the .npy extension and minimal_diff_<type> part
    base_name = filename.replace('.npy', '').replace('.png', '')
    
    # Find the last occurrence of '__minimal_diff_' to split
    parts = base_name.split('__minimal_diff_')
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}")
    
    file_patch_part = parts[0]
    data_type = parts[1]
    
    # Split file_patch_part by '__' to separate file info from patch info
    file_patch_parts = file_patch_part.split('__')
    if len(file_patch_parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # The last part contains patch coordinates
    patch_info = file_patch_parts[-1]
    file_info = '__'.join(file_patch_parts[:-1])
    
    # Parse patch coordinates: x<x1>_y<y1>_x<x2>_y<y2>_x<x3>_y<y3>_x<x4>_y<y4>
    coord_pattern = r'x(\d+)_y(\d+)_x(\d+)_y(\d+)_x(\d+)_y(\d+)_x(\d+)_y(\d+)'
    match = re.match(coord_pattern, patch_info)
    if not match:
        raise ValueError(f"Invalid patch coordinate format: {patch_info}")
    
    coords = [int(x) for x in match.groups()]
    x1, y1, x2, y2, x3, y3, x4, y4 = coords
    
    # Extract patch coordinates (top-left corner for compatibility)
    patch_x, patch_y = x1, y1
    
    # Store all 8 coordinate values
    patch_coords_8_values = coords
    
    # Reconstruct original file path
    # Replace double underscores with path separators, but handle file extensions properly
    file_path = file_info.replace('__', '/')
    
    # Fix file extensions: __png should become .png, __jpg should become .jpg, etc.
    file_path = file_path.replace('/png', '.png')
    file_path = file_path.replace('/jpg', '.jpg')
    file_path = file_path.replace('/jpeg', '.jpeg')
    file_path = file_path.replace('/tiff', '.tiff')
    file_path = file_path.replace('/bmp', '.bmp')
    
    # Fix Windows path: {drive}/Users/... should become {drive}:\Users\...
    if re.match(r'^[A-Z]/', file_path):
        # Extract the drive letter
        drive_letter = file_path[0]
        file_path = file_path.replace(f'{drive_letter}/', f'{drive_letter}:\\')
        file_path = file_path.replace('/', '\\')

    return {
        'file_path': file_path,
        'file_info': file_info, # Store the original file_info for reconstruction
        'patch_x': patch_x,
        'patch_y': patch_y,
        'patch_coords': patch_coords_8_values,  # Now returns all 8 values
        'data_type': data_type,
        'filename': filename
    }

def load_coordinates_directly(results_dir: str) -> Dict[str, List[int]]:
    """
    Load coordinates directly from _coords.npy files.
    
    Returns:
        Dict mapping base_filename to 8-value coordinates list
    """
    coords_files = glob.glob(os.path.join(results_dir, "**/*_coords.npy"), recursive=True)
    print(f"Found {len(coords_files)} coordinate files to process")
    coordinates = {}
    failed_files = 0
    
    for i, coords_file in enumerate(coords_files):
        try:
            # Build a stable key shared across all artifacts of the same patch
            base_key = _get_patch_base_key_from_filename(coords_file)
            
            # Load 8-value coordinates
            coords_raw = np.load(coords_file)
            
            # Only show debug info for first few files to avoid overwhelming output
            if i < 3:
                print(f"Debug: Loading coordinates from {coords_file}")
                print(f"  Raw coords shape: {coords_raw.shape}")
                print(f"  Raw coords dtype: {coords_raw.dtype}")
                print(f"  Raw coords: {coords_raw}")
            
            # Fix: Handle case where coordinates are saved as 2D arrays
            if len(coords_raw.shape) == 2:
                if i < 3:
                    print(f"  Warning: 2D coordinate array detected with shape {coords_raw.shape}")
                    print(f"  This suggests coordinates were saved incorrectly during the save phase")
                
                # If it's [8, N] shape, take the first N values
                # If it's [N, 8] shape, transpose and take the first N values
                if coords_raw.shape[0] == 8:
                    # Shape is [8, N] - take first value from each coordinate
                    coords_8_values = coords_raw[:, 0].tolist()
                    if i < 3:
                        print(f"  Fixed [8, N] coordinates to: {coords_8_values}")
                elif coords_raw.shape[1] == 8:
                    # Shape is [N, 8] - take first row
                    coords_8_values = coords_raw[0, :].tolist()
                    if i < 3:
                        print(f"  Fixed [N, 8] coordinates to: {coords_8_values}")
                else:
                    # Unexpected shape, try to extract 8 values
                    if i < 3:
                        print(f"  Unexpected 2D shape, attempting to extract 8 values...")
                    coords_8_values = coords_raw.flatten()[:8].tolist()
                    if i < 3:
                        print(f"  Extracted coordinates: {coords_8_values}")
            else:
                # Normal 1D array case
                coords_8_values = coords_raw.tolist()
            
            if i < 3:
                print(f"  Converted to list: {coords_8_values}")
                print(f"  List type: {type(coords_8_values)}")
                print(f"  List length: {len(coords_8_values)}")
                if len(coords_8_values) > 0:
                    print(f"  First element type: {type(coords_8_values[0])}")
                    print(f"  First element: {coords_8_values[0]}")
            
            # Fix: If coordinates are nested lists, flatten them
            if len(coords_8_values) == 8 and isinstance(coords_8_values[0], (list, tuple)):
                if i < 3:
                    print(f"  Warning: Nested coordinates detected, flattening...")
                # Extract first value from each coordinate list
                coords_8_values = [coord[0] if isinstance(coord, (list, tuple)) else coord for coord in coords_8_values]
                if i < 3:
                    print(f"  Flattened coordinates: {coords_8_values}")
            
            # Additional fix: Handle case where coordinates are 2D arrays (e.g., [8, 16] shape)
            # This happens when coordinates were saved incorrectly during the save phase
            if len(coords_8_values) == 8 and isinstance(coords_8_values[0], (list, tuple)):
                # Check if this is a 2D array case where each coordinate has multiple values
                first_coord_length = len(coords_8_values[0])
                if first_coord_length > 1:
                    if i < 3:
                        print(f"  Warning: 2D coordinate array detected with {first_coord_length} values per coordinate")
                        print(f"  This suggests coordinates were saved incorrectly during the save phase")
                    
                    # Take the first value from each coordinate (most common case)
                    # Alternative: could take mean, median, or other aggregation
                    coords_8_values = [coord[0] if isinstance(coord, (list, tuple)) else coord for coord in coords_8_values]
                    if i < 3:
                        print(f"  Fixed 2D coordinates to: {coords_8_values}")
            
            # Validate: Ensure all coordinates are integers
            try:
                coords_8_values = [int(coord) for coord in coords_8_values]
                if i < 3:
                    print(f"  Validated coordinates: {coords_8_values}")
            except (ValueError, TypeError) as e:
                print(f"  Error: Invalid coordinate values: {e}")
                print(f"  Raw coordinates: {coords_8_values}")
                continue  # Skip this file if coordinates are invalid
            
            # Final validation: Ensure we have exactly 8 coordinates
            if len(coords_8_values) != 8:
                print(f"  Error: Expected 8 coordinates, got {len(coords_8_values)}")
                print(f"  Raw coordinates: {coords_8_values}")
                continue  # Skip this file if coordinate count is wrong
            
            # Final validation: Ensure all coordinates are single integers (not lists)
            if any(isinstance(coord, (list, tuple)) for coord in coords_8_values):
                print(f"  Error: Coordinates still contain nested structures after processing")
                print(f"  Final coordinates: {coords_8_values}")
                continue  # Skip this file if coordinates are still nested
            
            coordinates[base_key] = coords_8_values
            
        except Exception as e:
            print(f"Warning: Could not load coordinates from {coords_file}: {e}")
            failed_files += 1
    
    print(f"Loaded coordinates for {len(coordinates)} patches from _coords.npy files")
    if failed_files > 0:
        print(f"Failed to load {failed_files} coordinate files")
    
    # Show summary of coordinate validation
    if coordinates:
        sample_coords = list(coordinates.values())[0]
        print(f"Sample coordinates: {sample_coords}")
        print(f"Coordinate format: {len(sample_coords)} values, all integers")
        
        # Additional validation
        all_valid = all(
            isinstance(coord, int) and not isinstance(coord, (list, tuple)) 
            for coord in sample_coords
        )
        print(f"Coordinate validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
        
        if not all_valid:
            print(f"  Invalid coordinate types found: {[type(coord) for coord in sample_coords]}")
            print(f"  This indicates a serious issue with coordinate loading")
    else:
        print("⚠️  No valid coordinates loaded!")
        print("   This will cause the process to fail. Check the coordinate files above.")
        print("   Possible causes:")
        print("   1. All coordinate files have invalid formats")
        print("   2. Coordinate files were corrupted during save")
        print("   3. File permissions or path issues")
    
    return coordinates


def load_raw_data_files(results_dir: str, visualize: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all raw data files and organize them by patch.
    
    Args:
        results_dir: Directory containing raw data files
        visualize: Enable visualization of diff images (default: False)
        
    Returns:
        Dict mapping patch_key to dict of data arrays
    """
    # Load coordinates directly from _coords.npy files first
    direct_coordinates = load_coordinates_directly(results_dir)
    
    # If no coordinates were loaded, warn the user
    if not direct_coordinates:
        print("⚠️  Warning: No coordinates loaded from _coords.npy files")
        print("   Will fall back to filename parsing for coordinates")
        print("   This may result in incomplete coordinate information")
    
    # Find all .npy files
    npy_files = glob.glob(os.path.join(results_dir, "**/*.npy"), recursive=True)
    
    # Filter for minimal_diff files
    minimal_diff_files = [f for f in npy_files if 'minimal_diff_' in f]
    
    print(f"Found {len(minimal_diff_files)} minimal_diff files")
    
    # Group files by patch
    patch_data = {}
    
    for file_path in minimal_diff_files:
        filename = os.path.basename(file_path)
        
        try:
            info = parse_filename_to_info(filename)
            patch_key = f"{info['file_path']}_{info['patch_x']}_{info['patch_y']}"
            
            # Load the data
            data = np.load(file_path)
            
            # Check if filename contains "108826_198" for debug and visualization
            should_debug = "108826_198" in filename
            should_visualize = visualize and "108826_198" in filename
            
            # Debug messages for files containing "108826_198"
            if should_debug:
                print(f"Processing file: {filename}")
                print(f"  Data shape: {data.shape}")
                print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"  Data mean: {data.mean():.3f}")
                print(f"  Data std: {data.std():.3f}")
            
            # Visualization for diff images
            if should_visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    
                    # Show the raw diff data
                    axes[0].imshow(data, cmap='gray')
                    axes[0].set_title(f'Raw Diff Data\nShape: {data.shape}')
                    axes[0].axis('off')
                    
                    # Show histogram of values
                    axes[1].hist(data.flatten(), bins=50, alpha=0.7)
                    axes[1].set_title(f'Value Distribution\nMin: {data.min():.3f}, Max: {data.max():.3f}')
                    axes[1].set_xlabel('Diff Value')
                    axes[1].set_ylabel('Frequency')
                    
                    plt.suptitle(f'Diff Image: {filename}')
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    
                except ImportError:
                    print("Warning: matplotlib not available for visualization")
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
            
            if patch_key not in patch_data:
                # Use a stable base key to look up coordinates loaded from _coords.npy
                base_key = _get_patch_base_key_from_filename(filename)

                if base_key in direct_coordinates:
                    coords_8_values = direct_coordinates[base_key]
                    #print(f"Debug: Using direct coords for {base_key}: {coords_8_values}")
                else:
                    coords_8_values = info['patch_coords']  # From filename parsing (8 values)
                    #print(f"Debug: Using filename coords for {base_key}: {coords_8_values}")

                # Extract the actual patch coordinates from the 8-value coordinates
                # Ensure we have proper x, y coordinates for the top-left corner
                if isinstance(coords_8_values, (list, tuple)) and len(coords_8_values) == 8:
                    x1, y1 = coords_8_values[0], coords_8_values[1]  # Top-left corner
                    # Ensure these are integers
                    if isinstance(x1, (list, tuple)):
                        x1 = x1[0] if len(x1) > 0 else 0
                    if isinstance(y1, (list, tuple)):
                        y1 = y1[0] if len(y1) > 0 else 0
                    actual_patch_x = int(x1)
                    actual_patch_y = int(y1)
                    
                    # Debug output for first few patches
                    if len(patch_data) < 3:
                        print(f"Debug: Extracted coordinates for {patch_key}")
                        print(f"  coords_8_values: {coords_8_values}")
                        print(f"  x1: {x1} (type: {type(x1)})")
                        print(f"  y1: {y1} (type: {type(y1)})")
                        print(f"  actual_patch_x: {actual_patch_x} (type: {type(actual_patch_x)})")
                        print(f"  actual_patch_y: {actual_patch_y} (type: {type(actual_patch_y)})")
                else:
                    # Fallback to filename-parsed coordinates
                    actual_patch_x = info['patch_x']
                    actual_patch_y = info['patch_y']
                    
                    # Debug output for fallback case
                    if len(patch_data) < 3:
                        print(f"Debug: Using fallback coordinates for {patch_key}")
                        print(f"  coords_8_values: {coords_8_values}")
                        print(f"  fallback patch_x: {actual_patch_x} (type: {type(actual_patch_x)})")
                        print(f"  fallback patch_y: {actual_patch_y} (type: {type(actual_patch_y)})")
                
                patch_data[patch_key] = {
                    'file_path': info['file_path'],
                    'file_path_original': info['file_info'],  # Store original format with __png
                    'patch_x': actual_patch_x,  # Use the properly extracted coordinates
                    'patch_y': actual_patch_y,  # Use the properly extracted coordinates
                    'patch_coords': coords_8_values  # Always 8 values
                }
            
            patch_data[patch_key][info['data_type']] = data
            
        except Exception as e:
            print(f"Warning: Could not process {filename}: {e}")
            continue
    
    # Verify that each patch has all three required data types
    complete_patches = {}
    for patch_key, data in patch_data.items():
        required_types = ['encodedrecon', 'latent', 'anomaly_map_arithmetic']
        if all(t in data for t in required_types):
            complete_patches[patch_key] = data
        else:
            missing = [t for t in required_types if t not in data]
            print(f"Warning: Patch {patch_key} missing data types: {missing}")
    
    print(f"Found {len(complete_patches)} complete patches")
    return complete_patches

def reconstruct_records_from_raw_data(
    patch_data: Dict[str, Dict[str, np.ndarray]],
    ground_truth_map: Dict[str, Set[Tuple[int, int]]] = None,
    original_images: Dict[str, np.ndarray] = None,
    anomaly_binary_threshold: int = 5,
    anomaly_pixel_num_threshold: int = 0,
    adaptive_threshold: float = 0.1,
    patch_size: int = 128,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> List[Record]:
    """
    Reconstruct evaluation records from raw data.
    """
    records = []
    
    for patch_key, data in tqdm(patch_data.items(), desc="Processing patches"):
        file_path = data['file_path']
        patch_coords_8_values = data['patch_coords']  # Now contains 8 values
        
        # Debug: Print coordinate information for first few patches
        if len(records) < 3:
            print(f"\nDebug: Processing patch {len(records)+1}")
            print(f"  File: {file_path}")
            print(f"  Coordinates type: {type(patch_coords_8_values)}")
            print(f"  Coordinates length: {len(patch_coords_8_values) if hasattr(patch_coords_8_values, '__len__') else 'N/A'}")
            print(f"  First coordinate: {patch_coords_8_values[0] if hasattr(patch_coords_8_values, '__len__') and len(patch_coords_8_values) > 0 else 'N/A'}")
            if hasattr(patch_coords_8_values, '__len__') and len(patch_coords_8_values) > 0:
                print(f"  First coordinate type: {type(patch_coords_8_values[0])}")
                if isinstance(patch_coords_8_values[0], (list, tuple)):
                    print(f"  First coordinate length: {len(patch_coords_8_values[0])}")
                    print(f"  First coordinate values: {patch_coords_8_values[0]}")
        
        # Additional debug: Check the data structure
        if len(records) < 3:
            print(f"  Data keys: {list(data.keys())}")
            print(f"  patch_x from data: {data.get('patch_x', 'MISSING')} (type: {type(data.get('patch_x', 'MISSING'))})")
            print(f"  patch_y from data: {data.get('patch_y', 'MISSING')} (type: {type(data.get('patch_y', 'MISSING'))})")
            print(f"  patch_coords from data: {data.get('patch_coords', 'MISSING')} (type: {type(data.get('patch_coords', 'MISSING'))})")
        
        # Extract top-left corner coordinates for compatibility
        # Handle both flat lists and nested lists (in case coordinates were saved incorrectly)
        if isinstance(patch_coords_8_values, (list, tuple)):
            if len(patch_coords_8_values) == 8:
                # Check if we have nested lists (each coordinate is a list)
                if isinstance(patch_coords_8_values[0], (list, tuple)):
                    # Nested list case: extract first value from each coordinate
                    x1, y1, x2, y2, x3, y3, x4, y4 = [coord[0] if isinstance(coord, (list, tuple)) else coord for coord in patch_coords_8_values]
                    print(f"Warning: Nested coordinate format detected, using first value from each coordinate")
                else:
                    # Flat list case: direct extraction
                    x1, y1, x2, y2, x3, y3, x4, y4 = patch_coords_8_values
                
                patch_x, patch_y = int(x1), int(y1)  # Top-left corner, ensure integers
                #print(f"Extracted coordinates: patch_x={patch_x}, patch_y={patch_y}")
            else:
                raise ValueError(f"Expected 8-value patch coordinates, got {len(patch_coords_8_values)} values: {patch_coords_8_values}")
        else:
            raise ValueError(f"Expected list/tuple of patch coordinates, got: {type(patch_coords_8_values)}")
        
        # Additional safety check: ensure patch_x and patch_y are integers
        if not isinstance(patch_x, int) or not isinstance(patch_y, int):
            print(f"Error: patch_x and patch_y must be integers, got patch_x={type(patch_x)}:{patch_x}, patch_y={type(patch_y)}:{patch_y}")
            print(f"Original coordinates: {patch_coords_8_values}")
            # Try to convert them to integers
            try:
                patch_x = int(patch_x)
                patch_y = int(patch_y)
                print(f"Converted to integers: patch_x={patch_x}, patch_y={patch_y}")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not convert coordinates to integers: {e}. Original: {patch_coords_8_values}")
        
        # Load the raw data arrays
        encodedrecon_raw = data['encodedrecon']
        latent_raw = data['latent']
        anomaly_map_arithmetic_raw = data['anomaly_map_arithmetic']
        
        # Convert to torch tensors for processing
        encodedrecon_tensor = torch.from_numpy(encodedrecon_raw).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        latent_tensor = torch.from_numpy(latent_raw).float().unsqueeze(0).unsqueeze(0)
        anomaly_map_arithmetic_tensor = torch.from_numpy(anomaly_map_arithmetic_raw).float().unsqueeze(0).unsqueeze(0)
        
        # Create binary masks
        #anomaly_map_arithmetic_binary = _binary_mask(anomaly_map_arithmetic_tensor, anomaly_binary_threshold)
        anomaly_map_arithmetic_binary = _binary_mask_exclude_boundary3(anomaly_map_arithmetic_tensor, anomaly_binary_threshold, visualize=True, debug=True, filename=file_path)
        #anomaly_map_arithmetic_binary_raw = _binary_mask(anomaly_map_arithmetic_tensor, anomaly_binary_threshold)
        
        # Debug and visualize anomaly_map_arithmetic_binary if file_path includes "108826_198"
        if file_path and "108826_198" in file_path:
            print(f"\n=== Debug: anomaly_map_arithmetic_binary for {file_path} ===")
            
            # Convert to numpy for analysis
            if anomaly_map_arithmetic_binary.dim() == 4:
                binary_np = anomaly_map_arithmetic_binary.squeeze(0).squeeze(0).cpu().numpy()
            elif anomaly_map_arithmetic_binary.dim() == 3:
                binary_np = anomaly_map_arithmetic_binary.squeeze(0).cpu().numpy()
            else:
                binary_np = anomaly_map_arithmetic_binary.cpu().numpy()
            
            # Print debug information
            print(f"Binary mask shape: {binary_np.shape}")
            print(f"Binary mask dtype: {binary_np.dtype}")
            print(f"Unique values: {np.unique(binary_np)}")
            print(f"Number of anomaly pixels: {np.sum(binary_np > 0)}")
            print(f"Total pixels: {binary_np.size}")
            print(f"Anomaly percentage: {np.sum(binary_np > 0) / binary_np.size * 100:.2f}%")
            
            # Visualize the binary mask
            try:
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original anomaly map (before binary conversion)
                if anomaly_map_arithmetic_tensor.dim() == 4:
                    original_np = anomaly_map_arithmetic_tensor.squeeze(0).squeeze(0).cpu().numpy()
                elif anomaly_map_arithmetic_tensor.dim() == 3:
                    original_np = anomaly_map_arithmetic_tensor.squeeze(0).cpu().numpy()
                else:
                    original_np = anomaly_map_arithmetic_tensor.cpu().numpy()
                
                # Original anomaly map
                im1 = axes[0].imshow(original_np, cmap='viridis')
                axes[0].set_title(f'Original Anomaly Map\nMin: {original_np.min():.3f}, Max: {original_np.max():.3f}')
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                
                # Binary mask
                im2 = axes[1].imshow(binary_np, cmap='gray')
                axes[1].set_title(f'Binary Mask\n({np.sum(binary_np > 0)} anomaly pixels)')
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.suptitle(f'Anomaly Map Analysis for {file_path}', y=1.02)
                
                # Save the visualization
                import os
                debug_dir = "debug_visualizations"
                os.makedirs(debug_dir, exist_ok=True)
                safe_filename = path_to_safe_filename(file_path)
                save_path = os.path.join(debug_dir, f"anomaly_binary_debug_{safe_filename}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Binary mask visualization saved to: {save_path}")
                
                plt.show()
                plt.close()
                
            except ImportError:
                print("Warning: matplotlib not available for visualization")
            except Exception as e:
                print(f"Warning: Visualization failed: {e}")
            
            print("=== End Debug ===\n")
        
        # Calculate metrics using tensor operations (more efficient)
        anomaly_max = int(round(anomaly_map_arithmetic_tensor.max().item() * 255))
        
        # Get the actual patch dimensions for consistent cropping
        original_image = original_images[file_path]
        h, w = original_image.shape[:2]
        actual_patch_height = min(patch_size, h - patch_y)
        actual_patch_width = min(patch_size, w - patch_x)
        
        # Crop the binary mask tensor to match the actual patch size for consistent counting
        # Shape: (1, 1, H, W) -> crop to (1, 1, actual_patch_height, actual_patch_width)
        anomaly_binary_cropped = anomaly_map_arithmetic_binary[:, :, :actual_patch_height, :actual_patch_width]
        
        # Calculate anomaly pixels using tensor operations (more efficient)
        anomaly_pixels = torch.sum(anomaly_binary_cropped).item()
        is_predicted_defective = anomaly_pixels > anomaly_pixel_num_threshold


        # Get ground truth defective patches for this image
        ground_truth_defective = ground_truth_map.get(file_path, set()) if ground_truth_map else set()

        # Convert pixel coordinates to grid coordinates for ground truth comparison
        grid_row = patch_y // patch_size
        grid_col = patch_x // patch_size
        
        # Determine status
        status = "TP" if is_predicted_defective and (grid_row, grid_col) in ground_truth_defective else \
                 "FP" if is_predicted_defective else \
                 "FN" if (grid_row, grid_col) in ground_truth_defective else "TN"

        original_patch = original_image[patch_y:patch_y + actual_patch_height, patch_x:patch_x + actual_patch_width]

        # Store the binary map with proper shape (ensure it's 2D)
        binary_map_numpy = _to_numpy(anomaly_binary_cropped)
        if len(binary_map_numpy.shape) == 4:
            binary_map_numpy = binary_map_numpy.squeeze()  # Remove batch and channel dims
        elif len(binary_map_numpy.shape) == 3:
            binary_map_numpy = binary_map_numpy.squeeze()  # Remove single channel dim
        
        # Create required record
        required_rec = make_record(
            split=("meta", "test"),  # Default to test split
            image_path=("meta", file_path),  # Reconstructed path for loading original images
            image_path_original=("meta", data['file_path_original']),  # Original format for output filenames
            anomaly_class=("meta", "all"),  # Default anomaly class
            patch_coords=("meta", patch_coords_8_values),
            anomaly_max=("meta", anomaly_max),
            anomaly_pixels=("meta", anomaly_pixels),
            #anomaly_pixels_raw=("meta", anomaly_pixels_raw),
            is_predicted_defective=("meta", is_predicted_defective),  # Add this field
            status=("meta", status),
            # Use original image data if available
            orig=("image", original_patch),
            dod_recon=("image", encodedrecon_raw),  # Use encodedrecon as placeholder
            encoded_recon=("image", encodedrecon_raw),
            anomaly_map_arithmetic=("image", anomaly_map_arithmetic_raw),
            anomaly_map_arithmetic_binary=("image", binary_map_numpy),  # Store the cropped version
            # Add geometric maps as placeholders (using arithmetic maps)
            anomaly_map_geometric=("image", anomaly_map_arithmetic_raw),
            anomaly_map_geometric_binary=("image", binary_map_numpy),  # Store the cropped version
            encoded=("image", latent_raw),  # Use latent as placeholder for encoded
        )
        
        # Add metric fields manually since we don't have proper RGB images for LPIPS
        # We'll set default values for metrics that require RGB images
        required_rec["lpips"] = ("metric", 0.0)  # Default value since we don't have RGB images
        required_rec["ssim"] = ("metric", 0.0)   # Default value since we don't have RGB images
        required_rec["mse"] = ("metric", 0.0)    # Default value since we don't have RGB images
        
        records.append(required_rec)
    
    return records

def compute_simple_metrics(records: List[Record]) -> Dict[str, float]:
    """
    Compute simple evaluation metrics from records.
    This function computes metrics based on the existing status assignments in the records,
    ensuring consistency with the patch categorization.
    """
    if not records:
        return {
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "best_threshold": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0
        }
    
    # Count confusion matrix elements based on existing status assignments
    tp = fp = tn = fn = 0
    
    for rec in records:
        status = rec["status"][1]
        if status == "TP":
            tp += 1
        elif status == "FP":
            fp += 1
        elif status == "TN":
            tn += 1
        elif status == "FN":
            fn += 1
    
    # Calculate metrics based on existing status assignments
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # For ROC AUC, we need to compute it using the anomaly scores
    y_true = []
    y_score = []
    
    for rec in records:
        # Determine if this patch is actually defective based on status
        is_defective = rec["status"][1] in ["TP", "FN"]  # True positive or false negative
        
        y_true.append(1 if is_defective else 0)
        
        # Use anomaly_pixels as the score
        y_score.append(float(rec["anomaly_pixels"][1]))  # Convert to float
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Compute ROC AUC
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.0
    
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "best_threshold": 0.0,  # Not applicable since we use existing status
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }

def process_raw_data_to_results(
    results_dir: str,
    output_dir: str = None,
    patch_size: int = 128,
    annotation_dir: str = None,
    anomaly_binary_threshold: int = 5,
    anomaly_pixel_num_threshold: int = 0,
    adaptive_threshold: float = 0.1,
    enable_excel_report: bool = False,
    enable_save_image_results: bool = False,
    enable_save_optional_image_results: bool = False,
    enable_save_whole_image_results: bool = False,
    enable_save_json_results: bool = False,
    enable_confusion_matrix: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> List[Record]:
    """
    Main function to process raw data files and generate evaluation results.
    """
    # Set default output directory inside results_dir with timestamp
    output_dir = os.path.join(results_dir, output_dir)
    
    print(f"Loading raw data from: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load ground truth map once
    ground_truth_map = load_ground_truth_map(annotation_dir)
    print(f"Loaded ground truth for {len(ground_truth_map)} images")
    
    # Load all raw data files
    patch_data = load_raw_data_files(results_dir, visualize=True)
    
    # Extract unique image paths from patch data
    image_paths = set(data['file_path'] for data in patch_data.values())
    
    # Load original images once
    original_images = load_original_images(image_paths)
    print(f"Loaded {len(original_images)} original images")
    
    if not patch_data:
        print("No complete patch data found!")
        return []
    
    print(f"Reconstructing records from {len(patch_data)} patches...")
    
    # Reconstruct records
    records = reconstruct_records_from_raw_data(
        patch_data,
        ground_truth_map=ground_truth_map,
        original_images=original_images,
        anomaly_binary_threshold=anomaly_binary_threshold,
        anomaly_pixel_num_threshold=anomaly_pixel_num_threshold,
        adaptive_threshold=adaptive_threshold,
        patch_size=patch_size,  # Use 128x128 patches
        device=device
    )
    
    print(f"Generated {len(records)} records")
    
    # Compute evaluation metrics
    if records:
        print("Computing evaluation metrics...")
        metrics = compute_simple_metrics(records)
        
        print("Evaluation Results:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        
        # Save results if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metrics to JSON
            metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_file}")

            # Create confusion matrix if enabled
            if enable_confusion_matrix:
                create_confusion_matrix_from_records(
                    records,
                    output_dir,
                    annotation_dir=annotation_dir,
                    patch_size=patch_size
                )
                        
            # Save JSON results if enabled
            if enable_save_json_results:
                save_json_results_from_raw_data(
                    records,
                    output_dir,
                    patch_size=patch_size
                )
            
            # Save image results if enabled
            if enable_save_image_results:
                save_image_results_from_raw_data(
                    records,
                    output_dir,
                    ground_truth_map=ground_truth_map,
                    original_images=original_images,
                    enable_save_optional_image_results=enable_save_optional_image_results,
                    enable_save_whole_image_results=enable_save_whole_image_results,
                    patch_size=patch_size
                )

            # Generate Excel report if enabled
            if enable_excel_report:
                excel_files = make_excel(
                    records,
                    image_size=patch_size,  # Assuming 128x128 patches
                    save_dir=output_dir,
                    save_filename="raw_data_evaluation_report"
                )
                print(f"Generated Excel report: {excel_files}")
    
    return records

def main():
    parser = argparse.ArgumentParser(description="Process raw data files to generate evaluation results")
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Directory containing the raw data files (defaults to JSON key name)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory name inside results-dir (optional, defaults to timestamp)")
    parser.add_argument("--annotation-dir", type=str, default=None,
                       help="Directory containing annotation files (optional)")
    parser.add_argument("--patch-size", type=int, default=128,
                       help="Patch size for image processing")
    parser.add_argument("--anomaly-binary-threshold", type=int, default=225,
                       help="Binary threshold for anomaly detection")
    parser.add_argument("--anomaly-pixel-num-threshold", type=int, default=0,
                       help="Pixel number threshold for anomaly detection")
    parser.add_argument("--adaptive-threshold", type=float, default=0.1,
                       help="Adaptive threshold for contour-based binary masks")
    parser.add_argument("--enable-excel-report", action="store_true",
                       help="Generate Excel report")
    parser.add_argument("--enable-save-image-results", action="store_true",
                       help="Save image results (marked images and anomaly maps)")
    parser.add_argument("--enable-save-optional-image-results", action="store_true",
                       help="Save optional image results")
    parser.add_argument("--enable-save-whole-image-results", action="store_true",
                       help="Save whole image results")
    parser.add_argument("--enable-save-json-results", action="store_true",
                       help="Save JSON results (evaluation files per image)")
    parser.add_argument("--enable-confusion-matrix", action="store_true",
                       help="Create confusion matrix visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for processing")
    parser.add_argument(
        "--input-json",
        type=str,
        help="Path to JSON file containing multiple test configurations"
    )
    
    args = parser.parse_args()
            
    # Handle input JSON if provided
    if args.input_json:
        import json
        with open(args.input_json, 'r') as f:
            test_configs = json.load(f)
            
        # Run processing for each test configuration
        for test_name, test_args in test_configs.items():
            print(f"\nRunning processing for {test_name}")
            print(test_args)
            
            # Update args with test configuration
            for key, value in test_args.items():
                # Convert key from kebab-case to snake_case
                key = key.replace('-', '_')
                if hasattr(args, key):
                    # Convert string values to appropriate types
                    if key in ['anomaly_binary_threshold', 'anomaly_pixel_num_threshold', 'patch_size']:
                        value = int(value)
                    elif key == 'adaptive_threshold':
                        value = float(value)
                    elif key in ['results_dir', 'output_dir', 'annotation_dir']:
                        value = os.path.expanduser(value)
                    elif key in ['enable_excel_report', 'enable_save_optional_image_results', 
                               'enable_save_image_results', 'enable_save_json_results', 
                               'enable_confusion_matrix', 'enable_save_whole_image_results']:
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    setattr(args, key, value)
    
    device = torch.device(args.device)
    # Set up results directory with timestamp
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"{current_time}"
    else:
        args.output_dir = f"{args.output_dir}_{current_time}"
    print(f"output_dir: {args.output_dir}")

    if args.input_json:
        # Save the current test_args (key-value pairs) into the results_dir as a JSON file
        config_save_path = os.path.join(args.results_dir, args.output_dir, "config.json")
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, "w") as config_file:
            json.dump(test_args, config_file, indent=2)
        
    # Process the data
    records = process_raw_data_to_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        annotation_dir=args.annotation_dir,
        patch_size=args.patch_size,
        anomaly_binary_threshold=args.anomaly_binary_threshold,
        anomaly_pixel_num_threshold=args.anomaly_pixel_num_threshold,
        adaptive_threshold=args.adaptive_threshold,
        enable_excel_report=args.enable_excel_report,
        enable_save_image_results=args.enable_save_image_results,
        enable_save_optional_image_results=args.enable_save_optional_image_results,
        enable_save_whole_image_results=args.enable_save_whole_image_results,
        enable_save_json_results=args.enable_save_json_results,
        enable_confusion_matrix=args.enable_confusion_matrix,
        device=device
    )
    
    print(f"Processing complete. Generated {len(records)} records.")

if __name__ == "__main__":
    main() 