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

# Import necessary functions from evaluation_DeCo_Diff2.py
from evaluation_DeCo_Diff2 import (
    make_record, _to_numpy, add_metric_fields, _binary_mask,
    _get_largest_connected_component_pixels, _create_contour_based_binary_mask_single,
    CheckpointManager, save_image_results_from_records, save_patch_results_from_records,
    determine_image_status, compute_y_true_y_score, compute_metrics_from_y_true_y_score,
    make_excel, plot_accuracy_results, save_perturbation_results, draw_patch_rectangles_on_image
)

# Type definitions
Kinded = Tuple[str, Any]  # (kind, value)
Record = OrderedDict[str, Kinded]

def _binary_mask_exclude_boundary(diff: torch.Tensor, threshold: int = 5) -> torch.Tensor:
    """
    Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude,
    but exclude pixels that are adjacent to the boundary of the image.
    
    Args:
        diff: Input tensor with shape (H, W) or (1, H, W)
        threshold: Threshold value for binary conversion (0-255, default: 5)
        
    Returns:
        Binary tensor with same shape as input where boundary-adjacent pixels are excluded
    """
    import cv2
    import numpy as np
    
    # Create initial binary mask
    binary_mask = _binary_mask(diff, threshold)
    
    # Convert to numpy for OpenCV operations
    if binary_mask.dim() == 4 and binary_mask.shape[0] == 1 and binary_mask.shape[1] == 1:
        binary_np = binary_mask.squeeze(0).squeeze(0).cpu().numpy()
    elif binary_mask.dim() == 3 and binary_mask.shape[0] == 1:
        binary_np = binary_mask.squeeze(0).cpu().numpy()
    else:
        binary_np = binary_mask.cpu().numpy()
    
    # Ensure binary values (0 or 1)
    binary_np = (binary_np > 0).astype(np.uint8)
    
    # If the mask is all zeros, return as is
    if np.all(binary_np == 0):
        return binary_mask
    
    # Ensure we have a 2D array
    if binary_np.ndim != 2:
        print(f"Warning: Expected 2D array, got shape {binary_np.shape}")
        return binary_mask
    
    # Create a mask for boundary pixels
    h, w = binary_np.shape
    boundary_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Mark boundary pixels (first and last row/column)
    boundary_mask[0, :] = 1      # Top row
    boundary_mask[-1, :] = 1     # Bottom row
    boundary_mask[:, 0] = 1      # Left column
    boundary_mask[:, -1] = 1     # Right column
    
    # Find contours of the binary mask
    contours, _ = cv2.findContours(binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for pixels adjacent to boundary
    adjacent_to_boundary = np.zeros_like(binary_np)
    
    for contour in contours:
        # Create a mask for this contour
        contour_mask = np.zeros_like(binary_np)
        cv2.fillPoly(contour_mask, [contour], (1,))
        
        # Check if this contour touches the boundary
        if np.any(contour_mask * boundary_mask):
            # This contour touches the boundary, mark all pixels in this contour as adjacent to boundary
            adjacent_to_boundary = np.logical_or(adjacent_to_boundary, contour_mask)
    
    # Remove pixels adjacent to boundary from the original binary mask
    result_np = binary_np * (1 - adjacent_to_boundary)
    
    # Convert back to tensor
    result_tensor = torch.from_numpy(result_np).float()
    
    # Move to the same device as input
    if diff.is_cuda:
        result_tensor = result_tensor.cuda()
    
    return result_tensor

def path_to_safe_filename(path: str) -> str:
    """Convert a file path to a safe filename by replacing invalid characters."""
    return path.replace('/', '__').replace('\\', '__').replace(':', '__')

def save_image_results_from_raw_data(
    records: List[Record],
    output_dir: str,
    enable_save_optional_image_results: bool = False,
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
    
    # Group records by image path for image-level processing
    image_records = {}
    for record in records:
        image_path = record["image_path"][1]
        if image_path not in image_records:
            image_records[image_path] = []
        image_records[image_path].append(record)
    
    # Process each patch individually for patch-level images
    patch_idx = 0
    for record in tqdm(records, desc="Saving patch-level images"):
        # Get patch information
        patch_x, patch_y = record["patch_coords"][1]
        status = record["status"][1]
        image_path = record["image_path"][1]
        
        # Create patch-level image
        patch_data = record["orig"][1]  # encodedrecon data
        
        # Convert to RGB image
        if len(patch_data.shape) == 2:
            patch_image = np.stack([patch_data] * 3, axis=-1)
        else:
            patch_image = patch_data
        
        # Convert to uint8 range [0, 255]
        patch_image = (patch_image * 255).astype(np.uint8)
        
        # Ensure it's RGB
        if patch_image.shape[-1] == 1:
            patch_image = np.repeat(patch_image, 3, axis=-1)
        
        # Create unique name for this patch
        if "image_path_original" in record:
            file_info_original = record["image_path_original"][1]
        else:
            file_info_original = record["image_path"][1]
        
        patch_name = f"{patch_idx:04d}_{file_info_original}_x{patch_x}_y{patch_y}"
        
        # Save patch-level image in status folder
        marked_path = os.path.join(status_folders[status], f"{patch_name}__marked.png")
        PILImage.fromarray(patch_image).save(marked_path)
        
        patch_idx += 1
    
    # Process image-level images (full image reconstructions)
    for image_idx, (image_path, image_record_list) in enumerate(tqdm(image_records.items(), desc="Saving image-level images")):
        # Determine image status
        image_status = determine_image_status(image_record_list)
        
        # Build predicted defective set and ground truth defective set
        predicted_defective_set = set()
        ground_truth_defective = set()
        
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
            
            # Add to ground truth defective set if TP or FN
            if status in ["TP", "FN"]:
                ground_truth_defective.add((grid_row, grid_col))
        
        # Calculate overlapping regions
        overlapping = predicted_defective_set.intersection(ground_truth_defective)
        
        # Create image-level reconstruction (full image, not patch-level)
        # Estimate full image dimensions from all patch coordinates
        max_x = max(record["patch_coords"][1][0] for record in image_record_list) + patch_size
        max_y = max(record["patch_coords"][1][1] for record in image_record_list) + patch_size
        
        # Create full image reconstruction
        full_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        
        # Fill the full image with patch data
        for record in image_record_list:
            patch_x, patch_y = record["patch_coords"][1]
            patch_data = record["orig"][1]  # encodedrecon data
            
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
            base_name = f"{image_idx:04d}_{file_info_original}"
        else:
            base_name = f"{image_idx:04d}_unknown_image"
        
        # Save in image_level directory
        image_level_path = os.path.join(image_level_dir, f"{base_name}__marked.png")
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
            
            # Ensure patch data fits in the map
            patch_height = min(patch_size, max_y - patch_y)
            patch_width = min(patch_size, max_x - patch_x)
            
            if patch_height > 0 and patch_width > 0:
                anomaly_maps['arithmetic'][patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = \
                    patch_arithmetic[:patch_height, :patch_width]
                anomaly_maps['arithmetic_binary'][patch_y:patch_y + patch_height, patch_x:patch_x + patch_width] = \
                    patch_arithmetic_binary[:patch_height, :patch_width]
        
        # Save anomaly maps
        for map_name, map_data in anomaly_maps.items():
            # Convert to uint8 for saving
            if map_name.endswith('_binary'):
                # Binary maps: 0 or 255 (proper binary)
                map_img = (map_data > 0).astype(np.uint8) * 255
            else:
                # Continuous maps: use cv2.applyColorMap with COLORMAP_HOT (matching evaluation_DeCo_Diff2.py)
                # Convert [0, 1] range to [0, 255] range for colormap
                anomaly_map_uint8 = (map_data * 255).astype(np.uint8)
                # Apply HOT colormap to the map (returns BGR)
                anomaly_colored_bgr = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_HOT)
                # Convert BGR to RGB for proper display
                map_img = cv2.cvtColor(anomaly_colored_bgr, cv2.COLOR_BGR2RGB)
            
            map_path = os.path.join(anomaly_maps_dir, f"{base_name}__{map_name}.png")
            PILImage.fromarray(map_img).save(map_path)
            
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
            
            # Create overlay using the same method as evaluation_DeCo_Diff2.py
            if map_name.endswith('_binary'):
                # Binary overlay using pure red
                overlay = original_image.copy()
                mask = map_data > 0
                # Create pure red overlay for anomaly regions
                h, w = map_data.shape
                anomaly_colored = np.zeros((h, w, 3), dtype=np.uint8)
                # Set red channel for anomaly regions (pure red)
                anomaly_colored[mask, 0] = 255  # Red channel (RGB format)
                # Resize anomaly map to match original image size
                original_h, original_w = original_image.shape[:2]
                anomaly_colored_resized = cv2.resize(anomaly_colored, (original_w, original_h))
                # Apply the red anomaly regions to the overlay
                mask_resized = cv2.resize(mask.astype(np.uint8), (original_w, original_h)) > 0
                overlay[mask_resized] = anomaly_colored_resized[mask_resized]
            else:
                # Continuous overlay using HOT colormap
                # Convert [0, 1] range to [0, 255] range for colormap
                anomaly_map_uint8 = (map_data * 255).astype(np.uint8)
                # Apply HOT colormap to the map (returns BGR)
                anomaly_colored_bgr = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_HOT)
                # Convert BGR to RGB for proper overlay
                anomaly_colored = cv2.cvtColor(anomaly_colored_bgr, cv2.COLOR_BGR2RGB)
                # Resize anomaly map to match original image size
                original_h, original_w = original_image.shape[:2]
                anomaly_colored_resized = cv2.resize(anomaly_colored, (original_w, original_h))
                # Create overlay using alpha blending
                overlay = cv2.addWeighted(original_image, 0.2, anomaly_colored_resized, 0.8, 0)
            
            # Save overlay image
            overlay_path = os.path.join(anomaly_maps_dir, f"{base_name}__ao_{map_name}.png")
            PILImage.fromarray(overlay.astype(np.uint8)).save(overlay_path)
            
            # Create marked overlay (overlay with patch rectangles)
            marked_overlay = draw_patch_rectangles_on_image(
                overlay, predicted_defective_set, ground_truth_defective, overlapping, 
                patch_size=patch_size, grid_thickness=1
            )
            
            # Save marked overlay image
            marked_overlay_path = os.path.join(anomaly_maps_dir, f"{base_name}__mo_{map_name}.png")
            PILImage.fromarray(marked_overlay).save(marked_overlay_path)
    
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
                "anomaly_max": record["anomaly_max"][1],
                "anomaly_pixels": record["anomaly_pixels"][1],
                "anomaly_pixels_raw": record["anomaly_pixels_raw"][1],
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
    
    # Extract patch coordinates (top-left corner)
    patch_x, patch_y = x1, y1
    
    # Reconstruct original file path
    # Replace double underscores with path separators, but handle file extensions properly
    file_path = file_info.replace('__', '/')
    
    # Fix file extensions: __png should become .png, __jpg should become .jpg, etc.
    file_path = file_path.replace('/png', '.png')
    file_path = file_path.replace('/jpg', '.jpg')
    file_path = file_path.replace('/jpeg', '.jpeg')
    file_path = file_path.replace('/tiff', '.tiff')
    file_path = file_path.replace('/bmp', '.bmp')
    
    # Fix Windows path: C/Users/... should become C:\Users\...
    if file_path.startswith('C/'):
        file_path = file_path.replace('C/', 'C:\\')
        file_path = file_path.replace('/', '\\')
    
    return {
        'file_path': file_path,
        'file_info': file_info, # Store the original file_info for reconstruction
        'patch_x': patch_x,
        'patch_y': patch_y,
        'patch_coords': (patch_x, patch_y),
        'data_type': data_type,
        'filename': filename
    }

def load_raw_data_files(results_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all raw data files and organize them by patch.
    
    Returns:
        Dict mapping patch_key to dict of data arrays
    """
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
            
            if patch_key not in patch_data:
                patch_data[patch_key] = {
                    'file_path': info['file_path'],
                    'file_path_original': info['file_info'],  # Store original format with __png
                    'patch_x': info['patch_x'],
                    'patch_y': info['patch_y'],
                    'patch_coords': info['patch_coords']
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
    annotation_dir: str = None,
    anomaly_binary_threshold: int = 5,
    anomaly_pixel_num_threshold: int = 0,
    adaptive_threshold: float = 0.1,
    device: torch.device = torch.device("cpu")
) -> List[Record]:
    """
    Reconstruct evaluation records from raw data.
    """
    records = []
    
    for patch_key, data in tqdm(patch_data.items(), desc="Processing patches"):
        file_path = data['file_path']
        patch_x, patch_y = data['patch_coords']
        
        # Load the raw data arrays
        encodedrecon_raw = data['encodedrecon']
        latent_raw = data['latent']
        anomaly_map_arithmetic_raw = data['anomaly_map_arithmetic']
        
        # Convert to torch tensors for processing
        encodedrecon_tensor = torch.from_numpy(encodedrecon_raw).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        latent_tensor = torch.from_numpy(latent_raw).float().unsqueeze(0).unsqueeze(0)
        anomaly_map_arithmetic_tensor = torch.from_numpy(anomaly_map_arithmetic_raw).float().unsqueeze(0).unsqueeze(0)
        
        # Create binary masks
        anomaly_map_arithmetic_binary = _binary_mask_exclude_boundary(anomaly_map_arithmetic_tensor, anomaly_binary_threshold)
        anomaly_map_arithmetic_binary_raw = _binary_mask(anomaly_map_arithmetic_tensor, anomaly_binary_threshold)
        
        # Calculate metrics
        anomaly_max = int(round(anomaly_map_arithmetic_raw.max() * 255))
        anomaly_pixels = torch.sum(anomaly_map_arithmetic_binary).item()
        anomaly_pixels_raw = torch.sum(anomaly_map_arithmetic_binary_raw).item()
        is_predicted_defective = anomaly_pixels > anomaly_pixel_num_threshold
        
        # Load ground truth if annotation directory is provided
        ground_truth_defective = set()
        if annotation_dir:
            annotation_filename = f"{os.path.basename(file_path).replace('.png', '')}__annotations.json"
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)
                    ground_truth_defective = set(tuple(x) for x in annotation.get("defective_patches", []))
        
        # Determine status
        status = "TP" if is_predicted_defective and (patch_x, patch_y) in ground_truth_defective else \
                 "FP" if is_predicted_defective else \
                 "FN" if (patch_x, patch_y) in ground_truth_defective else "TN"
        
        # Create required record
        required_rec = make_record(
            split=("meta", "test"),  # Default to test split
            image_path=("meta", file_path),  # Reconstructed path for loading original images
            image_path_original=("meta", data['file_path_original']),  # Original format for output filenames
            anomaly_class=("meta", "all"),  # Default anomaly class
            patch_coords=("meta", (patch_x, patch_y)),
            anomaly_max=("meta", anomaly_max),
            anomaly_pixels=("meta", anomaly_pixels),
            anomaly_pixels_raw=("meta", anomaly_pixels_raw),
            is_predicted_defective=("meta", is_predicted_defective),  # Add this field
            status=("meta", status),
            # Note: We don't have the original images, so we'll use the raw data as placeholders
            orig=("image", encodedrecon_raw),  # Use encodedrecon as placeholder for original
            dod_recon=("image", encodedrecon_raw),  # Use encodedrecon as placeholder
            encoded_recon=("image", encodedrecon_raw),
            anomaly_map_arithmetic=("image", anomaly_map_arithmetic_raw),
            anomaly_map_arithmetic_binary=("image", _to_numpy(anomaly_map_arithmetic_binary)),
            # Add geometric maps as placeholders (using arithmetic maps)
            anomaly_map_geometric=("image", anomaly_map_arithmetic_raw),
            anomaly_map_geometric_binary=("image", _to_numpy(anomaly_map_arithmetic_binary)),
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
    """
    y_true = []
    y_score = []
    
    for rec in records:
        # Determine if this patch is actually defective based on status
        is_defective = rec["status"][1] in ["TP", "FN"]  # True positive or false negative
        
        y_true.append(1 if is_defective else 0)
        
        # Use anomaly_pixels as the score
        y_score.append(rec["anomaly_pixels"][1])
    
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Compute basic metrics
    if len(y_true) > 0:
        # Find optimal threshold using ROC curve
        from sklearn.metrics import roc_curve, auc, accuracy_score
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Find threshold that maximizes accuracy
        accuracies = []
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred))
        
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        
        # Compute confusion matrix
        y_pred = (y_score >= best_threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": best_accuracy,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "best_threshold": best_threshold,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        }
    else:
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

def process_raw_data_to_results(
    results_dir: str,
    output_dir: str = None,
    annotation_dir: str = None,
    anomaly_binary_threshold: int = 5,
    anomaly_pixel_num_threshold: int = 0,
    adaptive_threshold: float = 0.1,
    enable_excel_report: bool = False,
    enable_save_optional_image_results: bool = False,
    enable_save_image_results: bool = False,
    enable_save_json_results: bool = False,
    device: torch.device = torch.device("cpu")
) -> List[Record]:
    """
    Main function to process raw data files and generate evaluation results.
    """
    print(f"Loading raw data from: {results_dir}")
    
    # Load all raw data files
    patch_data = load_raw_data_files(results_dir)
    
    if not patch_data:
        print("No complete patch data found!")
        return []
    
    print(f"Reconstructing records from {len(patch_data)} patches...")
    
    # Reconstruct records
    records = reconstruct_records_from_raw_data(
        patch_data,
        annotation_dir=annotation_dir,
        anomaly_binary_threshold=anomaly_binary_threshold,
        anomaly_pixel_num_threshold=anomaly_pixel_num_threshold,
        adaptive_threshold=adaptive_threshold,
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
            
            # Generate Excel report if enabled
            if enable_excel_report:
                excel_files = make_excel(
                    records,
                    image_size=128,  # Assuming 128x128 patches
                    save_dir=output_dir,
                    save_filename="raw_data_evaluation_report"
                )
                print(f"Generated Excel report: {excel_files}")
            
            # Save image results if enabled
            if enable_save_image_results:
                save_image_results_from_raw_data(
                    records,
                    output_dir,
                    enable_save_optional_image_results=enable_save_optional_image_results,
                    patch_size=128
                )
            
            # Save JSON results if enabled
            if enable_save_json_results:
                save_json_results_from_raw_data(
                    records,
                    output_dir,
                    patch_size=128
                )
    
    return records

def main():
    parser = argparse.ArgumentParser(description="Process raw data files to generate evaluation results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing the raw data files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (optional)")
    parser.add_argument("--annotation_dir", type=str, default=None,
                       help="Directory containing annotation files (optional)")
    parser.add_argument("--anomaly_binary_threshold", type=int, default=5,
                       help="Binary threshold for anomaly detection")
    parser.add_argument("--anomaly_pixel_num_threshold", type=int, default=0,
                       help="Pixel number threshold for anomaly detection")
    parser.add_argument("--adaptive_threshold", type=float, default=0.1,
                       help="Adaptive threshold for contour-based binary masks")
    parser.add_argument("--enable_excel_report", action="store_true",
                       help="Generate Excel report")
    parser.add_argument("--enable_save_optional_image_results", action="store_true",
                       help="Save optional image results")
    parser.add_argument("--enable_save_image_results", action="store_true",
                       help="Save image results (marked images and anomaly maps)")
    parser.add_argument("--enable_save_json_results", action="store_true",
                       help="Save JSON results (evaluation files per image)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use for processing")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Process the data
    records = process_raw_data_to_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        annotation_dir=args.annotation_dir,
        anomaly_binary_threshold=args.anomaly_binary_threshold,
        anomaly_pixel_num_threshold=args.anomaly_pixel_num_threshold,
        adaptive_threshold=args.adaptive_threshold,
        enable_excel_report=args.enable_excel_report,
        enable_save_optional_image_results=args.enable_save_optional_image_results,
        enable_save_image_results=args.enable_save_image_results,
        enable_save_json_results=args.enable_save_json_results,
        device=device
    )
    
    print(f"Processing complete. Generated {len(records)} records.")

if __name__ == "__main__":
    main() 