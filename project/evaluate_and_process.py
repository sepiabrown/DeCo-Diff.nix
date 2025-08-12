#!/usr/bin/env python3
"""
Combined Evaluation and Processing Script

This script combines the functionality of evaluation_DeCo_Diff_raw.py and process_raw_data_to_results.py
to provide 4 different execution modes:

1. save_only: Save .npy files and diff images only (no categorization)
2. process_only: Read existing .npy files and generate categorization results
3. save_and_process: Save .npy files and immediately process them for categorization
4. full_pipeline: Complete pipeline without saving intermediates (evaluation to categorization)

Usage examples:
  # Mode 1: Save only
  python evaluate_and_process.py --mode save_only --annotation-dir path/to/annotations --pretrained path/to/model.pt

  # Mode 2: Process only  
  python evaluate_and_process.py --mode process_only --annotation-dir path/to/annotations
  # Or with explicit results directory:
  python evaluate_and_process.py --mode process_only --results-dir path/to/results --annotation-dir path/to/annotations

  # Mode 3: Save and process
  python evaluate_and_process.py --mode save_and_process --annotation-dir path/to/annotations --pretrained path/to/model.pt

  # Mode 4: Full pipeline
  python evaluate_and_process.py --mode full_pipeline --annotation-dir path/to/annotations --pretrained path/to/model.pt

  # Using JSON configuration only (mode specified in JSON)
  python evaluate_and_process.py --input-json config.json

  # Using JSON configuration with mode override
  python evaluate_and_process.py --mode save_only --input-json config.json

  # Enable debug logging for troubleshooting
  python evaluate_and_process.py --mode save_only --annotation-dir path/to/annotations --pretrained model.pt --debug
"""

from __future__ import annotations
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import warnings
warnings.filterwarnings(
    "ignore",
    message="A new version of Albumentations is available.*",
    category=UserWarning
)

import argparse
import json
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
import re
import platform

# Import from evaluation_DeCo_Diff_raw.py
from diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from models import UNET_models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys

# Import from process_raw_data_to_results.py
from evaluation_DeCo_Diff2 import (
    make_record, _to_numpy, add_metric_fields, _binary_mask,
    _get_largest_connected_component_pixels, _create_contour_based_binary_mask_single,
    CheckpointManager, save_image_results_from_records, save_patch_results_from_records,
    determine_image_status, compute_y_true_y_score, compute_metrics_from_y_true_y_score,
    make_excel, plot_accuracy_results, save_perturbation_results, draw_patch_rectangles_on_image,
    EvaluationMetrics
)

# Import utility functions
from utils import path_to_safe_filename, safe_filename_to_path

# Import from process_raw_data_to_results.py
from process_raw_data_to_results import (
    load_ground_truth_map, load_raw_data_files, load_original_images,
    reconstruct_records_from_raw_data
)

# Set up device
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("GPU not found. Using CPU instead.")

# Constants
_LATENT_SCALE = 0.18215

# Type definitions
Kinded = Tuple[str, Any]  # (kind, value)
Record = OrderedDict[str, Kinded]

# Global debug flag (will be set from args)
DEBUG_ENABLED = False

def debug_print(*args, **kwargs):
    """Print debug messages only if debug mode is enabled."""
    if DEBUG_ENABLED:
        print(*args, **kwargs)

class AnnotatedImageDataset(Dataset):
    """Dataset for images with JSON annotations for defective regions."""
    
    def __init__(
        self,
        annotation_dir: str,
        patch_size: int = 128,
        transform=None,
        object_class: str = "pcb",
    ):
        self.annotation_dir = annotation_dir
        self.patch_size = patch_size
        self.transform = transform
        self.object_class = object_class
        
        # Load all annotation files
        self.annotation_files = []
        for root, dirs, files in os.walk(annotation_dir):
            for file in files:
                if file.endswith('.json'):
                    self.annotation_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.annotation_files)} annotation files")
        
    def __len__(self):
        return len(self.annotation_files)
        
    def __getitem__(self, index):
        annotation_file = self.annotation_files[index]
        
        # Load annotation data
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        
        image_path = annotation_data.get('image_path', '')

        # Load image
        if os.path.exists(image_path):
            image = PILImage.open(image_path).convert('RGB')
        else:
            # Try to construct path from annotation file name
            base_name = os.path.splitext(os.path.basename(annotation_file))[0]
            # Remove the '_annotations' suffix if present
            if base_name.endswith('_annotations'):
                base_name = base_name[:-12]
            # Try different extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(os.path.dirname(image_path), base_name + ext)
                if os.path.exists(potential_path):
                    image = PILImage.open(potential_path).convert('RGB')
                    break
            else:
                raise FileNotFoundError(f"Image not found for annotation: {annotation_file}")
        
        # Convert to numpy for patch extraction
        image_np = np.array(image)
        
        # Pad image to ensure dimensions are divisible by patch_size
        image_np = self._pad_image_to_patch_size(image_np)
        
        # Extract patches
        patches, coords = self._extract_patches(image_np)
        
        # Return all patches from the image
        if patches:
            # Convert all patches to tensors
            patch_tensors = []
            for patch in patches:
                patch_tensor = self.transform(PILImage.fromarray(patch))
                patch_tensors.append(patch_tensor)
            
            # Stack all patches into a single tensor
            x = torch.stack(patch_tensors)
            
            # Create dummy segmentation for all patches (all zeros for now)
            seg = torch.zeros(len(patches), 1, patches[0].shape[0], patches[0].shape[1])
            
            # Create dummy object class tensor for all patches
            object_cls = torch.zeros(len(patches), dtype=torch.long)  # Assuming single class
            
            # Create dummy anomaly class for all patches
            anomaly_classes = ["all"] * len(patches)
            
            # Create image paths for all patches
            image_paths = [image_path] * len(patches)
            
            return x, seg, object_cls, image_paths, anomaly_classes, coords
        else:
            raise ValueError(f"No patches extracted from image: {image_path}")
    
    def _pad_image_to_patch_size(self, img):
        """
        Pad image to ensure dimensions are exactly divisible by patch_size.
        This eliminates the need for overlapping edge patches.
        
        Args:
            img: numpy array of shape (height, width, channels)
            
        Returns:
            padded_img: numpy array with dimensions divisible by patch_size
        """
        height, width = img.shape[:2]
        
        # Calculate required padding
        pad_height = (self.patch_size - (height % self.patch_size)) % self.patch_size
        pad_width = (self.patch_size - (width % self.patch_size)) % self.patch_size
        
        if pad_height == 0 and pad_width == 0:
            # No padding needed
            debug_print(f"  📐 No padding needed. Image dimensions: {height}x{width}")
            return img
        
        # Calculate padding for each side (distribute evenly, with extra on bottom/right if odd)
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # Apply padding with reflection or edge mode to maintain image characteristics
        if len(img.shape) == 3:
            # Color image
            padded_img = np.pad(img, 
                              ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='reflect')
        else:
            # Grayscale image
            padded_img = np.pad(img, 
                              ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='reflect')
        
        new_height, new_width = padded_img.shape[:2]
        debug_print(f"  📐 Padded image from {height}x{width} to {new_height}x{new_width}")
        debug_print(f"  📐 Padding applied: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        
        return padded_img
    
    def _extract_patches(self, img):
        """Extract patches from padded image - simple non-overlapping grid."""
        patches = []
        coords = []
        
        height, width = img.shape[:2]
        stride = self.patch_size  # No overlap - patches are adjacent
        debug_print(f"  📐 Padded image dimensions: {height}x{width}, patch size: {self.patch_size}")
        
        # Verify image dimensions are divisible by patch_size
        assert height % self.patch_size == 0, f"Height {height} not divisible by patch_size {self.patch_size}"
        assert width % self.patch_size == 0, f"Width {width} not divisible by patch_size {self.patch_size}"
        
        # Simple grid extraction - no edge handling needed
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract patch (guaranteed to be exactly patch_size x patch_size)
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                
                # Calculate all 4 corner coordinates for the patch (8 values)
                x1, y1 = x, y  # Top-left
                x2, y2 = x + self.patch_size, y  # Top-right
                x3, y3 = x + self.patch_size, y + self.patch_size  # Bottom-right
                x4, y4 = x, y + self.patch_size  # Bottom-left
                coords_8_values = (x1, y1, x2, y2, x3, y3, x4, y4)
                
                patches.append(patch)
                coords.append(coords_8_values)
        
        debug_print(f"  ✅ Extracted {len(patches)} non-overlapping patches from padded image")
        return patches, coords

def _compute_abs_diff_mean(a: torch.Tensor, b: torch.Tensor, diff_scale: float = 1.0) -> torch.Tensor:
    return torch.abs(a - b).mean(dim=1, keepdim=True) * diff_scale

def _compute_abs_diff_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b).max(dim=1, keepdim=True)[0]

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized batches.
    Each item in the batch is a tuple of (x, seg, object_cls, image_paths, anomaly_classes, patch_coords)
    where x has shape [num_patches, channels, height, width] and num_patches varies between images.
    """
    # Separate the components
    x_list, seg_list, object_cls_list, image_paths_list, anomaly_classes_list, patch_coords_list = zip(*batch)
    
    # For variable-sized tensors, we need to handle them differently
    # Instead of stacking, we'll concatenate along a new dimension
    # But first, let's flatten all patches from all images into a single list
    
    all_x = []
    all_seg = []
    all_object_cls = []
    all_image_paths = []
    all_anomaly_classes = []
    all_patch_coords = []
    
    for i, (x, seg, object_cls, image_paths, anomaly_classes, patch_coords) in enumerate(batch):
        # Add all patches from this image
        all_x.append(x)
        all_seg.append(seg)
        all_object_cls.append(object_cls)
        all_image_paths.extend(image_paths)
        all_anomaly_classes.extend(anomaly_classes)
        all_patch_coords.extend(patch_coords)
    
    # Concatenate all tensors along the first dimension
    x_combined = torch.cat(all_x, dim=0)
    seg_combined = torch.cat(all_seg, dim=0)
    object_cls_combined = torch.cat(all_object_cls, dim=0)
    
    return x_combined, seg_combined, object_cls_combined, all_image_paths, all_anomaly_classes, all_patch_coords

def process_patches_in_chunks(patches_tensor, chunk_size=32):
    """
    Process patches in smaller chunks to avoid memory issues.
    """
    total_patches = patches_tensor.size(0)
    results = []
    
    for start_idx in range(0, total_patches, chunk_size):
        end_idx = min(start_idx + chunk_size, total_patches)
        chunk = patches_tensor[start_idx:end_idx]
        results.append(chunk)
    
    return results

def _process_batch_inference(x_chunk, object_cls_chunk, model, vae, diffusion, reverse_steps, device, epoch_metrics=None):
    """
    Shared inference logic for processing a batch chunk.
    Returns the computed difference tensors.
    """
    debug_print(f"   🔄 Moving {x_chunk.size(0)} patches to device: {device}")
    # Move chunk to device
    x_device = x_chunk.to(device)
    object_cls_device = object_cls_chunk.to(device)
    
    debug_print(f"   🎨 VAE encoding...")
    # Forward pass through VAE encoder (to latent space)
    encoded = vae.encode(x_device).latent_dist.mean * _LATENT_SCALE
    debug_print(f"   ✅ VAE encoding completed, latent shape: {encoded.shape}")

    # Reverse DDIM sampling conditioned on encoder latents
    model_kwargs = {"context": object_cls_device.unsqueeze(1), "mask": None}
    
    debug_print(f"   🔄 Starting DDIM sampling with {reverse_steps} steps...")
    latent_samples_list = []
    step_count = 0
    for samples in diffusion.ddim_deviation_sample_loop_progressive(
        model,
        shape=encoded.shape,
        noise=encoded,
        clip_denoised=False,
        start_t=reverse_steps,
        model_kwargs=model_kwargs,
        progress=False,
        device=device,
        eta=0.0,
    ):
        step_count += 1
        if step_count % max(1, reverse_steps // 5) == 0:  # Print every 20% of steps
            debug_print(f"     📈 DDIM step {step_count}/{reverse_steps}")
        latent_samples_list.append(samples["sample"])
    latent_samples_final = latent_samples_list[-1]
    debug_print(f"   ✅ DDIM sampling completed after {step_count} steps")

    debug_print(f"   🎨 VAE decoding...")
    # Decode final latent samples
    image_samples = vae.decode(latent_samples_final / _LATENT_SCALE).sample
    x0 = vae.decode(encoded / _LATENT_SCALE).sample
    debug_print(f"   ✅ VAE decoding completed")
    
    debug_print(f"   📊 Computing differences...")
    # Core difference computations
    encodedrecon_dodrecon_diff_raw = _compute_abs_diff_max(x0, image_samples)
    encodedrecon_dodrecon_diff = torch.clamp(encodedrecon_dodrecon_diff_raw, 0.0, 0.05) * 20

    encoded_latent_diff_raw = _compute_abs_diff_mean(latent_samples_final, encoded)
    encoded_latent_diff = torch.clamp(encoded_latent_diff_raw, 0.0, 0.05) * 20

    # Resize encoded_latent_diff to match the spatial dimensions
    patch_size_actual = x_device.shape[-1]
    encoded_latent_diff_resized = F.interpolate(
        encoded_latent_diff,
        size=(patch_size_actual, patch_size_actual),
        mode="bilinear",
        align_corners=False,
    )
    anomaly_map_arithmetic = 0.5 * (encodedrecon_dodrecon_diff + encoded_latent_diff_resized)
    debug_print(f"   ✅ Difference computation completed")
    
    # Collect epoch-wise statistics if enabled
    if epoch_metrics is not None:
        debug_print(f"   📊 Collecting epoch statistics...")
        # Use the same logic as in evaluation_DeCo_Diff2.py
        epoch_metrics.add_batch_stats(
            encodedrecon_dodrecon_diff_raw, 
            encoded_latent_diff_raw, 
            anomaly_map_arithmetic, 
            anomaly_map_arithmetic  # Use same for geometric since we don't compute separate geometric
        )
        debug_print(f"   ✅ Epoch statistics collected")
    
    # Clear memory
    del x_device, object_cls_device, encoded, latent_samples_list, latent_samples_final
    del image_samples, x0, encodedrecon_dodrecon_diff_raw, encoded_latent_diff_raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic


def _extract_patch_info(patch_coords, image_paths_batch, b, patch_size=128):
    """
    Shared logic for extracting patch coordinates and image paths.
    Expects 8-value coordinate format: (x1, y1, x2, y2, x3, y3, x4, y4)
    """
    # Get patch coordinates
    if b < len(patch_coords):
        patch_coord_tensor = patch_coords[b]
    else:
        # If we don't have enough coordinates, use the last available one or default
        default_coords = (0, 0, patch_size, 0, patch_size, patch_size, 0, patch_size)
        patch_coord_tensor = patch_coords[-1] if patch_coords else default_coords
    
    # Extract 8-value coordinates
    if isinstance(patch_coord_tensor, (list, tuple)) and len(patch_coord_tensor) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = patch_coord_tensor
        x_coord, y_coord = x1, y1  # Use top-left corner
    else:
        raise ValueError(f"Expected 8-value coordinate format, got {type(patch_coord_tensor)} with {len(patch_coord_tensor) if hasattr(patch_coord_tensor, '__len__') else 'unknown'} values")
    
    # Get image path
    if b < len(image_paths_batch):
        if isinstance(image_paths_batch[b], str):
            image_path = image_paths_batch[b]
        elif isinstance(image_paths_batch[b], (list, tuple)):
            image_path = image_paths_batch[b][0] if image_paths_batch[b] else ""
        else:
            image_path = str(image_paths_batch[b])
    else:
        image_path = str(image_paths_batch[-1]) if image_paths_batch else ""
    
    return x_coord, y_coord, image_path


def _process_chunked_inference(x, object_cls, model, vae, diffusion, reverse_steps, device, chunk_size=16, epoch_metrics=None):
    """
    Process patches in chunks and return concatenated results.
    """
    debug_print(f"📦 Creating chunks with size {chunk_size} from {x.size(0)} patches")
    x_chunks = process_patches_in_chunks(x, chunk_size=chunk_size)
    object_cls_chunks = process_patches_in_chunks(object_cls, chunk_size=chunk_size)
    
    debug_print(f"📊 Created {len(x_chunks)} chunks")
    
    all_encodedrecon_diffs = []
    all_encoded_latent_diffs = []
    all_anomaly_maps = []
    
    for chunk_idx, (x_chunk, object_cls_chunk) in enumerate(zip(x_chunks, object_cls_chunks)):
        debug_print(f"🔧 Processing chunk {chunk_idx+1}/{len(x_chunks)} with {x_chunk.size(0)} patches")
        encodedrecon_diff, encoded_latent_diff, anomaly_map = _process_batch_inference(
            x_chunk, object_cls_chunk, model, vae, diffusion, reverse_steps, device, epoch_metrics
        )
        debug_print(f"✅ Chunk {chunk_idx+1} completed")
        
        all_encodedrecon_diffs.append(encodedrecon_diff)
        all_encoded_latent_diffs.append(encoded_latent_diff)
        all_anomaly_maps.append(anomaly_map)
    
    # Concatenate all results
    encodedrecon_dodrecon_diff = torch.cat(all_encodedrecon_diffs, dim=0)
    encoded_latent_diff_resized = torch.cat(all_encoded_latent_diffs, dim=0)
    anomaly_map_arithmetic = torch.cat(all_anomaly_maps, dim=0)
    
    return encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic


def _save_results(
    args,
    dataloader,
    split: str,
    diffusion,
    model,
    vae,
    reverse_steps: int,
    batch_num: int,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "minimal_diff_results",
    checkpoint_manager: "CheckpointManager" | None = None
) -> None:
    """
    Save only .npy files and images without processing records.
    This is mode 1: save_only
    """
    # Initialize epoch metrics if enabled
    epoch_metrics = EvaluationMetrics() if args.enable_epoch_stats else None
    
    # Debug print checkpoint manager info
    debug_print(f"🔍 Checkpoint Manager: {checkpoint_manager}")
    debug_print(f"🔍 Checkpoint Manager type: {type(checkpoint_manager)}")
    if checkpoint_manager:
        debug_print(f"🔍 Checkpoint dir: {getattr(checkpoint_manager, 'base_checkpoint_dir', 'N/A')}")
        debug_print(f"🔍 Force rerun: {getattr(checkpoint_manager, 'force_rerun', 'N/A')}")
        # Check if checkpoint directory exists
        checkpoint_dir = getattr(checkpoint_manager, 'base_checkpoint_dir', None)
        if checkpoint_dir:
            debug_print(f"🔍 Checkpoint dir exists: {os.path.exists(checkpoint_dir)}")
            if os.path.exists(checkpoint_dir):
                checkpoint_files = os.listdir(checkpoint_dir)
                debug_print(f"🔍 Checkpoint files: {checkpoint_files}")
            else:
                debug_print(f"🔍 Checkpoint directory does not exist yet")
        # Check checkpoint manager methods
        debug_print(f"🔍 Has save_checkpoint method: {hasattr(checkpoint_manager, 'save_checkpoint')}")
        debug_print(f"🔍 Has get_processed_images method: {hasattr(checkpoint_manager, 'get_processed_images')}")
        # Try to get current processed images
        try:
            processed_images = checkpoint_manager.get_processed_images()
            debug_print(f"🔍 Currently processed images count: {len(processed_images)}")
            if processed_images:
                debug_print(f"🔍 Sample processed images: {list(processed_images)[:3]}")
        except Exception as e:
            debug_print(f"🔍 Error getting processed images: {e}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    idx = -1
    
    try:
        for idx, (x, seg, object_cls, image_paths, anomaly_classes, patch_coords) in enumerate(
            tqdm(dataloader, desc=f"{split} split")
        ):
            if idx >= batch_num:
                break

            debug_print(f"🔄 Processing batch {idx+1}/{min(batch_num, len(dataloader))}")
            debug_print(f"📊 Batch size: {x.size(0)} patches")
            debug_print(f"🖼️  Patch shape: {x.shape}")
            debug_print(f"🏷️  Object classes shape: {object_cls.shape}")
            
            with torch.no_grad():
                debug_print(f"🧠 Starting inference...")
                # Use shared inference logic with smaller chunk size for debugging
                encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic = _process_chunked_inference(
                    x, object_cls, model, vae, diffusion, reverse_steps, device, chunk_size=1, epoch_metrics=epoch_metrics
                )
                debug_print(f"✅ Inference completed")
            
            # ---------------------------------------------------------------------
            # Per‑sample aggregation and saving
            # ---------------------------------------------------------------------
            batch_size = x.size(0)
            
            for b in range(batch_size):
                # Use shared coordinate and path extraction
                x_coord, y_coord, image_path = _extract_patch_info(patch_coords, image_paths, b, args.patch_size)
                
                # Get all 4 corner coordinates for irregular patches
                x1, y1 = x_coord, y_coord  # Top-left corner
                
                # Since we use padding, all patches are regular patch_size x patch_size
                patch_size_actual = args.patch_size  # All patches are now regular due to padding
                x2, y2 = x_coord + patch_size_actual, y_coord  # Top-right
                x3, y3 = x_coord + patch_size_actual, y_coord + patch_size_actual  # Bottom-right
                x4, y4 = x_coord, y_coord + patch_size_actual  # Bottom-left
                
                # Create base filename with file info and patch info (all 4 corners)
                if b < len(image_paths):
                    if isinstance(image_paths[b], str):
                        image_path = image_paths[b]
                    elif isinstance(image_paths[b], (list, tuple)):
                        image_path = image_paths[b][0] if image_paths[b] else ""
                    else:
                        image_path = str(image_paths[b])
                else:
                    # If we don't have enough image paths, use the last available one or default
                    if image_paths:
                        if isinstance(image_paths[-1], str):
                            image_path = image_paths[-1]
                        elif isinstance(image_paths[-1], (list, tuple)):
                            image_path = image_paths[-1][0] if image_paths[-1] else ""
                        else:
                            image_path = str(image_paths[-1])
                    else:
                        image_path = ""
                file_info = path_to_safe_filename(image_path)
                patch_info = f"x{x1}_y{y1}_x{x2}_y{y2}_x{x3}_y{y3}_x{x4}_y{y4}"
                base_filename = f"{file_info}__{patch_info}__minimal_diff"
                
                # Save difference maps using PIL for robust tensor handling
                # Convert to uint8 for maximum I/O performance and remove single dimensions
                # Get raw values in [0,1] range for efficient storage
                encodedrecon_raw = _to_numpy(encodedrecon_dodrecon_diff[b])
                latent_raw = _to_numpy(encoded_latent_diff_resized[b])
                anomaly_map_arithmetic_raw = _to_numpy(anomaly_map_arithmetic[b])
                
                # Remove single dimensions
                encodedrecon_raw = encodedrecon_raw.squeeze()
                latent_raw = latent_raw.squeeze()
                anomaly_map_arithmetic_raw = anomaly_map_arithmetic_raw.squeeze()

                # Save raw values efficiently as numpy arrays (preserves [0,1] range)
                np.save(os.path.join(save_dir, f"{base_filename}_encodedrecon.npy"), encodedrecon_raw)
                np.save(os.path.join(save_dir, f"{base_filename}_latent.npy"), latent_raw)
                np.save(os.path.join(save_dir, f"{base_filename}_anomaly_map_arithmetic.npy"), anomaly_map_arithmetic_raw)
                
                # Save 8-value coordinates as a separate .npy file (more efficient than filename parsing)
                patch_coords_8_values = np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.int32)
                np.save(os.path.join(save_dir, f"{base_filename}_coords.npy"), patch_coords_8_values)
                
                # Convert to [0,255] range for image saving
                encodedrecon_img = (encodedrecon_raw * 255).astype(np.uint8)
                latent_img = (latent_raw * 255).astype(np.uint8)
                anomaly_map_arithmetic_img = (anomaly_map_arithmetic_raw * 255).astype(np.uint8)
                
                # Save as images for quick visual inspection
                PILImage.fromarray(encodedrecon_img).save(os.path.join(save_dir, f"{base_filename}_encodedrecon.png"))
                PILImage.fromarray(latent_img).save(os.path.join(save_dir, f"{base_filename}_latent.png"))
                PILImage.fromarray(anomaly_map_arithmetic_img).save(os.path.join(save_dir, f"{base_filename}_anomaly_map_arithmetic.png"))
            
            # After finishing all patches for this image, update checkpoint
            if checkpoint_manager is not None:
                debug_print(f"🔍 Attempting to save checkpoint for batch {idx}")
                try:
                    # Collect valid image paths (deduplicated)
                    processed_image_paths = []
                    for p in image_paths:
                        if isinstance(p, str) and p:
                            processed_image_paths.append(p)
                        elif isinstance(p, (list, tuple)) and len(p) > 0:
                            first = p[0]
                            if isinstance(first, str) and first:
                                processed_image_paths.append(first)
                    unique_processed = sorted(set(processed_image_paths))
                    debug_print(f"🔍 Unique processed image paths: {unique_processed}")

                    # Merge with previously processed images (from latest checkpoint)
                    previously_processed = checkpoint_manager.get_processed_images()
                    debug_print(f"🔍 Previously processed images count: {len(previously_processed)}")
                    merged = list(set(previously_processed).union(set(unique_processed)))
                    debug_print(f"🔍 Merged processed images count: {len(merged)}")

                    # Save a new checkpoint reflecting progress up to current image index
                    debug_print(f"🔍 Calling save_checkpoint with image_index={idx + 1}")
                    checkpoint_manager.save_checkpoint(
                        current_image_index=idx + 1,
                        processed_images=merged,
                    )
                    debug_print(f"🔍 ✅ Checkpoint saved successfully for batch {idx}")
                except Exception as e:
                    print(f"Warning: failed to save checkpoint after image {idx}: {e}")
                    debug_print(f"🔍 ❌ Checkpoint save failed: {e}")

            # Memory optimization: Clear cache every 10 batches
            if idx % 10 == 0 and idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        raise
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise

    print(f"Raw data saving completed. Results saved to {save_dir}")
    
    # Print epoch statistics if enabled
    if epoch_metrics is not None:
        epoch_metrics.print_epoch_stats()
    else:
        debug_print("Skipping epoch statistics (disabled)")

# Import processing functions from process_raw_data_to_results.py
from process_raw_data_to_results import (
    _binary_mask_exclude_boundary3,
    load_ground_truth_map,
    load_original_images,
    parse_filename_to_info,
    load_raw_data_files,
    reconstruct_records_from_raw_data,
    compute_simple_metrics,
    save_image_results_from_raw_data,
    save_json_results_from_raw_data,
    create_confusion_matrix_from_records
)

def save_all_records_json(records: List[Record], output_dir: str, filename: str = "all_records.json", patch_size: int = 128) -> None:
    """
    Save all records in a single comprehensive JSON file.
    
    Args:
        records: List of all evaluation records
        output_dir: Directory to save the JSON file
        filename: Name of the output JSON file (default: "all_records.json")
        patch_size: Size of patches for grid coordinate calculation
    """
    print(f"Saving all {len(records)} records to comprehensive JSON...")
    
    # Convert records to a JSON-serializable format
    all_records_data = {
        "total_records": len(records),
        "records": []
    }
    
    for i, record in enumerate(records):
        # Extract all the key information from each record
        record_data = {
            "record_id": i,
            "split": record["split"][1] if "split" in record else None,
            "image_path": record["image_path"][1] if "image_path" in record else None,
            "image_path_original": record["image_path_original"][1] if "image_path_original" in record else None,
            "anomaly_class": record["anomaly_class"][1] if "anomaly_class" in record else None,
            "patch_coords": record["patch_coords"][1] if "patch_coords" in record else None,
            "anomaly_max": int(record["anomaly_max"][1]) if "anomaly_max" in record else None,
            "anomaly_pixels": int(record["anomaly_pixels"][1]) if "anomaly_pixels" in record else None,
            "is_predicted_defective": record["is_predicted_defective"][1] if "is_predicted_defective" in record else None,
            "status": record["status"][1] if "status" in record else None,
            # Add metric information if available
            "lpips": float(record["lpips"][1]) if "lpips" in record else None,
            "ssim": float(record["ssim"][1]) if "ssim" in record else None,
            "mse": float(record["mse"][1]) if "mse" in record else None,
        }
        
        # Add computed grid coordinates (simple division since images are padded)
        if record_data["patch_coords"]:
            patch_coords = record_data["patch_coords"]
            debug_print(f"🔍 Processing patch_coords: {patch_coords} (type: {type(patch_coords)}, len: {len(patch_coords)})")
            if len(patch_coords) == 8:
                # 8-value format: (x1, y1, x2, y2, x3, y3, x4, y4)
                x1, y1, x2, y2, x3, y3, x4, y4 = patch_coords
                patch_x, patch_y = x1, y1  # Top-left corner
            elif len(patch_coords) == 2:
                # Legacy 2-value format detected - convert to 8-value format
                debug_print(f"⚠️  Converting 2-value coordinates {patch_coords} to 8-value format")
                x1, y1 = patch_coords[0], patch_coords[1]
                x2, y2 = x1 + patch_size, y1  # Top-right
                x3, y3 = x1 + patch_size, y1 + patch_size  # Bottom-right
                x4, y4 = x1, y1 + patch_size  # Bottom-left
                
                # Update the record with converted 8-value coordinates
                record_data["patch_coords"] = [x1, y1, x2, y2, x3, y3, x4, y4]
                patch_x, patch_y = x1, y1
                debug_print(f"✅ Converted to 8-value coordinates: {record_data['patch_coords']}")
            else:
                raise ValueError(f"Expected 8-value patch coordinates, got {len(patch_coords)} values. All coordinates should be 8-value format: (x1, y1, x2, y2, x3, y3, x4, y4)")
            
            # Simple grid calculation since all patches are regular grid-aligned with padding
            record_data["grid_row"] = patch_y // patch_size
            record_data["grid_col"] = patch_x // patch_size
        
        all_records_data["records"].append(record_data)
    
    # Add summary statistics
    if records:
        statuses = [r["status"][1] for r in records if "status" in r]
        all_records_data["summary"] = {
            "total_patches": len(records),
            "status_counts": {
                "TP": sum(1 for s in statuses if s == "TP"),
                "FP": sum(1 for s in statuses if s == "FP"), 
                "TN": sum(1 for s in statuses if s == "TN"),
                "FN": sum(1 for s in statuses if s == "FN")
            }
        }
        
        # Add accuracy metrics if possible
        tp = all_records_data["summary"]["status_counts"]["TP"]
        fp = all_records_data["summary"]["status_counts"]["FP"]
        tn = all_records_data["summary"]["status_counts"]["TN"]
        fn = all_records_data["summary"]["status_counts"]["FN"]
        
        total = tp + fp + tn + fn
        if total > 0:
            all_records_data["summary"]["metrics"] = {
                "accuracy": (tp + tn) / total,
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            }
    
    # Save to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(all_records_data, f, indent=2)
    
    print(f"All records saved to: {output_path}")
    print(f"Total records: {len(records)}")
    if "summary" in all_records_data:
        print(f"Status distribution: {all_records_data['summary']['status_counts']}")

def load_model_and_components(args):
    """Load VAE, model, and diffusion components."""
    # Load VAE
    if os.path.exists("./models/config.json"):
        vae = AutoencoderKL.from_pretrained("./models", local_files_only=True).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
    vae.eval()
    
    # Load model
    try:
        if args.pretrained != "":
            # Handle relative paths by checking if they exist relative to current directory
            # or parent directory (project root)
            ckpt = args.pretrained
            if not os.path.exists(ckpt):
                # Try relative to parent directory (project root)
                parent_ckpt = os.path.join("..", ckpt)
                if os.path.exists(parent_ckpt):
                    ckpt = parent_ckpt
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_class}_{args.model_size}_{args.patch_size}"
            try:
                ckpt = sorted(glob(f"{path}/last.pt"))[-1]
            except (IndexError, FileNotFoundError):
                ckpt = sorted(glob(f"{path}/*/last.pt"))[-1]
    except (IndexError, FileNotFoundError, OSError) as e:
        raise Exception(f"Please provide the model's pretrained path using --pretrained. Error: {e}")

    latent_size = int(args.patch_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size)

    state_dict = torch.load(ckpt)["model"]
    print(model.load_state_dict(state_dict))
    model.eval()  # important!
    model.cuda()
    print("model loaded")

    # Create diffusion
    diffusion = create_diffusion(
        f"ddim{args.reverse_steps}",
        predict_deviation=True,
        sigma_small=False,
        predict_xstart=False,
        diffusion_steps=1000,
    )
    
    return vae, model, diffusion

def get_transform():
    """Get the transform for input images."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_dataloader(dataset, batch_size=1, irregular_patch=False):
    """Get dataloader for the dataset."""
    if irregular_patch:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows-friendly
            drop_last=False,
            collate_fn=custom_collate_fn
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Windows-friendly
            drop_last=False
        )

def _before_saving_results(args):
    """
    Shared logic for preparing evaluation components.
    Returns: (vae, model, diffusion, dataset, loader, evaluation_results_dir, checkpoint_manager)
    """
    # Load model components
    vae, model, diffusion = load_model_and_components(args)
    
    # Create dataset
    dataset = AnnotatedImageDataset(
        annotation_dir=args.annotation_dir,
        patch_size=args.patch_size,
        transform=get_transform(),
        object_class=args.object_class,
    )
    loader = get_dataloader(dataset, batch_size=16, irregular_patch=args.irregular_patch)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(args.results_dir, args.annotation_dir, args.force_rerun)
    
    # Create evaluation results directory
    evaluation_results_dir = os.path.join(args.results_dir, "evaluation_results")
    
    return vae, model, diffusion, dataset, loader, evaluation_results_dir, checkpoint_manager


def _reading_saved_results(args):
    """
    Read saved .npy files and reconstruct records with 8-value coordinates.
    Returns: (records, ground_truth_map, original_images)
    """
    import glob
    import re
    
    debug_print("📂 Reading saved .npy files...")
    
    # Load ground truth information using existing dataset functionality
    dataset = AnnotatedImageDataset(
        annotation_dir=args.annotation_dir,
        patch_size=args.patch_size,
        transform=get_transform(),
        object_class=args.object_class,
    )
    
    # Create empty ground truth map for compatibility
    ground_truth_map = {}
    original_images = {}
    
    # Find all .npy files
    npy_pattern = os.path.join(args.results_dir, "evaluation_results", "*_encodedrecon.npy")
    npy_files = glob.glob(npy_pattern)
    debug_print(f"📁 Found {len(npy_files)} .npy files")
    
    records = []
    
    for npy_file in npy_files:
        # Load coordinates from dedicated .npy file (much more efficient than filename parsing)
        base_name = npy_file.replace("_encodedrecon.npy", "")
        coords_file = f"{base_name}_coords.npy"
        latent_file = f"{base_name}_latent.npy"
        anomaly_file = f"{base_name}_anomaly_map_arithmetic.npy"
        
        if os.path.exists(coords_file) and os.path.exists(latent_file) and os.path.exists(anomaly_file):
            # Load 8-value coordinates directly from .npy file
            patch_coords_8_values = np.load(coords_file).tolist()  # Convert to list for consistency
            debug_print(f"🔍 Loaded coords from .npy: {patch_coords_8_values} (len: {len(patch_coords_8_values)})")
            
            # Extract image path from filename (before first __)
            filename = os.path.basename(npy_file)
            file_info = filename.split("__")[0]
            image_path = safe_filename_to_path(file_info)
            
            # Load data
            encodedrecon_data = np.load(npy_file)
            latent_data = np.load(latent_file)
            anomaly_data = np.load(anomaly_file)
            
            # Create record with 8-value coordinates
            record = make_record(
                split=("meta", args.split),
                image_path=("meta", image_path),
                image_path_original=("meta", file_info),
                anomaly_class=("meta", "all"),
                patch_coords=("meta", patch_coords_8_values),  # 8 values from .npy file
                encodedrecon_dodrecon_diff=("tensor", torch.from_numpy(encodedrecon_data)),
                encoded_latent_diff_resized=("tensor", torch.from_numpy(latent_data)),
                anomaly_map_arithmetic=("tensor", torch.from_numpy(anomaly_data))
            )
            
            # Debug: Check what make_record actually created
            debug_print(f"🔍 make_record created patch_coords: {record.get('patch_coords', 'MISSING')} (type: {type(record.get('patch_coords', [None, None])[1]) if 'patch_coords' in record else 'N/A'})")
            
            records.append(record)
            debug_print(f"✅ Loaded record with 8-value coords from .npy: {patch_coords_8_values}")
        else:
            # Fallback: try to extract from filename for backward compatibility with old data
            filename = os.path.basename(npy_file)
            debug_print(f"🔍 No _coords.npy found, trying filename parsing for: {filename}")
            coord_pattern = r"__x(\d+)_y(\d+)_x(\d+)_y(\d+)_x(\d+)_y(\d+)_x(\d+)_y(\d+)__minimal_diff"
            match = re.search(coord_pattern, filename)
            
            if match and os.path.exists(latent_file) and os.path.exists(anomaly_file):
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, match.groups())
                patch_coords_8_values = [x1, y1, x2, y2, x3, y3, x4, y4]
                debug_print(f"🔍 Extracted coords from filename: {patch_coords_8_values}")
                
                file_info = filename.split("__")[0]
                image_path = safe_filename_to_path(file_info)
                
                # Load data
                encodedrecon_data = np.load(npy_file)
                latent_data = np.load(latent_file)
                anomaly_data = np.load(anomaly_file)
                
                # Create record with 8-value coordinates
                record = make_record(
                    split=("meta", args.split),
                    image_path=("meta", image_path),
                    image_path_original=("meta", file_info),
                    anomaly_class=("meta", "all"),
                    patch_coords=("meta", patch_coords_8_values),  # 8 values from filename
                    encodedrecon_dodrecon_diff=("tensor", torch.from_numpy(encodedrecon_data)),
                    encoded_latent_diff_resized=("tensor", torch.from_numpy(latent_data)),
                    anomaly_map_arithmetic=("tensor", torch.from_numpy(anomaly_data))
                )
                
                # Debug: Check what make_record actually created (filename path)
                debug_print(f"🔍 make_record (filename) created patch_coords: {record.get('patch_coords', 'MISSING')} (type: {type(record.get('patch_coords', [None, None])[1]) if 'patch_coords' in record else 'N/A'})")
                
                records.append(record)
                debug_print(f"✅ Loaded record with 8-value coords from filename: {patch_coords_8_values}")
            else:
                debug_print(f"⚠️  Missing required files for {npy_file}")
                debug_print(f"     coords.npy: {os.path.exists(coords_file)}")
                debug_print(f"     latent.npy: {os.path.exists(latent_file)}")
                debug_print(f"     anomaly.npy: {os.path.exists(anomaly_file)}")
                debug_print(f"     regex match: {match is not None}")
                if not match:
                    debug_print(f"     filename pattern expected: file__x0_y0_x128_y0_x128_y128_x0_y128__minimal_diff_*")
    
    debug_print(f"📊 Loaded {len(records)} records from saved data")
    return records, ground_truth_map, original_images


def _after_reading_saved_results(
    args,
    records,
    ground_truth_map,
    original_images,
    output_subdir_name: str = "processed_results",
):
    """
    Process records after reading from saved data and generate outputs.
    Returns: (metrics, output_dir)
    """
    # Generate timestamp for output directory if requested
    if args.append_timestamp:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.results_dir, f"{output_subdir_name}_{current_time}")
    else:
        output_dir = os.path.join(args.results_dir, output_subdir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    debug_print(f"📁 Output directory: {output_dir}")
    
    # Calculate metrics from records
    debug_print("📊 Computing evaluation metrics...")
    
    # compute_y_true_y_score expects a list of (records, records_defect) tuples
    # Since we have a single list of records, we'll separate them based on defect status
    normal_records = [r for r in records if not r.get('is_predicted_defective', [False, False])[1]]
    defect_records = [r for r in records if r.get('is_predicted_defective', [False, False])[1]]
    
    # Format as expected by compute_y_true_y_score
    all_records = [(normal_records, defect_records)]
    
    try:
        y_true, y_score = compute_y_true_y_score(all_records)
        metrics = compute_metrics_from_y_true_y_score(y_true, y_score)
    except Exception as e:
        debug_print(f"⚠️  Error computing metrics: {e}")
        # Fallback to simple metrics calculation
        metrics = {
            "total_records": len(records),
            "defective_records": len(defect_records),
            "normal_records": len(normal_records)
        }
    
    # Save results in various formats
    if args.enable_save_json_results:
        debug_print("💾 Saving JSON results...")
        debug_print("📄 Saving comprehensive JSON with all records...")
        save_all_records_json(
            records,
            output_dir,
            filename="all_evaluation_records.json",
            patch_size=args.patch_size
        )

    if args.enable_save_image_results:
        debug_print("🖼️ Saving image results...")
        # Build per-image groupings and compute predicted/ground-truth sets
        checkpoint_manager = CheckpointManager(args.results_dir, args.annotation_dir, args.force_rerun)
        from collections import defaultdict
        image_to_records = defaultdict(list)
        for rec in records:
            img_path = rec.get("image_path", (None, None))[1]
            if img_path:
                image_to_records[img_path].append(rec)

        for img_path, image_records in image_to_records.items():
            # Predicted defective set from records (expects 8-value patch_coords)
            predicted_defective_set = set()
            for rec in image_records:
                try:
                    is_def = rec.get("is_predicted_defective", (None, False))[1]
                    if not is_def:
                        continue
                    coords = rec.get("patch_coords", (None, []))[1]
                    if not (isinstance(coords, (list, tuple)) and len(coords) == 8):
                        raise ValueError(f"Expected 8-value patch_coords in records, got: {coords}")
                    x1, y1 = int(coords[0]), int(coords[1])
                    row = y1 // args.patch_size
                    col = x1 // args.patch_size
                    predicted_defective_set.add((row, col))
                except Exception:
                    continue

            ground_truth_defective = ground_truth_map.get(img_path, set()) if isinstance(ground_truth_map, dict) else set()
            overlapping = set()

            # Delegate to library function to save image-level marked outputs
            save_image_results_from_records(
                checkpoint_manager,
                img_path,
                image_records,
                predicted_defective_set,
                ground_truth_defective,
                overlapping,
                enable_save_optional_image_results=args.enable_save_optional_image_results,
                patch_size=args.patch_size,
            )
    
    if args.enable_excel_report:
        debug_print("📊 Creating Excel report...")
        make_excel(records, output_dir, args.split, args.object_class)
    
    debug_print("✅ Processing completed successfully!")
    return metrics, output_dir


def mode_save_only(args):
    """Mode 1: Save .npy files and diff images only."""
    print("=== Mode 1: Save Only ===")
    
    # Before saving results
    vae, model, diffusion, dataset, loader, evaluation_results_dir, checkpoint_manager = _before_saving_results(args)
    
    # Save raw data directly
    _save_results(
        args=args,
        dataloader=loader,
        split=args.split,
        diffusion=diffusion,
        model=model,
        vae=vae,
        reverse_steps=args.reverse_steps,
        batch_num=args.batch_num,
        device=device,
        save_dir=evaluation_results_dir,
        checkpoint_manager=checkpoint_manager
    )
    
    print(f"Raw data saved to: {evaluation_results_dir}")

def _reading_saved_results(args):
    """
    Shared logic for loading and reconstructing records from saved .npy files.
    Returns: (records, ground_truth_map, original_images)
    """
    # Load ground truth map
    ground_truth_map = load_ground_truth_map(args.annotation_dir)
    print(f"Loaded ground truth for {len(ground_truth_map)} images")
    # Load all raw data files
    patch_data = load_raw_data_files(args.results_dir, visualize=False)
    
    if not patch_data:
        print("No complete patch data found!")
        return [], ground_truth_map, {}
    
    # Extract unique image paths from patch data
    image_paths = set(data['file_path'] for data in patch_data.values())
    
    # Load original images
    original_images = load_original_images(image_paths)
    print(f"Loaded {len(original_images)} original images")
    
    print(f"Reconstructing records from {len(patch_data)} patches...")
    
    # Debug: Check patch_data coordinates before reconstruction
    debug_print("🔍 Checking patch_data coordinates...")
    for key, data in list(patch_data.items())[:3]:  # Check first 3 items
        coords = data.get('patch_coords', 'MISSING')
        debug_print(f"   {key}: patch_coords = {coords} (type: {type(coords)}, len: {len(coords) if hasattr(coords, '__len__') else 'N/A'})")
    
    # Verify coordinates are now 8-value format
    debug_print("🔍 Verifying 8-value coordinates from fixed load_raw_data_files...")
    for key, data in list(patch_data.items())[:3]:  # Check first 3 items
        coords = data.get('patch_coords', 'MISSING')
        debug_print(f"   {key}: patch_coords = {coords} (type: {type(coords)}, len: {len(coords) if hasattr(coords, '__len__') else 'N/A'})")
    
    # Reconstruct records
    debug_print("🔍 Calling reconstruct_records_from_raw_data...")
    records = reconstruct_records_from_raw_data(
        patch_data,
        ground_truth_map=ground_truth_map,
        original_images=original_images,
        anomaly_binary_threshold=args.anomaly_binary_threshold,
        anomaly_pixel_num_threshold=args.anomaly_pixel_num_threshold,
        adaptive_threshold=args.adaptive_threshold,
        patch_size=args.patch_size,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Debug: Check the first few records that were created
    debug_print("🔍 Checking created records...")
    for i, record in enumerate(records[:3]):  # Check first 3 records
        coords = record.get('patch_coords', 'MISSING')
        debug_print(f"   Record {i}: patch_coords = {coords} (value: {coords[1] if coords != 'MISSING' and len(coords) > 1 else 'N/A'})")
    
    print(f"Generated {len(records)} records")
    return records, ground_truth_map, original_images


def mode_process_only(args):
    """Mode 2: Read existing .npy files and generate categorization results."""
    print("=== Mode 2: Process Only ===")
    
    # Reading saved results
    records, ground_truth_map, original_images = _reading_saved_results(args)
    
    # After reading saved results
    metrics, output_dir = _after_reading_saved_results(args, records, ground_truth_map, original_images)
    
    return records

def _generate_records_directly(args, vae, model, diffusion, loader):
    """
    Generate records directly from evaluation without saving intermediate files.
    This combines the "before saving results" logic with direct record generation.
    Returns: (records, ground_truth_map, original_images)
    """
    # Initialize epoch metrics if enabled
    epoch_metrics = EvaluationMetrics() if args.enable_epoch_stats else None
    # Load ground truth map
    ground_truth_map = load_ground_truth_map(args.annotation_dir)
    print(f"Loaded ground truth for {len(ground_truth_map)} images")
    
    # Extract unique image paths from dataset
    image_paths = set()
    for i in range(len(loader.dataset)):
        annotation_file = loader.dataset.annotation_files[i]
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        image_path = annotation_data.get('image_path', '')
        if image_path:
            image_paths.add(image_path)
    
    # Load original images
    original_images = load_original_images(image_paths)
    print(f"Loaded {len(original_images)} original images")
    
    # Process data directly without saving intermediates
    records = []
    
    idx = -1
    try:
        for idx, (x, seg, object_cls, image_paths_batch, anomaly_classes, patch_coords) in enumerate(
            tqdm(loader, desc="Processing patches")
        ):
            if idx >= args.batch_num:
                break

            with torch.no_grad():
                # Use shared inference logic
                encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic = _process_chunked_inference(
                    x, object_cls, model, vae, diffusion, args.reverse_steps, device, chunk_size=16, epoch_metrics=epoch_metrics
                )
            
            # Process each patch to create records
            batch_size = x.size(0)
            
            for b in range(batch_size):
                # Use shared coordinate and path extraction
                x_coord, y_coord, image_path = _extract_patch_info(patch_coords, image_paths_batch, b, args.patch_size)
                
                # Get the 8-value coordinates from the dataset
                patch_coords_raw = patch_coords[b]  # Should already be 8 values from dataset
                debug_print(f"  🔍 Raw patch_coords[{b}]: {patch_coords_raw} (type: {type(patch_coords_raw)}, len: {len(patch_coords_raw) if hasattr(patch_coords_raw, '__len__') else 'N/A'})")
                
                # Ensure we have 8-value coordinates and convert to list if needed
                if isinstance(patch_coords_raw, (list, tuple)) and len(patch_coords_raw) == 8:
                    patch_coords_8_values = list(patch_coords_raw)
                elif hasattr(patch_coords_raw, '__len__') and len(patch_coords_raw) == 8:
                    # Handle numpy arrays or other array-like objects
                    patch_coords_8_values = list(patch_coords_raw)
                else:
                    raise ValueError(f"Expected 8-value coordinates from dataset, got {type(patch_coords_raw)} with {len(patch_coords_raw) if hasattr(patch_coords_raw, '__len__') else 'unknown'} values: {patch_coords_raw}")
                
                debug_print(f"  ✅ Using 8-value coordinates: {patch_coords_8_values}")
                
                # Convert tensors to numpy
                encodedrecon_raw = _to_numpy(encodedrecon_dodrecon_diff[b]).squeeze()
                latent_raw = _to_numpy(encoded_latent_diff_resized[b]).squeeze()
                anomaly_map_arithmetic_raw = _to_numpy(anomaly_map_arithmetic[b]).squeeze()
                
                # Convert to torch tensors for processing
                anomaly_map_arithmetic_tensor = torch.from_numpy(anomaly_map_arithmetic_raw).float().unsqueeze(0).unsqueeze(0)
                
                # Create binary mask
                anomaly_map_arithmetic_binary = _binary_mask_exclude_boundary3(
                    anomaly_map_arithmetic_tensor, 
                    args.anomaly_binary_threshold, 
                    visualize=False, 
                    debug=False, 
                    filename=image_path
                )
                
                # Calculate metrics
                anomaly_max = int(round(anomaly_map_arithmetic_tensor.max().item() * 255))
                
                # Get actual patch dimensions for consistent cropping
                if image_path in original_images:
                    original_image = original_images[image_path]
                    h, w = original_image.shape[:2]
                    actual_patch_height = min(args.patch_size, h - y_coord)
                    actual_patch_width = min(args.patch_size, w - x_coord)
                    
                    # Crop the binary mask tensor to match the actual patch size
                    anomaly_binary_cropped = anomaly_map_arithmetic_binary[:, :, :actual_patch_height, :actual_patch_width]
                    anomaly_pixels = torch.sum(anomaly_binary_cropped).item()
                    is_predicted_defective = anomaly_pixels > args.anomaly_pixel_num_threshold
                    
                    # Get ground truth defective patches for this image
                    ground_truth_defective = ground_truth_map.get(image_path, set()) if ground_truth_map else set()
                    
                    # Convert pixel coordinates to grid coordinates (simple division since no overlapping)
                    # With padded images, all patches are regular grid-aligned patches
                    grid_row = y_coord // args.patch_size
                    grid_col = x_coord // args.patch_size
                    
                    # Determine status
                    status = "TP" if is_predicted_defective and (grid_row, grid_col) in ground_truth_defective else \
                             "FP" if is_predicted_defective else \
                             "FN" if (grid_row, grid_col) in ground_truth_defective else "TN"
                    
                    # Get original patch
                    original_patch = original_image[y_coord:y_coord + actual_patch_height, x_coord:x_coord + actual_patch_width]
                    
                    # Store the binary map with proper shape
                    binary_map_numpy = _to_numpy(anomaly_binary_cropped).squeeze()
                    
                    # Create record
                    debug_print(f"🔍 Creating record with patch_coords_8_values: {patch_coords_8_values}")
                    record = make_record(
                        split=("meta", args.split),
                        image_path=("meta", image_path),
                        image_path_original=("meta", path_to_safe_filename(image_path)),
                        anomaly_class=("meta", "all"),
                        patch_coords=("meta", patch_coords_8_values),
                        anomaly_max=("meta", anomaly_max),
                        anomaly_pixels=("meta", anomaly_pixels),
                        is_predicted_defective=("meta", is_predicted_defective),
                        status=("meta", status),
                        orig=("image", original_patch),
                        dod_recon=("image", encodedrecon_raw),
                        encoded_recon=("image", encodedrecon_raw),
                        anomaly_map_arithmetic=("image", anomaly_map_arithmetic_raw),
                        anomaly_map_arithmetic_binary=("image", binary_map_numpy),
                        anomaly_map_geometric=("image", anomaly_map_arithmetic_raw),
                        anomaly_map_geometric_binary=("image", binary_map_numpy),
                        encoded=("image", latent_raw),
                    )
                    
                    # Add metric fields
                    record["lpips"] = ("metric", 0.0)
                    record["ssim"] = ("metric", 0.0)
                    record["mse"] = ("metric", 0.0)
                    
                    # Debug: Check what make_record actually stored
                    debug_print(f"🔍 make_record result patch_coords: {record.get('patch_coords', 'MISSING')} (value: {record.get('patch_coords', [None, None])[1] if 'patch_coords' in record else 'N/A'})")
                    
                    records.append(record)

            # Memory optimization
            if idx % 10 == 0 and idx > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

    except KeyboardInterrupt:
        print("\nProcess interrupted.")
        raise
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise

    print(f"Generated {len(records)} records")
    
    # Print epoch statistics if enabled
    if epoch_metrics is not None:
        epoch_metrics.print_epoch_stats()
    else:
        debug_print("Skipping epoch statistics (disabled)")
    
    return records, ground_truth_map, original_images


def mode_save_and_process(args):
    """Mode 3: Save .npy files and immediately process them for categorization."""
    print("=== Mode 3: Save and Process ===")
    
    # First save the raw data (Mode 1)
    mode_save_only(args)
    
    # Then process the saved data (Mode 2)
    return mode_process_only(args)

def mode_full_pipeline(args):
    """
    Mode 4: Complete pipeline without saving intermediates.
    Composed from shared components: _before_saving_results + _generate_records_directly + _after_reading_saved_results
    """
    print("=== Mode 4: Full Pipeline ===")
    
    # Before saving results: Load model components and prepare evaluation setup
    vae, model, diffusion, dataset, loader, evaluation_results_dir, checkpoint_manager = _before_saving_results(args)
    
    # Generate records directly (bypassing disk I/O): Evaluate and create records in memory
    records, ground_truth_map, original_images = _generate_records_directly(args, vae, model, diffusion, loader)
    
    # After reading saved results: Process records and generate evaluation outputs
    metrics, output_dir = _after_reading_saved_results(args, records, ground_truth_map, original_images)
    
    return records

def validate_mode_arguments(args):
    """
    Validate that required arguments are provided for each mode.
    Provides clear error messages to guide users.
    """
    mode = args.mode
    errors = []
    
    # Define required arguments for each mode
    if mode == "save_only":
        # Mode 1: Save .npy files and diff images only
        required_args = [
            ("annotation_dir", "Directory containing annotation files"),
            ("pretrained", "Path to pretrained model (use --pretrained)"),
        ]
        optional_but_recommended = [
            ("model_size", "Model size (UNet_XS, UNet_S, UNet_M, UNet_L, UNet_XL)"),
            ("patch_size", "Center size for model"),
            ("reverse_steps", "Number of reverse steps"),
            ("batch_num", "Number of batches to process"),
            ("append_timestamp", "Append timestamp to output directories (use --append-timestamp)"),
            ("enable_epoch_stats", "Enable detailed epoch-wise statistics (use --enable-epoch-stats)"),
            
            ("debug", "Enable detailed debug logging (use --debug)"),
        ]
        
    elif mode == "process_only":
        # Mode 2: Read existing .npy files and generate categorization results
        required_args = [
            ("annotation_dir", "Directory containing annotation files"),
        ]
        optional_but_recommended = [
            ("results_dir", "Directory containing saved .npy files"),
            ("anomaly_binary_threshold", "Binary threshold for anomaly detection"),
            ("anomaly_pixel_num_threshold", "Pixel number threshold"),
            ("enable_excel_report", "Generate Excel report (use --enable-excel-report)"),
            ("enable_save_image_results", "Save image results (use --enable-save-image-results)"),
            ("enable_save_optional-image-results", "Save optional image results (use --enable-save-optional-image-results)"),
            ("enable_save_whole-image-results", "Save whole image results (use --enable-save-whole-image-results)"),
            ("append_timestamp", "Append timestamp to output directories (use --append-timestamp)"),
            ("enable_epoch_stats", "Enable detailed epoch-wise statistics (use --enable-epoch-stats)"),
            ("debug", "Enable detailed debug logging (use --debug)"),
        ]
        
    elif mode == "save_and_process":
        # Mode 3: Save .npy files and immediately process them
        required_args = [
            ("annotation_dir", "Directory containing annotation files"),
            ("pretrained", "Path to pretrained model (use --pretrained)"),
        ]
        optional_but_recommended = [
            ("model_size", "Model size"),
            ("patch_size", "Center size for model"),
            ("reverse_steps", "Number of reverse steps"),
            ("batch_num", "Number of batches to process"),
            ("anomaly_binary_threshold", "Binary threshold for anomaly detection"),
            ("enable_excel_report", "Generate Excel report (use --enable-excel-report)"),
            ("append_timestamp", "Append timestamp to output directories (use --append-timestamp)"),
            ("enable_epoch_stats", "Enable detailed epoch-wise statistics (use --enable-epoch-stats)"),
            ("debug", "Enable detailed debug logging (use --debug)"),
        ]
        
    elif mode == "full_pipeline":
        # Mode 4: Complete pipeline without saving intermediates
        required_args = [
            ("annotation_dir", "Directory containing annotation files"),
            ("pretrained", "Path to pretrained model (use --pretrained)"),
        ]
        optional_but_recommended = [
            ("model_size", "Model size"),
            ("patch_size", "Center size for model"),
            ("reverse_steps", "Number of reverse steps"),
            ("batch_num", "Number of batches to process"),
            ("anomaly_binary_threshold", "Binary threshold for anomaly detection"),
            ("enable_excel_report", "Generate Excel report (use --enable-excel-report)"),
            ("append_timestamp", "Append timestamp to output directories (use --append-timestamp)"),
            ("enable_epoch_stats", "Enable detailed epoch-wise statistics (use --enable-epoch-stats)"),
            ("debug", "Enable detailed debug logging (use --debug)"),
        ]
    
    # Check required arguments
    for arg_name, description in required_args:
        value = getattr(args, arg_name, None)
        if not value or (isinstance(value, str) and value.strip() == ""):
            errors.append(f"❌ REQUIRED for mode '{mode}': --{arg_name.replace('_', '-')} ({description})")
    
    # Show helpful information about the mode
    mode_descriptions = {
        "save_only": "Save raw .npy files and diff images without processing",
        "process_only": "Process existing .npy files to generate evaluation results", 
        "save_and_process": "Save raw files AND process them immediately",
        "full_pipeline": "Complete evaluation pipeline without saving intermediate files"
    }
    
    if errors:
        print(f"\n🚨 ARGUMENT VALIDATION FAILED for mode: {mode}")
        print(f"📝 Mode description: {mode_descriptions.get(mode, 'Unknown mode')}")
        print(f"\n❌ Missing required arguments:")
        for error in errors:
            print(f"   {error}")
        
        print(f"\n💡 Recommended optional arguments for mode '{mode}':")
        for arg_name, description in optional_but_recommended:
            value = getattr(args, arg_name, None)
            status = "✅" if value else "⚠️"
            print(f"   {status} --{arg_name.replace('_', '-')}: {description}")
        
        print(f"\n📚 Example usage for mode '{mode}':")
        if mode == "save_only":
            print(f"   python evaluate_and_process.py --mode save_only \\")
            print(f"       --annotation-dir path/to/annotations \\")
            print(f"       --pretrained path/to/model.pt \\")
            print(f"       --results-dir ./results \\")
            print(f"       --append-timestamp \\")
            print(f"       --debug")
            
        elif mode == "process_only":
            print(f"   python evaluate_and_process.py --mode process_only \\")
            print(f"       --annotation-dir path/to/annotations \\")
            print(f"       --enable-excel-report \\")
            print(f"       --append-timestamp \\")
            print(f"       --debug")
            print(f"   # Or with explicit results directory:")
            print(f"   python evaluate_and_process.py --mode process_only \\")
            print(f"       --results-dir path/to/saved_results \\")
            print(f"       --annotation-dir path/to/annotations")
            
        elif mode == "save_and_process":
            print(f"   python evaluate_and_process.py --mode save_and_process \\")
            print(f"       --annotation-dir path/to/annotations \\")
            print(f"       --pretrained path/to/model.pt \\")
            print(f"       --enable-excel-report \\")
            print(f"       --append-timestamp \\")
            print(f"       --debug")
            
        elif mode == "full_pipeline":
            print(f"   python evaluate_and_process.py --mode full_pipeline \\")
            print(f"       --annotation-dir path/to/annotations \\")
            print(f"       --pretrained path/to/model.pt \\")
            print(f"       --enable-excel-report \\")
            print(f"       --append-timestamp \\")
            print(f"       --debug")
        
        print(f"\n💭 Need help? Check the script header comments for detailed usage information.")
        return False
    
    # Validation passed
    print(f"✅ Argument validation passed for mode: {mode}")
    print(f"📝 Mode description: {mode_descriptions.get(mode, 'Unknown mode')}")
    return True


def main():
    global DEBUG_ENABLED
    
    parser = argparse.ArgumentParser(description="Combined Evaluation and Processing Script")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["save_only", "process_only", "save_and_process", "full_pipeline"],
        required=False,  # Made optional when using --input-json
        help="""Execution mode:
        save_only: Save .npy files and diff images only (needs: --annotation-dir, --pretrained)
        process_only: Process existing .npy files to generate results (needs: --annotation-dir)
        save_and_process: Save AND process immediately (needs: --annotation-dir, --pretrained)
        full_pipeline: Complete pipeline without saving intermediates (needs: --annotation-dir, --pretrained)
        Note: Can be omitted if specified in --input-json file"""
    )
    
    # Common arguments
    parser.add_argument("--results-dir", type=str, default="./results", 
                       help="Results directory (optional for process_only mode - can be specified in JSON)")
    parser.add_argument("--annotation-dir", type=str, 
                       help="Directory containing annotation files (REQUIRED for all modes)")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size for image processing")
    parser.add_argument("--irregular-patch", action="store_true", help="Use irregular patch for image processing")
    # Model arguments (REQUIRED for modes: save_only, save_and_process, full_pipeline)
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa", "pcb"], default="pcb",
                       help="Dataset type (for model loading modes)")
    parser.add_argument("--model-size", type=str, choices=["UNet_XS", "UNet_S", "UNet_M", "UNet_L", "UNet_XL"], default="UNet_L",
                       help="Model size (for model loading modes)")
    parser.add_argument("--pretrained", type=str, default="", 
                       help="Path to pretrained model (REQUIRED for save_only, save_and_process, full_pipeline)")
    parser.add_argument("--reverse-steps", type=int, default=5,
                       help="Number of reverse steps (for model loading modes)")
    parser.add_argument("--batch-num", type=int, default=12,
                       help="Number of batches to process (for model loading modes)")
    parser.add_argument("--split", type=str, default="test",
                       help="Data split to process (for model loading modes)")
    parser.add_argument("--object-class", type=str, default="all",
                       help="Object class to process (for model loading modes)")
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema",
                       help="VAE type (for model loading modes)")
    parser.add_argument("--force-rerun", action="store_true", 
                       help="Force rerun evaluation (for save_only mode)")
    
    # Processing arguments (used in modes: process_only, save_and_process, full_pipeline)
    parser.add_argument("--anomaly-binary-threshold", type=int, default=5, 
                       help="Binary threshold for anomaly detection (for processing modes)")
    parser.add_argument("--anomaly-pixel-num-threshold", type=int, default=0, 
                       help="Pixel number threshold (for processing modes)")
    parser.add_argument("--adaptive-threshold", type=float, default=0.1, 
                       help="Adaptive threshold for contour-based masks (for processing modes)")
    
    # Output options
    parser.add_argument("--enable-excel-report", action="store_true", help="Generate Excel report")
    parser.add_argument("--enable-save-image-results", action="store_true", help="Save image results")
    parser.add_argument("--enable-save-optional-image-results", action="store_true", help="Save optional image results")
    parser.add_argument("--enable-save-whole-image-results", action="store_true", help="Save whole image results")
    parser.add_argument("--enable-save-json-results", action="store_true", help="Save JSON results")
    parser.add_argument("--enable-confusion-matrix", action="store_true", help="Create confusion matrix")
    parser.add_argument("--enable-epoch-stats", action="store_true", help="Enable detailed epoch-wise statistics collection and printing")
    
    # Input JSON for batch processing
    parser.add_argument("--input-json", type=str, 
                       help="""JSON file with test configurations. Can be used instead of --mode.
                       Example format: {"test_name": {"mode": "save_only", "annotation-dir": "path/to/annotations", ...}}
                       Or simple format: {"mode": "save_only", "annotation-dir": "path/to/annotations", ...}""")
    
    # Timestamp control
    parser.add_argument("--append-timestamp", action="store_true", 
                       help="Append timestamp to output directory names (default: False)")
    
    # Debug control
    parser.add_argument("--debug", action="store_true",
                       help="Enable detailed debug logging during processing (default: False)")
    
    args = parser.parse_args()
    
    # Set global debug flag
    DEBUG_ENABLED = args.debug
    
    # Handle mode extraction from JSON if needed
    if not args.mode and not args.input_json:
        print("❌ Error: Either --mode or --input-json must be provided!")
        print("💡 Use --mode to specify execution mode directly, or --input-json to load configuration from file.")
        sys.exit(1)
    
    # Extract mode from JSON if not provided via command line
    if not args.mode and args.input_json:
        try:
            with open(args.input_json, 'r') as f:
                config_data = json.load(f)
            
            # Look for mode in the JSON structure
            if isinstance(config_data, dict):
                # Check if there's a mode at the top level
                if 'mode' in config_data:
                    args.mode = config_data['mode']
                else:
                    # Check if there's a test configuration with mode
                    first_test = next(iter(config_data.values()))
                    if isinstance(first_test, dict) and 'mode' in first_test:
                        args.mode = first_test['mode']
                    else:
                        print("❌ Error: No 'mode' found in JSON configuration!")
                        print("💡 Add 'mode': 'save_only' (or other mode) to your JSON file.")
                        sys.exit(1)
            else:
                print("❌ Error: JSON configuration must be a dictionary!")
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"❌ Error: JSON file not found: {args.input_json}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON file: {e}")
            sys.exit(1)
    
    # Validate that mode is valid
    if args.mode not in ["save_only", "process_only", "save_and_process", "full_pipeline"]:
        print(f"❌ Error: Invalid mode '{args.mode}'. Must be one of: save_only, process_only, save_and_process, full_pipeline")
        sys.exit(1)
    
    # Handle input JSON if provided (BEFORE validation so JSON values are loaded)
    if args.input_json:
        with open(args.input_json, 'r') as f:
            test_configs = json.load(f)
            
        # Run processing for each test configuration
        for test_name, test_args in test_configs.items():
            print(f"\n📄 Loading configuration for: {test_name}")
            print(f"🔧 Mode: {test_args.get('mode', args.mode)}")
            print(f"📂 Annotation dir: {test_args.get('annotation-dir', 'Not specified')}")
            print(f"🤖 Model: {test_args.get('pretrained', 'Not specified')}")
            print(f"⚙️  Loading {len(test_args)} configuration parameters...")
            
            # Update args with test configuration
            for key, value in test_args.items():
                key = key.replace('-', '_')
                if hasattr(args, key):
                    if key in ['anomaly_binary_threshold', 'anomaly_pixel_num_threshold', 'patch_size', 
                              'patch_size', 'batch_num', 'reverse_steps']:
                        value = int(value)
                    elif key == 'adaptive_threshold':
                        value = float(value)
                    elif key in ['results_dir', 'annotation_dir', 'pretrained']:
                        value = os.path.expanduser(value)
                        # Convert to absolute path if it's relative
                        if not os.path.isabs(value):
                            value = os.path.abspath(value)
                    elif key in ['irregular_patch', 'enable_excel_report', 'enable_save_optional_image_results', 
                               'enable_save_image_results', 'enable_save_json_results', 'enable_save_whole_image_results',
                               'enable_confusion_matrix', 'force_rerun', 'append_timestamp', 'enable_epoch_stats']:
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    elif key == 'debug':
                        DEBUG_ENABLED = value.lower() in ('yes', 'true', 't', 'y', '1') if isinstance(value, str) else bool(value)
                    setattr(args, key, value)
    
    # Validate mode-specific arguments AFTER JSON is loaded
    if not validate_mode_arguments(args):
        print(f"\n❌ Argument validation failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Set up results directory with optional timestamp for modes that need it
    base_name = f"DeCo-Diff_{args.dataset}_{args.object_class}_{args.model_size}_{args.patch_size}"
    
    if args.append_timestamp:
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        if args.input_json:
            # Use the test name from JSON
            test_name = list(json.load(open(args.input_json, 'r')).keys())[0]
            args.results_dir = f"results/{test_name}_{current_time}"
        else:
            args.results_dir = f"results/{base_name}_{current_time}"
    else:
        if args.input_json:
            # Use the test name from JSON without timestamp
            test_name = list(json.load(open(args.input_json, 'r')).keys())[0]
            args.results_dir = f"results/{test_name}"
        else:
            args.results_dir = f"results/{base_name}"
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save config if using JSON
    if args.input_json:
        config_save_path = os.path.join(args.results_dir, "config.json")
        with open(args.input_json, 'r') as f:
            config_data = json.load(f)
        with open(config_save_path, "w") as config_file:
            json.dump(config_data, config_file, indent=2)
    
    # Execute the selected mode
    if args.mode == "save_only":
        mode_save_only(args)
    elif args.mode == "process_only":
        mode_process_only(args)
    elif args.mode == "save_and_process":
        mode_save_and_process(args)
    elif args.mode == "full_pipeline":
        mode_full_pipeline(args)
    
    print(f"Mode {args.mode} completed successfully!")

if __name__ == "__main__":
    main()
