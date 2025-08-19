#!/usr/bin/env python3
"""
Combined Evaluation and Processing Script

This script combines the functionality of evaluation_DeCo_Diff_raw.py and process_raw_data_to_results.py
to provide 4 different execution modes:

1. save_only: Save .npy files and diff images only (no categorization)
2. process_only: Read existing .npy files and generate categorization results
3. save_and_process: Save .npy files and immediately process them for categorization
4. full_pipeline: Complete pipeline without saving intermediates (evaluation to categorization)

IMPORTANT: This script handles patch-level processing:
- The dataset returns individual patches (each __getitem__ returns a single patch)
- The DataLoader naturally batches individual patches together
- Each patch is a 3D tensor with shape [3, patch_size, patch_size]
- The default PyTorch collate function works perfectly for this use case

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
from collections import deque
from tqdm import tqdm
import glob
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
import re
import platform
from concurrent.futures import ThreadPoolExecutor

# Import from evaluation_DeCo_Diff_raw.py
from diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from models import UNET_models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import sys

# Import utility functions
from utils import (
    path_to_safe_filename,
    safe_filename_to_path,
    _binary_mask,
    _to_numpy
)

# Import from process_raw_data_to_results.py
from evaluation_DeCo_Diff2 import (
    make_record, add_metric_fields,
    _get_largest_connected_component_pixels, _create_contour_based_binary_mask_single,
    CheckpointManager, save_image_results_from_records, save_patch_results_from_records,
    determine_image_status, compute_y_true_y_score, compute_metrics_from_y_true_y_score,
    make_excel, plot_accuracy_results, save_perturbation_results, draw_patch_rectangles_on_image,
    EvaluationMetrics
)

# Import from process_raw_data_to_results.py
from process_raw_data_to_results import (
    load_ground_truth_map,
    load_original_images,
    parse_filename_to_info,
    load_raw_data_files,
    reconstruct_records_from_raw_data,
    compute_simple_metrics,
    save_image_results_from_raw_data,
    save_json_results_from_raw_data
)

# Set up device
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("GPU not found. Using CPU instead.")
else:
    # Enable performance knobs for GPU
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        print("GPU not found. Using CPU instead.")
        pass

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

# Helper: create a safe filename component from an image path or string
# Ensures no illegal characters (like ':' on Windows) and provides a fallback
# when input is empty or invalid
import re as _re_for_safe_name

def _safe_filename_component(value: Any) -> str:
    try:
        # Normalize input
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        value = str(value or "")

        # First try provided utility to preserve mapping if available
        if value:
            try:
                safe = path_to_safe_filename(value)
            except Exception:
                safe = ""
        else:
            safe = ""

        # Fallback: sanitize basename or raw value; strip drive letters and illegal chars
        if not safe or safe.strip() == "" or safe == ":":
            base = os.path.basename(value) or value.split(":")[-1]
            safe = _re_for_safe_name.sub(r"_", base)

        # Final cleanup: remove any remaining illegal characters, especially ':'
        safe = safe.replace(":", "_")
        safe = safe.strip("._ ")
        if not safe:
            safe = "unknown"
        return safe
    except Exception:
        return "unknown"

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

        # Build a per-patch index at initialization time.
        # Each entry is a tuple: (resolved_image_path, coords_8_values)
        self.patch_items = []
        self._unique_image_paths = set()

        for annotation_file in self.annotation_files:
            # Load annotation data to get image path
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)

            image_path_raw = annotation_data.get('image_path', '')
            resolved_image_path = self._resolve_image_path(annotation_file, image_path_raw)

            # Open image only to read its size; do not load into memory
            with PILImage.open(resolved_image_path) as img:
                width, height = img.size

            # Compute padded dimensions (divisible by patch_size)
            pad_height = (self.patch_size - (height % self.patch_size)) % self.patch_size
            pad_width = (self.patch_size - (width % self.patch_size)) % self.patch_size
            padded_height = height + pad_height
            padded_width = width + pad_width

            # Generate 8-value coordinates for each patch on the padded image grid
            for y in range(0, padded_height, self.patch_size):
                for x in range(0, padded_width, self.patch_size):
                    # Standard parallel patch (aligned with image axes)
                    x1, y1 = x, y  # Top-left
                    x2, y2 = x + self.patch_size, y  # Top-right
                    x3, y3 = x + self.patch_size, y + self.patch_size  # Bottom-right
                    x4, y4 = x, y + self.patch_size  # Bottom-left
                    coords_8_values = (x1, y1, x2, y2, x3, y3, x4, y4)
                    self.patch_items.append((resolved_image_path, coords_8_values))

            self._unique_image_paths.add(resolved_image_path)

        print(f"Indexed {len(self.patch_items)} patches from {len(self._unique_image_paths)} images")
        
        # Image cache to avoid reloading the same image multiple times
        self._image_cache = {}
        self._cache_size_limit = 10  # Limit cache to prevent memory issues
        
    def __len__(self):
        # Dataset unit is now a single patch
        return len(self.patch_items)
        
    def __getitem__(self, index):
        debug_print(f"!!!!!!🔍 starting __getitem__ index: {index}")

        # Retrieve per-patch unit: (image_path, coords_8_values)
        image_path, coords_8_values = self.patch_items[index]

        # Resolve image path
        resolved_image_path = self._resolve_image_path(None, image_path)
        
        # Get or load image from cache
        image_np = self._get_cached_image(resolved_image_path)

        # Extract patch using all 8 coordinate values
        patch_np = self._extract_patch_from_coords(image_np, coords_8_values)

        if patch_np.shape[0] != self.patch_size or patch_np.shape[1] != self.patch_size:
            raise ValueError(f"Extracted patch has wrong size: {patch_np.shape} at index {index}")

        # Convert to tensor [3, patch_size, patch_size]
        x = self.transform(PILImage.fromarray(patch_np)) if self.transform else transforms.ToTensor()(PILImage.fromarray(patch_np))

        # Sanity check: ensure 3D tensor
        if len(x.shape) == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        if len(x.shape) != 3 or x.shape[0] != 3:
            raise ValueError(f"Dataset[{index}]: Expected 3D tensor with 3 channels, got shape {x.shape}")

        # Dummy segmentation and class per single patch item
        seg = torch.zeros(1, self.patch_size, self.patch_size)
        object_cls = torch.zeros(1, dtype=torch.long)

        # Metadata
        anomaly_classes = "all"
        # Convert patch coordinates to tensor for proper batching
        patch_coords = torch.tensor(coords_8_values, dtype=torch.long)

        debug_print(f"🔍 Dataset[{index}]: Final shapes - x: {x.shape}, seg: {seg.shape}, object_cls: {object_cls.shape}")
        debug_print(f"🔍 Dataset[{index}]: patch_coords shape: {patch_coords.shape}, dtype: {patch_coords.dtype}")
        debug_print(f"!!!!!!🔍 ending __getitem__ index: {index}")
        return x, seg, object_cls, anomaly_classes, image_path, patch_coords

    def _resolve_image_path(self, annotation_file, image_path):
        """Resolve the actual image path, trying common fallbacks if needed."""
        # If valid absolute/relative path is provided and exists
        if image_path and os.path.exists(image_path):
            return image_path

        # If annotation_file is provided, try constructing from its base name
        base_name = None
        if annotation_file is not None:
            base_name = os.path.splitext(os.path.basename(annotation_file))[0]
        else:
            # If we don't have the annotation file, try to parse from provided path
            provided = image_path or ""
            base_name = os.path.splitext(os.path.basename(provided))[0]

        if base_name and base_name.endswith('_annotations'):
            base_name = base_name[:-12]

        candidate_dirs = [os.path.dirname(image_path) if image_path else ""]
        if annotation_file is not None:
            candidate_dirs.append(os.path.dirname(annotation_file))

        for d in candidate_dirs:
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = os.path.join(d, f"{base_name}{ext}") if base_name else ""
                if potential_path and os.path.exists(potential_path):
                    return potential_path

        raise FileNotFoundError(f"Image not found for annotation: {annotation_file if annotation_file else image_path}")

    def get_all_image_paths(self):
        """Return a list of unique image paths indexed by this dataset."""
        return list(self._unique_image_paths)
    
    def _extract_patch_from_coords(self, image_np, coords_8_values):
        """
        Extract patch from image using 8-value coordinates.
        Optimizes for parallel patches (faster) vs non-parallel patches (slower but more flexible).
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = coords_8_values
        
        # Check if patch is parallel to image axes (faster extraction)
        is_parallel = (y1 == y2 and y3 == y4 and x1 == x4 and x2 == x3)
        
        if is_parallel:
            # Fast path: rectangular patch aligned with image axes
            # Use top-left corner and dimensions
            x_min, y_min = int(x1), int(y1)
            x_max, y_max = int(x3), int(y3)
            
            # Ensure coordinates are within image bounds
            height, width = image_np.shape[:2]
            x_min = max(0, min(x_min, width - 1))
            y_min = max(0, min(y_min, height - 1))
            x_max = max(x_min + 1, min(x_max, width))
            y_max = max(y_min + 1, min(y_max, height))
            
            # Extract patch
            patch = image_np[y_min:y_max, x_min:x_max]
            
            # Resize to target patch_size if needed
            if patch.shape[:2] != (self.patch_size, self.patch_size):
                patch_pil = PILImage.fromarray(patch)
                patch_pil = patch_pil.resize((self.patch_size, self.patch_size), PILImage.Resampling.LANCZOS)
                patch = np.array(patch_pil)
            
            debug_print(f"🔧 Fast parallel patch extraction: {patch.shape}")
            return patch
        else:
            # Slow path: non-parallel patch requires perspective transform
            # This is more computationally expensive but handles rotated patches
            debug_print(f"🔧 Slow non-parallel patch extraction (perspective transform)")
            
            # Convert coordinates to numpy arrays for OpenCV
            src_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            dst_points = np.float32([[0, 0], [self.patch_size, 0], 
                                   [self.patch_size, self.patch_size], [0, self.patch_size]])
            
            # Calculate perspective transform matrix
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transform
            patch = cv2.warpPerspective(image_np, transform_matrix, (self.patch_size, self.patch_size))
            
            return patch

    
    def _get_cached_image(self, image_path):
        """
        Get image from cache or load and cache it if not present.
        Implements LRU-style cache management to prevent memory issues.
        """
        if image_path in self._image_cache:
            debug_print(f"🔧 Using cached image: {image_path}")
            return self._image_cache[image_path]
        
        # Load and cache new image
        debug_print(f"🔧 Loading new image into cache: {image_path}")
        image = PILImage.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Pad image to ensure dimensions are divisible by patch_size
        image_np = self._pad_image_to_patch_size(image_np)
        
        # Manage cache size
        if len(self._image_cache) >= self._cache_size_limit:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._image_cache))
            del self._image_cache[oldest_key]
            debug_print(f"🔧 Removed oldest image from cache: {oldest_key}")
        
        # Add to cache
        self._image_cache[image_path] = image_np
        debug_print(f"🔧 Cached image: {image_path} (cache size: {len(self._image_cache)})")
        
        return image_np
    
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
                
                # Debug: Check coordinate types
                debug_print(f"  🔍 Created coordinates: {coords_8_values}")
                debug_print(f"  🔍 Coordinate types: {[type(coord) for coord in coords_8_values]}")
                debug_print(f"  🔍 All integers: {all(isinstance(coord, int) for coord in coords_8_values)}")
                
                patches.append(patch)
                coords.append(coords_8_values)
        
        debug_print(f"  ✅ Extracted {len(patches)} non-overlapping patches from padded image")
        return patches, coords

def _compute_abs_diff_mean(a: torch.Tensor, b: torch.Tensor, diff_scale: float = 1.0) -> torch.Tensor:
    return torch.abs(a - b).mean(dim=1, keepdim=True) * diff_scale

def _compute_abs_diff_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b).max(dim=1, keepdim=True)[0]






def _process_batch_inference(x, object_cls, model, vae, diffusion, reverse_steps, device, epoch_metrics=None):
    """
    Shared inference logic for processing a batch.
    Returns the computed difference tensors.
    """
    debug_print(f"   🔄 Moving {x.size(0)} patches to device: {device}")
    debug_print(f"   🔍 x shape before device move: {x.shape}")
    debug_print(f"   🔍 object_cls shape: {object_cls.shape}")
    
    # Validate tensor dimensions before processing
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D input tensor for VAE, got shape: {x.shape}. Expected: [batch, channels, height, width]")
    
    if x.shape[1] != 3:  # Check channels
        raise ValueError(f"Expected 3 channels (RGB), got {x.shape[1]} channels")
    
    # Move batch to device
    x_device = x.to(device)
    object_cls_device = object_cls.to(device)
    
    debug_print(f"   🔍 x_device shape after device move: {x_device.shape}")
    
    debug_print(f"   🎨 VAE encoding...")
    # Forward pass through VAE encoder (to latent space)
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            encoded = vae.encode(x_device).latent_dist.mean * _LATENT_SCALE
        # Ensure downstream diffusion/model (kept in FP32) receives FP32 tensors
        encoded = encoded.float()
    else:
        encoded = vae.encode(x_device).latent_dist.mean * _LATENT_SCALE
    debug_print(f"   ✅ VAE encoding completed, latent shape: {encoded.shape}")

    # Reverse DDIM sampling conditioned on encoder latents
    # Ensure object_cls has the correct shape and dtype for the model's class embedder
    # The UNet's ClassEmbedder expects Long indices shaped [batch_size, 1] (so embedding -> [batch_size, 1, dim])
    if len(object_cls_device.shape) == 2:
        # Use as-is, expected shape: [batch_size, 1]
        context = object_cls_device
    elif len(object_cls_device.shape) == 1:
        # [batch_size] -> [batch_size, 1]
        context = object_cls_device.unsqueeze(1)
    else:
        # Squeeze/reshape unexpected shapes to [batch_size, 1]
        context = object_cls_device.view(object_cls_device.shape[0], -1)
        if context.shape[1] != 1:
            context = context[:, :1]
    
    debug_print(f"   🔍 Context tensor shape (indices): {context.shape}")
    
    # Validate context tensor shape for embedding: must be 2D [batch_size, 1]
    if len(context.shape) != 2 or context.shape[1] != 1:
        raise ValueError(f"Context indices must be 2D [batch_size, 1], got: {context.shape}")

    # Ensure dtype is long for embedding indices
    if context.dtype != torch.long:
        context = context.long()
    
    # Additional validation: ensure the input tensor has the correct shape for the model
    debug_print(f"   🔍 Input tensor x_device shape: {x_device.shape}")
    debug_print(f"   🔍 Input tensor x_device dtype: {x_device.dtype}")
    debug_print(f"   🔍 Input tensor x_device device: {x_device.device}")
    
    # The model might expect a specific input format
    # Try to ensure the input tensor is in the right format
    if len(x_device.shape) == 4:
        # Standard format: [batch, channels, height, width]
        debug_print(f"   ✅ Input tensor has correct 4D format: {x_device.shape}")
        
        # The model architecture might have specific input requirements
        # Let's check if there's a mismatch between what we're providing and what the model expects
        debug_print(f"   🔍 Model type: {type(model)}")
        debug_print(f"   🔍 Model device: {next(model.parameters()).device}")
        
        # Check if the model has any specific input requirements
        if hasattr(model, 'config'):
            debug_print(f"   🔍 Model config: {model.config}")
        
        # The issue might be that the model expects a different input format
        # Let's try to understand what the model actually expects
        debug_print(f"   🔍 Input tensor shape: {x_device.shape}")
        debug_print(f"   🔍 Context tensor shape: {context.shape}")
    else:
        debug_print(f"   ⚠️  Warning: Input tensor has unexpected shape: {x_device.shape}")
    
    model_kwargs = {"context": context, "mask": None}
    
    debug_print(f"   🔄 Starting DDIM sampling with {reverse_steps} steps...")
    debug_print(f"   🔍 Encoded latent shape: {encoded.shape}")
    debug_print(f"   🔍 Model kwargs: {model_kwargs}")
    
    # Try to catch the error earlier by testing the model with a simple forward pass
    try:
        debug_print(f"   🔍 Testing model forward pass...")
        with torch.no_grad():
            # Create a simple test input with the same shape as encoded
            test_input = torch.randn_like(encoded)
            test_output = model(test_input, torch.zeros(1, device=device), **model_kwargs)
            debug_print(f"   ✅ Model forward pass successful, output shape: {test_output.shape}")
    except Exception as e:
        debug_print(f"   ❌ Model forward pass failed: {e}")
        debug_print(f"   🔍 This suggests the model has input format requirements we're not meeting")
        # Continue anyway to see the full error
    
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
    if torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            image_samples = vae.decode(latent_samples_final / _LATENT_SCALE).sample
            x0 = vae.decode(encoded / _LATENT_SCALE).sample
        # Bring back to FP32 for subsequent ops
        image_samples = image_samples.float()
        x0 = x0.float()
    else:
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


def _validate_and_fix_coordinates(coords_8_values):
    """
    Validate and fix 8-value coordinates to ensure they are single integers.
    
    Args:
        coords_8_values: List, tuple, or array of 8 coordinates
        
    Returns:
        List of 8 integers representing the coordinates
    """
    if not isinstance(coords_8_values, (list, tuple)) or len(coords_8_values) != 8:
        raise ValueError(f"Expected 8-value coordinates, got {type(coords_8_values)} with length {len(coords_8_values) if hasattr(coords_8_values, '__len__') else 'unknown'}")
    
    fixed_coords = []
    for i, coord in enumerate(coords_8_values):
        # Handle torch tensors explicitly
        if 'torch' in globals() and isinstance(coord, torch.Tensor):
            if coord.numel() == 0:
                raise ValueError(f"Coordinate[{i}] is an empty tensor: {coord}")
            fixed_value = int(coord.view(-1)[0].item())
            fixed_coords.append(fixed_value)
            debug_print(f"  🔧 Fixed coordinate[{i}] tensor -> {fixed_value}")
        elif isinstance(coord, (list, tuple)):
            # If coordinate is a list/array, take the first value
            if len(coord) > 0:
                fixed_coords.append(int(coord[0]))
                debug_print(f"  🔧 Fixed coordinate[{i}]: {coord} -> {coord[0]}")
            else:
                raise ValueError(f"Coordinate[{i}] is empty list/array: {coord}")
        else:
            # If coordinate is already a single value, convert to int
            fixed_coords.append(int(coord))
    
    debug_print(f"  ✅ Validated coordinates: {fixed_coords}")
    return fixed_coords

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
    
    # Check for existing checkpoints and resume from where we left off
    start_idx = 0
    if checkpoint_manager is not None:
        try:
            # Get already processed images
            already_processed = checkpoint_manager.get_processed_images()
            debug_print(f"🔍 Found {len(already_processed)} already processed images")
            
            # Determine checkpoint behavior based on command line arguments
            if args.checkpoint_mode == "overwrite":
                debug_print(f"🔍 Checkpoint mode: overwrite - ignoring existing checkpoints")
                start_idx = 0
            elif args.checkpoint_mode == "resume" or (args.checkpoint_mode == "auto" and already_processed and not args.force_rerun):
                debug_print(f"🔍 Checkpoint mode: resume - attempting to resume from existing checkpoint")
                
                # Check if we have a checkpoint file with batch information
                checkpoint_dir = getattr(checkpoint_manager, 'base_checkpoint_dir', None)
                if checkpoint_dir and os.path.exists(checkpoint_dir):
                    # Look for the latest checkpoint file
                    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]
                    if checkpoint_files:
                        # Sort by modification time to get the latest
                        checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                        latest_checkpoint = checkpoint_files[-1]
                        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                        
                        try:
                            with open(checkpoint_path, 'r') as f:
                                checkpoint_data = json.load(f)
                            
                            # Extract the last processed batch index
                            if 'current_image_index' in checkpoint_data:
                                last_processed_batch = checkpoint_data['current_image_index']
                                start_idx = max(0, last_processed_batch)
                                debug_print(f"🔍 Found checkpoint with last processed batch: {last_processed_batch}")
                                debug_print(f"🔍 Resuming from batch index: {start_idx}")
                            else:
                                debug_print(f"🔍 Checkpoint file doesn't contain batch index, estimating...")
                                # Fallback to estimation
                                estimated_batches = len(already_processed) // 16  # Assuming ~16 images per batch
                                start_idx = max(0, estimated_batches - 1)
                                debug_print(f"🔍 Estimated resume from batch index: {start_idx}")
                        except Exception as e:
                            debug_print(f"⚠️  Error reading checkpoint file: {e}, using estimation")
                            estimated_batches = len(already_processed) // 16
                            start_idx = max(0, estimated_batches - 1)
                    else:
                        debug_print(f"🔍 No checkpoint files found, using estimation")
                        estimated_batches = len(already_processed) // 16
                        start_idx = max(0, estimated_batches - 1)
                else:
                    debug_print(f"🔍 No checkpoint directory, using estimation")
                    estimated_batches = len(already_processed) // 16
                    start_idx = max(0, estimated_batches - 1)
                
                debug_print(f"🔍 This will skip approximately {start_idx * 16} already processed images")
            else:
                debug_print(f"🔍 Checkpoint mode: auto - no existing checkpoints found, starting fresh")
        except Exception as e:
            debug_print(f"⚠️  Error checking existing checkpoints: {e}, starting from beginning")
            start_idx = 0
    
    idx = start_idx - 1  # Start from the batch before start_idx since we increment at the beginning
    
    # Setup async saving executor (Windows-safe). Default to 1 worker on Windows.
    async_workers = getattr(args, "async_save_workers", None)
    if async_workers is None:
        # If not provided via CLI or JSON, set default: 1 for Windows, 4 otherwise
        async_workers = 1 if platform.system().lower().startswith("win") else max(2, min(8, os.cpu_count() or 4))
    # Ensure at least 1
    async_workers = max(1, int(async_workers))

    executor = ThreadPoolExecutor(max_workers=async_workers)
    pending = deque()
    # throttle pending tasks to avoid unbounded memory when disk is slow
    # allow more queueing so we can exit the batch quickly and overlap with next batch
    max_pending = max(8, async_workers * 8)

    def _submit_save_job(save_dir_local, base_filename_local,
                         encodedrecon_raw_local, latent_raw_local, anomaly_raw_local,
                         coords_array_local, save_preview_local, save_dtype_is_f16_local):
        def _job():
            # Save npy arrays
            if save_dtype_is_f16_local:
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_encodedrecon.npy"), encodedrecon_raw_local.astype(np.float16))
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_latent.npy"), latent_raw_local.astype(np.float16))
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_anomaly_map_arithmetic.npy"), anomaly_raw_local.astype(np.float16))
            else:
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_encodedrecon.npy"), encodedrecon_raw_local.astype(np.float32))
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_latent.npy"), latent_raw_local.astype(np.float32))
                np.save(os.path.join(save_dir_local, f"{base_filename_local}_anomaly_map_arithmetic.npy"), anomaly_raw_local.astype(np.float32))

            # Save coords
            np.save(os.path.join(save_dir_local, f"{base_filename_local}_coords.npy"), coords_array_local.astype(np.int32))

            # Save previews if requested
            if save_preview_local:
                PILImage.fromarray((encodedrecon_raw_local * 255).astype(np.uint8)).save(os.path.join(save_dir_local, f"{base_filename_local}_encodedrecon.png"))
                PILImage.fromarray((latent_raw_local * 255).astype(np.uint8)).save(os.path.join(save_dir_local, f"{base_filename_local}_latent.png"))
                PILImage.fromarray((anomaly_raw_local * 255).astype(np.uint8)).save(os.path.join(save_dir_local, f"{base_filename_local}_anomaly_map_arithmetic.png"))

        # throttle pending jobs
        while len(pending) >= max_pending:
            fut = pending.popleft()
            fut.result()
        fut = executor.submit(_job)
        pending.append(fut)

    try:
        for idx, (x, seg, object_cls, anomaly_classes, image_path, patch_coords) in enumerate(
            tqdm(dataloader, desc=f"{split} split")
        ):
            if idx >= batch_num:
                break

            # Skip batches that have already been processed
            if idx < start_idx:
                debug_print(f"⏭️  Skipping batch {idx+1} (already processed)")
                continue

            # Additional check: see if output files already exist for this batch
            # This provides a more direct way to determine if we should skip processing
            batch_already_processed = False
            if checkpoint_manager is not None:
                try:
                    # Check if any output files exist for this batch
                    # We'll check the first few patches to see if they're already processed
                    sample_patch_count = min(3, x.size(0))  # Check first 3 patches
                    for b in range(sample_patch_count):
                        if b < len(image_path):
                            sample_image_path = image_path[b] if isinstance(image_path[b], str) else str(image_path[b])
                            file_info = path_to_safe_filename(sample_image_path)
                            
                            # Check if the output files exist
                            base_filename = f"{file_info}__minimal_diff"
                            encodedrecon_file = os.path.join(save_dir, f"{base_filename}_encodedrecon.npy")
                            latent_file = os.path.join(save_dir, f"{base_filename}_latent.npy")
                            anomaly_file = os.path.join(save_dir, f"{base_filename}_anomaly_map_arithmetic.npy")
                            
                            if os.path.exists(encodedrecon_file) and os.path.exists(latent_file) and os.path.exists(anomaly_file):
                                batch_already_processed = True
                                debug_print(f"🔍 Batch {idx+1} appears to be already processed (output files exist)")
                                break
                except Exception as e:
                    debug_print(f"⚠️  Error checking existing output files: {e}")
            
            if batch_already_processed and not args.force_rerun:
                debug_print(f"⏭️  Skipping batch {idx+1} (output files already exist)")
                continue

            debug_print(f"🔄 Processing batch {idx+1}/{min(batch_num, len(dataloader))}")
            debug_print(f"📊 Batch size: {x.size(0)} patches")
            debug_print(f"🖼️  Patch shape: {x.shape}")
            debug_print(f"🏷️  Object classes shape: {object_cls.shape}")
            
            # Validate tensor dimensions before processing
            if len(x.shape) != 4:
                raise ValueError(f"Expected 4D input tensor, got shape: {x.shape}. Expected: [batch, channels, height, width]")
            
            if x.shape[1] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {x.shape[1]} channels")
            
            debug_print(f"✅ Tensor shape validation passed")
            
            with torch.no_grad():
                debug_print(f"🧠 Starting inference...")
                
                # Final validation: ensure tensor is 4D before inference
                if len(x.shape) != 4:
                    debug_print(f"⚠️  CRITICAL: Tensor has wrong shape {x.shape} before inference, attempting to fix")
                    if len(x.shape) == 5:
                        # Flatten first two dimensions
                        batch_size = x.shape[0] * x.shape[1]
                        x = x.view(batch_size, *x.shape[2:])
                        debug_print(f"🔧 Fixed 5D tensor to: {x.shape}")
                    elif len(x.shape) == 3:
                        # Add batch dimension
                        x = x.unsqueeze(0)
                        debug_print(f"🔧 Added batch dimension to 3D tensor: {x.shape}")
                    else:
                        raise ValueError(f"Cannot fix tensor with shape {x.shape}")
                
                # Validate final shape
                if len(x.shape) != 4:
                    raise ValueError(f"Failed to create 4D tensor before inference. Final shape: {x.shape}")
                
                if x.shape[1] != 3:
                    raise ValueError(f"Expected 3 channels (RGB), got {x.shape[1]} channels")
                
                debug_print(f"✅ Tensor validated before inference: {x.shape}")
                
                # Direct inference on the batch (no chunking needed since dataset returns individual patches)
                encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic = _process_batch_inference(
                    x, object_cls, model, vae, diffusion, reverse_steps, device, epoch_metrics=epoch_metrics
                )
                debug_print(f"✅ Inference completed")
            
            # ---------------------------------------------------------------------
            # Per‑sample aggregation and async saving (Windows-safe threading)
            # ---------------------------------------------------------------------
            batch_size = x.size(0)
            save_f16 = getattr(args, "save_npy_dtype", "float16") == "float16"
            save_preview = getattr(args, "save_preview_images", False)
            
            for b in range(batch_size):
                # Get the image path directly from the batch
                if b < len(image_path):
                    if isinstance(image_path[b], str):
                        current_image_path = image_path[b]
                    elif isinstance(image_path[b], (list, tuple)):
                        current_image_path = image_path[b][0] if image_path[b] else ""
                    else:
                        current_image_path = str(image_path[b])
                else:
                    current_image_path = str(image_path[-1]) if image_path else ""
                
                # Get the 8-value coordinates directly from the batch
                if isinstance(patch_coords, torch.Tensor) and len(patch_coords.shape) == 2:
                    if b < patch_coords.size(0):
                        coords_8_values = patch_coords[b].tolist()  # Get [8] tensor for this batch item
                    else:
                        coords_8_values = patch_coords[-1].tolist() if patch_coords.size(0) > 0 else [0, 0, args.patch_size, 0, args.patch_size, args.patch_size, 0, args.patch_size]
                else:
                    raise ValueError(f"Expected patch_coords to be [batch_size, 8] tensor, got {type(patch_coords)} with shape {patch_coords.shape if hasattr(patch_coords, 'shape') else 'unknown'}")
                
                # Extract individual coordinates from the 8-value format
                x1, y1, x2, y2, x3, y3, x4, y4 = coords_8_values
                
                # Debug: Log the image path extraction
                debug_print(f"🔍 Patch {b}: current_image_path = '{current_image_path}'")
                
                # Ensure we have a valid file_info, fallback to "unknown" if empty
                file_info = _safe_filename_component(current_image_path)
                if file_info == "unknown":
                    debug_print(f"⚠️  Warning: Empty/invalid image path for patch {b}, using fallback filename")
                
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

                # Prepare coords as numpy and submit async save
                validated_coords = _validate_and_fix_coordinates(coords_8_values)
                debug_print(f"🔍 Saving coordinates for patch {b}: {validated_coords}")
                debug_print(f"  Coordinate types: {[type(coord) for coord in validated_coords]}")
                debug_print(f"  All integers: {all(isinstance(coord, int) for coord in validated_coords)}")
                patch_coords_array = np.array(validated_coords, dtype=np.int32)
                if len(patch_coords_array.shape) > 1:
                    debug_print(f"  ⚠️  Warning: Coordinates have unexpected shape {patch_coords_array.shape}, flattening...")
                    patch_coords_array = patch_coords_array.flatten()
                    debug_print(f"  Flattened shape: {patch_coords_array.shape}")

                _submit_save_job(
                    save_dir,
                    base_filename,
                    encodedrecon_raw,
                    latent_raw,
                    anomaly_map_arithmetic_raw,
                    patch_coords_array,
                    save_preview,
                    save_f16,
                )
            
            # After finishing all patches for this image, update checkpoint
            if checkpoint_manager is not None:
                # Only save checkpoint at specified intervals to avoid excessive I/O
                if (idx + 1) % args.checkpoint_interval == 0:
                    debug_print(f"🔍 Saving checkpoint for batch {idx + 1}")
                    try:
                        # Collect valid image paths (deduplicated)
                        processed_image_paths = []
                        for p in image_path:
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
                        debug_print(f"🔍 ✅ Checkpoint saved successfully for batch {idx + 1}")
                    except Exception as e:
                        print(f"Warning: failed to save checkpoint after image {idx + 1}: {e}")
                        debug_print(f"🔍 ❌ Checkpoint save failed: {e}")
                else:
                    debug_print(f"🔍 Skipping checkpoint save for batch {idx + 1} (interval: {args.checkpoint_interval})")

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
    finally:
        # Drain pending save jobs and shutdown executor
        while pending:
            fut = pending.popleft()
            try:
                fut.result()
            except Exception as e:
                debug_print(f"❌ Error in async save job: {e}")
        executor.shutdown(wait=True)

    # Print summary of processing
    total_batches = min(batch_num, len(dataloader))
    processed_batches = idx + 1 - start_idx
    skipped_batches = start_idx
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total batches in dataset: {len(dataloader)}")
    print(f"   Target batches to process: {batch_num}")
    print(f"   Batches skipped (already processed): {skipped_batches}")
    print(f"   Batches newly processed: {processed_batches}")
    print(f"   Total batches handled: {skipped_batches + processed_batches}")
    
    # Save final checkpoint
    if checkpoint_manager is not None:
        try:
            debug_print(f"🔍 Saving final checkpoint after completion")
            final_processed = checkpoint_manager.get_processed_images()
            checkpoint_manager.save_checkpoint(
                current_image_index=idx + 1,
                processed_images=final_processed,
            )
            debug_print(f"🔍 ✅ Final checkpoint saved successfully")
            print(f"   Total images processed (including previous runs): {len(final_processed)}")
        except Exception as e:
            print(f"   Could not save final checkpoint: {e}")
            debug_print(f"🔍 ❌ Final checkpoint save failed: {e}")

    print(f"Raw data saving completed. Results saved to {save_dir}")
    
    # Print epoch statistics if enabled
    if epoch_metrics is not None:
        epoch_metrics.print_epoch_stats()
    else:
        debug_print("Skipping epoch statistics (disabled)")


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
    
    # Simply count the status values directly from the records instead of re-deriving them
    for record in records:
        status = record["status"][1]
        if status == "TP":
            all_TP += 1
        elif status == "FP":
            all_FP += 1
        elif status == "FN":
            all_FN += 1
        elif status == "TN":
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

def save_all_records_json(records: List[Record], output_dir: str, filename: str = "all_records.json", patch_size: int = 128, sort_records: bool = True) -> None:
    """
    Save all records in a single comprehensive JSON file.
    
    Args:
        records: List of all evaluation records
        output_dir: Directory to save the JSON file
        filename: Name of the output JSON file (default: "all_records.json")
        patch_size: Size of patches for grid coordinate calculation
        sort_records: Whether to sort records by anomaly_pixels (default: True)
    """
    print(f"Saving all {len(records)} records to comprehensive JSON...")
    
    # Sort records by anomaly_pixels from largest to smallest if requested
    if sort_records:
        print(f"🔍 Sorting records by anomaly_pixels (largest to smallest)...")
        sorted_records = sorted(records, key=lambda record: record.get("anomaly_pixels", [None, 0])[1], reverse=True)
        print(f"✅ Records sorted successfully")
    else:
        print(f"📊 Keeping records in original order (no sorting)")
        sorted_records = records
    
    # Convert records to a JSON-serializable format
    all_records_data = {
        "total_records": len(records),
        "records": []
    }
    
    for i, record in enumerate(sorted_records):
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
    
    # Add sorting information to the header
    if sort_records:
        all_records_data["sorting_info"] = {
            "sorted_by": "anomaly_pixels",
            "order": "descending (largest to smallest)",
            "note": "Records are sorted by anomaly_pixels value for easier analysis of most anomalous patches"
        }
    else:
        all_records_data["sorting_info"] = {
            "sorted_by": "none",
            "order": "original order",
            "note": "Records are kept in their original order as processed"
        }
    
    # Save to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(all_records_data, f, indent=2)
    
    print(f"All records saved to: {output_path}")
    print(f"Total records: {len(records)}")
    
    if sort_records:
        print(f"📊 Records sorted by anomaly_pixels (largest to smallest)")
        
        # Show some statistics about the sorting
        if records:
            anomaly_pixels_values = [r.get("anomaly_pixels", [None, 0])[1] for r in records if "anomaly_pixels" in r]
            if anomaly_pixels_values:
                max_pixels = max(anomaly_pixels_values)
                min_pixels = min(anomaly_pixels_values)
                avg_pixels = sum(anomaly_pixels_values) / len(anomaly_pixels_values)
                print(f"   Max anomaly_pixels: {max_pixels}")
                print(f"   Min anomaly_pixels: {min_pixels}")
                print(f"   Avg anomaly_pixels: {avg_pixels:.1f}")
    else:
        print(f"📊 Records kept in original order (no sorting)")
    
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
    
    # Debug: Check model configuration
    debug_print(f"🔍 Model loaded successfully")
    debug_print(f"🔍 Model type: {type(model)}")
    debug_print(f"🔍 Model device: {next(model.parameters()).device}")
    
    # Check if the model has any specific input requirements
    if hasattr(model, 'config'):
        debug_print(f"🔍 Model config: {model.config}")
    
    # Check the model's expected input format
    debug_print(f"🔍 Model latent_size: {latent_size}")
    debug_print(f"🔍 Model patch_size: {args.patch_size}")
    
    # Check if there's a mismatch between the model architecture and input
    if hasattr(model, 'image_size'):
        debug_print(f"🔍 Model expected image_size: {model.image_size}")
    
    if hasattr(model, 'in_channels'):
        debug_print(f"🔍 Model expected in_channels: {model.in_channels}")
    
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

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    if args.num_workers and args.num_workers > 0:
        dataloader_kwargs.update({
            "prefetch_factor": 2,
            "persistent_workers": True,
        })

    loader = DataLoader(**dataloader_kwargs)

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
        object_class=args.object_class
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
    
    if args.enable_confusion_matrix:
        create_confusion_matrix_from_records(
            records,
            output_dir,
            annotation_dir=args.annotation_dir,
            patch_size=args.patch_size
        )
    # Save results in various formats
    if args.enable_save_json_results:
        debug_print("💾 Saving JSON results...")
        debug_print("📄 Saving comprehensive JSON with all records...")
        save_all_records_json(
            records,
            output_dir,
            filename="all_evaluation_records.json",
            patch_size=args.patch_size,
            sort_records=not args.no_sort_records_by_anomaly
        )

    if args.enable_save_image_results or args.enable_save_whole_image_results:
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

            # Patch-level image save per patch (same as in evaluation_DeCo_Diff2)
            if args.enable_save_image_results:
                for rec in image_records:
                    coords = rec.get("patch_coords", (None, []))[1]
                    if not (isinstance(coords, (list, tuple)) and len(coords) == 8):
                        continue
                    patch_x, patch_y = int(coords[0]), int(coords[1])
                    # Build a single-patch record list
                    patch_records = [rec]
                    # Compute predicted set for this single patch
                    anomaly_map = rec["anomaly_map_arithmetic_binary"][1]
                    anomaly_pixels = int(np.sum(anomaly_map)) if hasattr(anomaly_map, 'sum') else 0
                    patch_pred_set = set()
                    if anomaly_pixels > 0:
                        grid_row = patch_y // args.patch_size
                        grid_col = patch_x // args.patch_size
                        patch_pred_set.add((grid_row, grid_col))
                    try:
                        save_patch_results_from_records(
                            checkpoint_manager,
                            img_path,
                            patch_records,
                            patch_pred_set,
                            ground_truth_defective,
                            overlapping,
                            enable_save_optional_image_results=args.enable_save_optional_image_results,
                            patch_size=args.patch_size,
                            patch_x=patch_x,
                            patch_y=patch_y,
                        )
                    except Exception as e:
                        debug_print(f"⚠️  Failed to save patch-level result for {img_path}: {e}")

            # Image-level image save (only when explicitly enabled)
            if args.enable_save_whole_image_results:
                try:
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
                except Exception as e:
                    debug_print(f"⚠️  Failed to save image-level results for {img_path}: {e}")
    
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
    
    # Extract unique image paths from dataset (dataset is patch-level now)
    image_paths = set(loader.dataset.get_all_image_paths())
    
    # Load original images
    original_images = load_original_images(image_paths)
    print(f"Loaded {len(original_images)} original images")
    
    # Process data directly without saving intermediates
    records = []
    
    idx = -1
    try:
        for idx, (x, seg, object_cls, anomaly_classes, image_paths_batch, patch_coords) in enumerate(
            tqdm(loader, desc="Processing patches")
        ):
            if idx >= args.batch_num:
                break
            debug_print(f"!!!!!!🔍 idx: {idx}")
            debug_print(f"!!!!!!🔍 Input tensor x shape: {x.shape}")
            debug_print(f"!!!!!!🔍 Input tensor x type: {type(x)}")
            debug_print(f"!!!!!!🔍 Input tensor x dtype: {x.dtype}")
            debug_print(f"!!!!!!🔍 Input tensor x device: {x.device}")
            with torch.no_grad():
                # Direct inference on the batch (no chunking needed since dataset returns individual patches)
                encodedrecon_dodrecon_diff, encoded_latent_diff_resized, anomaly_map_arithmetic = _process_batch_inference(
                    x, object_cls, model, vae, diffusion, args.reverse_steps, device, epoch_metrics=epoch_metrics
                )
            
            # Process each patch to create records
            batch_size = x.size(0)
            
            for b in range(batch_size):
                # Get the image path directly from the batch
                if b < len(image_paths_batch):
                    if isinstance(image_paths_batch[b], str):
                        current_image_path = image_paths_batch[b]
                    elif isinstance(image_paths_batch[b], (list, tuple)):
                        current_image_path = image_paths_batch[b][0] if image_paths_batch[b] else ""
                    else:
                        current_image_path = str(image_paths_batch[b])
                else:
                    current_image_path = str(image_paths_batch[-1]) if image_paths_batch else ""
                
                # Extract x_coord and y_coord from the patch coordinates
                if isinstance(patch_coords, torch.Tensor) and len(patch_coords.shape) == 2:
                    if b < patch_coords.size(0):
                        coords_8_values = patch_coords[b].tolist()
                    else:
                        coords_8_values = patch_coords[-1].tolist() if patch_coords.size(0) > 0 else [0, 0, args.patch_size, 0, args.patch_size, args.patch_size, 0, args.patch_size]
                else:
                    raise ValueError(f"Expected patch_coords to be [batch_size, 8] tensor, got {type(patch_coords)} with shape {patch_coords.shape if hasattr(patch_coords, 'shape') else 'unknown'}")
                
                # Extract x_coord and y_coord from the first two values (top-left corner)
                x_coord, y_coord = coords_8_values[0], coords_8_values[1]
                
                # We already have coords_8_values from above
                debug_print(f"  ✅ Using 8-value coordinates: {coords_8_values}")
                
                # Convert tensors to numpy
                encodedrecon_raw = _to_numpy(encodedrecon_dodrecon_diff[b]).squeeze()
                latent_raw = _to_numpy(encoded_latent_diff_resized[b]).squeeze()
                anomaly_map_arithmetic_raw = _to_numpy(anomaly_map_arithmetic[b]).squeeze()
                
                # Convert to torch tensors for processing
                anomaly_map_arithmetic_tensor = torch.from_numpy(anomaly_map_arithmetic_raw).float().unsqueeze(0).unsqueeze(0)
                
                # Create binary mask
                anomaly_map_arithmetic_binary = _binary_mask(
                    anomaly_map_arithmetic_tensor, 
                    args.anomaly_binary_threshold
                )
                #anomaly_map_arithmetic_binary = _binary_mask_exclude_boundary3(
                #    anomaly_map_arithmetic_tensor, 
                #    args.anomaly_binary_threshold, 
                #    visualize=False, 
                #    debug=False, 
                #    filename=image_path
                #)
                
                # Calculate metrics
                anomaly_max = int(round(anomaly_map_arithmetic_tensor.max().item() * 255))
                
                # Get actual patch dimensions for consistent cropping
                if current_image_path in original_images:
                    original_image = original_images[current_image_path]
                    h, w = original_image.shape[:2]
                    actual_patch_height = min(args.patch_size, h - y_coord)
                    actual_patch_width = min(args.patch_size, w - x_coord)
                    
                    # Crop the binary mask tensor to match the actual patch size
                    anomaly_binary_cropped = anomaly_map_arithmetic_binary[:, :, :actual_patch_height, :actual_patch_width]
                    anomaly_pixels = torch.sum(anomaly_binary_cropped).item()
                    is_predicted_defective = anomaly_pixels > args.anomaly_pixel_num_threshold
                    
                    # Get ground truth defective patches for this image
                    ground_truth_defective = ground_truth_map.get(current_image_path, set()) if ground_truth_map else set()
                    
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
                    debug_print(f"🔍 Creating record with patch_coords_8_values: {coords_8_values}")
                    record = make_record(
                        split=("meta", args.split),
                        image_path=("meta", current_image_path),
                        image_path_original=("meta", path_to_safe_filename(current_image_path)),
                        anomaly_class=("meta", "all"),
                        patch_coords=("meta", coords_8_values),
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
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for processing (for model loading modes)")
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
    parser.add_argument("--no-sort-records-by-anomaly", action="store_true", help="Disable sorting records by anomaly_pixels (default: sorted by anomaly_pixels, largest to smallest)")
    parser.add_argument("--save-preview-images", action="store_true", help="Save PNG previews; disable to reduce I/O overhead in save_only mode")
    parser.add_argument("--save-npy-dtype", type=str, choices=["float16", "float32"], default="float16",
                        help="Data type for saved .npy arrays (float16 reduces disk I/O, float32 for full precision)")
    
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
    
    # Checkpoint control
    parser.add_argument("--checkpoint-mode", type=str, choices=["auto", "resume", "overwrite"], default="auto",
                       help="Checkpoint behavior: auto=resume if possible, resume=force resume, overwrite=ignore existing checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                       help="Save checkpoint every N batches (default: 1)")
    # Performance knobs

    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader workers (Windows often needs 0; try >0 if stable)")
    parser.add_argument("--async-save-workers", type=int, default=4,
                        help="Number of async saving threads. Default: 4 on Windows, 2-8 elsewhere.")
    
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
                              'patch_size', 'batch_num', 'batch_size', 'reverse_steps', 'async_save_workers']:
                        value = int(value)
                    elif key in ['adaptive_threshold']:
                        value = float(value)
                    elif key in ['results_dir', 'annotation_dir', 'pretrained']:
                        value = os.path.expanduser(value)
                        # Convert to absolute path if it's relative
                        if not os.path.isabs(value):
                            value = os.path.abspath(value)
                    elif key in ['irregular_patch', 'enable_excel_report', 'enable_save_optional_image_results', 
                               'enable_save_image_results', 'enable_save_json_results', 'enable_save_whole_image_results',
                               'enable_confusion_matrix', 'force_rerun', 'append_timestamp', 'enable_epoch_stats', 'save_preview_images']:
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    elif key in ['save_npy_dtype']:
                        value = value.lower() in ('float16', 'float32')
                    elif key == 'debug':
                        DEBUG_ENABLED = value.lower() in ('yes', 'true', 't', 'y', '1') if isinstance(value, str) else bool(value)
                    setattr(args, key, value)
    
    # Validate mode-specific arguments AFTER JSON is loaded
    if not validate_mode_arguments(args):
        print(f"\n❌ Argument validation failed. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Debug: Print final argument values
    debug_print(f"🔍 Final argument values:")
    debug_print(f"   mode: {args.mode}")
    debug_print(f"   irregular_patch: {args.irregular_patch}")
    debug_print(f"   batch_num: {args.batch_num}")
    debug_print(f"   patch_size: {args.patch_size}")
    debug_print(f"   annotation_dir: {args.annotation_dir}")
    debug_print(f"   pretrained: {args.pretrained}")
    
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

def _get_memory_usage():
    """Get current memory usage for monitoring."""
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Total GPU memory in GB
            return allocated, reserved, total
        except Exception:
            return 0, 0, 0
    return 0, 0, 0



if __name__ == "__main__":
    main()
