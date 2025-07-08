# %%
from __future__ import annotations
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import warnings
warnings.filterwarnings(
    "ignore",
    message="A new version of Albumentations is available.*",
    category=UserWarning
)
from datetime import datetime
import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
import torch.nn.functional as F

from glob import glob

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from PCBDataLoader import PCBDataset
from scipy.ndimage import gaussian_filter

from anomalib import metrics
from sklearn.metrics import average_precision_score
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

import sys
from typing import List
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage
from typing import Sequence

from io import BytesIO
from pathlib import Path

from typing import Any, Tuple, cast

from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity as _lpips,
    structural_similarity_index_measure as _ssim,
)
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
from utils import path_to_safe_filename

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("GPU not found. Using CPU instead.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LATENT_SCALE = 0.18215

Kinded = Tuple[str, Any]  # (kind, value)
Record = OrderedDict[str, Kinded]

# ---------------------------------------------------------------------------
# Base Classes for Reducing Boilerplate
# ---------------------------------------------------------------------------

class BaseEvaluator:
    """Base class for evaluation operations to reduce boilerplate code."""
    
    def __init__(self, args, diffusion, model, vae, device=None):
        self.args = args
        self.diffusion = diffusion
        self.model = model
        self.vae = vae
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = self._get_transform()
        self.results_dir = args.results_dir
        
    def _get_transform(self):
        """Get standard transform for all datasets."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    
    def _get_dataloader(self, dataset, batch_size=None, shuffle=False):
        """Get standard dataloader configuration."""
        batch_size = batch_size or getattr(self.args, 'batch_size', 64)
        
        # Memory optimization: Reduce batch size for large datasets
        if hasattr(self.args, 'memory_optimization') and self.args.memory_optimization:
            batch_size = min(batch_size, 1)  # Force batch size of 1 for memory optimization
            print(f"Memory optimization enabled: using batch_size={batch_size}")
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=2 if hasattr(self.args, 'memory_optimization') and self.args.memory_optimization else 4, 
            drop_last=False
        )
    
    def _create_output_dirs(self):
        """Create standard output directories."""
        dirs = {
            'marked_images': os.path.join(self.results_dir, "marked_images"),
            'evaluation_results': os.path.join(self.results_dir, "evaluation_results"),
            'accuracy_plots': os.path.join(self.results_dir, "accuracy_vs_param")
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        return dirs
    
    def _save_image(self, img, path, description=""):
        """Save image to path."""
        import os
        from PIL import Image
        import numpy as np
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            # Handle different array formats
            if img.dtype == np.uint8:
                pil_img = Image.fromarray(img)
            else:
                # Convert float arrays to uint8
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                pil_img = Image.fromarray(img)
        elif hasattr(img, 'save'):
            # Already a PIL Image
            pil_img = img
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        pil_img.save(path)
        print(f"Saved {description}: {path}")
    
    def _save_json(self, data, path, description=""):
        """Save JSON with standard error handling."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            if description:
                print(f"Saved {description}: {path}")
        except Exception as e:
            print(f"Error saving {description} to {path}: {e}")

class ImageProcessor:
    """Utility class for common image processing operations."""
    
    @staticmethod
    def create_anomaly_map_image(anomaly_map, is_binary=True, patch_size=128, add_grid=True, grid_color=(255, 255, 255), grid_thickness=1, patch_results=None, ground_truth_patches=None):
        """Create anomaly map visualization with optional grid overlay and patch prediction rectangles."""
        if is_binary:
            # Binary map: 0 or 1 - create custom red colormap
            anomaly_map_img = (anomaly_map * 255).astype(np.uint8)
            # Create custom colormap: 0 -> transparent (black), 255 -> pure red
            h, w = anomaly_map_img.shape
            anomaly_map_colored_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            # Set red channel based on anomaly values (0 = transparent/black, 255 = pure red)
            anomaly_map_colored_bgr[:, :, 2] = anomaly_map_img  # Red channel (BGR format)
        else:
            anomaly_map_img = (anomaly_map * 255).astype(np.uint8)
            anomaly_map_colored_bgr = cv2.applyColorMap(anomaly_map_img, cv2.COLORMAP_HOT)

        # Add grid overlay if requested
        if add_grid:
            h, w = anomaly_map.shape
            
            # Draw vertical grid lines
            for x in range(patch_size, w, patch_size):
                cv2.line(anomaly_map_colored_bgr, (x, 0), (x, h), grid_color, grid_thickness)
            
            # Draw horizontal grid lines
            for y in range(patch_size, h, patch_size):
                cv2.line(anomaly_map_colored_bgr, (0, y), (w, y), grid_color, grid_thickness)
            
            # Draw border around the entire image
            cv2.rectangle(anomaly_map_colored_bgr, (0, 0), (w-1, h-1), grid_color, grid_thickness)
        
        # Convert BGR to RGB for proper display
        anomaly_map_colored = cv2.cvtColor(anomaly_map_colored_bgr, cv2.COLOR_BGR2RGB)
        
        # Add patch prediction rectangles if provided
        if patch_results is not None:
            anomaly_map_colored = draw_patch_rectangles_on_image(
                anomaly_map_colored, 
                patch_results, 
                ground_truth_patches, 
                patch_size=patch_size, 
                stride=patch_size, 
                grid_thickness=grid_thickness
            )
        
        return anomaly_map_colored
    
    @staticmethod
    def create_anomaly_overlay(original_img, anomaly_map, alpha=0.6, is_binary=True):
        """Create overlay of anomaly map on original image."""
        if is_binary:
            # Binary overlay using pure red
            overlay = original_img.copy()
            mask = anomaly_map > 0
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
            overlay = cv2.addWeighted(original_img, 1-alpha, anomaly_colored, alpha, 0)
        
        return overlay.astype(np.uint8)
    


class EvaluationMetrics:
    """Utility class for computing and storing evaluation metrics with memory-efficient approach."""
    
    def __init__(self):
        # Use running statistics instead of storing all values
        self.epoch_stats = {
            'encodedrecon_values': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'min': float('inf'), 'max': float('-inf'), 'values': []},
            'latent_values': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'min': float('inf'), 'max': float('-inf'), 'values': []},
            'anomaly_map_arithmetic_values': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'min': float('inf'), 'max': float('-inf'), 'values': []},
            'anomaly_map_geometric_values': {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'min': float('inf'), 'max': float('-inf'), 'values': []}
        }
        # Store histogram bins for distribution analysis
        self.hist_bins = np.arange(0, 0.4, 0.01)
        self.histograms = {
            'encodedrecon_values': np.zeros(len(self.hist_bins) - 1, dtype=np.int64),
            'latent_values': np.zeros(len(self.hist_bins) - 1, dtype=np.int64),
            'anomaly_map_arithmetic_values': np.zeros(len(self.hist_bins) - 1, dtype=np.int64),
            'anomaly_map_geometric_values': np.zeros(len(self.hist_bins) - 1, dtype=np.int64)
        }
    
    def add_batch_stats(self, encodedrecon_diff, latent_diff, anomaly_map_arithmetic, anomaly_map_geometric):
        """Add batch statistics to epoch collection using running statistics."""
        # Process encodedrecon_diff
        encodedrecon_flat = encodedrecon_diff.flatten().cpu().numpy()
        self._update_running_stats('encodedrecon_values', encodedrecon_flat)
        self._update_histogram('encodedrecon_values', encodedrecon_flat)
        
        # Process latent_diff
        latent_flat = latent_diff.flatten().cpu().numpy()
        self._update_running_stats('latent_values', latent_flat)
        self._update_histogram('latent_values', latent_flat)

        # Process anomaly_map_arithmetic
        anomaly_map_arithmetic_flat = anomaly_map_arithmetic.flatten().cpu().numpy()
        self._update_running_stats('anomaly_map_arithmetic_values', anomaly_map_arithmetic_flat)
        self._update_histogram('anomaly_map_arithmetic_values', anomaly_map_arithmetic_flat)
        
        # Process anomaly_map_geometric
        anomaly_map_geometric_flat = anomaly_map_geometric.flatten().cpu().numpy()
        self._update_running_stats('anomaly_map_geometric_values', anomaly_map_geometric_flat)
        self._update_histogram('anomaly_map_geometric_values', anomaly_map_geometric_flat)

    
    def _update_running_stats(self, key, values):
        """Update running statistics for a given key."""
        stats = self.epoch_stats[key]
        n = len(values)
        
        if n == 0:
            return
            
        # Update count
        stats['count'] += n
        
        # Update sum and sum of squares
        stats['sum'] += np.sum(values)
        stats['sum_sq'] += np.sum(values ** 2)
        
        # Update min and max
        stats['min'] = min(stats['min'], np.min(values))
        stats['max'] = max(stats['max'], np.max(values))
        
        stats['values'].extend(values.tolist())
        ## Store values for quantile calculation (memory efficient: only store every 10th value for large datasets)
        #if stats['count'] <= 10000:  # Store all values if dataset is small
        #    stats['values'].extend(values.tolist())
        #else:
        #    # For large datasets, sample every 10th value to keep memory usage reasonable
        #    if 'sampling_counter' not in stats:
        #        stats['sampling_counter'] = 0
        #    stats['sampling_counter'] += 1
        #    if stats['sampling_counter'] % 10 == 0:
        #        stats['values'].extend(values[::10].tolist())
    
    def _update_histogram(self, key, values):
        """Update histogram for a given key."""
        hist, _ = np.histogram(values, bins=self.hist_bins, range=(0, 3))
        self.histograms[key] += hist
    
    def print_epoch_stats(self):
        """Print epoch-wise statistics."""
        print(f'\n=== EPOCH-WISE STATISTICS ===')
        
        for name, stats in self.epoch_stats.items():
            if stats['count'] > 0:
                # Calculate statistics from running values
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))  # Ensure non-negative
                
                print(f'{name} epoch stats:')
                print(f'  min: {stats["min"]:.6f}, max: {stats["max"]:.6f}')
                print(f'  mean: {mean:.6f}, std: {std:.6f}')
                
                # Calculate quantiles if we have stored values
                if len(stats['values']) > 0:
                    values_array = np.array(stats['values'])
                    q1 = np.percentile(values_array, 25)
                    median = np.percentile(values_array, 50)
                    q3 = np.percentile(values_array, 75)
                    print(f'  Q1: {q1:.6f}, median: {median:.6f}, Q3: {q3:.6f}')
                
                # Print histogram
                hist = self.histograms[name]
                print(f'{name} epoch distribution:')
                for i, (count, edge) in enumerate(zip(hist, self.hist_bins[:-1])):
                    if stats['count'] > 0:
                        percentage = (count / stats['count']) * 100
                        print(f'  [{edge:.2f}-{edge+0.01:.2f}): {count:6d} ({percentage:5.1f}%)')

# ---------------------------------------------------------------------------
# New Dataset Class for Irregular Images
# ---------------------------------------------------------------------------

class AnnotatedImageDataset(Dataset):
    """Dataset for handling images with JSON annotations for defective regions."""
    
    def __init__(
        self,
        annotation_dir: str,
        patch_size: int = 128,
        transform=None,
        object_class: str = "pcb",
    ):
        """
        Args:
            annotation_dir: Directory containing JSON annotation files
            patch_size: Size of patches to extract (default: 128)
            transform: Optional transform to apply to patches
            object_class: Object class for classification
        """
        self.annotation_dir = annotation_dir
        self.patch_size = patch_size
        self.transform = transform
        self.object_class = object_class
        
        # Find all annotation files
        annotation_files = glob(os.path.join(annotation_dir, "*__annotations.json"))
        
        self.images = []
        self.patches = []
        self.patch_coords = []  # (x, y) coordinates of each patch in original image
        self.image_paths = []
        self.anomaly_classes = []
        self.is_defective = []  # Boolean indicating if patch is defective
        
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            image_path = annotation["image_path"]
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            # Load original image
            img = np.array(PILImage.open(image_path).convert('RGB'))
            original_height, original_width = img.shape[:2]
            
            # Get defective patch coordinates
            defective_patches = set()
            for patch_coord in annotation["defective_patches"]:
                grid_row, grid_col = patch_coord
                defective_patches.add((grid_row, grid_col))
            
            # Generate all possible patches
            grid_rows = original_height // patch_size
            grid_cols = original_width // patch_size
            
            for grid_row in range(grid_rows):
                for grid_col in range(grid_cols):
                    # Calculate pixel coordinates
                    x = grid_col * patch_size
                    y = grid_row * patch_size
                    
                    # Extract patch
                    patch = img[y:y + patch_size, x:x + patch_size]
                    
                    # Determine if this patch is defective
                    is_defective = (grid_row, grid_col) in defective_patches
                    
                    self.images.append(img)  # Store original image
                    self.patches.append(patch)
                    self.patch_coords.append((x, y))
                    self.image_paths.append(image_path)
                    self.anomaly_classes.append("defect" if is_defective else "normal")
                    self.is_defective.append(is_defective)
        
        print(f"Created {len(self.patches)} patches from {len(set(self.image_paths))} images")
        print(f"Defective patches: {sum(self.is_defective)}, Normal patches: {len(self.is_defective) - sum(self.is_defective)}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        patch = self.patches[index].astype(np.float32) / 255.0
        original_img = self.images[index]
        x, y = self.patch_coords[index]
        image_path = self.image_paths[index]
        anomaly_class = self.anomaly_classes[index]
        is_defective = self.is_defective[index]
        
        # Create segmentation mask (all zeros for normal patches, all ones for defective)
        seg = np.ones((self.patch_size, self.patch_size), dtype=np.float32) if is_defective else np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        
        # Apply transform if provided
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = torch.from_numpy(patch.transpose(2, 0, 1))
            patch = (patch - 0.5) / 0.5
        
        # Convert coordinates to tensors for proper batching
        coords_tensor = torch.tensor([x, y], dtype=torch.int32)
        
        # Don't return original_img in the batch to avoid tensor size mismatch
        # We'll handle original image loading separately in the processing function
        return patch, seg, 0, image_path, anomaly_class, coords_tensor


class IrregularImageDataset(Dataset):
    """Dataset for handling irregular-sized images by splitting them into patches."""
    
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 128,
        stride: int = 64,  # Overlap between patches
        transform=None,
        object_class: str = "pcb",
        anomaly_class: str = "all",
        split_csv_path: str | None = None,
    ):
        """
        Args:
            data_dir: Directory containing the original images
            patch_size: Size of patches to extract (default: 128)
            stride: Stride between patches (default: 64 for overlap)
            transform: Optional transform to apply to patches
            object_class: Object class for classification
            anomaly_class: Anomaly class filter
            split_csv_path: Path to CSV file with image information
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.object_class = object_class
        self.anomaly_class = anomaly_class
        
        # Load image information from CSV
        if split_csv_path and os.path.exists(split_csv_path):
            df = pd.read_csv(split_csv_path)
            if anomaly_class != "all":
                df = df.query(f'category=="{anomaly_class}"')
        else:
            # If no CSV, scan directory for images
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(data_dir, f"*{ext}")))
                image_files.extend(glob(os.path.join(data_dir, f"*{ext.upper()}")))
            
            # Create a simple dataframe
            df = pd.DataFrame({
                'image': [os.path.basename(f) for f in image_files],
                'category': ['unknown'] * len(image_files),
                'object': [object_class] * len(image_files)
            })
        
        self.images = []
        self.patches = []
        self.patch_coords = []  # (x, y) coordinates of each patch in original image
        self.image_paths = []
        self.anomaly_classes = []
        
        object_cls_dict = {"pcb": 0}
        
        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, str(row['image']))
            if not os.path.exists(image_path):
                continue

            # Load original image
            img = np.array(PILImage.open(image_path).convert('RGB'))
            original_height, original_width = img.shape[:2]
            
            # Generate patches
            patches, coords = self._extract_patches(img)
            
            for patch, (x, y) in zip(patches, coords):
                self.images.append(img)  # Store original image
                self.patches.append(patch)
                self.patch_coords.append((x, y))
                self.image_paths.append(image_path)
                self.anomaly_classes.append(row['category'])
        
        print(f"Created {len(self.patches)} patches from {len(set(self.image_paths))} images")
    
    def _extract_patches(self, img):
        """Extract patches from an image with overlap."""
        height, width = img.shape[:2]
        patches = []
        coords = []
        
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)
                coords.append((x, y))
        
        # Handle edge cases if image is smaller than patch_size
        if height < self.patch_size or width < self.patch_size:
            # Pad the image to patch_size
            padded_img = np.zeros((max(height, self.patch_size), max(width, self.patch_size), 3), dtype=img.dtype)
            padded_img[:height, :width] = img
            patch = padded_img[:self.patch_size, :self.patch_size]
            patches = [patch]
            coords = [(0, 0)]
        
        return patches, coords
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, index):
        patch = self.patches[index].astype(np.float32) / 255.0
        original_img = self.images[index]
        x, y = self.patch_coords[index]
        image_path = self.image_paths[index]
        anomaly_class = self.anomaly_classes[index]
        
        # Create dummy segmentation (all zeros for normal patches)
        seg = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        
        # Apply transform if provided
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = torch.from_numpy(patch.transpose(2, 0, 1))
            patch = (patch - 0.5) / 0.5
        
        # Convert coordinates to tensors for proper batching
        coords_tensor = torch.tensor([x, y], dtype=torch.int32)
        
        return patch, seg, 0, image_path, anomaly_class, coords_tensor, original_img

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def add_metric_fields(rec: Record, *, device=torch.device("cpu")) -> None:

    def to4d(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype != torch.float32:
                x = x.float()
            if x.ndim == 3 and x.shape[-1] == 3:  # HWC ➜ CHW
                x = x.permute(2, 0, 1)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            return x.to(device).clamp(-1, 1)
        return None

    a = to4d(rec["encoded_recon"][1])
    b = to4d(rec["dod_recon"][1])
    
    # Add null checks before computing metrics
    if a is not None and b is not None:
        rec["lpips"] = ("metric", _lpips(a, b, net_type="alex").item())
        ssim_result = _ssim(a, b)
        rec["ssim"] = ("metric", ssim_result[0].item() if isinstance(ssim_result, tuple) else ssim_result.item())
        rec["mse"] = ("metric", F.mse_loss(a, b).item())
    else:
        # Set default values if conversion fails
        rec["lpips"] = ("metric", 0.0)
        rec["ssim"] = ("metric", 0.0)
        rec["mse"] = ("metric", 0.0)


def make_record(**kwargs) -> Record:
    """Return an **ordered** dict whose values are (kind, value) pairs."""
    return OrderedDict(kwargs)


def _compute_diff_mean(a: torch.Tensor, b: torch.Tensor, diff_scale: float = 1.0) -> torch.Tensor:
    """Return the mean channel‑wise difference *scaled* by ``diff_scale``."""
    return (a - b).mean(dim=1, keepdim=True) / diff_scale

def _compute_abs_diff_mean(a: torch.Tensor, b: torch.Tensor, diff_scale: float = 1.0) -> torch.Tensor:
    """Return the mean channel‑wise absolute difference *scaled* by ``diff_scale``."""
    return torch.abs(a - b).mean(dim=1, keepdim=True) / diff_scale

def _compute_abs_diff_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the maximum channel‑wise absolute difference."""
    return torch.abs(a - b).max(dim=1, keepdim=True).values

def _binary_mask(diff: torch.Tensor, threshold: int = 5) -> torch.Tensor:
    """Return a binary mask in ``{0, 1}`` based on *absolute* diff magnitude."""
    return (diff.abs() > (threshold / 255.0)).float()


def _get_largest_connected_component_pixels(anomaly_binary: torch.Tensor) -> int:
    """
    Calculate the number of pixels in the largest connected component of white pixels.
    
    Args:
        anomaly_binary: Binary tensor with shape (H, W) or (1, H, W) where 1 indicates white pixels
        
    Returns:
        Number of pixels in the largest connected component
    """
    import cv2
    import numpy as np
    
    # Convert to numpy and ensure 2D shape
    if anomaly_binary.dim() == 3 and anomaly_binary.shape[0] == 1:
        binary_np = anomaly_binary.squeeze(0).cpu().numpy()
    else:
        binary_np = anomaly_binary.cpu().numpy()
    
    # Ensure binary values (0 or 1)
    binary_np = (binary_np > 0).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_np, connectivity=8)
    
    if num_labels <= 1:  # Only background (label 0) or no components
        return 0
    
    # Find the largest component (excluding background which is label 0)
    largest_component_size = 0
    for i in range(1, num_labels):  # Skip background (i=0)
        component_size = stats[i, cv2.CC_STAT_AREA]
        if component_size > largest_component_size:
            largest_component_size = component_size
    
    return largest_component_size


def _create_contour_based_binary_mask(anomaly_map: torch.Tensor, adaptive_threshold: float = 0.1) -> torch.Tensor:
    """
    Create binary mask based on contours with adaptive selection based on distribution.
    
    Args:
        anomaly_map: Anomaly map tensor with shape (batch_size, 1, H, W) with values in [0, 1]
        adaptive_threshold: Threshold for adaptive contour selection (default: 0.1)
                          - Lower values select more contours
                          - Higher values select fewer contours
        
    Returns:
        Binary tensor with same shape as input where selected contour pixels are 1, others are 0
    """
    import cv2
    import numpy as np
    
    # Handle batch processing for shape (batch_size, 1, H, W)
    if anomaly_map.dim() == 4:
        batch_size = anomaly_map.shape[0]
        binary_masks = []
        
        for b in range(batch_size):
            # Extract single image from batch
            single_map = anomaly_map[b, 0]  # Shape: (H, W)
            single_binary = _create_contour_based_binary_mask_single(single_map, adaptive_threshold)
            binary_masks.append(single_binary)
        
        # Stack back into batch
        return torch.stack(binary_masks, dim=0).unsqueeze(1)  # Shape: (batch_size, 1, H, W)
    else:
        # Handle single image case
        return _create_contour_based_binary_mask_single(anomaly_map, adaptive_threshold)


def _create_contour_based_binary_mask_single(anomaly_map: torch.Tensor, adaptive_threshold: float = 0.1) -> torch.Tensor:
    """
    Create binary mask for a single image based on adaptive contour selection.
    
    Args:
        anomaly_map: Anomaly map tensor with shape (H, W) with values in [0, 1]
        adaptive_threshold: Threshold for adaptive contour selection (default: 0.1)
        
    Returns:
        Binary tensor with shape (H, W) where selected contour pixels are 1, others are 0
    """
    import cv2
    import numpy as np
    
    # Convert to numpy
    map_np = anomaly_map.cpu().numpy()
    
    # Ensure the map is 2D
    if map_np.ndim != 2:
        print(f"Warning: Expected 2D array, got shape {map_np.shape}")
        return torch.zeros_like(anomaly_map)
    
    # Handle negative values and ensure proper range
    map_np = np.clip(map_np, 0, 1)  # Clip to [0, 1] range
    
    # Convert to uint8 for contour detection (0-255 range)
    map_uint8 = (map_np * 255).astype(np.uint8)
    
    # Check if the image is all zeros (no contours possible)
    if np.all(map_uint8 == 0):
        return torch.zeros_like(anomaly_map)
    
    try:
        # Find contours
        contours, _ = cv2.findContours(map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as e:
        print(f"OpenCV error in findContours: {e}")
        print(f"Map shape: {map_uint8.shape}, dtype: {map_uint8.dtype}")
        print(f"Map min: {map_uint8.min()}, max: {map_uint8.max()}")
        return torch.zeros_like(anomaly_map)
    
    if not contours:
        # No contours found, return all zeros
        return torch.zeros_like(anomaly_map)
    
    # Calculate contour statistics (sum of pixel values within each contour)
    contour_stats = []
    for i, contour in enumerate(contours):
        # Create a mask for this contour
        contour_mask = np.zeros_like(map_uint8)
        cv2.fillPoly(contour_mask, [contour], (255,))
        
        # Calculate sum of pixel values within this contour
        contour_sum = np.sum(map_np * (contour_mask > 0))
        contour_area = cv2.contourArea(contour)
        
        contour_stats.append({
            'index': i,
            'contour': contour,
            'sum': contour_sum,
            'area': contour_area
        })
    
    # Sort by sum (descending)
    contour_stats.sort(key=lambda x: x['sum'], reverse=True)
    
    if not contour_stats:
        return torch.zeros_like(anomaly_map)
    
    # Extract sums for adaptive selection
    sums = np.array([stat['sum'] for stat in contour_stats])
    
    # Adaptive selection based on distribution
    if len(sums) == 1:
        # Only one contour, select it
        selected_contours = contour_stats
    else:
        # Calculate statistics for adaptive selection
        total_sum = np.sum(sums)
        max_sum = np.max(sums)
        mean_sum = np.mean(sums)
        std_sum = np.std(sums)
        
        # Multiple adaptive criteria
        # 1. Contours with sum > threshold * max_sum
        threshold_max = adaptive_threshold * max_sum
        
        # 2. Contours with sum > mean + threshold * std
        threshold_stat = mean_sum + adaptive_threshold * std_sum
        
        # 3. Contours contributing > threshold of total sum
        threshold_total = adaptive_threshold * total_sum
        
        # Select contours that meet any of the criteria
        selected_contours = []
        for stat in contour_stats:
            if (stat['sum'] >= threshold_max or 
                stat['sum'] >= threshold_stat or 
                stat['sum'] >= threshold_total):
                selected_contours.append(stat)
        
        # If no contours meet criteria, select at least the top one
        if not selected_contours:
            selected_contours = [contour_stats[0]]
    
    # Create binary mask with selected contours
    binary_mask = np.zeros_like(map_uint8)
    for contour_info in selected_contours:
        cv2.fillPoly(binary_mask, [contour_info['contour']], (255,))
    
    # Convert back to tensor and normalize to [0, 1]
    binary_tensor = torch.from_numpy(binary_mask).float() / 255.0
    
    return binary_tensor.to(anomaly_map.device)


def _to_numpy(
    t: torch.Tensor,
) -> np.ndarray:  # keep Images API compatibility
    """Detach, move to CPU and convert to ``numpy`` if ``t`` is a tensor."""
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def _tensor_to_xlimage(arr, size: int) -> XLImage:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c == 4:
        # Split the image into 4 quadrants if c == 4 (e.g., 4-channel image)
        # We'll arrange the 4 channels as 2x2 grid: [0|1]
        #                                             [2|3]
        # h, w = arr.shape[0], h2, w2 = h // 2, w // 2
        # If the image is not square, just split in half along each axis
        # Each channel is a grayscale image, so we tile them
        q0 = arr[..., 0]
        q1 = arr[..., 1]
        q2 = arr[..., 2]
        q3 = arr[..., 3]
        # Stack as 2x2 grid
        top = np.concatenate([q0, q1], axis=1)
        bottom = np.concatenate([q2, q3], axis=1)
        arr = np.stack(
            [np.concatenate([top, bottom], axis=0)] * 3, axis=2
        )  # make 3 channels
    elif c != 3:
        raise ValueError(f"unsupported channels: {c}")
    arr = ((np.clip(arr, -1, 1) + 1) / 2 * 255).astype(np.uint8)

    buf = BytesIO()
    PILImage.fromarray(arr, mode="RGB").save(buf, format="PNG")
    buf.seek(0)
    img = XLImage(buf)
    img.width = img.height = size
    return img


def _write_row(ws, row_idx: int, rec: dict, size: int):
    scalars, embeds = [], []
    for col_idx, key in enumerate(rec.keys(), 1):
        kind, val = rec[key]
        if kind == "image":
            embeds.append((col_idx, _tensor_to_xlimage(val, size)))
            scalars.append("")
        else:
            # Convert tuples and other non-serializable types to strings
            if isinstance(val, (tuple, list)):
                scalars.append(str(val))
            elif isinstance(val, (np.ndarray, torch.Tensor)):
                scalars.append(str(val.tolist() if hasattr(val, 'tolist') else val))
            else:
                scalars.append(val)
    ws.append(scalars)
    ws.row_dimensions[row_idx].height = size * 0.75
    for col_idx, img in embeds:
        ws.add_image(img, f"{get_column_letter(col_idx)}{row_idx}")


def make_excel(
    records: List[Record],
    image_size: int,
    save_dir: str | Path = "report",
    save_filename: str | None = datetime.now().strftime("%y%m%d_%H%M%S"),
) -> Path:
    """Create Excel report with all evaluation records and images."""
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Header comes from the first record's keys (order preserved)
    header = list(records[0].keys())

    wb = Workbook()
    ws = wb.active
    if ws is not None:
        ws.title = "Report"
        ws.append(header)

        for r, rec in enumerate(records, start=2):
            _write_row(ws, r, rec, image_size)
        for c in range(1, len(header) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 18

    out_path = save_dir / f"report_{save_filename}.xlsx"
    wb.save(out_path)
    print(f"Report saved to {out_path}")
    return out_path


def draw_patch_rectangles_on_image(base_img, patch_results, ground_truth_patches=None, patch_size=128, stride=64, grid_thickness=1):
    """
    Draw patch rectangles (TP/FP/FN) on top of an image.
    
    Args:
        base_img: The image to draw on (np.uint8, HxWx3)
        patch_results: List of (x, y, is_defective) tuples for each patch
        ground_truth_patches: List of [grid_row, grid_col] coordinates for ground truth defective patches
        patch_size: Size of patches
        stride: Stride used for patch extraction
        grid_thickness: Thickness of the rectangle lines (default: 1)
    
    Returns:
        Image with rectangles drawn:
        - Yellow rectangles around predicted defective regions
        - Red rectangles around ground truth defective regions  
        - HOT colormap colored rectangles where prediction and ground truth overlap
    """
    img = base_img.copy()
    
    # Create sets for efficient lookup
    predicted_defective = set()
    ground_truth_defective = set()
    
    # Collect predicted defective patches
    for x, y, anomaly_pixels in patch_results:
        # Consider patch defective if it has any anomaly pixels (anomaly_pixels > 0)
        if anomaly_pixels > 0:
            grid_row = y // patch_size
            grid_col = x // patch_size
            predicted_defective.add((grid_row, grid_col))
    
    # Collect ground truth defective patches
    if ground_truth_patches:
        for grid_row, grid_col in ground_truth_patches:
            ground_truth_defective.add((grid_row, grid_col))
    
    # Draw predicted defective regions (yellow)
    for grid_row, grid_col in predicted_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (255, 255, 0), grid_thickness)
    
    # Draw ground truth defective regions (red)
    for grid_row, grid_col in ground_truth_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), grid_thickness)
    
    # Draw overlapping regions (green) - where prediction and ground truth match
    overlapping = predicted_defective.intersection(ground_truth_defective)
    for grid_row, grid_col in overlapping:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), grid_thickness)
    
    return img





# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def process_split_irregular(
    dataloader,
    split: str,
    diffusion,
    model,
    vae,
    reverse_steps: int,
    center_size: int,
    batch_num: int,
    device: torch.device = torch.device("cpu"),
    anomaly_binary_threshold: int = 0,
    anomaly_pixel_num_threshold: int = 0,
    adaptive_threshold: float = 0.1,
    enable_epoch_stats: bool = True,
) -> tuple[List[Record], defaultdict[str, list[tuple[int, int, bool]]], defaultdict[str, list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]]:
    """Run a forward‑&‑reverse pass on irregular images and collect metrics."""
    # Initialize evaluation metrics
    metrics = EvaluationMetrics()

    results: List[Record] = []
    image_patch_results = defaultdict(list)  # Track results per original image
    image_anomaly_maps = defaultdict(list)  # Track anomaly maps per original image
    
    # Memory optimization: Clear cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for idx, (x, seg, object_cls, image_paths, anomaly_classes, patch_coords) in enumerate(
        tqdm(dataloader, desc=f"{split} split")
    ):
        if idx >= batch_num:
            break

        with torch.no_grad():
            # -----------------------------------------------------------------
            # Forward pass through VAE encoder (to latent space)
            # -----------------------------------------------------------------
            x = x.to(device)
            object_cls = object_cls.to(device)

            encoded = vae.encode(x).latent_dist.mean * _LATENT_SCALE

            # -----------------------------------------------------------------
            # Reverse DDIM sampling conditioned on encoder latents
            # -----------------------------------------------------------------
            model_kwargs = {"context": object_cls.unsqueeze(1), "mask": None}
            
            latent_samples_list = []
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
                latent_samples_list.append(samples["sample"])
            latent_samples_final = latent_samples_list[-1]

            image_samples_list = []
            for latent_samples in latent_samples_list:
                image_samples_list.append(
                    vae.decode(latent_samples / _LATENT_SCALE).sample
                )

            # -----------------------------------------------------------------
            # Reconstructions & other intermediate images
            # -----------------------------------------------------------------
            image_samples = vae.decode(latent_samples_final / _LATENT_SCALE).sample
            x0 = vae.decode(encoded / _LATENT_SCALE).sample

            # -----------------------------------------------------------------
            # Difference / binary maps
            # -----------------------------------------------------------------
            orig_dodrecon_diff = _compute_diff_mean(x, image_samples)
            orig_dodrecon_binary = _binary_mask(orig_dodrecon_diff, anomaly_binary_threshold)

            orig_encodedrecon_diff = _compute_diff_mean(x, x0)
            orig_encodedrecon_binary = _binary_mask(orig_encodedrecon_diff, anomaly_binary_threshold)
            
            encodedrecon_dodrecon_diff_raw = _compute_abs_diff_max(x0, image_samples)
            encodedrecon_dodrecon_diff = torch.clamp(encodedrecon_dodrecon_diff_raw, 0.0, 0.05) * 20

            encoded_latent_diff_raw = _compute_abs_diff_mean(latent_samples_final,encoded)
            encoded_latent_diff = torch.clamp(encoded_latent_diff_raw, 0.0, 0.05) * 20

            # Resize encoded_latent_diff to match the spatial dimensions of encodedrecon_dodrecon_diff
            # For irregular images, we want to use the patch size (128) as the target size
            patch_size = x.shape[-1]  # Should be 128 for patches
            encoded_latent_diff_resized = F.interpolate(
                encoded_latent_diff,
                size=(patch_size, patch_size),
                mode="bilinear",
                align_corners=False,
            )

            # -----------------------------------------------------------------
            # Composite anomaly maps
            # -----------------------------------------------------------------
            anomaly_map_arithmetic = 0.5 * (
                encodedrecon_dodrecon_diff + encoded_latent_diff_resized
            )
            # Use contour-based binary mask instead of fixed threshold
            anomaly_map_arithmetic_binary = _binary_mask(anomaly_map_arithmetic, anomaly_binary_threshold)
            #anomaly_map_arithmetic_binary = _create_contour_based_binary_mask(anomaly_map_arithmetic, adaptive_threshold=adaptive_threshold)
            anomaly_map_geometric = (
                encodedrecon_dodrecon_diff * encoded_latent_diff_resized
            )
            # Use contour-based binary mask instead of fixed threshold
            anomaly_map_geometric_binary = _binary_mask(anomaly_map_geometric, anomaly_binary_threshold)
            #anomaly_map_geometric_binary = _create_contour_based_binary_mask(anomaly_map_geometric, adaptive_threshold=adaptive_threshold)

            # Collect epoch-wise statistics
            metrics.add_batch_stats(encodedrecon_dodrecon_diff_raw, encoded_latent_diff_raw, anomaly_map_arithmetic, anomaly_map_geometric)
        # ---------------------------------------------------------------------
        # Per‑sample aggregation
        # ---------------------------------------------------------------------
        batch_size = x.size(0)
        
        for b in range(batch_size):
            # Determine if this patch is defective
            anomaly_binary = anomaly_map_arithmetic_binary[b]
            #encoded_latent = encoded_latent_diff[b]
            #x_original = x[b]
            #
            #img_path = image_paths[b]
            #
            ## Save anomaly_binary to tmp directory
            #tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp')
            #os.makedirs(tmp_dir, exist_ok=True)
            #
            ## Convert to numpy and save as image
            #anomaly_binary_np = _to_numpy(anomaly_binary)
            #encoded_latent_np = _to_numpy(encoded_latent)
            #x_original_np = _to_numpy(x_original)
            #x_original_min = np.min(x_original_np)
            #x_original_max = np.max(x_original_np)
            #if len(anomaly_binary_np.shape) == 3 and anomaly_binary_np.shape[0] == 1:
            #    anomaly_binary_np = anomaly_binary_np[0]  # Remove channel dimension if single channel
            #    encoded_latent_np = encoded_latent_np[0]
            #    x_original_np = x_original_np[0]
            ## Save as PNG image
            #from PIL import Image
            #anomaly_img = Image.fromarray((anomaly_binary_np * 255).astype(np.uint8))
            #anomaly_save_path = os.path.join(tmp_dir, f'anomaly_binary_batch_{b}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            #anomaly_img.save(anomaly_save_path)
            #encoded_latent_img = Image.fromarray((encoded_latent_np * 255).astype(np.uint8))
            #encoded_latent_save_path = os.path.join(tmp_dir, f'encoded_latent_batch_{b}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            #encoded_latent_img.save(encoded_latent_save_path)
            #x_original_img = Image.fromarray(((x_original_np + 1) / 2 * 255).astype(np.uint8))
            #x_original_save_path = os.path.join(tmp_dir, f'original_batch_{b}_{x_original_min:.3f}_{x_original_max:.3f}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            #x_original_img.save(x_original_save_path)
            # Use largest connected component instead of total white pixels
            anomaly_pixels = torch.sum(anomaly_binary).item()
            #anomaly_pixels = _get_largest_connected_component_pixels(anomaly_binary)
            is_defective = anomaly_pixels > anomaly_pixel_num_threshold  # Largest connected component size
            
            # Store patch result for original image marking
            # patch_coords[b] is now a tensor [x, y]
            patch_coord_tensor = patch_coords[b]
            if isinstance(patch_coord_tensor, torch.Tensor):
                x_coord = int(patch_coord_tensor[0].item())
                y_coord = int(patch_coord_tensor[1].item())
            elif isinstance(patch_coord_tensor, (list, tuple)) and len(patch_coord_tensor) == 2:
                x_coord, y_coord = patch_coord_tensor
            else:
                # Fallback
                print(f"Warning: unexpected patch_coord format, using default coordinates")
                x_coord, y_coord = 0, 0
            
            # Store (x_coord, y_coord, anomaly_pixels) instead of (x_coord, y_coord, is_defective)
            image_patch_results[image_paths[b]].append((x_coord, y_coord, anomaly_pixels))
            
            # Store anomaly maps for overlay creation
            anomaly_map_arithmetic_np = _to_numpy(anomaly_map_arithmetic[b])
            anomaly_map_arithmetic_binary_np = _to_numpy(anomaly_map_arithmetic_binary[b])
            anomaly_map_geometric_np = _to_numpy(anomaly_map_geometric[b])
            anomaly_map_geometric_binary_np = _to_numpy(anomaly_map_geometric_binary[b])
            image_anomaly_maps[image_paths[b]].append((x_coord, y_coord, anomaly_map_arithmetic_np, anomaly_map_arithmetic_binary_np, anomaly_map_geometric_np, anomaly_map_geometric_binary_np))
            
            rec = make_record(
                split=("meta", split),
                image_path=("meta", image_paths[b]),
                anomaly_class=("meta", anomaly_classes[b]),
                patch_coords=("meta", (x_coord, y_coord)),
                is_defective=("meta", is_defective),
                orig=("image", _to_numpy(x[b])),
                dod_recon=("image", _to_numpy(image_samples[b])),
                encoded_recon=("image", _to_numpy(x0[b])),
                orig_dodrecon_diff=("image", _to_numpy(orig_dodrecon_diff[b])),
                orig_dodrecon_binary=("image", _to_numpy(orig_dodrecon_binary[b])),
                orig_encodedrecon_diff=("image", _to_numpy(orig_encodedrecon_diff[b])),
                orig_encodedrecon_binary=(
                    "image",
                    _to_numpy(orig_encodedrecon_binary[b]),
                ),
                encodedrecon_dodrecon_diff=(
                    "image",
                    _to_numpy(encodedrecon_dodrecon_diff[b]),
                ),
                encoded_latent_diff=("image", _to_numpy(encoded_latent_diff[b])),
                anomaly_map_arithmetic=("image", _to_numpy(anomaly_map_arithmetic[b])),
                anomaly_map_geometric=("image", _to_numpy(anomaly_map_geometric[b])),
                anomaly_map_arithmetic_binary=(
                    "image",
                    _to_numpy(anomaly_map_arithmetic_binary[b]),
                ),
                anomaly_map_geometric_binary=(
                    "image",
                    _to_numpy(anomaly_map_geometric_binary[b]),
                ),
                encoded=("image", _to_numpy(encoded[b])),
            )

            add_metric_fields(rec, device=device)
            results.append(rec)
        
        # Memory optimization: Clear cache every 10 batches
        if idx % 10 == 0 and idx > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    # Print epoch-wise statistics (if enabled)
    if enable_epoch_stats:
        metrics.print_epoch_stats()
    else:
        print("Skipping epoch statistics (disabled)")

    return results, image_patch_results, image_anomaly_maps


def process_split(
    dataloader,
    split: str,
    diffusion,
    model,
    vae,
    reverse_steps: int,
    center_size: int,
    batch_num: int,
    device: torch.device | None = None,
) -> List[Record]:
    """Run a forward‑&‑reverse pass on *one* dataset split and collect metrics."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Record] = []

    for idx, (x, seg, object_cls, image_paths, anomaly_classes) in enumerate(  # noqa: B905
        tqdm(dataloader, desc=f"{split} split")
    ):
        if idx >= batch_num:
            break

        with torch.no_grad():
            # -----------------------------------------------------------------
            # Forward pass through VAE encoder (to latent space)
            # -----------------------------------------------------------------
            x = x.to(device)
            object_cls = object_cls.to(device)

            encoded = vae.encode(x).latent_dist.mean * _LATENT_SCALE

            # -----------------------------------------------------------------
            # Reverse DDIM sampling conditioned on encoder latents
            # -----------------------------------------------------------------
            model_kwargs = {"context": object_cls.unsqueeze(1), "mask": None}
            
            latent_samples_list = []
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
                latent_samples_list.append(samples["sample"])
            latent_samples_final = latent_samples_list[-1]

            image_samples_list = []
            for latent_samples in latent_samples_list:
                image_samples_list.append(
                    vae.decode(latent_samples / _LATENT_SCALE).sample
                )

            # -----------------------------------------------------------------
            # Reconstructions & other intermediate images
            # -----------------------------------------------------------------
            image_samples = vae.decode(latent_samples_final / _LATENT_SCALE).sample
            x0 = vae.decode(encoded / _LATENT_SCALE).sample

            # -----------------------------------------------------------------
            # Difference / binary maps
            # -----------------------------------------------------------------
            orig_dodrecon_diff = _compute_diff_mean(x, image_samples)
            orig_encodedrecon_diff = _compute_diff_mean(x, x0)
            encodedrecon_dodrecon_diff = _compute_diff_mean(x0, image_samples)

            orig_dodrecon_binary = _binary_mask(orig_dodrecon_diff)
            orig_encodedrecon_binary = _binary_mask(orig_encodedrecon_diff)
            encodedrecon_dodrecon_diff = _binary_mask(encodedrecon_dodrecon_diff)

            encoded_latent_diff = (
                (latent_samples_final - encoded).max(dim=1, keepdim=True).values
            )
            encoded_latent_diff = _binary_mask(encoded_latent_diff)

            encoded_latent_abs_diff_resized = F.interpolate(
                encoded_latent_diff.abs(),
                size=(center_size, center_size),
                mode="bilinear",
                align_corners=False,
            )

            # -----------------------------------------------------------------
            # Composite anomaly maps
            # -----------------------------------------------------------------
            anomaly_map_arithmetic = 0.5 * (
                encodedrecon_dodrecon_diff + encoded_latent_abs_diff_resized
            )
            anomaly_map_arithmetic_binary = _binary_mask(anomaly_map_arithmetic)
            anomaly_map_geometric = (
                encodedrecon_dodrecon_diff * encoded_latent_abs_diff_resized
            )
            anomaly_map_geometric_binary = _binary_mask(anomaly_map_geometric)

        # ---------------------------------------------------------------------
        # Per‑sample aggregation
        # ---------------------------------------------------------------------
        batch_size = x.size(0)
        for b in range(batch_size):
            rec = make_record(
                split=("meta", split),
                image_path=("meta", image_paths[b]),
                anomaly_class=("meta", anomaly_classes[b]),
                orig=("image", _to_numpy(x[b])),
                dod_recon=("image", _to_numpy(image_samples[b])),
                encoded_recon=("image", _to_numpy(x0[b])),
                orig_dodrecon_diff=("image", _to_numpy(orig_dodrecon_diff[b])),
                orig_dodrecon_binary=("image", _to_numpy(orig_dodrecon_binary[b])),
                orig_encodedrecon_diff=("image", _to_numpy(orig_encodedrecon_diff[b])),
                orig_encodedrecon_binary=(
                    "image",
                    _to_numpy(orig_encodedrecon_binary[b]),
                ),
                encodedrecon_dodrecon_diff=(
                    "image",
                    _to_numpy(encodedrecon_dodrecon_diff[b]),
                ),
                encoded_latent_diff=("image", _to_numpy(encoded_latent_diff[b])),
                anomaly_map_arithmetic=("image", _to_numpy(anomaly_map_arithmetic[b])),
                anomaly_map_geometric=("image", _to_numpy(anomaly_map_geometric[b])),
                anomaly_map_arithmetic_binary=(
                    "image",
                    _to_numpy(anomaly_map_arithmetic_binary[b]),
                ),
                anomaly_map_geometric_binary=(
                    "image",
                    _to_numpy(anomaly_map_geometric_binary[b]),
                ),
                encoded=("image", _to_numpy(encoded[b])),
            )

            add_metric_fields(rec, device=device)
            results.append(rec)

    return results


def evaluation(args):
    if os.path.exists("./models/config.json"):
        vae = cast(AutoencoderKL, AutoencoderKL.from_pretrained("./models", local_files_only=True)).to(
            device
        )
    else:
        vae = cast(AutoencoderKL, AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}")).to(
            device
        )
    vae.eval()
    try:
        if args.pretrained != "":
            ckpt = args.pretrained
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_class}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f"{path}/last.pt"))[-1]
            except (IndexError, FileNotFoundError):
                ckpt = sorted(glob(f"{path}/*/last.pt"))[-1]
    except (IndexError, FileNotFoundError, OSError):
        raise Exception("Please provide the model's pretrained path using --pretrained")

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size)

    state_dict = torch.load(ckpt)["model"]
    print(model.load_state_dict(state_dict))
    model.eval()  # important!
    model.cuda()
    print("model loaded")

    print("==" * 30)
    print("Starting Evaluation...")
    print("==" * 30)
    diffusion = create_diffusion(
        f"ddim{args.reverse_steps}",
        predict_deviation=True,
        sigma_small=False,
        predict_xstart=False,
        diffusion_steps=1000,
    )

    # Check if we're processing annotated images
    if hasattr(args, 'annotation_dir') and args.annotation_dir:
        print("Processing images with JSON annotations...")
        evaluation_annotated_images(args, diffusion, model, vae)
    # Check if we're processing irregular images
    elif hasattr(args, 'irregular_images') and args.irregular_images:
        print("Processing irregular-sized images with patch-based approach...")
        evaluation_irregular_images(args, diffusion, model, vae)
    else:
        print("Processing regular-sized images...")
        evaluation_regular_images(args, diffusion, model, vae)


def evaluation_annotated_images(args, diffusion, model, vae):
    """Evaluate images with JSON annotations for defective regions."""
    # Create base evaluator
    evaluator = BaseEvaluator(args, diffusion, model, vae)
    
    # Create dataset for annotated images
    dataset = AnnotatedImageDataset(
        annotation_dir=args.annotation_dir,
        patch_size=128,
        transform=evaluator.transform,
        object_class=args.object_class,
    )
    
    loader = evaluator._get_dataloader(dataset, batch_size=8)
    
    # For annotated images, use patch size as center_size to ensure consistent dimensions
    patch_center_size = 128
    
    # Process patches
    records, image_patch_results, image_anomaly_maps = process_split_irregular(
        loader,
        args.split,
        diffusion,
        model,
        vae,
        args.reverse_steps,
        patch_center_size,  # Use patch size instead of args.center_size
        args.batch_num,
        device,
        args.anomaly_binary_threshold,
        args.anomaly_pixel_num_threshold,
        0.1,  # adaptive_threshold
        args.enable_epoch_stats,  # Pass only the boolean flag
    )
    
    # Mark defective regions on original images
    print("Marking defective regions on original images...")
    output_dirs = evaluator._create_output_dirs()
    
    for image_path, patch_results in image_patch_results.items():
        # Load original image
        original_img = np.array(PILImage.open(image_path).convert('RGB'))
        
        # Load ground truth annotations for this image
        ground_truth_patches = None
        annotation_filename = f"{path_to_safe_filename(image_path)}__annotations.json"
        annotation_path = os.path.join(args.annotation_dir, annotation_filename)
        
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
                ground_truth_patches = annotation.get("defective_patches", [])
        
        # Mark defective regions (both predicted and ground truth)
        marked_img = draw_patch_rectangles_on_image(
            original_img, patch_results, ground_truth_patches, patch_size=128, stride=128, grid_thickness=1
        )
        
        safe_name = path_to_safe_filename(image_path)
        # Save marked image
        marked_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__marked.png")
        evaluator._save_image(marked_img, marked_path, "marked image")

        # --- Create and save anomaly overlay image ---
        # Reconstruct the full anomaly maps for this image
        anomaly_map_list = image_anomaly_maps[image_path]
        # Assume all patches are 128x128 and non-overlapping
        h, w, _ = original_img.shape
        full_anomaly_map_arithmetic = np.zeros((h, w), dtype=np.float32)
        full_anomaly_map_arithmetic_binary = np.zeros((h, w), dtype=np.float32)
        full_anomaly_map_geometric = np.zeros((h, w), dtype=np.float32)
        full_anomaly_map_geometric_binary = np.zeros((h, w), dtype=np.float32)
        
        for x, y, patch_map_arithmetic, patch_map_arithmetic_binary, patch_map_geometric, patch_map_geometric_binary in anomaly_map_list:
            full_anomaly_map_arithmetic[y:y+128, x:x+128] = patch_map_arithmetic.squeeze()
            full_anomaly_map_arithmetic_binary[y:y+128, x:x+128] = patch_map_arithmetic_binary.squeeze()
            full_anomaly_map_geometric[y:y+128, x:x+128] = patch_map_geometric.squeeze()
            full_anomaly_map_geometric_binary[y:y+128, x:x+128] = patch_map_geometric_binary.squeeze()
        
        # Create and save standalone arithmetic anomaly maps
        anomaly_map_arithmetic_img = ImageProcessor.create_anomaly_map_image(full_anomaly_map_arithmetic, patch_size=128, add_grid=True, patch_results=patch_results, ground_truth_patches=ground_truth_patches, is_binary=False)
        anomaly_map_arithmetic_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__am_arithmetic.png")
        evaluator._save_image(anomaly_map_arithmetic_img, anomaly_map_arithmetic_path, "arithmetic anomaly map image")
        
        anomaly_map_arithmetic_binary_img = ImageProcessor.create_anomaly_map_image(full_anomaly_map_arithmetic_binary, patch_size=128, add_grid=True, patch_results=patch_results, ground_truth_patches=ground_truth_patches)
        anomaly_map_arithmetic_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__am_arithmetic_binary.png")
        evaluator._save_image(anomaly_map_arithmetic_binary_img, anomaly_map_arithmetic_binary_path, "arithmetic anomaly map binary image")

        # Create and save standalone geometric anomaly maps
        anomaly_map_geometric_img = ImageProcessor.create_anomaly_map_image(full_anomaly_map_geometric, patch_size=128, add_grid=True, patch_results=patch_results, ground_truth_patches=ground_truth_patches, is_binary=False)
        anomaly_map_geometric_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__am_geometric.png")
        evaluator._save_image(anomaly_map_geometric_img, anomaly_map_geometric_path, "geometric anomaly map image")
        
        anomaly_map_geometric_binary_img = ImageProcessor.create_anomaly_map_image(full_anomaly_map_geometric_binary, patch_size=128, add_grid=True, patch_results=patch_results, ground_truth_patches=ground_truth_patches)
        anomaly_map_geometric_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__am_geometric_binary.png")
        evaluator._save_image(anomaly_map_geometric_binary_img, anomaly_map_geometric_binary_path, "geometric anomaly map binary image")

        # Create arithmetic overlay images
        overlay_arithmetic_img = ImageProcessor.create_anomaly_overlay(original_img, full_anomaly_map_arithmetic, alpha=0.8, is_binary=False)
        overlay_arithmetic_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__ao_arithmetic.png")
        evaluator._save_image(overlay_arithmetic_img, overlay_arithmetic_path, "arithmetic anomaly overlay image")

        overlay_arithmetic_binary_img = ImageProcessor.create_anomaly_overlay(original_img, full_anomaly_map_arithmetic_binary, alpha=0.8)
        overlay_arithmetic_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__ao_arithmetic_binary.png")
        evaluator._save_image(overlay_arithmetic_binary_img, overlay_arithmetic_binary_path, "arithmetic anomaly overlay binary image")

        # Create geometric overlay images
        overlay_geometric_img = ImageProcessor.create_anomaly_overlay(original_img, full_anomaly_map_geometric, alpha=0.8, is_binary=False)
        overlay_geometric_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__ao_geometric.png")
        evaluator._save_image(overlay_geometric_img, overlay_geometric_path, "geometric anomaly overlay image")

        overlay_geometric_binary_img = ImageProcessor.create_anomaly_overlay(original_img, full_anomaly_map_geometric_binary, alpha=0.8)
        overlay_geometric_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__ao_geometric_binary.png")
        evaluator._save_image(overlay_geometric_binary_img, overlay_geometric_binary_path, "geometric anomaly overlay binary image")

        # Save overlay+patches images for arithmetic maps
        marked_overlay_arithmetic_img = draw_patch_rectangles_on_image(overlay_arithmetic_img, patch_results, ground_truth_patches, patch_size=128, stride=128, grid_thickness=1)
        marked_overlay_arithmetic_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__mo_arithmetic.png")
        evaluator._save_image(marked_overlay_arithmetic_img, marked_overlay_arithmetic_path, "marked arithmetic overlay image")

        marked_overlay_arithmetic_binary_img = draw_patch_rectangles_on_image(overlay_arithmetic_binary_img, patch_results, ground_truth_patches, patch_size=128, stride=128, grid_thickness=1)
        marked_overlay_arithmetic_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__mo_arithmetic_binary.png")
        evaluator._save_image(marked_overlay_arithmetic_binary_img, marked_overlay_arithmetic_binary_path, "marked arithmetic overlay binary image")

        # Save overlay+patches images for geometric maps
        marked_overlay_geometric_img = draw_patch_rectangles_on_image(overlay_geometric_img, patch_results, ground_truth_patches, patch_size=128, stride=128, grid_thickness=1)
        marked_overlay_geometric_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__mo_geometric.png")
        evaluator._save_image(marked_overlay_geometric_img, marked_overlay_geometric_path, "marked geometric overlay image")

        marked_overlay_geometric_binary_img = draw_patch_rectangles_on_image(overlay_geometric_binary_img, patch_results, ground_truth_patches, patch_size=128, stride=128, grid_thickness=1)
        marked_overlay_geometric_binary_path = os.path.join(output_dirs['marked_images'], f"{safe_name}__mo_geometric_binary.png")
        evaluator._save_image(marked_overlay_geometric_binary_img, marked_overlay_geometric_binary_path, "marked geometric overlay binary image")
    
    # Save evaluation results with white pixel counts
    for image_path, patch_results in image_patch_results.items():
        # Convert pixel coordinates back to grid coordinates and include white pixel counts
        patch_analysis = []
        for x, y, anomaly_pixels in patch_results:
            grid_row = y // 128
            grid_col = x // 128
            patch_analysis.append({
                "grid_row": grid_row,
                "grid_col": grid_col,
                "anomaly_pixels": int(anomaly_pixels),
                "is_defective": bool(anomaly_pixels > 0)
            })
        
        result_filename = f"{path_to_safe_filename(image_path)}__evaluation.json"
        result_path = os.path.join(output_dirs['evaluation_results'], result_filename)
        evaluation_result = {
            "image_path": image_path,
            "patch_analysis": patch_analysis,
            "grid_size": 128
        }
        evaluator._save_json(evaluation_result, result_path, "evaluation result")
    
    # Create Excel report (if enabled)
    if hasattr(args, 'enable_excel_report') and args.enable_excel_report:
        print("Creating Excel report...")
        excel_path = make_excel(
            records=records,
            image_size=128,
            save_dir=output_dirs['evaluation_results'],
            save_filename=f"report_{datetime.now().strftime('%y%m%d_%H%M%S')}"
        )
        print(f"Excel report saved to: {excel_path}")
    else:
        print("Skipping Excel report generation (disabled)")
    
    print("==" * 30)
    # Compute confusion matrix and accuracy
    compute_confusion_matrix_and_accuracy(args.annotation_dir, output_dirs['evaluation_results'])


def evaluation_irregular_images(args, diffusion, model, vae):
    """Evaluate irregular-sized images by splitting into patches."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    
    # Create dataset for irregular images
    dataset = IrregularImageDataset(
        data_dir=args.data_dir,
        patch_size=128,
        stride=128,  # 50% overlap
        transform=transform,
        object_class=args.object_class,
        anomaly_class=args.anomaly_class,
        split_csv_path=args.split_csv_path,
    )
    
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )
    
    # For irregular images, use patch size as center_size to ensure consistent dimensions
    patch_center_size = 128
    
    # Process patches
    _recs, image_patch_results, image_anomaly_maps = process_split_irregular(
        loader,
        args.split,
        diffusion,
        model,
        vae,
        args.reverse_steps,
        patch_center_size,  # Use patch size instead of args.center_size
        args.batch_num,
        device,
        args.anomaly_binary_threshold,
        args.anomaly_pixel_num_threshold,
        0.1,  # adaptive_threshold
        args.enable_epoch_stats,  # Pass only the boolean flag
    )
    
    # Mark defective regions on original images
    print("Marking defective regions on original images...")
    marked_images_dir = os.path.join(args.results_dir, "marked_images")
    
    for image_path, patch_results in image_patch_results.items():
        # Load original image
        original_img = np.array(PILImage.open(image_path).convert('RGB'))
        
        # Mark defective regions
        marked_img = draw_patch_rectangles_on_image(
            original_img, patch_results, patch_size=128, stride=128, grid_thickness=1
        )
        
        safe_name = path_to_safe_filename(image_path)
        # Save marked image
        marked_path = os.path.join(marked_images_dir, f"{safe_name}_marked.png")
        PILImage.fromarray(marked_img).save(marked_path)
        print(f"Saved marked image: {marked_path}")
    
    # Save patch analysis results
    patch_analysis_path = os.path.join(args.results_dir, "patch_analysis.json")
    patch_analysis = {}
    for image_path, patch_results in image_patch_results.items():
        patch_analysis[image_path] = [
            {"x": x, "y": y, "anomaly_pixels": int(anomaly_pixels), "is_defective": bool(anomaly_pixels > 0)}
            for x, y, anomaly_pixels in patch_results
        ]
    
    with open(patch_analysis_path, 'w') as f:
        json.dump(patch_analysis, f, indent=2)
    
    print(f"Patch analysis saved to: {patch_analysis_path}")
    print("==" * 30)


def evaluation_regular_images(args, diffusion, model, vae):
    """Evaluate regular-sized images using the original approach."""
    for object_class in args.object_classes:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        # Dataset-specific arguments
        dataset_args: dict = dict(
            split=args.split,
            object_class=object_class,
            rootdir=args.data_dir,
            transform=transform,
            anomaly_class=args.anomaly_class,
            image_size=args.image_size,
            center_size=args.actual_image_size,
            center_crop=True,
            split_csv_path=args.split_csv_path,
        )
        
        # Function-specific arguments
        function_args = dict(
            process_split_fn=process_split,
            diffusion=diffusion,
            model=model,
            vae=vae,
            reverse_steps=args.reverse_steps,
            batch_num=args.batch_num,
            device=device,
        )
        
        if args.dataset == "pcb":
            dataset_args["dataset_class"] = PCBDataset
        elif args.dataset == "mvtec":
            dataset_args["dataset_class"] = MVTECDataset
        elif args.dataset == "visa":
            dataset_args["dataset_class"] = VISADataset
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        if args.perturbation is not None:
            if args.perturbation == "brightness":
                param_values = np.arange(-20, 21, 1)
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="brightness", param_values=param_values, **all_args
                )
            if args.perturbation == "shift_x":
                param_values = np.arange(-20, 21, 1)
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="shift_x", param_values=param_values, **all_args
                )
            if args.perturbation == "shift_y":
                param_values = np.arange(-20, 21, 1)
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="shift_y", param_values=param_values, **all_args
                )
            if args.perturbation == "noise":
                param_values = np.arange(0, 21, 1)
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="noise", param_values=param_values, **all_args
                )
            if args.perturbation == "blur":
                param_values = np.arange(1, 42, 2)
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="blur", param_values=param_values, **all_args
                )
            if args.perturbation == "scratch":
                param_values = [0]
                all_args = {**dataset_args, **function_args}
                record_pairs = collect_records_for_params(
                    param_name="brightness", param_values=param_values, **all_args
                )
            y_true_score_list = compute_y_true_y_score(record_pairs)
            roc_stats = compute_metrics_from_y_true_y_score(y_true_score_list)
            save_perturbation_results(
                param_name=args.perturbation,
                roc_stats=roc_stats,
                param_values=param_values.tolist() if isinstance(param_values, np.ndarray) else param_values,
                save_dir=args.results_dir,
            )

            plot_accuracy_results(
                param_name=args.perturbation,
                param_values=param_values,
                accuracies=roc_stats["accuracies"],
                color="red",
                save_dir=args.results_dir,
            )

        print("==" * 30)


def collect_records_for_params(
    *,
    param_name: str,
    param_values,
    split: str,
    object_class: str,
    rootdir: str,
    transform,
    anomaly_class: str,
    image_size: int,
    center_size: int,
    center_crop: bool,
    process_split_fn,
    diffusion,
    model,
    vae,
    reverse_steps,
    batch_num,
    device=None,
    dataset_class,
    split_csv_path: str | None = None,
):
    common_args = dict(
        mode=split,
        object_class=object_class,
        rootdir=rootdir,
        transform=transform,
        anomaly_class=anomaly_class,
        image_size=image_size,
        center_size=center_size,
        center_crop=center_crop,
        split_csv_path=split_csv_path,
    )
    all_records = []
    for val in param_values:
        print(f"Processing {param_name} = {val}")
        kwargs = common_args.copy()
        kwargs[param_name] = val
        dataset = dataset_class(**kwargs)
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False
        )
        records = process_split_fn(
            loader,
            split,
            diffusion,
            model,
            vae,
            reverse_steps,
            center_size,
            batch_num,
            device,
        )

        kwargs["scratch"] = True
        dataset_defect = dataset_class(**kwargs)
        loader_defect = DataLoader(
            dataset_defect, batch_size=8, shuffle=False, num_workers=4, drop_last=False
        )
        records_defect = process_split_fn(
            loader_defect,
            split,
            diffusion,
            model,
            vae,
            reverse_steps,
            center_size,
            batch_num,
            device,
        )
        all_records.append((records, records_defect))
    return all_records


def compute_y_true_y_score(all_records):
    """
    For each param value, compute y_true and y_score arrays from records.
    Returns: list of (y_true, y_score) tuples, one for each param value.
    """
    y_true_score_list = []
    for i, (records, records_defect) in enumerate(all_records):
        y_true = []
        y_score = []
        for rec in records:
            y_true.append(0)
            mask = (
                rec["anomaly_map_arithmetic_binary"][1]
                if isinstance(rec["anomaly_map_arithmetic_binary"], tuple)
                else rec["anomaly_map_arithmetic_binary"]
            )
            num_white = np.sum(mask == 1)
            y_score.append(num_white)
        for rec in records_defect:
            y_true.append(1)
            mask = (
                rec["anomaly_map_arithmetic_binary"][1]
                if isinstance(rec["anomaly_map_arithmetic_binary"], tuple)
                else rec["anomaly_map_arithmetic_binary"]
            )
            num_white = np.sum(mask == 1)
            y_score.append(num_white)
        y_score = np.array(y_score)
        y_true = np.array(y_true)
        y_true_score_list.append((y_true, y_score))
    return y_true_score_list


def compute_metrics_from_y_true_y_score(y_true_score_list):
    """
    For each (y_true, y_score), compute accuracy, threshold, and ROC stats.
    Returns: accuracies, thresholds, and a dict of lists for each ROC metric.
    """
    accuracies = []
    fpr_list = []
    tpr_list = []
    thresholds_list = []
    best_thresholds = []
    best_idxs = []
    aucs = []
    y_trues = []
    y_preds = []
    y_scores = []

    for i, (y_true, y_score) in enumerate(y_true_score_list):
        fpr, tpr, thresholds_, best_threshold, best_idx, auc_score = compute_roc_stats(
            y_true, y_score
        )
        y_pred = (y_score >= best_threshold).astype(int)
        y_preds.append(y_pred)
        accuracy = np.mean(y_pred == y_true)
        accuracies.append(accuracy)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        # thresholds_list.append(thresholds_)
        best_thresholds.append(best_threshold)
        best_idxs.append(best_idx)
        aucs.append(auc_score)
        y_trues.append(y_true)
        y_scores.append(y_score)
        print(f"Accuracy {accuracy:.4f} (threshold={best_threshold})")

    roc_stats = {
        "fpr": fpr_list,
        "tpr": tpr_list,
        # "thresholds": thresholds_list,
        "best_threshold": best_thresholds,
        "best_idx": best_idxs,
        "auc": aucs,
        "y_true": y_trues,
        "y_pred": y_preds,
        "y_score": y_scores,
        "accuracies": accuracies,
    }
    return roc_stats


def plot_accuracy_results(
    param_values,
    accuracies,
    param_name: str,
    save_dir="accuracy_vs_param",
    save_filename=datetime.now().strftime("%y%m%d_%H%M%S"),
    title=None,
    xlabel=None,
    ylabel="Accuracy",
    grid=True,
    marker="o",
    **plot_kwargs,
):
    """
    Plot accuracy results with customizable parameters.

    Args:
        param_values: List of parameter values
        accuracies: List of corresponding accuracies
        param_name: Name of the parameter being varied
        save_dir: Directory to save the plot
        save_filename: Filename for saving the plot
        title: Custom title for the plot
        xlabel: Custom x-axis label
        ylabel: Custom y-axis label
        grid: Whether to show grid
        marker: Marker style for the plot
        **plot_kwargs: Additional plotting parameters
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(param_values, accuracies, marker=marker, **plot_kwargs)
    plt.xlabel(xlabel or param_name)
    plt.ylabel(ylabel)
    plt.title(title or f"Accuracy vs {param_name.capitalize()} (synthetic defect)")
    plt.ylim(0.5, 1.0)
    plt.grid(grid)
    out_path = os.path.join(save_dir, f"accuracy_vs_{param_name}_{save_filename}.png")
    plt.savefig(out_path)
    print(f"Accuracy vs {param_name} saved to {out_path}")
    plt.close()


def save_perturbation_results(
    param_name: str,
    roc_stats: dict,
    param_values: list,
    save_dir: str,
):
    """
    Save perturbation experiment data (roc_stats and param_values) to a specified folder in JSON format.

    Args:
        param_name: Name of the perturbation parameter
        roc_stats: Dictionary containing ROC statistics
        param_values: List of parameter values
        save_dir: Directory to save the results
    """
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        else:
            return obj

    # Convert roc_stats to JSON-serializable format
    roc_stats_json = convert_for_json(roc_stats)
    
    # Convert param_values to JSON-serializable format
    param_values_json = convert_for_json(param_values)

    # Save both roc_stats and param_values in a single JSON file
    results_data = {
        "param_name": param_name,
        "param_values": param_values_json,
        "roc_stats": roc_stats_json
    }
    
    json_path = os.path.join(save_dir, f"{param_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"Perturbation results saved to {json_path}")


def compute_roc_stats(y_true, y_score):
    """
    Compute ROC statistics and best threshold using Youden's J statistic.
    Returns: fpr, tpr, thresholds, best_threshold, best_idx, auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, best_threshold, best_idx, auc_score


def compute_confusion_matrix_and_accuracy(annotation_dir, evaluation_results_dir):
    import glob
    from collections import Counter
    import numpy as np
    import os
    import json

    # Check if evaluation results directory exists
    if not os.path.exists(evaluation_results_dir):
        print(f"Warning: Evaluation results directory {evaluation_results_dir} does not exist")
        return
    
    # Find all evaluation result files
    eval_files = glob.glob(os.path.join(evaluation_results_dir, '*__evaluation.json'))
    
    if not eval_files:
        print(f"Warning: No evaluation result files found in {evaluation_results_dir}")
        return
    all_TP = all_FP = all_FN = all_TN = 0
    for eval_file in eval_files:
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Corrupted JSON file {eval_file}: {e}")
            print(f"Skipping this file and continuing...")
            continue
        except Exception as e:
            print(f"Warning: Error reading file {eval_file}: {e}")
            print(f"Skipping this file and continuing...")
            continue
        image_path = eval_data['image_path']
        # Handle new format with patch_analysis
        if 'patch_analysis' in eval_data:
            # New format: extract defective patches from patch_analysis
            predicted = set()
            for patch in eval_data['patch_analysis']:
                if patch['is_defective']:
                    predicted.add((patch['grid_row'], patch['grid_col']))
        else:
            # Old format: direct defective_patches list
            predicted = set(tuple(x) for x in eval_data['defective_patches'])
        grid_size = eval_data['grid_size']
        annotation_file = os.path.join(annotation_dir, f"{path_to_safe_filename(image_path)}__annotations.json")
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation for {image_path}")
            continue
        try:
            with open(annotation_file, 'r') as f:
                anno_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Corrupted annotation file {annotation_file}: {e}")
            print(f"Skipping this file and continuing...")
            continue
        except Exception as e:
            print(f"Warning: Error reading annotation file {annotation_file}: {e}")
            print(f"Skipping this file and continuing...")
            continue
        gt = set(tuple(x) for x in anno_data['defective_patches'])
        # Get all possible grid cells
        # (Assume image is divisible by grid_size)
        img = PILImage.open(image_path)
        h, w = img.height, img.width
        n_rows = h // grid_size
        n_cols = w // grid_size
        all_cells = set((r, c) for r in range(n_rows) for c in range(n_cols))
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
    cm = np.array([[all_TN, all_FP], [all_FN, all_TP]])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
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
    plt.xticks(tick_marks, ['Normal', 'Defective'], fontsize=12)
    plt.yticks(tick_marks, ['Normal', 'Defective'], fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1_score:.4f}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save the confusion matrix plot
    cm_plot_path = os.path.join(evaluation_results_dir, "confusion_matrix.png")
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
    with open(os.path.join(evaluation_results_dir, "confusion_matrix.json"), "w") as f:
        json.dump(result, f, indent=2)











def main():
    REPO_ROOT = os.environ.get("REPO_ROOT", None)
    if REPO_ROOT is not None:
        os.chdir(os.path.dirname(REPO_ROOT))
        print("Current path:", os.getcwd())
    if "ipykernel_launcher" in sys.argv[0]:
        print("Running in IPython kernel")
        sys.argv = [
            "",
            "--dataset",
            "pcb",
            "--data-dir",
            os.path.expanduser(
                "~/dataset/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff"
            ),
            "--model-size",
            "UNet_L",
            "--object-class",
            "all",
            "--anomaly-class",
            "all",
            "--image-size",
            "128",
            "--center-size",
            "128",
            "--center-crop",
            "False",
            "--batch-num",
            "1",
            "--pretrained",
            "DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L/checkpoints/best.pt",
            "--split",
            "train",
            "--perturbation",
            "noise",
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["mvtec", "visa", "pcb"], default="mvtec"
    )
    parser.add_argument("--data-dir", type=str, default="./mvtec-dataset/")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["UNet_XS", "UNet_S", "UNet_M", "UNet_L", "UNet_XL"],
        default="UNet_L",
    )
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--batch-num", type=int, default=12)
    parser.add_argument(
        "--center-crop",
        type=lambda v: True if v.lower() in ("yes", "true", "t", "y", "1") else False,
        default=True,
    )
    parser.add_argument(
        "--vae-type", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-class", type=str, default="all")
    parser.add_argument("--pretrained", type=str, default=".")
    parser.add_argument("--anomaly-class", type=str, default="all")
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--perturbation",
        type=str,
        choices=[None, "brightness", "shift_x", "shift_y", "noise", "blur", "scratch"],
        default=None,
    )
    parser.add_argument("--split-csv-path", type=str, default=None)
    parser.add_argument(
        "--input-json",
        type=str,
        help="Path to JSON file containing multiple test configurations"
    )
    parser.add_argument(
        "--irregular-images",
        action="store_true",
        help="Process irregular-sized images by splitting into 128x128 patches"
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        help="Directory containing JSON annotation files for defective regions"
    )
    parser.add_argument(
        "--anomaly-binary-threshold",
        type=int,
        default=5,
        help="Threshold for determining if a patch is defective based on number of anomaly pixels (default: 5)"
    )
    parser.add_argument(
        "--anomaly-pixel-num-threshold",
        type=int,
        default=0,
        help="Threshold for determining if a patch is defective based on number of anomaly pixels (default: 0)"
    )
    parser.add_argument(
        "--memory-optimization",
        action="store_true",
        help="Enable memory optimization for large datasets (reduces batch size and workers)"
    )
    parser.add_argument(
        "--enable-epoch-stats",
        action="store_true",
        default=False,
        help="Enable epoch statistics printing (memory intensive for large datasets)"
    )
    parser.add_argument(
        "--enable-excel-report",
        action="store_true",
        default=False,
        help="Enable Excel report generation (memory intensive for large datasets)"
    )
    args = parser.parse_args()
    
    # Handle input JSON if provided
    if args.input_json:
        import json
        with open(args.input_json, 'r') as f:
            test_configs = json.load(f)
            
        # Run evaluation for each test configuration
        for test_name, test_args in test_configs.items():
            print(f"\nRunning evaluation for {test_name}")
            print(test_args)
            # Update args with test configuration
            for key, value in test_args.items():
                # Convert key from kebab-case to snake_case
                key = key.replace('-', '_')
                if hasattr(args, key):
                    # Convert string values to appropriate types
                    if key in ['image_size', 'center_size', 'batch_num', 'reverse_steps', 'anomaly_binary_threshold', 'anomaly_pixel_num_threshold']:
                        value = int(value)
                    elif key == 'center_crop':
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    elif key in ['pretrained', 'data_dir', 'split_csv_path', 'annotation_dir']:
                        value = os.path.expanduser(value)
                    elif key in ['irregular_images', 'memory_optimization', 'enable_epoch_stats', 'enable_excel_report']:
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    setattr(args, key, value)
            
            # Set up derived arguments
            if args.dataset == "mvtec":
                args.num_classes = 15
            elif args.dataset == "visa":
                args.num_classes = 12
            elif args.dataset == "pcb":
                args.num_classes = 1
            current_time = datetime.now().strftime("%y%m%d_%H%M%S")
            args.results_dir = f"results/{test_name}_{current_time}"
            os.makedirs(args.results_dir, exist_ok=True)
            # INSERT_YOUR_CODE
            # Save the current test_args (key-value pairs) into the results_dir as a JSON file
            config_save_path = os.path.join(args.results_dir, "config.json")
            with open(config_save_path, "w") as config_file:
                json.dump(test_args, config_file, indent=2)
            if args.center_crop:
                args.actual_image_size = args.center_size
            else:
                args.actual_image_size = args.image_size

            # Set up object classes
            if args.object_class == "all" and args.dataset == "mvtec":
                args.object_classes = [
                    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
                    "pill", "screw", "toothbrush", "transistor", "zipper",
                    "carpet", "grid", "leather", "tile", "wood",
                ]
            elif args.object_class == "all" and args.dataset == "visa":
                args.object_classes = [
                    "candle", "cashew", "fryum", "macaroni2", "pcb2", "pcb4",
                    "capsules", "chewinggum", "macaroni1", "pcb1", "pcb3", "pipe_fryum",
                ]
            elif args.object_class == "all" and args.dataset == "pcb":
                args.object_classes = ["pcb"]
            else:
                args.object_classes = [args.object_class]
                
            # Run evaluation for this configuration
            evaluation(args)
    else:
        # Original single configuration evaluation
        if args.dataset == "mvtec":
            args.num_classes = 15
        elif args.dataset == "visa":
            args.num_classes = 12
        elif args.dataset == "pcb":
            args.num_classes = 1
        args.results_dir = f"./DeCo-Diff_{args.dataset}_{args.object_class}_{args.model_size}_{args.center_size}"
        if args.center_crop:
            args.results_dir += "_CenterCrop"
            args.actual_image_size = args.center_size
        else:
            args.actual_image_size = args.image_size

        if args.object_class == "all" and args.dataset == "mvtec":
            args.object_classes = [
                "bottle",
                "cable",
                "capsule",
                "hazelnut",
                "metal_nut",
                "pill",
                "screw",
                "toothbrush",
                "transistor",
                "zipper",
                "carpet",
                "grid",
                "leather",
                "tile",
                "wood",
            ]
        elif args.object_class == "all" and args.dataset == "visa":
            args.object_classes = [
                "candle",
                "cashew",
                "fryum",
                "macaroni2",
                "pcb2",
                "pcb4",
                "capsules",
                "chewinggum",
                "macaroni1",
                "pcb1",
                "pcb3",
                "pipe_fryum",
            ]
        elif args.object_class == "all" and args.dataset == "pcb":
            args.object_classes = [
                "pcb",
            ]
        else:
            args.object_classes = [args.object_class]

        evaluation(args)


# %%
if __name__ == "__main__":
    main()
# Below are cell makrkers used in VSCode
# %%
#
# %%
