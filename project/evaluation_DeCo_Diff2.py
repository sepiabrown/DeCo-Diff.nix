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

from typing import Any, Tuple

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

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("GPU not found. Using CPU instead.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIFF_SCALE = 2.0
_THRESHOLD = 5.0 / 255.0
_LATENT_SCALE = 0.18215

Kinded = Tuple[str, Any]  # (kind, value)
Record = OrderedDict[str, Kinded]

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
        split_csv_path: str = None,
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
            image_path = os.path.join(data_dir, row['image'])
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

    a = to4d(rec["encoded_recon"][1])
    b = to4d(rec["dod_recon"][1])
    rec["lpips"] = ("metric", _lpips(a, b, net_type="alex").item())
    rec["ssim"] = ("metric", _ssim(a, b).item())
    rec["mse"] = ("metric", F.mse_loss(a, b).item())


def make_record(**kwargs) -> Record:
    """Return an **ordered** dict whose values are (kind, value) pairs."""
    return OrderedDict(kwargs)


def _compute_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the mean channel‑wise difference *scaled* by ``_DIFF_SCALE``."""
    return (a - b).mean(dim=1, keepdim=True) / _DIFF_SCALE


def _binary_mask(diff: torch.Tensor, threshold: float = _THRESHOLD) -> torch.Tensor:
    """Return a binary mask in ``{-1, 1}`` based on *absolute* diff magnitude."""
    return (diff.abs() > threshold).float() * 2.0 - 1.0


def _to_numpy(
    t: torch.Tensor,
) -> "Sequence | torch.Tensor":  # keep Images API compatibility
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
            scalars.append(val)
    ws.append(scalars)
    ws.row_dimensions[row_idx].height = size * 0.75
    for col_idx, img in embeds:
        ws.add_image(img, f"{get_column_letter(col_idx)}{row_idx}")


def mark_defective_regions_on_image(original_img, patch_results, ground_truth_patches=None, patch_size=128, stride=64):
    """
    Mark defective regions on the original image based on patch analysis results and ground truth.
    
    Args:
        original_img: Original image as numpy array
        patch_results: List of (x, y, is_defective) tuples for each patch
        ground_truth_patches: List of [grid_row, grid_col] coordinates for ground truth defective patches
        patch_size: Size of patches
        stride: Stride used for patch extraction
    
    Returns:
        Marked image with:
        - Red rectangles around predicted defective regions
        - Green rectangles around ground truth defective regions
        - Yellow rectangles where prediction and ground truth overlap
    """
    marked_img = original_img.copy()
    
    # Create sets for efficient lookup
    predicted_defective = set()
    ground_truth_defective = set()
    
    # Collect predicted defective patches
    for x, y, is_defective in patch_results:
        if is_defective:
            grid_row = y // patch_size
            grid_col = x // patch_size
            predicted_defective.add((grid_row, grid_col))
    
    # Collect ground truth defective patches
    if ground_truth_patches:
        for grid_row, grid_col in ground_truth_patches:
            ground_truth_defective.add((grid_row, grid_col))
    
    # Draw predicted defective regions (red)
    for grid_row, grid_col in predicted_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(marked_img, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 2)
    
    # Draw ground truth defective regions (yellow)
    for grid_row, grid_col in ground_truth_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(marked_img, (x, y), (x + patch_size, y + patch_size), (255, 255, 0), 2)
    
    # Draw overlapping regions (green) - where prediction and ground truth match
    overlapping = predicted_defective.intersection(ground_truth_defective)
    for grid_row, grid_col in overlapping:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(marked_img, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 3)
    
    return marked_img


def save_marked_image(marked_img, original_path, output_dir):
    """Save the marked image to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_marked{ext}")
    
    # Save image
    PILImage.fromarray(marked_img).save(output_path)
    return output_path


def create_anomaly_overlay(original_img, anomaly_map, alpha=0.6):
    """
    Create an overlay of the anomaly map on top of the original image.
    
    Args:
        original_img: Original image as numpy array (H, W, 3)
        anomaly_map: Anomaly map as numpy array (H, W) or (H, W, 1)
        alpha: Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)
    
    Returns:
        Overlay image as numpy array
    """
    # Ensure anomaly_map is 2D
    if anomaly_map.ndim == 3 and anomaly_map.shape[2] == 1:
        anomaly_map = anomaly_map.squeeze()
    
    # Normalize anomaly map to 0-1 range
    if anomaly_map.max() > 0:
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    
    # Create lime color overlay (lime = [0, 255, 0])
    lime_color = np.array([0, 255, 0], dtype=np.uint8)
    
    # Create colored overlay
    overlay = np.zeros_like(original_img)
    overlay[..., 0] = lime_color[0] * anomaly_map
    overlay[..., 1] = lime_color[1] * anomaly_map
    overlay[..., 2] = lime_color[2] * anomaly_map
    
    # Blend with original image
    result = original_img.astype(np.float32) * (1 - alpha * anomaly_map[..., np.newaxis]) + \
             overlay.astype(np.float32) * alpha * anomaly_map[..., np.newaxis]
    
    return result.astype(np.uint8)


def save_anomaly_overlay(overlay_img, original_path, output_dir):
    """Save the anomaly overlay image to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_anomaly_overlay{ext}")
    
    # Save image
    PILImage.fromarray(overlay_img).save(output_path)
    return output_path


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
) -> List[Record]:
    """Run a forward‑&‑reverse pass on irregular images and collect metrics."""

    results: List[Record] = []
    image_patch_results = defaultdict(list)  # Track results per original image
    image_anomaly_maps = defaultdict(list)  # Track anomaly maps per original image

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
            orig_dodrecon_diff = _compute_diff(x, image_samples)
            orig_encodedrecon_diff = _compute_diff(x, x0)
            encodedrecon_dodrecon_diff = _compute_diff(x0, image_samples)

            orig_dodrecon_binary = _binary_mask(orig_dodrecon_diff)
            orig_encodedrecon_binary = _binary_mask(orig_encodedrecon_diff)
            encodedrecon_dodrecon_binary = _binary_mask(encodedrecon_dodrecon_diff)

            encoded_latent_diff = (
                (latent_samples_final - encoded).max(dim=1, keepdim=True).values
            )
            encoded_latent_binary = _binary_mask(encoded_latent_diff)

            # Resize encoded_latent_diff to match the spatial dimensions of encodedrecon_dodrecon_diff
            # For irregular images, we want to use the patch size (128) as the target size
            patch_size = x.shape[-1]  # Should be 128 for patches
            encoded_latent_abs_diff_resized = F.interpolate(
                encoded_latent_diff.abs(),
                size=(patch_size, patch_size),
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
            # Determine if this patch is defective
            anomaly_binary = anomaly_map_arithmetic_binary[b]
            #encoded_latent = encoded_latent_binary[b]
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
            anomaly_pixels = torch.sum(anomaly_binary > 0).item()
            is_defective = anomaly_pixels > anomaly_binary_threshold  # Any white pixels
            
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
            
            image_patch_results[image_paths[b]].append((x_coord, y_coord, is_defective))
            
            # Store anomaly map for overlay creation
            anomaly_map_np = _to_numpy(anomaly_map_arithmetic[b])
            image_anomaly_maps[image_paths[b]].append((x_coord, y_coord, anomaly_map_np))
            
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
                encodedrecon_dodrecon_binary=(
                    "image",
                    _to_numpy(encodedrecon_dodrecon_binary[b]),
                ),
                encoded_latent_diff=("image", _to_numpy(encoded_latent_diff[b])),
                encoded_latent_binary=("image", _to_numpy(encoded_latent_binary[b])),
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
            orig_dodrecon_diff = _compute_diff(x, image_samples)
            orig_encodedrecon_diff = _compute_diff(x, x0)
            encodedrecon_dodrecon_diff = _compute_diff(x0, image_samples)

            orig_dodrecon_binary = _binary_mask(orig_dodrecon_diff)
            orig_encodedrecon_binary = _binary_mask(orig_encodedrecon_diff)
            encodedrecon_dodrecon_binary = _binary_mask(encodedrecon_dodrecon_diff)

            encoded_latent_diff = (
                (latent_samples_final - encoded).max(dim=1, keepdim=True).values
            )
            encoded_latent_binary = _binary_mask(encoded_latent_diff)

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
                encodedrecon_dodrecon_binary=(
                    "image",
                    _to_numpy(encodedrecon_dodrecon_binary[b]),
                ),
                encoded_latent_diff=("image", _to_numpy(encoded_latent_diff[b])),
                encoded_latent_binary=("image", _to_numpy(encoded_latent_binary[b])),
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
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}"  # @param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
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
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    
    # Create dataset for annotated images
    dataset = AnnotatedImageDataset(
        annotation_dir=args.annotation_dir,
        patch_size=128,
        transform=transform,
        object_class=args.object_class,
    )
    
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, num_workers=4, drop_last=False
    )
    
    # For annotated images, use patch size as center_size to ensure consistent dimensions
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
    )
    
    # Mark defective regions on original images
    print("Marking defective regions on original images...")
    marked_images_dir = os.path.join(args.results_dir, "marked_images")
    os.makedirs(marked_images_dir, exist_ok=True)
    
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
        marked_img = mark_defective_regions_on_image(
            original_img, patch_results, ground_truth_patches, patch_size=128, stride=128
        )
        
        safe_name = path_to_safe_filename(image_path)
        # Save marked image
        marked_path = os.path.join(marked_images_dir, f"{safe_name}__marked.png")
        PILImage.fromarray(marked_img).save(marked_path)
        print(f"Saved marked image: {marked_path}")

        # --- Create and save anomaly overlay image ---
        # Reconstruct the full anomaly map for this image
        anomaly_map_list = image_anomaly_maps[image_path]
        # Assume all patches are 128x128 and non-overlapping
        h, w, _ = original_img.shape
        full_anomaly_map = np.zeros((h, w), dtype=np.float32)
        for x, y, patch_map in anomaly_map_list:
            full_anomaly_map[y:y+128, x:x+128] = patch_map.squeeze()
        overlay_img = create_anomaly_overlay(original_img, full_anomaly_map, alpha=0.8)
        overlay_path = os.path.join(marked_images_dir, f"{safe_name}__anomaly_overlay.png")
        PILImage.fromarray(overlay_img).save(overlay_path)
        print(f"Saved anomaly overlay image: {overlay_path}")

        # Save overlay+patches image
        marked_overlay_img = draw_patch_rectangles_on_image(overlay_img, patch_results, ground_truth_patches, patch_size=128, stride=128)
        marked_overlay_path = os.path.join(marked_images_dir, f"{safe_name}__marked_overlay.png")
        PILImage.fromarray(marked_overlay_img).save(marked_overlay_path)
        print(f"Saved marked overlay image: {marked_overlay_path}")
    
    # Save evaluation results in the same format as annotation files
    evaluation_results_dir = os.path.join(args.results_dir, "evaluation_results")
    os.makedirs(evaluation_results_dir, exist_ok=True)
    
    for image_path, patch_results in image_patch_results.items():
        # Convert pixel coordinates back to grid coordinates
        predicted_defective_patches = []
        for x, y, is_defective in patch_results:
            if is_defective:
                grid_row = y // 128
                grid_col = x // 128
                predicted_defective_patches.append([grid_row, grid_col])
        result_filename = f"{path_to_safe_filename(image_path)}__evaluation.json"
        result_path = os.path.join(evaluation_results_dir, result_filename)
        evaluation_result = {
            "image_path": image_path,
            "defective_patches": predicted_defective_patches,
            "grid_size": 128
        }
        with open(result_path, 'w') as f:
            json.dump(evaluation_result, f, indent=2)
        print(f"Saved evaluation result: {result_path}")
    print("==" * 30)
    # Compute confusion matrix and accuracy
    compute_confusion_matrix_and_accuracy(args.annotation_dir, evaluation_results_dir)


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
        dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False
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
    )
    
    # Mark defective regions on original images
    print("Marking defective regions on original images...")
    marked_images_dir = os.path.join(args.results_dir, "marked_images")
    
    for image_path, patch_results in image_patch_results.items():
        # Load original image
        original_img = np.array(PILImage.open(image_path).convert('RGB'))
        
        # Mark defective regions
        marked_img = mark_defective_regions_on_image(
            original_img, patch_results, patch_size=128, stride=128
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
            {"x": x, "y": y, "is_defective": is_defective}
            for x, y, is_defective in patch_results
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
        common_args = dict(
            split=args.split,
            object_class=object_class,
            rootdir=args.data_dir,
            transform=transform,
            anomaly_class=args.anomaly_class,
            image_size=args.image_size,
            center_size=args.actual_image_size,
            center_crop=True,
            process_split_fn=process_split,
            diffusion=diffusion,
            model=model,
            vae=vae,
            reverse_steps=args.reverse_steps,
            batch_num=args.batch_num,
            device=device,
            split_csv_path=args.split_csv_path,
        )
        if args.dataset == "pcb":
            common_args["dataset_class"] = PCBDataset
        elif args.dataset == "mvtec":
            common_args["dataset_class"] = MVTECDataset
        elif args.dataset == "visa":
            common_args["dataset_class"] = VISADataset
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")
        if args.perturbation is not None:
            if args.perturbation == "brightness":
                param_values = np.arange(-20, 21, 1)
                record_pairs = collect_records_for_params(
                    param_name="brightness", param_values=param_values, **common_args
                )
            if args.perturbation == "shift_x":
                param_values = np.arange(-20, 21, 1)
                record_pairs = collect_records_for_params(
                    param_name="shift_x", param_values=param_values, **common_args
                )
            if args.perturbation == "shift_y":
                param_values = np.arange(-20, 21, 1)
                record_pairs = collect_records_for_params(
                    param_name="shift_y", param_values=param_values, **common_args
                )
            if args.perturbation == "noise":
                param_values = np.arange(0, 21, 1)
                record_pairs = collect_records_for_params(
                    param_name="noise", param_values=param_values, **common_args
                )
            if args.perturbation == "blur":
                param_values = np.arange(1, 42, 2)
                record_pairs = collect_records_for_params(
                    param_name="blur", param_values=param_values, **common_args
                )
            if args.perturbation == "scratch":
                param_values = [0]
                record_pairs = collect_records_for_params(
                    param_name="brightness", param_values=param_values, **common_args
                )
            y_true_score_list = compute_y_true_y_score(record_pairs)
            roc_stats = compute_metrics_from_y_true_y_score(y_true_score_list)
            save_perturbation_results(
                param_name=args.perturbation,
                roc_stats=roc_stats,
                param_values=param_values,
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
    split_csv_path: str = None,
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

    # Find all evaluation result files
    eval_files = glob.glob(os.path.join(evaluation_results_dir, '*__evaluation.json'))
    all_TP = all_FP = all_FN = all_TN = 0
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        image_path = eval_data['image_path']
        predicted = set(tuple(x) for x in eval_data['defective_patches'])
        grid_size = eval_data['grid_size']
        annotation_file = os.path.join(annotation_dir, f"{path_to_safe_filename(image_path)}__annotations.json")
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation for {image_path}")
            continue
        with open(annotation_file, 'r') as f:
            anno_data = json.load(f)
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


def path_to_safe_filename(file_path: str) -> str:
    """
    Convert an absolute file path to a safe filename by replacing path separators with underscores.
    Handles Windows drive letters and both types of path separators.
    Also replaces .png extension at the end with __png.
    """
    normalized_path = os.path.normpath(file_path)
    normalized_path = re.sub(r'^([a-zA-Z]):[\\/]', r'\1__', normalized_path)
    safe_name = re.sub(r'[\\/]', '__', normalized_path)
    # Replace .png at the end with __png
    safe_name = re.sub(r'\.png$', '__png', safe_name, flags=re.IGNORECASE)
    return safe_name


def draw_patch_rectangles_on_image(base_img, patch_results, ground_truth_patches=None, patch_size=128, stride=64):
    """
    Draw patch rectangles (TP/FP/FN) on top of an existing image (e.g., anomaly overlay).
    Args:
        base_img: The image to draw on (np.uint8, HxWx3)
        patch_results: List of (x, y, is_defective) tuples for each patch
        ground_truth_patches: List of [grid_row, grid_col] coordinates for ground truth defective patches
        patch_size: Size of patches
        stride: Stride used for patch extraction
    Returns:
        Image with rectangles drawn.
    """
    img = base_img.copy()
    predicted_defective = set()
    ground_truth_defective = set()
    for x, y, is_defective in patch_results:
        if is_defective:
            grid_row = y // patch_size
            grid_col = x // patch_size
            predicted_defective.add((grid_row, grid_col))
    if ground_truth_patches:
        for grid_row, grid_col in ground_truth_patches:
            ground_truth_defective.add((grid_row, grid_col))
    # Draw predicted defective regions (red)
    for grid_row, grid_col in predicted_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (255, 0, 0), 2)
    # Draw ground truth defective regions (yellow)
    for grid_row, grid_col in ground_truth_defective:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (255, 255, 0), 2)
    # Draw overlapping regions (green)
    overlapping = predicted_defective.intersection(ground_truth_defective)
    for grid_row, grid_col in overlapping:
        x = grid_col * patch_size
        y = grid_row * patch_size
        cv2.rectangle(img, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 3)
    return img


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
        type=float,
        default=0,
        help="Threshold for determining if a patch is defective based on number of anomaly pixels (default: 0)"
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
                    if key in ['image_size', 'center_size', 'batch_num', 'reverse_steps', 'anomaly_binary_threshold']:
                        value = int(value)
                    elif key == 'center_crop':
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    elif key in ['pretrained', 'data_dir', 'split_csv_path', 'annotation_dir']:
                        value = os.path.expanduser(value)
                    elif key == 'irregular_images':
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
