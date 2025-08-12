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
from diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PCBDataLoader import PCBDataset

import sys
from typing import List
from tqdm import tqdm

from PIL import Image as PILImage
from typing import Sequence

from typing import Any, Tuple, cast

import json
import cv2
import glob

import re
import platform
from utils import path_to_safe_filename

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cpu"):
    print("GPU not found. Using CPU instead.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LATENT_SCALE = 0.18215

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
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _get_dataloader(self, dataset, batch_size=None, shuffle=False):
        batch_size = batch_size or self.args.batch_num
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Windows-friendly
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        
    def _create_output_dirs(self):
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _save_image(self, img, path, description=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)
        if description:
            print(f"Saved {description}: {path}")
            
    def _save_json(self, data, path, description=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        if description:
            print(f"Saved {description}: {path}")

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
        
        # Extract image path and defective patches
        image_path = annotation_data.get('image_path', '')
        defective_patches = annotation_data.get('defective_patches', [])
        
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
    
    def _extract_patches(self, img):
        """Extract patches from image without overlap, including edge patches."""
        patches = []
        coords = []
        
        height, width = img.shape[:2]
        stride = self.patch_size  # No overlap - patches are adjacent
        
        # Calculate the range of patch positions
        y_positions = list(range(0, height, stride))
        x_positions = list(range(0, width, stride))
        
        # Ensure we include edge patches if they don't perfectly fit
        if y_positions and height - y_positions[-1] >= self.patch_size:
            # Last position already fits, no need to add edge patch
            pass
        elif y_positions and height - y_positions[-1] > 0:
            # Add edge patch that starts at the position to capture the far-bottom
            y_positions.append(max(0, height - self.patch_size))
        
        if x_positions and width - x_positions[-1] >= self.patch_size:
            # Last position already fits, no need to add edge patch
            pass
        elif x_positions and width - x_positions[-1] > 0:
            # Add edge patch that starts at the position to capture the far-right
            x_positions.append(max(0, width - self.patch_size))
        
        # Remove duplicates while preserving order
        y_positions = list(dict.fromkeys(y_positions))
        x_positions = list(dict.fromkeys(x_positions))
        
        for y in y_positions:
            for x in x_positions:
                # Calculate actual patch dimensions
                patch_width = min(self.patch_size, width - x)
                patch_height = min(self.patch_size, height - y)
                
                # Extract patch
                patch = img[y:y + patch_height, x:x + patch_width]
                
                # Pad patch to full size if necessary
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=patch.dtype)
                    padded_patch[:patch_height, :patch_width] = patch
                    patch = padded_patch
                
                patches.append(patch)
                coords.append((x, y))
        
        # Handle edge cases if image is smaller than patch_size
        if height < self.patch_size or width < self.patch_size:
            # Pad the entire image to patch_size and return as single patch
            if not patches:  # Only if no patches were extracted
                padded_img = np.zeros((self.patch_size, self.patch_size, 3), dtype=img.dtype)
                actual_height = min(height, self.patch_size)
                actual_width = min(width, self.patch_size)
                padded_img[:actual_height, :actual_width] = img[:actual_height, :actual_width]
                patches.append(padded_img)
                coords.append((0, 0))
        
        return patches, coords

def _compute_abs_diff_mean(a: torch.Tensor, b: torch.Tensor, diff_scale: float = 1.0) -> torch.Tensor:
    return torch.abs(a - b).mean(dim=1, keepdim=True) * diff_scale

def _compute_abs_diff_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.abs(a - b).max(dim=1, keepdim=True)[0]

def _to_numpy(
    t: torch.Tensor,
) -> np.ndarray:  # keep Images API compatibility
    return t.detach().cpu().numpy()

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

class CheckpointManager:
    """Manages checkpoint files for resuming evaluation."""
    
    def __init__(self, results_dir: str, annotation_dir: str | None = None, force_rerun: bool = False):
        self.results_dir = results_dir
        self.annotation_dir = annotation_dir
        self.force_rerun = force_rerun
        self.evaluation_results_dir = os.path.join(results_dir, "evaluation_results")
        os.makedirs(self.evaluation_results_dir, exist_ok=True)
        
        # Create checkpoint directory: <results_dir_without_timestamp>_checkpoints
        results_dir_without_timestamp = self._remove_timestamp_from_path(results_dir)
        self.checkpoint_dir = f"{results_dir_without_timestamp}_checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if force_rerun:
            self.clear_checkpoint_files()
    
    def _remove_timestamp_from_path(self, path: str) -> str:
        """Remove timestamp pattern from the full path."""
        import re
        # Remove timestamp patterns (YYMMDD_HHMMSS) from the end of the path
        return re.sub(r'_\d{6}_\d{6}$', '', path)
    
    def _extract_base_name(self, results_dir: str) -> str:
        """Extract base name from results directory path."""
        # Remove common prefixes and suffixes
        base_name = os.path.basename(results_dir)
        
        # Remove timestamp patterns (YYMMDD_HHMMSS)
        import re
        base_name = re.sub(r'_\d{6}_\d{6}$', '', base_name)
        
        return base_name
    
    def find_latest_checkpoint(self) -> str:
        """Find the checkpoint file."""
        base_name = self._extract_base_name(self.results_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{base_name}_checkpoint.json")
        
        if os.path.exists(checkpoint_file):
            return checkpoint_file
        else:
            return None
    
    def get_checkpoint_data(self) -> dict:
        """Get checkpoint data from the latest checkpoint file."""
        checkpoint_file = self.find_latest_checkpoint()
        if checkpoint_file and os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_processed_images_with_cache(self) -> set:
        """Get processed images from checkpoint with caching."""
        checkpoint_data = self.get_checkpoint_data()
        processed_images = set(checkpoint_data.get('processed_images', []))
        return processed_images
    
    def save_checkpoint(self, current_image_index: int, processed_images: list):
        """Save checkpoint data."""
        base_name = self._extract_base_name(self.results_dir)
        checkpoint_file = os.path.join(
            self.checkpoint_dir, 
            f"{base_name}_checkpoint.json"
        )
        
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        checkpoint_data = {
            'current_image_index': current_image_index,
            'processed_images': processed_images,
            'timestamp': timestamp
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_file}")
    
    def get_processed_images(self) -> set:
        """Get set of processed image paths."""
        return self._get_processed_images_with_cache()
    
    def mark_image_processed(self, image_path: str):
        """Mark an image as processed."""
        processed_images = self.get_processed_images()
        processed_images.add(image_path)
        
        # Save updated checkpoint
        checkpoint_data = self.get_checkpoint_data()
        checkpoint_data['processed_images'] = list(processed_images)
        
        # Save to checkpoint file
        checkpoint_file = self.find_latest_checkpoint()
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def is_image_processed(self, image_path: str) -> bool:
        """Check if an image has been processed."""
        processed_images = self.get_processed_images()
        return image_path in processed_images
    
    def get_resume_info(self, all_image_paths: list) -> tuple[int, list]:
        """Get resume information for evaluation."""
        if self.force_rerun:
            return 0, []
        
        checkpoint_data = self.get_checkpoint_data()
        current_image_index = checkpoint_data.get('current_image_index', 0)
        processed_images = set(checkpoint_data.get('processed_images', []))
        
        # Filter out already processed images
        remaining_images = [img for img in all_image_paths if img not in processed_images]
        
        return current_image_index, remaining_images
    
    def cleanup_checkpoint(self):
        """Clean up old checkpoint files, keeping only the latest."""
        # Since we now use a single checkpoint file, no cleanup is needed
        pass
    
    def clear_checkpoint_files(self):
        """Clear all checkpoint files."""
        base_name = self._extract_base_name(self.results_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{base_name}_checkpoint.json")
        
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"Removed checkpoint: {checkpoint_file}")
    
    def batch_mark_images_processed(self, image_paths: list):
        """Mark multiple images as processed in batch."""
        processed_images = self.get_processed_images()
        processed_images.update(image_paths)
        
        # Save updated checkpoint
        checkpoint_data = self.get_checkpoint_data()
        checkpoint_data['processed_images'] = list(processed_images)
        
        # Save to checkpoint file
        checkpoint_file = self.find_latest_checkpoint()
        if checkpoint_file:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def print_checkpoint_status(self):
        """Print current checkpoint status."""
        checkpoint_data = self.get_checkpoint_data()
        processed_count = len(checkpoint_data.get('processed_images', []))
        current_index = checkpoint_data.get('current_image_index', 0)
        
        print(f"Checkpoint Status:")
        print(f"  - Current image index: {current_index}")
        print(f"  - Processed images: {processed_count}")
        print(f"  - Force rerun: {self.force_rerun}")

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
            path = f"./DeCo-Diff_{args.dataset}_{args.object_class}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f"{path}/last.pt"))[-1]
            except (IndexError, FileNotFoundError):
                ckpt = sorted(glob(f"{path}/*/last.pt"))[-1]
    except (IndexError, FileNotFoundError, OSError) as e:
        raise Exception(f"Please provide the model's pretrained path using --pretrained. Error: {e}")

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

    evaluation_annotated_images(args, diffusion, model, vae)

def evaluation_annotated_images(args, diffusion, model, vae):
    """Evaluate images with JSON annotations for defective regions."""
    # Create base evaluator
    evaluator = BaseEvaluator(args, diffusion, model, vae)
    
    # Create dataset for annotated images
    dataset = AnnotatedImageDataset(
        annotation_dir=args.annotation_dir,
        patch_size=args.patch_size,
        transform=evaluator.transform,
        object_class=args.object_class,
    )
    
    loader = evaluator._get_dataloader(dataset, batch_size=1)  # Reduced batch size to avoid memory issues
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(args.results_dir, args.annotation_dir, args.force_rerun)
    
    # Print checkpoint status
    print("=== Checkpoint Status ===")
    checkpoint_manager.print_checkpoint_status()
    print("========================")

    # Process patches with minimal diff functionality (no records returned)
    process_split_irregular_minimal_diff(
        loader,
        args.split,
        diffusion,
        model,
        vae,
        args.reverse_steps,
        args.center_size,
        args.batch_num,
        device,
        checkpoint_manager.evaluation_results_dir,
        args.patch_size,
        checkpoint_manager=checkpoint_manager
    )

def process_split_irregular_minimal_diff(
    dataloader,
    split: str,
    diffusion,
    model,
    vae,
    reverse_steps: int,
    center_size: int,
    batch_num: int,
    device: torch.device = torch.device("cpu"),
    save_dir: str = "minimal_diff_results",
    patch_size: int = 256,
    checkpoint_manager: "CheckpointManager" | None = None
) -> None:
    """
    Minimal function that only saves encodedrecon_dodrecon_diff and encoded_latent_diff_resized.
    Performance optimized - no extra computations or features.
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    idx = -1
    
    try:
        for idx, (x, seg, object_cls, image_paths, anomaly_classes, patch_coords) in enumerate(
            tqdm(dataloader, desc=f"{split} split")
        ):
            if idx >= batch_num:
                break

            with torch.no_grad():
                # Process patches in chunks to avoid memory issues
                x_chunks = process_patches_in_chunks(x, chunk_size=16)  # Smaller chunks for memory efficiency
                object_cls_chunks = process_patches_in_chunks(object_cls, chunk_size=16)
                
                all_encodedrecon_diffs = []
                all_encoded_latent_diffs = []
                all_anomaly_maps = []
                
                for chunk_idx, (x_chunk, object_cls_chunk) in enumerate(zip(x_chunks, object_cls_chunks)):
                    # Move chunk to device
                    x_device = x_chunk.to(device)
                    object_cls_device = object_cls_chunk.to(device)
                    
                    # -----------------------------------------------------------------
                    # Forward pass through VAE encoder (to latent space)
                    # -----------------------------------------------------------------
                    encoded = vae.encode(x_device).latent_dist.mean * _LATENT_SCALE

                    # -----------------------------------------------------------------
                    # Reverse DDIM sampling conditioned on encoder latents
                    # -----------------------------------------------------------------
                    # The model expects raw class indices and will handle embedding internally
                    model_kwargs = {"context": object_cls_device.unsqueeze(1), "mask": None}
                    
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

                    # -----------------------------------------------------------------
                    # Core computations (only what we need)
                    # -----------------------------------------------------------------
                    # Decode final latent samples
                    image_samples = vae.decode(latent_samples_final / _LATENT_SCALE).sample
                    x0 = vae.decode(encoded / _LATENT_SCALE).sample
                    
                    # Core difference computations
                    encodedrecon_dodrecon_diff_raw = _compute_abs_diff_max(x0, image_samples)
                    encodedrecon_dodrecon_diff = torch.clamp(encodedrecon_dodrecon_diff_raw, 0.0, 0.05) * 20

                    encoded_latent_diff_raw = _compute_abs_diff_mean(latent_samples_final, encoded)
                    encoded_latent_diff = torch.clamp(encoded_latent_diff_raw, 0.0, 0.05) * 20

                    # Resize encoded_latent_diff to match the spatial dimensions
                    patch_size_actual = x_device.shape[-1]  # Should be 128 for patches
                    encoded_latent_diff_resized = F.interpolate(
                        encoded_latent_diff,
                        size=(patch_size_actual, patch_size_actual),
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map_arithmetic = 0.5 * (encodedrecon_dodrecon_diff + encoded_latent_diff_resized)
                    
                    # Store results
                    all_encodedrecon_diffs.append(encodedrecon_dodrecon_diff)
                    all_encoded_latent_diffs.append(encoded_latent_diff_resized)
                    all_anomaly_maps.append(anomaly_map_arithmetic)
                    
                    # Clear memory after each chunk
                    del x_device, object_cls_device, encoded, latent_samples_list, latent_samples_final
                    del image_samples, x0, encodedrecon_dodrecon_diff_raw, encoded_latent_diff_raw
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Concatenate all results
                encodedrecon_dodrecon_diff = torch.cat(all_encodedrecon_diffs, dim=0)
                encoded_latent_diff_resized = torch.cat(all_encoded_latent_diffs, dim=0)
                anomaly_map_arithmetic = torch.cat(all_anomaly_maps, dim=0)
            
            # ---------------------------------------------------------------------
            # Per‑sample aggregation and saving
            # ---------------------------------------------------------------------
            batch_size = x.size(0)
            
            for b in range(batch_size):
                # Get patch coordinates - handle case where patch_coords might not have enough elements
                if b < len(patch_coords):
                    patch_coord_tensor = patch_coords[b]
                else:
                    # If we don't have enough coordinates, use the last available one or default
                    patch_coord_tensor = patch_coords[-1] if patch_coords else [0, 0]
                
                # Debug print to see what we're getting
                #print(f"Debug - patch_coord_tensor type: {type(patch_coord_tensor)}, value: {patch_coord_tensor}")
                
                if isinstance(patch_coord_tensor, torch.Tensor):
                    # If it's a single tensor with 2 elements [x, y]
                    if patch_coord_tensor.numel() == 2:
                        x_coord = int(patch_coord_tensor[0].item())
                        y_coord = int(patch_coord_tensor[1].item())
                    else:
                        print(f"Warning: unexpected tensor shape, using default coordinates")
                        x_coord, y_coord = 0, 0
                elif isinstance(patch_coord_tensor, (list, tuple)) and len(patch_coord_tensor) == 2:
                    # Handle list/tuple of tensors or values
                    first_elem = patch_coord_tensor[0]
                    second_elem = patch_coord_tensor[1]
                    
                    if isinstance(first_elem, torch.Tensor):
                        if first_elem.numel() == 1:
                            x_coord = int(first_elem.item())
                        else:
                            print(f"Warning: first tensor has {first_elem.numel()} elements, using default")
                            x_coord = 0
                    else:
                        x_coord = int(first_elem)
                        
                    if isinstance(second_elem, torch.Tensor):
                        if second_elem.numel() == 1:
                            y_coord = int(second_elem.item())
                        else:
                            print(f"Warning: second tensor has {second_elem.numel()} elements, using default")
                            y_coord = 0
                    else:
                        y_coord = int(second_elem)
                else:
                    print(f"Warning: unexpected patch_coord format, using default coordinates")
                    x_coord, y_coord = 0, 0
                
                #print(f"Debug - extracted coordinates: x_coord={x_coord}, y_coord={y_coord}")
                
                # Get all 4 corner coordinates for irregular patches
                # Don't assume rectangular shape - capture all corners
                x1, y1 = x_coord, y_coord  # Top-left corner
                
                # For irregular patches, we need to get the actual patch coordinates
                # from the dataset to handle non-rectangular shapes
                if hasattr(dataloader.dataset, '_extract_patches'):
                    try:
                        # Get the original image to calculate actual patch boundaries
                        # Handle image_paths as list of paths - with safety check
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
                        original_img = np.array(PILImage.open(image_path).convert('RGB'))
                        #height, width = original_img.shape[:2]
                        
                        # Get the actual patch extraction coordinates
                        patches, coords = dataloader.dataset._extract_patches(original_img)
                        
                        # Find the specific patch coordinates for this patch
                        patch_idx = None
                        for i, (patch_x, patch_y) in enumerate(coords):
                            if patch_x == x_coord and patch_y == y_coord:
                                patch_idx = i
                                break
                        
                        if patch_idx is not None:
                            # Get the actual patch to determine its real dimensions
                            actual_patch = patches[patch_idx]
                            patch_height, patch_width = actual_patch.shape[:2]
                            
                            # Calculate all 4 corner coordinates based on actual patch
                            x1, y1 = x_coord, y_coord  # Top-left
                            x2, y2 = x_coord + patch_width, y_coord  # Top-right
                            x3, y3 = x_coord + patch_width, y_coord + patch_height  # Bottom-right
                            x4, y4 = x_coord, y_coord + patch_height  # Bottom-left
                        else:
                            # Fallback to rectangular coordinates
                            x2, y2 = x_coord + patch_size_actual, y_coord  # Top-right
                            x3, y3 = x_coord + patch_size_actual, y_coord + patch_size_actual  # Bottom-right
                            x4, y4 = x_coord, y_coord + patch_size_actual  # Bottom-left
                    except Exception as e:
                        print(f"Warning: Could not get actual patch coordinates, using rectangular: {e}")
                        # Fallback to rectangular coordinates
                        x2, y2 = x_coord + patch_size_actual, y_coord  # Top-right
                        x3, y3 = x_coord + patch_size_actual, y_coord + patch_size_actual  # Bottom-right
                        x4, y4 = x_coord, y_coord + patch_size_actual  # Bottom-left
                else:
                    # Fallback to rectangular coordinates
                    x2, y2 = x_coord + patch_size_actual, y_coord  # Top-right
                    x3, y3 = x_coord + patch_size_actual, y_coord + patch_size_actual  # Bottom-right
                    x4, y4 = x_coord, y_coord + patch_size_actual  # Bottom-left
                
                # Create base filename with file info and patch info (all 4 corners)
                # Handle image_paths as list of paths - with safety check
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

                    # Merge with previously processed images (from latest checkpoint)
                    previously_processed = checkpoint_manager.get_processed_images()
                    merged = list(set(previously_processed).union(set(unique_processed)))

                    # Save a new checkpoint reflecting progress up to current image index
                    checkpoint_manager.save_checkpoint(
                        current_image_index=idx + 1,
                        processed_images=merged,
                    )
                except Exception as e:
                    print(f"Warning: failed to save checkpoint after image {idx}: {e}")

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

    print(f"Minimal diff processing completed. Results saved to {save_dir}")

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
    parser.add_argument("--object-class", type=str, default="all")
    parser.add_argument("--pretrained", type=str, default=".")
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
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
        "--patch-size",
        type=int,
        default=128,
        help="Patch size for splitting images (default: 128)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Force rerun evaluation even if checkpoint exists"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        default=False,
        help="Do not append current time to results directory names (applies to --input-json batch mode)"
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
                    if key in ['image_size', 'center_size', 'patch_size', 'batch_num', 'reverse_steps']:
                        value = int(value)
                    elif key == 'center_crop':
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    elif key in ['pretrained', 'data_dir', 'annotation_dir']:
                        value = os.path.expanduser(value)
                    elif key in ['irregular_images', 'force_rerun', 'no_timestamp']:
                        value = value.lower() in ('yes', 'true', 't', 'y', '1')
                    setattr(args, key, value)
            
            # Set up derived arguments
            if args.dataset == "mvtec":
                args.num_classes = 15
            elif args.dataset == "visa":
                args.num_classes = 12
            elif args.dataset == "pcb":
                args.num_classes = 1
            if getattr(args, 'no_timestamp', False):
                args.results_dir = f"results/{test_name}"
            else:
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