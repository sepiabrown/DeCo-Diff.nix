import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import albumentations as A
from PCBDataLoader import random_brightness_contrast, path_to_safe_filename
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import synthetic scratch functionality if available
try:
    from synthetic_scratch import add_scratch_controlled
    SYNTHETIC_SCRATCH_AVAILABLE = True
except ImportError:
    SYNTHETIC_SCRATCH_AVAILABLE = False
    print("Warning: synthetic_scratch module not found, scratch generation will be disabled")

def albumentations_brightness_contrast(img, brightness_limit=0.05, contrast_limit=0.05, p=0.5):
    """
    Fast Albumentations-based brightness/contrast adjustment that tracks applied values.
    
    Args:
        img: Input image (numpy array)
        brightness_limit: Brightness adjustment limit
        contrast_limit: Contrast adjustment limit  
        p: Probability of applying the transform
        
    Returns:
        tuple: (transformed_img, brightness_factor, contrast_factor, applied)
    """
    # Initialize return values
    brightness_factor = 0.0
    contrast_factor = 0.0
    applied = False
    
    if np.random.rand() < p:
        # Create Albumentations transform
        transform = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=1.0  # Always apply if we reach this point
        )
        
        # Apply transform
        transformed = transform(image=img)
        transformed_img = transformed['image']
        
        # Extract the applied parameters from the transform
        # Albumentations uses alpha (contrast) and beta (brightness)
        if hasattr(transform, 'params'):
            # alpha is the contrast factor, beta is the brightness factor
            contrast_factor = transform.params.get('alpha', 1.0) - 1.0  # Convert to our format
            brightness_factor = transform.params.get('beta', 0.0)  # This is already in our format
            
        applied = True
        return transformed_img, brightness_factor, contrast_factor, applied
    
    return img, brightness_factor, contrast_factor, applied

class MixedFineTuningDataset(Dataset):
    """
    Dataset for mixed fine-tuning that combines:
    1. Small images from CSV (no augmentation)
    2. Existing large images with augmentation
    """
    
    def __init__(
        self,
        mode: str,
        object_class: str,
        rootdir="./pcb-dataset/",
        transform=None,
        anomaly_class="good",
        image_size=288,
        center_size=256,
        augment=False,
        center_crop=False,
        # Mixed fine-tuning specific parameters
        fine_tuning_csv: str = None,
        mixed_split_ratio: float = 0.5,
        # PCBDataset compatibility parameters
        scratch: bool = False,
        brightness: int = None,
        shift_x: int = None,
        shift_y: int = None,
        blur: int = None,
        noise: int = None,
        num_datafile: int = None,
        split_csv_path: str = None,
        split_json_path: str = None,
        fine_tuning_json: str = None,
        # Crop tracking parameters
        track_crop: bool = False,
        save_crop_visualizations: bool = False,
        crop_vis_dir: str = "./crop_visualizations",
        resume_dir: str = None,
        resume_epoch: int = None,
        cumulative_crops: dict = None,
        debug: bool = False,
    ):
        """
        Args:
            # Mixed fine-tuning parameters
            fine_tuning_csv: Path to CSV with small images (no augmentation)
            mixed_split_ratio: Ratio of CSV images vs existing images (default: 0.5)
            
            # PCBDataset compatibility parameters
            scratch: Whether to add synthetic scratches
            brightness: Brightness adjustment factor
            shift_x: Horizontal shift factor
            shift_y: Vertical shift factor
            blur: Blur factor
            noise: Noise factor
            num_datafile: Number of data files to use
            split_csv_path: Path to CSV split file (also used as existing_data_csv for mixed fine-tuning)
            split_json_path: Path to JSON split file
            fine_tuning_json: Path to fine-tuning JSON file
            
            # Crop tracking parameters
            track_crop: If True, track crop coordinates after rotation
            save_crop_visualizations: If True, save crop visualization images
            crop_vis_dir: Directory to save crop visualization images
            resume_dir: Directory to resume from (for loading existing crop annotations)
            resume_epoch: Epoch to resume from (for filtering crops)
            cumulative_crops: Dictionary to store cumulative crops
            debug: Enable debug mode
        """
        self.mode = mode
        self.center_size = center_size
        self.augment = augment
        self.center_crop = center_crop
        self.track_crop = track_crop
        self.save_crop_visualizations = save_crop_visualizations
        self.crop_vis_dir = crop_vis_dir
        self.resume_dir = resume_dir
        self.resume_epoch = resume_epoch
        self.current_epoch = 0
        self.mixed_split_ratio = mixed_split_ratio
        self.debug = debug
        
        # PCBDataset compatibility attributes
        self.scratch = scratch
        self.brightness = brightness
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.blur = blur
        self.noise = noise
        self.fine_tuning_json = fine_tuning_json
        self.false_positive_patches = {}  # {image_path: [patch_info, ...]}
        self._current_patch_info = None  # Initialize patch info for fine-tuning
        
        # Create crop visualization directory if needed
        if self.save_crop_visualizations:
            os.makedirs(self.crop_vis_dir, exist_ok=True)
            
        # Dictionary to store cumulative crop information per original image
        if cumulative_crops is not None:
            self.cumulative_crops = cumulative_crops
        else:
            self.cumulative_crops = defaultdict(list)
            # Only load from JSON if not provided
            if self.resume_dir and self.track_crop:
                self.load_crop_annotations(resume_epoch)
        
        self.transform = transform
        self.object_class = object_class
        self.image_size = image_size
        self.anomaly_class = anomaly_class
        
        # Determine loading strategy based on available parameters
        self._determine_loading_strategy(
            fine_tuning_csv, split_csv_path, split_json_path, 
            fine_tuning_json, num_datafile, object_class, mode, anomaly_class, rootdir
        )
        
        # Set up augmentations
        self._setup_augmentations()
        
        # Calculate total dataset size
        self.total_size = len(self.csv_image_paths) + len(self.existing_image_paths)
        print(f"Total dataset size: {self.total_size} (CSV: {len(self.csv_image_paths)}, Existing: {len(self.existing_image_paths)})")
        
        if self.total_size == 0:
            raise Exception("No images loaded from any source")

    def _determine_loading_strategy(self, fine_tuning_csv, split_csv_path, 
                                   split_json_path, fine_tuning_json, num_datafile, 
                                   object_class, mode, anomaly_class, rootdir):
        """
        Determine which loading strategy to use based on available parameters
        """
        # Initialize CSV data structures (similar to PCBDataLoader.py)
        self.csv_images = []
        self.csv_segs = []
        self.csv_object_classes = []
        self.csv_image_paths = []
        self.csv_anomaly_classes = []
        
        # Initialize existing image data structures (similar to PCBDataLoader.py)
        self.existing_images = []
        self.existing_segs = []
        self.existing_object_classes = []
        self.existing_image_paths = []
        self.existing_anomaly_classes = []
        
        # Strategy 1: Mixed fine-tuning mode (fine_tuning_csv + split_csv_path)
        if fine_tuning_csv and split_csv_path:
            print("Using mixed fine-tuning loading strategy")
            self._load_csv_image_paths(fine_tuning_csv, rootdir)
            self._load_existing_image_paths(split_csv_path, num_datafile, rootdir)
            
        # Strategy 2: Fine-tuning JSON mode (PCBDataset compatibility)
        elif fine_tuning_json:
            print("Using fine-tuning JSON loading strategy")
            self._load_fine_tuning_json_data(fine_tuning_json, rootdir)
            
        # Strategy 3: Fine-tuning CSV mode (PCBDataset compatibility)
        elif fine_tuning_csv:
            print("Using fine-tuning CSV loading strategy")
            self._load_fine_tuning_csv_data(fine_tuning_csv, rootdir)
            
        # Strategy 4: JSON split mode (PCBDataset compatibility)
        elif split_json_path:
            print("Using JSON split loading strategy")
            self._load_json_data(split_json_path, rootdir, num_datafile, object_class, mode, anomaly_class)
            
        # Strategy 5: CSV split mode (PCBDataset compatibility)
        elif split_csv_path:
            print("Using CSV split loading strategy")
            self._load_csv_data(split_csv_path, rootdir, num_datafile, object_class, mode, anomaly_class)
            
        # Strategy 6: Default CSV mode (PCBDataset compatibility)
        else:
            print("Using default CSV loading strategy")
            self._load_csv_data(None, rootdir, num_datafile, object_class, mode, anomaly_class)

    def _load_fine_tuning_json_data(self, fine_tuning_json, rootdir):
        """Load image paths and masks directly from the fine-tuning JSON file (PCBDataset compatibility)"""
        if os.path.exists(fine_tuning_json):
            with open(fine_tuning_json, 'r') as f:
                review_data = json.load(f)
            
            # Collect unique image paths from selected entries
            unique_images = set()
            for entry in review_data.get('entries', []):
                if entry.get('selected', True):  # Only use selected entries
                    image_path = entry.get('image_path')
                    if image_path:
                        unique_images.add(image_path)
            
            # Load images and set up dataset
            for image_path in unique_images:
                # Normalize the path to handle mixed separators
                normalized_path = os.path.normpath(str(image_path).strip())
                if os.path.exists(normalized_path):
                    self.existing_image_paths.append({
                        'path': normalized_path,
                        'object_class': 0,  # pcb
                        'anomaly_class': "good",
                        'is_csv': False
                    })
                else:
                    print(f"Warning: Image not found: {normalized_path}")
            
            print(f"Loaded {len(self.existing_image_paths)} unique images from fine-tuning JSON")
            
            # Load false positive patches
            self._load_false_positive_patches(fine_tuning_json)
            
        else:
            print(f"Warning: Fine-tuning JSON file not found: {fine_tuning_json}")
            raise Exception("Fine-tuning JSON file not found")

    def _load_fine_tuning_csv_data(self, fine_tuning_csv, rootdir):
        """Load image paths and masks directly from the fine-tuning CSV file (PCBDataset compatibility)"""
        if os.path.exists(fine_tuning_csv):
            # Read CSV file using pandas
            df = pd.read_csv(fine_tuning_csv)
            
            print(f"DEBUG: Loading fine-tuning data from CSV: {fine_tuning_csv}")
            print(f"DEBUG: CSV columns: {list(df.columns)}")
            print(f"DEBUG: Found {len(df)} entries in CSV file")
            
            # Filter entries based on split and category (similar to _load_csv_data)
            if 'split' in df.columns:
                df = df.query(f'split=="train"')  # Use train split for fine-tuning
            
            if 'category' in df.columns:
                df = df.query('category=="good"')  # Use good images for fine-tuning
            
            if len(df) == 0:
                raise Exception("No data found in CSV file after filtering")
            
            for i, row in df.iterrows():
                image_path = row.get('image')
                if not image_path:
                    continue
                    
                # Normalize the path to handle mixed separators
                normalized_path = os.path.normpath(str(image_path).strip())
                if os.path.exists(normalized_path):
                    self.existing_image_paths.append({
                        'path': normalized_path,
                        'object_class': 0,  # pcb
                        'anomaly_class': "good",
                        'is_csv': False
                    })
                else:
                    print(f"Warning: Image not found: {normalized_path}")
            
            print(f"Loaded {len(self.existing_image_paths)} images from fine-tuning CSV")
            
        else:
            print(f"Warning: Fine-tuning CSV file not found: {fine_tuning_csv}")
            raise Exception("Fine-tuning CSV file not found")

    def _load_json_data(self, split_json_path, rootdir, num_datafile, object_class, mode, anomaly_class):
        """Load image paths and masks from JSON file (PCBDataset compatibility)"""
        if split_json_path is None:
            raise Exception("split_json_path is required for JSON-based loading")
        
        with open(split_json_path, 'r') as f:
            json_data = json.load(f)
        
        # Extract entries from JSON
        entries = json_data.get('entries', [])
        
        if num_datafile is not None:
            # Ensure we don't sample more than available data
            if len(entries) > num_datafile:
                import random
                entries = random.sample(entries, num_datafile)
        
        # Filter entries based on object_class and mode
        filtered_entries = []
        for entry in entries:
            # Check if entry has required fields
            if 'image_path' in entry:
                # For now, we'll include all entries since JSON structure may vary
                filtered_entries.append(entry)
        
        if len(filtered_entries) == 0:
            raise Exception("No data found in JSON file")

        for entry in filtered_entries:
            image_path = entry.get('image_path')
            if not image_path:
                continue
                
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image path not found: {image_path}")
                continue
            
            try:
                # Check if this entry has grid coordinates (patch-specific)
                grid_row = entry.get('grid_row')
                grid_col = entry.get('grid_col')
                
                if grid_row is not None and grid_col is not None:
                    # This is a patch-specific entry - store as existing image for augmentation
                    self.existing_image_paths.append({
                        'path': image_path,
                        'object_class': 0,  # pcb
                        'anomaly_class': "good",
                        'is_csv': False,
                        'patch_info': {
                            'pixel_coordinates': [
                                grid_col * 128,  # Convert grid coordinates to pixel coordinates
                                grid_row * 128,
                                (grid_col + 1) * 128,
                                (grid_row + 1) * 128
                            ],
                            'grid_row': grid_row,
                            'grid_col': grid_col,
                            'anomaly_max': entry.get('anomaly_max', 0),
                            'status': entry.get('status', 'FP')
                        }
                    })
                else:
                    # This is a full image entry (no grid coordinates)
                    self.existing_image_paths.append({
                        'path': image_path,
                        'object_class': 0,  # pcb
                        'anomaly_class': "good",
                        'is_csv': False
                    })
                
            except Exception as e:
                print(f"Warning: Could not process image {image_path}: {e}")
                continue

    def _load_csv_data(self, split_csv_path, rootdir, num_datafile, object_class, mode, anomaly_class):
        """Load image paths and masks from CSV file (PCBDataset compatibility)"""
        if split_csv_path is None:
            df = pd.read_csv(os.path.join(".", "splits", "pcb-split.csv"))
        else:
            df = pd.read_csv(split_csv_path)
        if num_datafile is not None:
            # Ensure we don't sample more than available data
            df = df.sample(n=num_datafile, replace=True)
        if object_class == "all":
            df = df.query(f'split=="{mode}"')
        else:
            df = df.query(f'split=="{mode}" and object=="{object_class}"')

        if anomaly_class == "good":
            df = df.query('category=="good"')
        elif anomaly_class == "all":
            pass
        else:
            df = df.query(f'category=="{anomaly_class}"')

        if len(df) == 0:
            raise Exception("No data found")

        for i, row in df.iterrows():
            # Fix path handling to ensure consistent separators
            image_filename = str(row["image"]).strip()
            # Normalize the path to handle mixed separators
            data_path = os.path.normpath(os.path.join(rootdir, image_filename))
            
            # Check if file exists before trying to load it
            if not os.path.exists(data_path):
                print(f"Warning: Image file not found: {data_path}")
                print(f"  - rootdir: {rootdir}")
                print(f"  - image filename: {image_filename}")
                continue
                
            # Store as existing image for augmentation
            self.existing_image_paths.append({
                'path': data_path,
                'object_class': 0,  # pcb
                'anomaly_class': str(row["category"]),
                'is_csv': False
            })

    def _load_false_positive_patches(self, fp_review_file):
        """Load false positive patches from JSON file (PCBDataset compatibility)"""
        if os.path.exists(fp_review_file):
            with open(fp_review_file, 'r') as f:
                review_data = json.load(f)
            
            print(f"DEBUG: Loading patches from {fp_review_file}")
            print(f"DEBUG: Found {len(review_data.get('entries', []))} entries in review file")
            
            # Process entries from the review list
            for entry in review_data.get('entries', []):
                if entry.get('selected', True):  # Only use selected entries
                    image_path = entry.get('image_path')
                    if image_path:
                        # Convert the entry to the expected patch format
                        patch_info = {
                            'pixel_coordinates': [
                                entry.get('grid_col', 0) * 128,  # Convert grid coordinates to pixel coordinates
                                entry.get('grid_row', 0) * 128,
                                (entry.get('grid_col', 0) + 1) * 128,
                                (entry.get('grid_row', 0) + 1) * 128
                            ],
                            'anomaly_max': entry.get('anomaly_max', 0),
                            'status': entry.get('status', 'FP')
                        }
                        
                        if image_path not in self.false_positive_patches:
                            self.false_positive_patches[image_path] = []
                        self.false_positive_patches[image_path].append(patch_info)
            
            print(f"Loaded {sum(len(patches) for patches in self.false_positive_patches.values())} patches from {fp_review_file}")
        else:
            print(f"Warning: Fine-tuning JSON file not found: {fp_review_file}")

    def _select_fine_tuning_patch(self, img_path, index):
        """
        Select a patch for fine-tuning from the available false positive patches (PCBDataset compatibility)
        """
        if not self.fine_tuning_json or not hasattr(self, 'existing_image_paths'):
            return False, None, None, None, None, None
            
        # For JSON fine-tuning, select from available patches
        fp_patches = self.false_positive_patches.get(img_path, [])
        if not fp_patches:
            return False, None, None, None, None, None
            
        # Randomly select a false positive patch
        patch = np.random.choice(fp_patches)
        
        # patch['pixel_coordinates'] = [x1, y1, x2, y2]
        crop_x, crop_y, crop_x2, crop_y2 = patch['pixel_coordinates']
        crop_x = int(crop_x)
        crop_y = int(crop_y)
        crop_w = int(crop_x2 - crop_x)
        crop_h = int(crop_y2 - crop_y)
        
        # Create patch info for later saving
        patch_info = {
            'pixel_coordinates': [crop_x, crop_y, crop_x2, crop_y2],
            'anomaly_max': patch.get('anomaly_max', 0),
            'status': patch.get('status', 'FP')
        }
        
        return True, crop_x, crop_y, crop_w, crop_h, patch_info

    def _load_csv_image_paths(self, csv_path, rootdir):
        """Load small images from CSV file (no augmentation) - similar to _load_csv_data in PCBDataLoader.py"""
        df = pd.read_csv(csv_path)
        
        # Filter for train split and good category
        if 'split' in df.columns:
            df = df.query(f'split=="train"')
        
        if 'category' in df.columns:
            df = df.query('category=="good"')
            
        if len(df) == 0:
            print("Warning: No data found in CSV file after filtering")
            return
            
        # Define object class dictionary
        object_cls_dict = {"pcb": 0}
        
        # Initialize lists to store loaded data (similar to PCBDataLoader.py)
        self.csv_images = []
        self.csv_segs = []
        self.csv_object_classes = []
        self.csv_image_paths = []
        self.csv_anomaly_classes = []
        
        for i, row in df.iterrows():
            # Fix path handling to ensure consistent separators
            image_filename = str(row["image"]).strip()
            # Normalize the path to handle mixed separators
            data_path = os.path.normpath(os.path.join(rootdir, image_filename))
            
            # Check if file exists before trying to load it
            if not os.path.exists(data_path):
                print(f"Warning: Image file not found: {data_path}")
                print(f"  - rootdir: {rootdir}")
                print(f"  - image filename: {image_filename}")
                continue
                
            try:
                img = np.array(
                    Image.open(data_path).convert("RGB")
                    # .resize((self.image_size, self.image_size))
                ).astype(np.uint8)
                self.csv_image_paths.append(data_path)
                self.csv_images.append(img)
                self.csv_object_classes.append(object_cls_dict[str(row["object"])])
                self.csv_anomaly_classes.append(str(row["category"]))
                
                # For CSV images, we assume they are good images (no masks)
                #seg_shape = (self.image_size, self.image_size)
                #self.csv_segs.append(np.zeros(seg_shape))
                
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue

    def _load_existing_image_paths(self, csv_path, num_datafile, rootdir):
        """Load existing large images from CSV file (with augmentation) - similar to _load_csv_data in PCBDataLoader.py"""
        if csv_path is None:
            df = pd.read_csv(os.path.join(".", "splits", "pcb-split.csv"))
        else:
            df = pd.read_csv(csv_path)
        
        if num_datafile is not None:
            # Ensure we don't sample more than available data
            df = df.sample(n=num_datafile, replace=True)
        
        # Filter for train split and good category
        if 'split' in df.columns:
            df = df.query(f'split=="train"')
        
        if 'category' in df.columns:
            df = df.query('category=="good"')

        if len(df) == 0:
            print("Warning: No data found in existing CSV file after filtering")
            return
            
        # Define object class dictionary
        object_cls_dict = {"pcb": 0}
        
        # Initialize lists to store loaded data (similar to PCBDataLoader.py)
        self.existing_images = []
        self.existing_segs = []
        self.existing_object_classes = []
        self.existing_image_paths = []
        self.existing_anomaly_classes = []
        
        for i, row in df.iterrows():
            # Fix path handling to ensure consistent separators
            image_filename = str(row["image"]).strip()
            # Normalize the path to handle mixed separators
            data_path = os.path.normpath(os.path.join(rootdir, image_filename))
            
            # Check if file exists before trying to load it
            if not os.path.exists(data_path):
                print(f"Warning: Image file not found: {data_path}")
                print(f"  - rootdir: {rootdir}")
                print(f"  - image filename: {image_filename}")
                continue
                
            try:
                img = np.array(
                    Image.open(data_path).convert("RGB")
                    # .resize((self.image_size, self.image_size))
                ).astype(np.uint8)
                self.existing_image_paths.append(data_path)
                self.existing_images.append(img)
                self.existing_object_classes.append(object_cls_dict[str(row["object"])])
                self.existing_anomaly_classes.append(str(row["category"]))
                
                # For existing images, we assume they are good images (no masks)
                #seg_shape = img.shape[:2]  # Use actual image shape
                #self.existing_segs.append(np.zeros(seg_shape))
                
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue

    def _setup_augmentations(self):
        """Set up augmentation functions for existing images"""
        if self.track_crop:
            # Define tracking functions for existing images
            def rotate_and_crop_func(img, **kwargs):
                # Apply brightness/contrast first
                img, brightness_factor, contrast_factor, bc_applied = albumentations_brightness_contrast(
                    img, brightness_limit=0.05, contrast_limit=0.05, p=0.5
                )
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_angle = np.random.uniform(-5, 5)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                inverse_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_REPLICATE)
                crop_h = min(self.image_size, h)
                crop_w = min(self.image_size, w)
                max_h = h - crop_h
                max_w = w - crop_w
                
                # Random crop for existing images
                crop_y = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                crop_x = np.random.randint(0, max_w + 1) if max_w > 0 else 0
                
                cropped_rotated = rotated[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                crop_corners_rot = np.array([
                    [crop_x, crop_y],
                    [crop_x + crop_w, crop_y],
                    [crop_x + crop_w, crop_y + crop_h],
                    [crop_x, crop_y + crop_h]
                ], dtype=np.float32)
                crop_corners_orig = cv2.transform(crop_corners_rot.reshape(-1, 1, 2), inverse_matrix).reshape(-1, 2)
                crop_corners_orig_flat = [float(x) for x in np.array(crop_corners_orig).flatten()]
                return (cropped_rotated, 
                       {
                           'crop_coords': crop_corners_orig_flat,  # Use the rotated polygon coordinates for tracking
                           'rotation_angle': rotation_angle,
                           'brightness_factor': brightness_factor,
                           'contrast_factor': contrast_factor,
                           'brightness_contrast_applied': bc_applied,
                           'original_shape': img.shape[:2],
                           'is_csv_image': False,
                           'epoch': self.current_epoch,  # Add missing epoch key
                           'augmentation_applied': True
                       })
            
            # Create wrapper function for existing images
            #def transform_with_tracking(image, mask, index=None):
            def transform_with_tracking(image, index=None):
                cropped_img, transform_info = rotate_and_crop_func(image, index=index)
                # Apply the same transformations to mask
                #mask, _ = rotate_and_crop_func(mask, index=index)
                # Add missing keys to ensure consistency
                transform_info['epoch'] = self.current_epoch
                #return {'image': cropped_img, 'mask': mask, 'transform_info': transform_info}
                return {'image': cropped_img, 'transform_info': transform_info}
            
            self.aug = transform_with_tracking
        else:
            # No crop tracking - use standard augmentations with size enforcement
            #def transform_without_tracking(image, mask, index=None):
            def transform_without_tracking(image, index=None):
                # Apply brightness/contrast using Albumentations for speed, but don't track parameters
                if np.random.rand() < 0.5:
                    # Use Albumentations for fast transformation without parameter tracking
                    transform = A.RandomBrightnessContrast(
                        brightness_limit=0.05,
                        contrast_limit=0.05,
                        p=1.0
                    )
                    transformed = transform(image=image)
                    img = transformed['image']
                    bc_applied = True
                else:
                    img = image
                    bc_applied = False
                
                # Apply rotation
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_angle = np.random.uniform(-5, 5)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_REPLICATE)
                #rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h),
                #                            flags=cv2.INTER_NEAREST,
                #                            borderMode=cv2.BORDER_REPLICATE)
                
                # Random crop to exact size
                crop_h = min(self.image_size, h)
                crop_w = min(self.image_size, w)
                max_h = h - crop_h
                max_w = w - crop_w
                
                # Random crop for existing images
                crop_y = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                crop_x = np.random.randint(0, max_w + 1) if max_w > 0 else 0
                
                cropped_img = rotated[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                #cropped_mask = rotated_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                
                # Ensure exact size (in case the crop is smaller than target)
                #if cropped_img.shape[:2] != (self.image_size, self.image_size):
                #    cropped_img = cv2.resize(cropped_img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                #    cropped_mask = cv2.resize(cropped_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                
                # Convert rectangle to 8-value format for consistency
                x1, y1, x2, y2 = crop_x, crop_y, crop_x + crop_w, crop_y + crop_h
                transform_info = {
                    'crop_coords': [x1, y1, x2, y1, x2, y2, x1, y2],  # 8 values for consistency
                    'rotation_angle': rotation_angle,
                    'brightness_factor': 0.0,  # Emit 0 as requested
                    'contrast_factor': 0.0,    # Emit 0 as requested
                    'brightness_contrast_applied': bc_applied,
                    'original_shape': image.shape[:2],
                    'is_csv_image': False,
                    'epoch': self.current_epoch,  # Add missing epoch key
                    'augmentation_applied': True
                }
                
                #return {'image': cropped_img, 'mask': cropped_mask, 'transform_info': transform_info}
                return {'image': cropped_img, 'transform_info': transform_info}
            
            self.aug = transform_without_tracking

    def load_crop_annotations(self, resume_epoch):
        """Load existing crop annotations from JSON file"""
        if not self.resume_dir:
            return
            
        crop_annotations_file = os.path.join(self.resume_dir, "crop_annotations.json")
        if os.path.exists(crop_annotations_file):
            try:
                with open(crop_annotations_file, 'r') as f:
                    loaded_crops = json.load(f)
                
                # Clear existing crops and initialize new storage
                self.cumulative_crops.clear()
                
                # Track statistics for debug output
                kept_crops = 0
                removed_crops = 0
                epochs_seen = set()
                
                # Process each image's crops
                for image_path, crops in loaded_crops.items():
                    self.cumulative_crops[image_path] = []
                    for crop in crops:
                        crop_epoch = crop.get("epoch", 0)
                        epochs_seen.add(crop_epoch)
                        # Keep crops up to and including resume_epoch
                        if self.resume_epoch is None or crop_epoch <= self.resume_epoch:
                            self.cumulative_crops[image_path].append(crop)
                            kept_crops += 1
                        else:
                            removed_crops += 1
                
                if self.resume_epoch is not None:
                    print(f"DEBUG: Loading crops for resume at epoch {self.resume_epoch}:")
                    print(f"  - Epochs seen in file: {sorted(list(epochs_seen))}")
                    print(f"  - Kept {kept_crops} crops from epochs up to {self.resume_epoch}")
                    print(f"  - Removed {removed_crops} crops from epochs after {self.resume_epoch}")
                else:
                    print(f"DEBUG: Loaded all {kept_crops} crops (no resume epoch specified)")
                    
            except Exception as e:
                print(f"Warning: Could not load crop annotations: {e}")
                self.cumulative_crops.clear()
        else:
            print("No existing crop annotations found.")

    def save_crop_annotations(self):
        """Save crop annotations to JSON file"""
        if not self.resume_dir or not self.track_crop:
            return
        crop_annotations_file = os.path.join(self.resume_dir, "crop_annotations.json")
        try:
            # Save to file
            with open(crop_annotations_file, 'w') as f:
                json.dump(self.cumulative_crops, f, indent=2)
                
        except Exception as e:
            print(f"Error saving crop annotations: {e}")

    def add_crop_to_cumulative_map(self, original_image_path, crop_info, current_epoch=None):
        """Add crop information to the cumulative map"""
        if crop_info is None or 'crop_coords' not in crop_info:
            return
            
        # Add epoch information to crop_info
        if current_epoch is not None:
            crop_info['epoch'] = current_epoch
        
        # Initialize list for this image if not exists
        if original_image_path not in self.cumulative_crops:
            self.cumulative_crops[original_image_path] = []
        
        self.cumulative_crops[original_image_path].append(crop_info)
        
        # Update visualization
        if self.save_crop_visualizations:
            self.save_crop_annotations()
            # Create the cumulative crop visualization
            self.create_cumulative_crop_visualization(original_image_path)
            
            # Save individual crop patches to tmp directory for testing (only in debug mode)
            if self.debug:
                self.save_crop_patch_to_tmp(original_image_path, crop_info)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        # Simple alternating logic based on mixed_split_ratio
        csv_count = len(self.csv_image_paths)
        existing_count = len(self.existing_image_paths)
        
        # Ensure mixed_split_ratio is a float
        if not isinstance(self.mixed_split_ratio, (int, float)):
            self.mixed_split_ratio = float(self.mixed_split_ratio)
        
        # Calculate how many CSV images to use before switching to existing images
        # For ratio 0.5: use 1 CSV, then 1 existing, repeat
        # For ratio 0.7: use 7 CSV, then 3 existing, repeat
        csv_per_cycle = int(10 * self.mixed_split_ratio)  # Convert to integers (e.g., 0.5 -> 5, 0.7 -> 7)
        existing_per_cycle = 10 - csv_per_cycle  # Remaining images in cycle
        
        # Calculate which cycle we're in and position within cycle
        cycle = index // 10
        position_in_cycle = index % 10
        
        if position_in_cycle < csv_per_cycle and csv_count > 0:
            # Use CSV image (small, no augmentation)
            # Ensure all CSV images are used by cycling through them properly
            csv_index = (cycle * csv_per_cycle + position_in_cycle) % csv_count
            if self.debug:
                print(f"DEBUG: Using CSV image {csv_index} (cycle {cycle}, position {position_in_cycle})")
            
            # Use pre-loaded images (similar to PCBDataLoader.py)
            img = self.csv_images[csv_index]
            #seg = self.csv_segs[csv_index]
            anomaly_class = self.csv_anomaly_classes[csv_index]
            img_path = self.csv_image_paths[csv_index]
            
            # Create data structure for consistency with existing image handling
            data = {
                'path': img_path,
                'object_class': self.csv_object_classes[csv_index],
                'anomaly_class': anomaly_class,
                'is_csv': True
            }
            
            # No augmentation for CSV images
            # Convert rectangle to 8-value format for consistency
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
            transform_info = {
                'crop_coords': [x1, y1, x2, y1, x2, y2, x1, y2],  # 8 values for consistency
                'rotation_angle': 0,
                'brightness_factor': 0,
                'contrast_factor': 0,
                'brightness_contrast_applied': False,
                'original_shape': img.shape[:2],
                'is_csv_image': True,
                'epoch': self.current_epoch,  # Always include epoch
                'augmentation_applied': False  # Ensure consistent keys
            }
            
        else:
            # Use existing image (large, with augmentation)
            if existing_count > 0:
                existing_index = (cycle * existing_per_cycle + (position_in_cycle - csv_per_cycle)) % existing_count
                if self.debug:
                    print(f"DEBUG: Using existing image {existing_index} (cycle {cycle}, position {position_in_cycle})")
                
                # Use pre-loaded images (similar to PCBDataLoader.py)
                img = self.existing_images[existing_index]
                #seg = self.existing_segs[existing_index]
                anomaly_class = self.existing_anomaly_classes[existing_index]
                img_path = self.existing_image_paths[existing_index]
                
                # Create data structure for consistency
                data = {
                    'path': img_path,
                    'object_class': self.existing_object_classes[existing_index],
                    'anomaly_class': anomaly_class,
                    'is_csv': False
                }
                
                # Handle patch-specific data from JSON loading (PCBDataset compatibility)
                patch_info = data.get('patch_info')
                
                # Fine-tuning patch selection (PCBDataset compatibility)
                fine_tuning_patch_info = None
                if self.fine_tuning_json:
                    use_fp_patch, crop_x, crop_y, crop_w, crop_h, patch_info_ft = self._select_fine_tuning_patch(img_path, index)
                    if use_fp_patch:
                        # Extract the patch from the original image (JSON fine-tuning)
                        img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                        #seg = seg[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                        fine_tuning_patch_info = patch_info_ft
                
                # Apply augmentation for existing images
                if self.augment:
                    # If we have patch info from JSON loading, we already have a specific patch
                    # and shouldn't apply additional cropping, just apply other augmentations
                    if patch_info is not None:
                        # Apply only non-cropping augmentations (brightness, contrast, rotation)
                        # but keep the original patch coordinates
                        # Create a simple augmentation that doesn't crop
                        img, brightness_factor, contrast_factor, bc_applied = albumentations_brightness_contrast(
                            img, brightness_limit=0.05, contrast_limit=0.05, p=0.5
                        )
                        
                        # Apply rotation without cropping
                        h, w = img.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_angle = np.random.uniform(-5, 5)
                        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                        img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REPLICATE)
                        #seg = cv2.warpAffine(seg, rotation_matrix, (w, h),
                        #                   flags=cv2.INTER_NEAREST,
                        #                   borderMode=cv2.BORDER_REPLICATE)
                        
                        # Create transform info that preserves the original patch coordinates
                        transform_info = {
                            'crop_coords': patch_info['pixel_coordinates'],
                            'rotation_angle': rotation_angle,
                            'brightness_factor': brightness_factor,
                            'contrast_factor': contrast_factor,
                            'brightness_contrast_applied': bc_applied,
                            'original_shape': img.shape[:2],
                            'patch_info': patch_info  # Include original patch info
                        }
                        
                    else:
                        # Standard augmentation with random cropping
                        #augmented = self.aug(image=img, mask=seg, index=index)
                        augmented = self.aug(image=img, index=index)
                        img = augmented["image"]
                        #seg = augmented["mask"]
                        transform_info = augmented["transform_info"]

                else:
                    # Center crop if no augmentation
                    h, w = img.shape[:2]
                    crop_h = min(self.image_size, h)
                    crop_w = min(self.image_size, w)
                    crop_y = (h - crop_h) // 2
                    crop_x = (w - crop_w) // 2
                    img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    #seg = seg[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    # Convert rectangle to 8-value format for consistency
                    x1, y1, x2, y2 = crop_x, crop_y, crop_x + crop_w, crop_y + crop_h
                    transform_info = {
                        'crop_coords': [x1, y1, x2, y1, x2, y2, x1, y2],  # 8 values for consistency
                        'rotation_angle': 0,
                        'brightness_factor': 0,
                        'contrast_factor': 0,
                        'brightness_contrast_applied': False,
                        'original_shape': img.shape[:2],
                        'is_csv_image': False,
                        'epoch': self.current_epoch,  # Always include epoch
                        'augmentation_applied': False
                    }
            else:
                # Fallback if no existing images
                csv_index = index % csv_count if csv_count > 0 else 0
                data = self.csv_image_paths[csv_index]
                img_path = data['path']
                
                # Load image on-demand with error handling
                try:
                    img = np.array(Image.open(img_path).convert("RGB")).astype(np.uint8)
                except Exception as e:
                    print(f"Error loading fallback CSV image {img_path}: {e}")
                    raise e
                    
                #seg = np.zeros((self.image_size, self.image_size))
                anomaly_class = data['anomaly_class']
                
                # Convert rectangle to 8-value format for consistency
                x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
                transform_info = {
                    'crop_coords': [x1, y1, x2, y1, x2, y2, x1, y2],  # 8 values for consistency
                    'rotation_angle': 0,
                    'brightness_factor': 0,
                    'contrast_factor': 0,
                    'brightness_contrast_applied': False,
                    'original_shape': img.shape[:2],
                    'is_csv_image': True,
                    'epoch': self.current_epoch,
                    'augmentation_applied': False
                }

        # Save crop annotations if tracking (only for existing images, not CSV images)
        if self.save_crop_visualizations and self.track_crop and not transform_info.get('is_csv_image', False):
            self.add_crop_to_cumulative_map(img_path, transform_info, self.current_epoch)

        # Debug: Check image sizes
        if img.shape[:2] != (self.image_size, self.image_size):
            print(f"ERROR: Image size mismatch! Expected {self.image_size}x{self.image_size}, got {img.shape[:2]}")
            print(f"  - Image path: {img_path}")
            print(f"  - Is CSV image: {transform_info.get('is_csv_image', 'unknown')}")
            print(f"  - Aborting training due to image size mismatch")
            import sys
            sys.exit(1)
        
        # Apply PCBDataset compatibility features (synthetic defects)
        if self.scratch:
            # Use imported synthetic scratch function if available
            if SYNTHETIC_SCRATCH_AVAILABLE:
                img, _, _ = add_scratch_controlled(img)
                anomaly_class = "defect"
            else:
                print("Warning: synthetic_scratch module not found, skipping scratch generation")
        
        # Randomly apply additional synthetic defects (PCBDataset compatibility)
        # 1. Random brightness
        if self.brightness is not None:
            factor = self.brightness
            img = np.clip(img.astype(np.float32) + factor, 0, 255).astype(np.uint8)
        # 2. Random shift
        if self.shift_x is not None:
            tx = self.shift_x
            assert abs(tx) < img.shape[0], "shift should be less than the image size"
            M = np.array([[1, 0, tx], [0, 1, 0]], dtype=np.float32)
            img = cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT
            )
        if self.shift_y is not None:
            ty = self.shift_y
            assert abs(ty) < img.shape[1], "shift should be less than the image size"
            M = np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32)
            img = cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT
            )
        # 3. Random blur
        if self.blur is not None:
            ksize = self.blur
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        # 4. Random noise
        if self.noise is not None:
            num_salt = self.noise
            coords = [np.random.randint(0, i, num_salt) for i in img.shape]
            img[tuple(coords)] = 255
        
        # Convert to tensor format
        img = img.astype(np.float32) / 255.0
        y = data['object_class']
        
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose((-1, 0, 1)))
            img = (img - 0.5) / 0.5
        
        return (
            img,
            0,#seg.astype(np.float32),
            int(y),
            img_path,
            anomaly_class,
            transform_info,
        )

    def create_cumulative_crop_visualization(self, original_image_path, save_path=None):
        """
        Create a cumulative visualization showing all crops for a specific original image
        
        Args:
            original_image_path: Path to the original image
            save_path: Optional path to save the visualization
        """
        # Check if we have crops for this image
        if original_image_path not in self.cumulative_crops:
            return None
            
        all_crops = self.cumulative_crops[original_image_path]
        if not all_crops:
            return None
            
        # Load the original image
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image not found: {original_image_path}")
            return None
            
        original_image = np.array(Image.open(original_image_path).convert("RGB"))
        original_image_basename = os.path.basename(original_image_path)
        original_filename = path_to_safe_filename(original_image_path)
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Display the original image
        ax.imshow(original_image)
        
        # Draw rectangles/polygons for all crops
        for i, crop_info in enumerate(all_crops):
            corners = crop_info['crop_coords']  # Could be 4 values [x1, y1, x2, y2] or 8 values [x1, y1, x2, y2, x3, y3, x4, y4]
            
            # Color scheme: green for latest, red for all others
            if i == len(all_crops) - 1:  # Latest crop
                color = 'green'
                linewidth = 3  # Make latest crop more prominent
            else:  # Previous crops
                color = 'red'
                linewidth = 2
            
            # Check if we have 4 values (rectangle) or 8 values (rotated polygon)
            if len(corners) == 4:
                # Create rectangle from the 4 corner points
                # Format: [x1, y1, x2, y2] -> width = x2-x1, height = y2-y1
                x1, y1, x2, y2 = corners
                width = x2 - x1
                height = y2 - y1
                
                # Create rectangle patch
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=linewidth, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            elif len(corners) == 8:
                # Create polygon from the 8 corner points (rotated crop)
                # Format: [x1, y1, x2, y2, x3, y3, x4, y4] -> polygon with 4 vertices
                polygon_points = [(corners[j], corners[j+1]) for j in range(0, 8, 2)]
                polygon = patches.Polygon(polygon_points, 
                                        linewidth=linewidth, edgecolor=color, facecolor='none')
                ax.add_patch(polygon)
            else:
                print(f"Warning: Unexpected crop_coords format with {len(corners)} values: {corners}")
        
        # Set title with all the information including epoch info
        title_text = f'Cumulative Crop Map - {original_image_basename}\nTotal Crops: {len(all_crops)} | Green: Latest | Red: Previous'
        ax.set_title(title_text, fontsize=14, weight='bold', pad=20)
        ax.axis('off')
        
        # Save the visualization with original filename
        if save_path is None:
            save_path = os.path.join(self.crop_vis_dir, f"{original_filename}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

    def save_crop_patch_to_tmp(self, original_image_path, crop_info):
        """
        TEMPORARY: Save individual crop patches to tmp directory for testing
        """
        try:
            # Create tmp directory if it doesn't exist
            tmp_dir = "./tmp"
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Load the original image
            if not os.path.exists(original_image_path):
                print(f"Warning: Original image not found for tmp save: {original_image_path}")
                return
                
            original_image = np.array(Image.open(original_image_path).convert("RGB"))
            
            # Extract crop coordinates
            crop_coords = crop_info['crop_coords']  # Could be 4 values [x1, y1, x2, y2] or 8 values [x1, y1, x2, y2, x3, y3, x4, y4]
            
            # Handle different coordinate formats
            if len(crop_coords) == 4:
                # Rectangle format: [x1, y1, x2, y2]
                x1, y1, x2, y2 = crop_coords
                # Extract the crop patch
                crop_patch = original_image[y1:y2, x1:x2]
            elif len(crop_coords) == 8:
                # Rotated polygon format: [x1, y1, x2, y2, x3, y3, x4, y4]
                # Extract the exact rotated patch using perspective transformation
                import cv2
                
                # Convert coordinates to numpy array for cv2
                src_points = np.array([
                    [crop_coords[0], crop_coords[1]],  # x1, y1
                    [crop_coords[2], crop_coords[3]],  # x2, y2
                    [crop_coords[4], crop_coords[5]],  # x3, y3
                    [crop_coords[6], crop_coords[7]]   # x4, y4
                ], dtype=np.float32)
                
                # Calculate the dimensions of the rotated crop
                # Use the width and height of the original crop before rotation
                # We'll use the distance between opposite corners to estimate the crop size
                width = int(np.sqrt((crop_coords[2] - crop_coords[0])**2 + (crop_coords[3] - crop_coords[1])**2))
                height = int(np.sqrt((crop_coords[4] - crop_coords[2])**2 + (crop_coords[5] - crop_coords[3])**2))
                
                # Ensure minimum dimensions
                width = max(width, 1)
                height = max(height, 1)
                
                # Define destination points for a straight rectangle
                dst_points = np.array([
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ], dtype=np.float32)
                
                # Calculate perspective transform matrix
                transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # Apply perspective transform to extract the exact rotated patch
                crop_patch = cv2.warpPerspective(original_image, transform_matrix, (width, height))
            else:
                print(f"Warning: Unexpected crop_coords format with {len(crop_coords)} values: {crop_coords}")
                return
            
            # Create filename with epoch and crop info
            original_basename = os.path.basename(original_image_path)
            epoch = crop_info.get('epoch', 'unknown')
            is_csv = crop_info.get('is_csv_image', False)
            
            # Create descriptive filename
            filename = f"epoch_{epoch}_csv_{is_csv}_{original_basename}_{crop_coords}"
            safe_filename = path_to_safe_filename(filename)
            
            # Save the crop patch
            save_path = os.path.join(tmp_dir, f"{safe_filename}.png")
            Image.fromarray(crop_patch).save(save_path)
            
            # Only print debug messages if debug mode is enabled
            if self.debug:
                print(f"DEBUG: Saved crop patch to {save_path}")
                print(f"  - Original: {original_image_path}")
                print(f"  - Crop coords: {crop_coords}")
                print(f"  - Is CSV image: {is_csv}")
                print(f"  - Epoch: {epoch}")
            
        except Exception as e:
            print(f"Error saving crop patch to tmp: {e}") 