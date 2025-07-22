from datetime import datetime
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# import torchio as tio
import albumentations as A
from synthetic_scratch import add_scratch_controlled
import cv2
import math
from albumentations.core.transforms_interface import BasicTransform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from matplotlib import transforms
import json
from utils import path_to_safe_filename

class CropTrackingTransform(BasicTransform):
    """Custom transform that tracks crop coordinates after rotation"""
    
    def __init__(self, height, width, p=1.0):
        super().__init__(p)
        self.height = height
        self.width = width
        self.crop_coords = None
        
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply}
        
    def apply(self, img, force_apply=False, **params):
        # Get original image dimensions
        h, w = img.shape[:2]
        
        # Calculate crop coordinates in original image space
        crop_h = min(self.height, h)
        crop_w = min(self.width, w)
        
        # Calculate crop coordinates (top-left corner)
        max_h = h - crop_h
        max_w = w - crop_w
        crop_y = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        crop_x = np.random.randint(0, max_w + 1) if max_w > 0 else 0
        
        # Store crop coordinates
        self.crop_coords = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        
        # Apply the crop
        cropped = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        
        return cropped
    
    def get_transform_init_args_names(self):
        return ("height", "width")
    
    def get_params_dependent_on_targets(self, params):
        return {
            'crop_coords': self.crop_coords
        }
    
    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)
        params['crop_coords'] = self.crop_coords
        return params

class RotationTrackingTransform(BasicTransform):
    """Custom transform that tracks rotation angle"""
    
    def __init__(self, limit=5, p=1.0):
        super().__init__(p)
        self.limit = limit
        self.rotation_angle = 0
        
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply}
        
    def apply(self, img, **params):
        # Generate random rotation angle
        self.rotation_angle = np.random.uniform(-self.limit, self.limit)
        
        # Get image center
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def get_params(self):
        return {"rotation_angle": self.rotation_angle}

class PCBDataset(Dataset):
    """ABIDE dataset."""

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
        scratch: bool = False,
        brightness: int = None,
        shift_x: int = None,
        shift_y: int = None,
        blur: int = None,
        noise: int = None,
        num_datafile: int = None,
        split_csv_path: str = None,
        track_crop: bool = False,  # New parameter to enable crop tracking
        save_crop_visualizations: bool = False,  # New parameter to save crop visualizations
        crop_vis_dir: str = "./crop_visualizations",  # Directory to save crop visualizations
        resume_dir: str = None,  # Directory to resume from (for loading existing crop annotations)
        resume_epoch: int = None,  # Epoch to resume from (for filtering crops)
        cumulative_crops: dict = None,  # Dictionary to store cumulative crops
        annotation_dir: str = None,  # Directory for annotation JSONs for patch selection
    ):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
            track_crop: If True, track crop coordinates after rotation
            save_crop_visualizations: If True, save crop visualization images
            crop_vis_dir: Directory to save crop visualization images
            resume_dir: Directory to resume from (for loading existing crop annotations)
            resume_epoch: Epoch to resume from (for filtering crops)
        """
        self.scratch = scratch
        self.mode = mode
        self.center_size = center_size
        self.augment = augment
        self.center_crop = center_crop
        self.track_crop = track_crop
        self.save_crop_visualizations = save_crop_visualizations
        self.crop_vis_dir = crop_vis_dir
        self.resume_dir = resume_dir
        self.resume_epoch = resume_epoch
        self.current_epoch = 0  # Track current epoch for crop annotations
        
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
        
        # Add a set to track unique crops
        self.added_crops = set()  # Track (image_path, coords, epoch) tuples
            
        object_cls_dict = {
            "pcb": 0,
        }
        self.anomaly_class = anomaly_class
        self.transform = transform
        self.object_class = object_class
        self.image_size = image_size
        self.brightness = brightness
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.blur = blur
        self.noise = noise
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

        self.images = []
        self.segs = []
        self.object_classes = []
        self.image_paths = []
        self.anomaly_classes = []
        for i, row in df.iterrows():
            data_path = os.path.join(rootdir, str(row["image"]))
            img = np.array(
                Image.open(data_path).convert("RGB")
                # .resize((self.image_size, self.image_size))
            ).astype(np.uint8)
            self.image_paths.append(data_path)
            self.images.append(img)
            self.object_classes.append(object_cls_dict[str(row["object"])])
            self.anomaly_classes.append(str(row["category"]))
            if str(row["category"]) != "good":
                seg_path = os.path.join(rootdir, str(row["mask"]))
                seg = (
                    np.array(
                        Image.open(seg_path).convert("L")
                        # .resize((self.image_size, self.image_size))
                    )
                    > 0
                ).astype(np.uint8)
                self.segs.append((seg))
            else:
                seg_path = os.path.join(rootdir, str(row["image"]))
                if os.path.exists(seg_path):
                    seg_shape = np.array(Image.open(seg_path)).shape
                else:
                    seg_shape = (self.image_size, self.image_size)
                self.segs.append(np.zeros(seg_shape))
        
        # First check if we're tracking crops
        if self.track_crop:
            # Define tracking functions that will be used for both augmented and non-augmented cases
            def rotate_and_crop_func(img, **kwargs):
                # Apply brightness/contrast first
                img, brightness_factor, contrast_factor, bc_applied = random_brightness_contrast(
                    img, brightness_limit=0.05, contrast_limit=0.05, p=0.5
                )
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_angle = 30
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                inverse_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_REPLICATE)
                crop_h = min(self.image_size, h)
                crop_w = min(self.image_size, w)
                max_h = h - crop_h
                max_w = w - crop_w
                # If annotation_dir is set and there are false positive patches for this image, use one
                use_fp_patch = False
                crop_x = crop_y = None
                if self.annotation_dir and hasattr(self, 'image_paths'):
                    # Find the original image path for this sample
                    # This function is called inside __getitem__, so self.image_paths[index] is available
                    # But we don't have index here, so we need to pass it in via kwargs
                    index = kwargs.get('index', None)
                    if index is not None:
                        img_path = self.image_paths[index]
                        fp_patches = self.false_positive_patches.get(img_path, [])
                        if fp_patches:
                            use_fp_patch = True
                            # Randomly select a false positive patch
                            patch = np.random.choice(fp_patches)
                            # patch['pixel_coordinates'] = [x1, y1, x2, y2]
                            crop_x, crop_y, crop_x2, crop_y2 = patch['pixel_coordinates']
                            crop_x = int(crop_x)
                            crop_y = int(crop_y)
                            crop_w = int(crop_x2 - crop_x)
                            crop_h = int(crop_y2 - crop_y)
                if not use_fp_patch:
                    if self.augment:
                        crop_y = np.random.randint(0, max_h + 1) if max_h > 0 else 0
                        crop_x = np.random.randint(0, max_w + 1) if max_w > 0 else 0
                    else:
                        crop_y = (h - crop_h) // 2
                        crop_x = (w - crop_w) // 2
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
                           'crop_coords': crop_corners_orig_flat,  # Return as tuple instead of list
                           'rotation_angle': rotation_angle,
                           'brightness_factor': brightness_factor,
                           'contrast_factor': contrast_factor,
                           'brightness_contrast_applied': bc_applied,
                           'original_shape': img.shape[:2]
                       })
            
            # Create wrapper function for both augmented and non-augmented cases
            def transform_with_tracking(image, mask, index=None):
                cropped_img, transform_info = rotate_and_crop_func(image, index=index)
                cropped_mask = rotate_and_crop_func(mask, index=index)[0]
                return {'image': cropped_img, 'mask': cropped_mask, 'transform_info': transform_info}
            
            self.aug = transform_with_tracking

        else:
            # No crop tracking - use standard augmentations
            if self.augment:
                self.aug = A.Compose(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.05, contrast_limit=0.05, p=0.5
                        ),
                        A.Rotate(
                            limit=5,
                            p=1,
                            interpolation=cv2.INTER_NEAREST,
                            border_mode=cv2.BORDER_REPLICATE,
                        ),
                        A.RandomCrop(p=1, height=self.image_size, width=self.image_size),
                    ]
                )
            else:
                self.aug = A.CenterCrop(p=1, height=self.image_size, width=self.image_size)

        self.annotation_dir = annotation_dir
        self.false_positive_patches = {}  # {image_path: [patch_info, ...]}
        if self.annotation_dir:
            self._load_false_positive_patches()

    def _load_false_positive_patches(self):
        import glob
        # Find all annotation JSONs
        annotation_files = glob.glob(os.path.join(self.annotation_dir, '*__false_positives.json'))
        for ann_file in annotation_files:
            with open(ann_file, 'r') as f:
                anno = json.load(f)
            image_path = anno.get('image_path')
            # Only consider if there are false positives
            fp_patches = anno.get('false_positives', [])
            if fp_patches:
                self.false_positive_patches[image_path] = fp_patches

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
                            print(f"DEBUG: Keeping crop from epoch {crop_epoch} (resume_epoch={self.resume_epoch})")
                        else:
                            removed_crops += 1
                            print(f"DEBUG: Removing crop from epoch {crop_epoch} (resume_epoch={self.resume_epoch})")
                
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
                
            #print(f"DEBUG:Saved {sum(len(crops) for crops in self.cumulative_crops.values())} total crop annotations to {crop_annotations_file}")
            
        except Exception as e:
            print(f"Error saving crop annotations: {e}")

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0, 1)))
        return x

    def __len__(self):
        return len(self.images)

    def get_crop_coordinates_after_rotation(self, crop_coords, rotation_angle, original_shape):
        """
        Return the crop coordinates which are already in original image space
        
        Args:
            crop_coords: (x1, y1, x2, y2) already in original image space
            rotation_angle: rotation angle in degrees (not used anymore)
            original_shape: (height, width) of original image (not used anymore)
            
        Returns:
            (x1, y1, x2, y2) in original image space
        """
        return crop_coords

    def add_crop_to_cumulative_map(self, original_image_path, crop_info, current_epoch=None):
        """
        Add crop information to the cumulative map for a specific original image
        and immediately save the updated visualization and annotations
        
        Args:
            original_image_path: Path to the original image
            crop_info: Dictionary containing crop coordinates and rotation info
            current_epoch: Current training epoch (for tracking when crop was created)
        """
        if crop_info is None or 'crop_coords' not in crop_info:
            print(f"DEBUG: crop_info is None or missing crop_coords: {crop_info}")
            return
            
        # Add epoch information to crop_info
        if current_epoch is not None:
            crop_info['epoch'] = current_epoch
            #print(f"DEBUG: Setting epoch to {current_epoch} for crop at path {original_image_path}")
        
        # Initialize list for this image if not exists
        if original_image_path not in self.cumulative_crops:
            self.cumulative_crops[original_image_path] = []
        
        # Check if this exact crop already exists
        #existing_crop_ids = {
        #    (tuple(c['crop_coords']), round(c.get('rotation_angle', 0), 6), c.get('epoch', 0))
        #    for c in self.cumulative_crops[original_image_path]
        #}
        
        #if crop_id not in existing_crop_ids:
        #print(f"DEBUG: Adding new crop with coords {crop_coords_tuple}, rotation {rotation_angle}, epoch {epoch}")
        self.cumulative_crops[original_image_path].append(crop_info)
         
        # Update visualization
        if self.save_crop_visualizations:
            # Get the index of this image in our paths list
            self.save_crop_annotations()
            try:
                image_index = self.image_paths.index(original_image_path)
                self.create_cumulative_crop_visualization(image_index)
            except ValueError:
                print(f"Warning: Could not find index for image path {original_image_path}")
        
        #print(f"DEBUG: Total unique crops for image {original_image_path}: {len(self.cumulative_crops[original_image_path])}")

    def create_cumulative_crop_visualization(self, original_image_index, save_path=None):
        """
        Create a cumulative visualization showing all crops for a specific original image
        
        Args:
            original_image_index: Index of the original image
            save_path: Optional path to save the visualization
        """
        # Get the original image
        original_image = self.images[original_image_index]
        original_image_path = self.image_paths[original_image_index]
        original_image_basename = os.path.basename(original_image_path)
        original_filename = path_to_safe_filename(original_image_path)
        
        # Load ALL crops from JSON file for this image
        all_crops = []
        if self.resume_dir:
            crop_annotations_file = os.path.join(self.resume_dir, "crop_annotations.json")
            if os.path.exists(crop_annotations_file):
                try:
                    with open(crop_annotations_file, 'r') as f:
                        loaded_crops = json.load(f)
                    
                    # Get crops for this specific image path
                    if original_image_path in loaded_crops:
                        all_crops = loaded_crops[original_image_path]
                except Exception as e:
                    print(f"Warning: Could not load crop annotations for visualization: {e}")

        # If no crops found in JSON, use in-memory crops as fallback
        if not all_crops and original_image_index in self.cumulative_crops:
            all_crops = self.cumulative_crops[original_image_path]
        
        if not all_crops:
            return
            
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Display the original image
        ax.imshow(original_image)
        
        # Draw rectangles for all crops
        for i, crop_info in enumerate(all_crops):
            corners = crop_info['crop_coords']  # List of 4 (x,y) corner points
            
            # Color scheme: green for latest, red for all others
            if i == len(all_crops) - 1:  # Latest crop
                color = 'green'
                linewidth = 3  # Make latest crop more prominent
            else:  # Previous crops
                color = 'red'
                linewidth = 2
            
            # Create polygon from the 4 corner points
            # Handle flattened array format: [x1, y1, x2, y2, x3, y3, x4, y4]
            # Reshape to (4, 2) format for matplotlib Polygon
            corners_array = np.array(corners)
            polygon_points = corners_array.reshape(-1, 2)
            
            # Create polygon patch
            polygon = patches.Polygon(polygon_points, 
                                    linewidth=linewidth, edgecolor=color, facecolor='none')
            
            ax.add_patch(polygon)
        
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

    def __getitem__(self, index):
        img = self.images[index].astype(np.uint8)
        seg = self.segs[index].astype(np.int32)
        anomaly_class = self.anomaly_classes[index]
        
        if self.augment:
            augmented = self.aug(image=img, mask=seg, index=index)
            img = augmented["image"]
            seg = augmented["mask"]
            transform_info = augmented["transform_info"]

            if self.track_crop:
                os.makedirs("tmp", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                corners = transform_info['crop_coords']
                # Handle flattened array format: [x1, y1, x2, y2, x3, y3, x4, y4]
                x1, y1 = corners[0], corners[1]  # First corner coordinates
                coords_str = f"x{x1}-y{y1}"
                tmp_filename = os.path.join("tmp", f"crop_verify_{coords_str}_{timestamp}.png")
                # Ensure img is uint8 for saving
                img_to_save = img.astype(np.uint8) if isinstance(img, np.ndarray) else img
                cv2.imwrite(tmp_filename, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    
            if self.save_crop_visualizations:
                self.add_crop_to_cumulative_map(self.image_paths[index], transform_info, self.current_epoch)


        if self.scratch:
            img, _, _ = add_scratch_controlled(img)
            anomaly_class = "defect"
        # Randomly apply additional synthetic defects
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
        img = img.astype(np.float32) / 255.0
        y = self.object_classes[index]
        if self.transform:
            img = self.transform(img)
        else:
            img = self.transform_volume(img)
            img = (img - 0.5) / 0.5
        
        # Always return 6 values, with crop_info being an empty dict when not tracking or no crop was performed
        default_crop_info = {
            'crop_coords': (-1, -1, -1, -1),  # Invalid coordinates to indicate no crop
            'rotation_angle': 0,
            'original_shape': self.images[index].shape[:2]
        }
        
        return (
            img,
            seg.astype(np.float32),
            int(y),
            self.image_paths[index],
            anomaly_class,
            transform_info if self.augment else default_crop_info,  # Return default dict if not tracking
        )
        
def random_brightness_contrast(img, brightness_limit=0.05, contrast_limit=0.05, p=0.5):
    import random
    import numpy as np

    brightness_factor = 0.0
    contrast_factor = 0.0
    applied = False

    if np.random.rand() < p:
        brightness_factor = np.random.uniform(-brightness_limit, brightness_limit)
        contrast_factor = np.random.uniform(-contrast_limit, contrast_limit)
        img = img.astype(np.float32)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * (1 + contrast_factor) + mean + brightness_factor * 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        applied = True

    return img, brightness_factor, contrast_factor, applied
