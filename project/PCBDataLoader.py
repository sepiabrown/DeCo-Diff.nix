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
        
    def apply(self, img, **params):
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
        return img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    
    def apply_to_bbox(self, bbox, **params):
        # Transform bbox coordinates
        if self.crop_coords is None:
            return bbox
            
        x1, y1, x2, y2 = bbox
        crop_x, crop_y, _, _ = self.crop_coords
        
        # Adjust bbox to cropped area
        new_x1 = max(0, x1 - crop_x)
        new_y1 = max(0, y1 - crop_y)
        new_x2 = min(self.width, x2 - crop_x)
        new_y2 = min(self.height, y2 - crop_y)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def get_params(self):
        return {"crop_coords": self.crop_coords}

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
        """
        self.scratch = scratch
        self.mode = mode
        self.center_size = center_size
        self.augment = augment
        self.center_crop = center_crop
        self.track_crop = track_crop
        self.save_crop_visualizations = save_crop_visualizations
        self.crop_vis_dir = crop_vis_dir
        
        # Create crop visualization directory if needed
        if self.save_crop_visualizations:
            os.makedirs(self.crop_vis_dir, exist_ok=True)
            
        # Dictionary to store cumulative crop information per original image
        self.cumulative_crops = defaultdict(list)
            
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
            data_path = os.path.join(rootdir, row["image"])
            img = np.array(
                Image.open(data_path).convert("RGB")
                # .resize((self.image_size, self.image_size))
            ).astype(np.uint8)
            self.image_paths.append(data_path)
            self.images.append(img)
            self.object_classes.append(object_cls_dict[row["object"]])
            self.anomaly_classes.append(row["category"])
            if row["category"] != "good":
                seg_path = os.path.join(rootdir, row["mask"])
                seg = (
                    np.array(
                        Image.open(seg_path).convert("L")
                        # .resize((self.image_size, self.image_size))
                    )
                    > 0
                ).astype(np.uint8)
                self.segs.append((seg))
            else:
                seg_path = os.path.join(rootdir, row["image"])
                if os.path.exists(seg_path):
                    seg_shape = np.array(Image.open(seg_path)).shape
                else:
                    seg_shape = (self.image_size, self.image_size)
                self.segs.append(np.zeros(seg_shape))
        
        if self.augment:
            if self.track_crop:
                # Use custom transforms that track coordinates
                self.aug = A.Compose(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.05, contrast_limit=0.05, p=0.5
                        ),
                        RotationTrackingTransform(limit=5, p=1),
                        CropTrackingTransform(height=self.image_size, width=self.image_size, p=1),
                    ]
                )
            else:
                # Original augmentation pipeline
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

    def transform_volume(self, x):
        x = torch.from_numpy(x.transpose((-1, 0, 1)))
        return x

    def __len__(self):
        return len(self.images)

    def get_crop_coordinates_after_rotation(self, crop_coords, rotation_angle, original_shape):
        """
        Calculate the crop coordinates in the original image space after rotation
        
        Args:
            crop_coords: (x1, y1, x2, y2) in rotated image space
            rotation_angle: rotation angle in degrees
            original_shape: (height, width) of original image
            
        Returns:
            (x1, y1, x2, y2) in original image space
        """
        if crop_coords is None or rotation_angle == 0:
            return crop_coords
            
        x1, y1, x2, y2 = crop_coords
        h_orig, w_orig = original_shape
        
        # Convert to radians
        angle_rad = math.radians(rotation_angle)
        
        # Get center of rotated image
        h_rot, w_rot = self.image_size, self.image_size
        center_rot = (w_rot // 2, h_rot // 2)
        
        # Calculate the four corners of the crop in rotated space
        corners_rot = np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ])
        
        # Create rotation matrix to go back to original space
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Transform corners back to original space
        corners_orig = []
        for corner in corners_rot:
            # Translate to origin
            x_rel = corner[0] - center_rot[0]
            y_rel = corner[1] - center_rot[1]
            
            # Apply inverse rotation
            x_rot = x_rel * cos_a - y_rel * sin_a
            y_rot = x_rel * sin_a + y_rel * cos_a
            
            # Translate back
            x_final = x_rot + center_rot[0]
            y_final = y_rot + center_rot[1]
            
            corners_orig.append([x_final, y_final])
        
        # Calculate bounding box in original space
        corners_orig = np.array(corners_orig)
        x1_orig = max(0, int(np.min(corners_orig[:, 0])))
        y1_orig = max(0, int(np.min(corners_orig[:, 1])))
        x2_orig = min(w_orig, int(np.max(corners_orig[:, 0])))
        y2_orig = min(h_orig, int(np.max(corners_orig[:, 1])))
        
        return (x1_orig, y1_orig, x2_orig, y2_orig)

    def add_crop_to_cumulative_map(self, original_image_index, crop_info):
        """
        Add crop information to the cumulative map for a specific original image
        and immediately save the updated visualization
        
        Args:
            original_image_index: Index of the original image
            crop_info: Dictionary containing crop coordinates and rotation info
        """
        if crop_info is None or 'crop_coords' not in crop_info:
            return
            
        # Add crop info to the cumulative list for this image
        self.cumulative_crops[original_image_index].append(crop_info)
        
        # Immediately save the updated cumulative visualization
        if self.save_crop_visualizations:
            self.create_cumulative_crop_visualization(original_image_index)

    def create_cumulative_crop_visualization(self, original_image_index, save_path=None):
        """
        Create a cumulative visualization showing all crops for a specific original image
        
        Args:
            original_image_index: Index of the original image
            save_path: Optional path to save the visualization
        """
        if original_image_index not in self.cumulative_crops or not self.cumulative_crops[original_image_index]:
            return
            
        # Get the original image
        original_image = self.images[original_image_index]
        crops = self.cumulative_crops[original_image_index]
        
        # Get original image filename for naming
        original_image_path = self.image_paths[original_image_index]
        original_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Display the original image
        ax.imshow(original_image)
        
        # Define colors for different crops (cycling through a color palette)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Draw rectangles for all crops
        for i, crop_info in enumerate(crops):
            x1, y1, x2, y2 = crop_info['crop_coords']
            rotation_angle = crop_info.get('rotation_angle', 0)
            color = colors[i % len(colors)]
            
            # Draw crop rectangle
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add crop number and coordinates
            ax.text(x1, y1 - 5, f'Crop {i+1}: ({x1}, {y1}) to ({x2}, {y2})', 
                    color=color, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # Add rotation info if applicable
            if rotation_angle != 0:
                ax.text(x1, y2 + 15, f'Rot: {rotation_angle:.1f}°', 
                        color=color, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
        
        # Add legend showing total number of crops
        ax.text(10, 30, f'Total Crops: {len(crops)}', 
                color='black', fontsize=14, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # Set title with original filename
        ax.set_title(f'Cumulative Crop Map - {original_filename} ({len(crops)} crops)', 
                    fontsize=16, weight='bold')
        ax.axis('off')
        
        # Save the visualization with original filename
        if save_path is None:
            save_path = os.path.join(self.crop_vis_dir, f"cumulative_crops_{original_filename}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative crop visualization saved to: {save_path}")
        return save_path

    def save_all_cumulative_visualizations(self):
        """
        Save cumulative crop visualizations for all images that have crops
        """
        for image_index in self.cumulative_crops:
            if self.cumulative_crops[image_index]:  # Only save if there are crops
                self.create_cumulative_crop_visualization(image_index)

    def __getitem__(self, index):
        img = self.images[index].astype(np.uint8)
        seg = self.segs[index].astype(np.int32)
        anomaly_class = self.anomaly_classes[index]
        
        crop_info = None
        if self.augment:
            if self.track_crop:
                # Apply augmentation and get transformation info
                augmented = self.aug(image=img, mask=seg)
                img = augmented["image"]
                seg = augmented["mask"]
                
                # Extract crop coordinates and rotation angle from transforms
                crop_coords = None
                rotation_angle = 0
                
                for transform in self.aug.transforms:
                    if hasattr(transform, 'crop_coords'):
                        crop_coords = transform.crop_coords
                    if hasattr(transform, 'rotation_angle'):
                        rotation_angle = transform.rotation_angle
                
                # Calculate final crop coordinates in original image space
                if crop_coords is not None:
                    original_shape = self.images[index].shape[:2]
                    final_crop_coords = self.get_crop_coordinates_after_rotation(
                        crop_coords, rotation_angle, original_shape
                    )
                    crop_info = {
                        'crop_coords': final_crop_coords,
                        'rotation_angle': rotation_angle,
                        'original_shape': original_shape
                    }
                    
                    # Add to cumulative map instead of creating individual visualizations
                    if self.save_crop_visualizations:
                        self.add_crop_to_cumulative_map(index, crop_info)
            else:
                # Original augmentation without tracking
                augmented = self.aug(image=img, mask=seg)
                img = augmented["image"]
                seg = augmented["mask"]

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
            M = np.float32([[1, 0, tx], [0, 1, 0]])
            img = cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT
            )
        if self.shift_y is not None:
            ty = self.shift_y
            assert abs(ty) < img.shape[1], "shift should be less than the image size"
            M = np.float32([[1, 0, 0], [0, 1, ty]])
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

        #import matplotlib.pyplot as plt
        ## Convert tensor back to numpy for saving
        #if isinstance(img, torch.Tensor):
        #    img_np = img.detach().cpu().numpy()
        #    if img_np.ndim == 3:
        #        img_np = img_np.transpose(1, 2, 0)  # CHW to HWC
        #    # Denormalize
        #    img_np = (img_np + 1) / 2  # [-1,1] to [0,1]
        #    img_np = np.clip(img_np, 0, 1)
        #else:
        #    img_np = img
        ## Save image
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        #filename = f"tmp/pcb_image_{index}_{timestamp}.png"
        #plt.imsave(filename, img_np)
        #print(f"Saved image to: {filename}")
        #breakpoint()
        #print(f"Image shape: {img_np.shape}, dtype: {img_np.dtype}")
        #print(f"Image range: [{img_np.min():.3f}, {img_np.max():.3f}]")
        
        if self.track_crop and crop_info is not None:
            return (
                img,
                seg.astype(np.float32),
                int(y),
                self.image_paths[index],
                anomaly_class,
                crop_info,  # Add crop information to the return tuple
            )
        else:
            return (
                img,
                seg.astype(np.float32),
                int(y),
                self.image_paths[index],
                anomaly_class,
            )
