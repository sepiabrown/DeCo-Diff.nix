import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# import torchio as tio
import warnings
import albumentations as A
from synthetic_scratch import add_scratch_controlled
import cv2
import random

warnings.filterwarnings("ignore")


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
    ):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.scratch = scratch
        self.mode = mode
        self.center_size = center_size
        self.augment = augment
        self.center_crop = center_crop
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
        df = pd.read_csv(os.path.join(".", "splits", "pcb-split.csv"))
        if num_datafile is not None:
            df = df.sample(n=num_datafile)
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

    def __getitem__(self, index):
        img = self.images[index].astype(np.uint8)
        seg = self.segs[index].astype(np.int32)
        anomaly_class = self.anomaly_classes[index]
        if self.augment:
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
        return (
            img,
            seg.astype(np.float32),
            int(y),
            self.image_paths[index],
            anomaly_class,
        )
