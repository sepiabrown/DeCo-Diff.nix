import re
import torch
import sys
import os

# Import your modules
sys.path.append("project")
from models import UNET_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from PCBDataLoader import PCBDataset
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image

log_path = "DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L/log_val.txt"
checkpoint_dir = "DeCo-Diff_pcb_all_UNet_L_128_CenterCrop/001-UNet_L/checkpoints"
csv_path = "splits/pcb-split.csv_val"  # <-- set this to your actual csv path

# Read the log file
with open(log_path, "r") as f:
    lines = f.readlines()

checkpoint_pattern = re.compile(r"Saved checkpoint to (.+/checkpoints/(\d+)\.pt)")

def get_val_loader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    # Create a custom validation dataset that reads from the correct CSV file
    # Read the validation CSV file
    df = pd.read_csv(csv_path)
    # Filter for validation data
    df = df.query('split=="val"')
    
    if len(df) == 0:
        raise Exception(f"No validation data found in {csv_path}")
    
    print(f"Found {len(df)} validation samples")
    
    # Create a simple dataset class for validation
    class SimpleValDataset(torch.utils.data.Dataset):
        def __init__(self, df, rootdir, transform, image_size, center_size):
            self.df = df
            self.rootdir = rootdir
            self.transform = transform
            self.image_size = image_size
            self.center_size = center_size
            
        def __len__(self):
            return len(self.df)
            
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            data_path = os.path.join(self.rootdir, row["image"])
            img = np.array(Image.open(data_path).convert("RGB")).astype(np.uint8)
            
            # Apply center crop if needed
            if self.center_size != self.image_size:
                h, w = img.shape[:2]
                crop_h = (h - self.center_size) // 2
                crop_w = (w - self.center_size) // 2
                img = img[crop_h:crop_h+self.center_size, crop_w:crop_w+self.center_size]
            
            img = img.astype(np.float32) / 255.0
            
            if self.transform:
                img = self.transform(img)
            
            # Return dummy values for compatibility
            return img, np.zeros((self.center_size, self.center_size)), 0, data_path, "good"
    
    val_dataset = SimpleValDataset(
        df=df,
        rootdir="/home/suwonp/dataset/PCB/Huang/PCB_DATASET/PCB-gray-128___deco-diff",
        transform=transform,
        image_size=args["image_size"],
        center_size=args["center_size"]
    )
    
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    return loader

def random_mask(x, mask_ratios, mask_patch_size=1):
    n, c, w, h = x.shape
    size = int(np.prod(x.shape[2:]) / (mask_patch_size**2))
    mask = torch.zeros((n, c, size)).to(x.device)
    for b in range(n):
        masked_indexes = np.arange(size)
        np.random.shuffle(masked_indexes)
        masked_indexes = masked_indexes[: int(size * (1 - mask_ratios[b]))]
        mask[b, :, masked_indexes] = 1
    mask = mask.reshape(n, c, int(w / mask_patch_size), int(w / mask_patch_size))
    mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)
    return mask

def shuffle_patches(image, patch_size):
    import torch.nn.functional as F
    N, C, H, W = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0
    unfolded = F.unfold(image, kernel_size=patch_size, stride=patch_size)
    num_patches = unfolded.shape[-1]
    unfolded = unfolded.view(N, C, P, P, num_patches)
    unfolded = unfolded.permute(0, 4, 1, 2, 3)
    unfolded = unfolded.reshape(N * num_patches, C, P, P)
    indices = torch.randperm(N * num_patches)
    shuffled_unfolded = unfolded[indices]
    shuffled_unfolded = shuffled_unfolded.view(N, num_patches, C, P, P)
    shuffled_unfolded = shuffled_unfolded.permute(0, 2, 3, 4, 1)
    shuffled_unfolded = shuffled_unfolded.contiguous().view(N * C * P * P, num_patches)
    folded = F.fold(shuffled_unfolded, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    folded = folded.view(N, C, H, W)
    return folded

def compute_val_loss(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    model = UNET_models[args["model_size"]](latent_size=args["center_size"] // 8)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args['vae_type']}").to(device)
    vae.eval()
    diffusion = create_diffusion(
        timestep_respacing="ddim10",
        predict_deviation=True,
        predict_xstart=False,
        sigma_small=False,
    )
    val_loader = get_val_loader(args)
    val_loss = 0
    val_mse = 0
    val_steps = 0
    with torch.no_grad():
        for val_x, _, val_y, _, _ in val_loader:
            val_x = val_x.to(device)
            val_x = vae.encode(val_x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (val_x.shape[0],), device=device)
            if args["image_size"] == 128:
                mask_patch_size = np.random.choice(
                    [1, 2, 4], 1, p=[0.443, 0.333, 0.224]
                ).item()
            if args["image_size"] == 224:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 7], 1, p=[0.4, 0.3, 0.2, 0.1]
                ).item()
            if args["image_size"] == 256:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 8], 1, p=[0.4, 0.3, 0.2, 0.1]
                ).item()
            if args["image_size"] == 320:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 8], 1, p=[0.4, 0.3, 0.2, 0.1]
                ).item()
            if args["image_size"] == 384:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 8, 12], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]
                ).item()
            if args["image_size"] == 448:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 8, 14], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]
                ).item()
            elif args["image_size"] == 512:
                mask_patch_size = np.random.choice(
                    [1, 2, 4, 8, 16], 1, p=[0.3, 0.25, 0.20, 0.15, 0.1]
                ).item()
            if args["mask_random_ratio"]:
                mask_ratios = np.random.uniform(low=0.0, high=0.7, size=val_x.shape[0])
            else:
                mask_ratio = args["mask_ratio"]
                mask_ratios = ([mask_ratio] * val_x.shape[0],)

            mask = random_mask(
                val_x, mask_ratios=mask_ratios, mask_patch_size=mask_patch_size
            )
            model_kwargs = {
                "context": torch.tensor(val_y).to(device).int().unsqueeze(1),
                "mask": mask,
            }
            noise_mask = random_mask(
                val_x,
                mask_ratios=np.random.uniform(
                    low=0.0, high=args["patch_shuffle_ratio"], size=val_x.shape[0]
                ),
                mask_patch_size=mask_patch_size,
            )
            noise = noise_mask * torch.randn_like(val_x, device=device) + (
                1 - noise_mask
            ) * shuffle_patches(val_x, mask_patch_size)
            loss_dict = diffusion.training_losses(
                model, val_x, t, model_kwargs, noise=noise
            )
            val_loss += loss_dict["loss"].mean().item()
            val_mse += loss_dict["mse"].mean().item()
            val_steps += 1
    val_loss /= max(val_steps, 1)
    val_mse /= max(val_steps, 1)
    return val_loss, val_mse

# Prepare to write new log
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    match = checkpoint_pattern.search(line)
    if match:
        ckpt_path = match.group(1)
        print(f'Processing checkpoint: {ckpt_path}')
        try:
            val_loss, val_mse = compute_val_loss(ckpt_path)
            val_line = f"[VAL] Validation loss for checkpoint {ckpt_path}: {val_mse:.4f}\n"
            print(val_line)
            new_lines.append(val_line)
        except Exception as e:
            val_line = f"[VAL] Could not compute validation loss for {ckpt_path}: {e}\n"
            print(val_line)

# Write back to the log file (or to a new file)
with open(log_path, "w") as f:
    f.writelines(new_lines)