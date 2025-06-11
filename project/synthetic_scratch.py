import argparse
import cv2
import numpy as np
import os
import random
import json
from typing import Tuple, List
from pathlib import Path
from tqdm import tqdm



def bezier_curve(P0, P1, P2, num_points=100):
    t = np.linspace(0, 1, num_points)[:, None]
    curve = (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2
    return curve.astype(np.int32)

def is_grayscale_soft(image: np.ndarray, atol: float = 1e-3) -> bool:
    """Returns True if the image is grayscale in format or content (R≈G≈B)."""
    if image.ndim == 2:
        return True  # Already grayscale
    if image.ndim == 3:
        if image.shape[2] == 1:
            return True  # Single-channel grayscale
        if image.shape[2] == 3:
            # Check that all RGB channels are approximately equal
            r, g, b = image[..., 0], image[..., 1], image[..., 2]
            return np.allclose(r, g, atol=atol) and np.allclose(r, b, atol=atol)
    return False

def add_scratch_controlled(
    image: np.ndarray,
    num_scratches: int = 1,
    thickness: int = 2,
    length_range: Tuple[int, int] = (30, 80),
    curvature: float = 0.0,
    deviation_range: Tuple[int, int] = (10, 10),
    sigma: float = 0.0  # Gaussian noise σ per pixel
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Adds controlled synthetic scratches to an image with adaptive intensity.

    Parameters:
    - image: np.ndarray, input RGB or grayscale image.
    - num_scratches: int, number of scratches to add.
    - thickness: int, pixel thickness of the scratch.
    - length_range: tuple(int, int), min and max scratch length.
    - curvature: float, 0 = straight, up to ~1 = highly curved.
    - deviation_range: tuple(int, int), range of intensity deviation from background.

    Returns:
    - img_with_scratch: image with synthetic scratches.
    - mask: binary mask of scratch pixels.
    - info: list of metadata for each scratch.
    """
    h, w = image.shape[:2]
    img_out = image.copy().astype(np.float32)  # use float temporarily
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    scratch_info = []

    for _ in range(num_scratches):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.uniform(0, 2 * np.pi)
        length = random.randint(*length_range)
        x2 = np.clip(x1 + int(length * np.cos(angle)), 0, w - 1)
        y2 = np.clip(y1 + int(length * np.sin(angle)), 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        mid = None
        if curvature > 0:
            mid_x = int((x1 + x2) / 2 + random.uniform(-curvature, curvature) * length)
            mid_y = int((y1 + y2) / 2 + random.uniform(-curvature, curvature) * length)
            mid = [mid_x, mid_y]
            pts = bezier_curve(np.array([x1, y1]), np.array(mid), np.array([x2, y2])).astype(int).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, 255, thickness)
        else:
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

        region = image[mask == 255]
        avg_color = np.median(region) if is_grayscale_soft(image) else np.median(region, axis=0)
        deviation = random.randint(*deviation_range)
        base_color = avg_color + deviation

        # --- SOFT EDGE GRADATION ---
        # Blur the mask to create soft edges
        blur_ksize = max(3, thickness * 4 // 2 * 2 + 1)  # odd, proportional to thickness
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_ksize, blur_ksize), sigmaX=thickness)
        soft_mask = soft_mask / 255.0  # normalize to [0, 1]

        # If grayscale, make sure base_color and img_out are 3-channel for blending
        if is_grayscale_soft(image):
            base_color = np.array([base_color] * 3)
            if img_out.ndim == 2:
                img_out = np.stack([img_out]*3, axis=-1)

        # Prepare scratch color image
        scratch_img = np.zeros_like(img_out, dtype=np.float32)
        if sigma > 0:
            noise = np.random.normal(0, sigma, img_out.shape)
        else:
            noise = np.zeros_like(img_out)
        scratch_img[:] = base_color
        scratch_img += noise
        scratch_img = np.clip(scratch_img, 0, 255)

        # Blend using the soft mask
        for c in range(3):
            img_out[..., c] = img_out[..., c] * (1 - soft_mask) + scratch_img[..., c] * soft_mask

        # Update binary mask for output (optional: threshold soft_mask)
        scratch_mask = np.maximum(scratch_mask, (soft_mask > 0.05).astype(np.uint8) * 255)

        scratch_info.append({
            "start": [int(x1), int(y1)],
            "end": [int(x2), int(y2)],
            "thickness": int(thickness),
            "avg_background": float(avg_color) if not isinstance(avg_color, np.ndarray) else [float(x) for x in avg_color],
            "deviation": int(deviation),
            "sigma": float(sigma),
            "curvature": float(curvature)
        })
        if mid:
            scratch_info[-1]["mid"] = [int(mid[0]), int(mid[1])]

    # Clip and convert back to uint8
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    return img_out, scratch_mask, scratch_info

# Dataset generator
def generate_synthetic_dataset(input_dir: str, output_dir: str, severities: dict):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    image_files = list(input_path.glob("*.png"))
    if not image_files:
        print(f"[ERROR] No .png files found in input directory: {input_dir}")
        exit(1)
    for severity, params in severities.items():
        img_out_dir = output_path / severity / "images"
        mask_out_dir = output_path / severity / "masks"
        meta_out_dir = output_path / severity / "metadata"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)
        meta_out_dir.mkdir(parents=True, exist_ok=True)

        for img_file in tqdm(image_files, desc=f"[{severity}]"):
            img = cv2.imread(str(img_file))
            img_defect, mask, meta = add_scratch_controlled(
                img,
                **params
            )
            base_name = img_file.stem
            cv2.imwrite(str(img_out_dir / f"{base_name}_defect.png"), img_defect)
            cv2.imwrite(str(mask_out_dir / f"{base_name}_mask.png"), mask)
            with open(meta_out_dir / f"{base_name}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    print("Synthetic scratch dataset generated successfully.")
    print(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--severities_json", required=True, type=Path, help="Path to severities.json")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {args.input_dir}")
        exit(1)

    # Load severity definitions
    with open(args.severities_json) as f:
        severities = json.load(f)

    # Convert length_range/deviation_range to tuples
    for k, v in severities.items():
        v["length_range"] = tuple(v["length_range"])
        v["deviation_range"] = tuple(v["deviation_range"])

    generate_synthetic_dataset(args.input_dir, args.output_dir, severities)


if __name__ == "__main__":
    main()
