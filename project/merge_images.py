import os
import sys
import cv2
from glob import glob
import numpy as np

def merge_images(img1_path, img2_path, out_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f"Error reading {img1_path} or {img2_path}")
        return
    # Resize to the same size if needed
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))
    # Create a 3-channel image: R=img1, G=img2, B=0
    merged = cv2.merge([np.zeros_like(img1), img2, img1])  # B, G, R order in OpenCV
    cv2.imwrite(out_path, merged)

def merge_from_two_folders(folder1, folder2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files1 = {os.path.basename(f): f for f in glob(os.path.join(folder1, '*'))}
    files2 = {os.path.basename(f): f for f in glob(os.path.join(folder2, '*'))}
    common = set(files1.keys()) & set(files2.keys())
    for fname in common:
        out_path = os.path.join(out_dir, fname)
        merge_images(files1[fname], files2[fname], out_path)
        print(f"Merged: {fname}")

def merge_from_one_folder(folder, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(folder, '*')))
    used = set()
    for i, f1 in enumerate(files):
        name1 = os.path.basename(f1)
        prefix1 = name1[:5]
        for j, f2 in enumerate(files):
            if i >= j or f2 in used:
                continue
            name2 = os.path.basename(f2)
            if name1[:5] == name2[:5]:
                out_name = f"{name1}_MERGED_{name2}"
                out_path = os.path.join(out_dir, out_name)
                merge_images(f1, f2, out_path)
                used.add(f1)
                used.add(f2)
                print(f"Merged: {name1} + {name2}")
                break

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Two folders: python merge_images.py folder1 folder2 out_dir
        merge_from_two_folders(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        # One folder: python merge_images.py folder out_dir
        merge_from_one_folder(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  python merge_images.py folder1 folder2 out_dir")
        print("  python merge_images.py folder out_dir")