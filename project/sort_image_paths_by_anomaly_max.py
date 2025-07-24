#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import re

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]

def collect_fp_patches(input_dir):
    fp_patches = []
    # Natural sort of JSON filenames
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    filenames.sort(key=natural_key)
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            image_path = data.get('image_path', '')
            for patch in data.get('patch_analysis', []):
                if patch.get('status') == 'FP':
                    anomaly_max = patch.get('anomaly_max')
                    grid_row = patch.get('grid_row')
                    grid_col = patch.get('grid_col')
                    if anomaly_max is not None and grid_row is not None and grid_col is not None:
                        fp_patches.append((anomaly_max, image_path, grid_row, grid_col))
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
    return fp_patches

def write_sorted_image_paths(fp_patches, output_path):
    # Sort by anomaly_max descending, then by image_path alphabetically
    fp_patches_sorted = sorted(fp_patches, key=lambda x: (-x[0], x[1]))
    with open(output_path, 'w') as f:
        for anomaly_max, image_path, grid_row, grid_col in fp_patches_sorted:
            f.write(f"{image_path}:{grid_row}:{grid_col}:{anomaly_max}\n")
    print(f"Sorted image paths written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Write image paths (with grid_row and grid_col) sorted by anomaly_max for status 'FP' from evaluation JSON files.")
    parser.add_argument('--input-dir', required=True, help='Directory containing evaluation JSON files')
    parser.add_argument('--output-path', help='Output text file (default: sorted_image_paths.txt in input-dir)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output_path or os.path.join(input_dir, 'sorted_image_paths.txt')

    fp_patches = collect_fp_patches(input_dir)
    write_sorted_image_paths(fp_patches, output_path)

if __name__ == "__main__":
    main() 