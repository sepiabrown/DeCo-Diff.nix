import os
import json
import glob
import argparse
from utils import path_to_safe_filename

def main():
    parser = argparse.ArgumentParser(description="Generate JSONs with false positive patch positions per evaluation file.")
    parser.add_argument("--annotation-dir", required=True, help="Directory containing annotation JSON files")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-evaluation false positive JSONs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_files = glob.glob(os.path.join(args.evaluation_dir, '*__evaluation.json'))
    if not eval_files:
        print(f"No evaluation files found in {args.evaluation_dir}")
        return

    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        image_path = eval_data['image_path']
        grid_size = eval_data['grid_size']
        predicted = set()
        for patch in eval_data['patch_analysis']:
            if patch['is_defective']:
                predicted.add((patch['grid_row'], patch['grid_col']))
        annotation_filename = f"{path_to_safe_filename(image_path)}__annotations.json"
        annotation_file = os.path.join(args.annotation_dir, annotation_filename)
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation file found: {annotation_file}")
            continue
        with open(annotation_file, 'r') as f:
            anno_data = json.load(f)
        gt = set(tuple(x) for x in anno_data['defective_patches'])
        # Get image dimensions to calculate grid
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            h, w = img.height, img.width
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}")
            continue
        n_rows = h // grid_size
        n_cols = w // grid_size
        all_cells = set((r, c) for r in range(n_rows) for c in range(n_cols))
        false_positives = []
        for cell in all_cells:
            grid_row, grid_col = cell
            pred = cell in predicted
            truth = cell in gt
            pixel_x = grid_col * grid_size
            pixel_y = grid_row * grid_size
            patch_info = {
                'grid_position': [grid_row, grid_col],
                'pixel_coordinates': [pixel_x, pixel_y, pixel_x + grid_size, pixel_y + grid_size],
                'patch_size': grid_size
            }
            if pred and not truth:
                false_positives.append(patch_info)
        # Write output JSON for this evaluation file
        eval_base = os.path.splitext(os.path.basename(eval_file))[0]
        output_json = os.path.join(args.output_dir, f"{eval_base}__false_positives.json")
        output_data = {
            'image_path': image_path,
            'false_positives': false_positives
        }
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Wrote {len(false_positives)} false positives to {output_json}")

if __name__ == "__main__":
    main() 