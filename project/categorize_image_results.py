import os
import json
import glob
import argparse
from collections import defaultdict
from analyze_confusion_matrix import path_to_safe_filename

def categorize_images(annotation_dir, evaluation_results_dir, output_file=None):
    eval_files = glob.glob(os.path.join(evaluation_results_dir, '*__evaluation.json'))
    categories = defaultdict(list)

    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        image_path = eval_data['image_path']
        predicted = set(tuple(x) for x in eval_data['defective_patches'])
        grid_size = eval_data['grid_size']

        annotation_filename = f"{path_to_safe_filename(image_path)}__annotations.json"
        annotation_file = os.path.join(annotation_dir, annotation_filename)
        if not os.path.exists(annotation_file):
            continue
        with open(annotation_file, 'r') as f:
            anno_data = json.load(f)
        gt = set(tuple(x) for x in anno_data['defective_patches'])

        # All possible grid cells
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        h, w = img.height, img.width
        n_rows = h // grid_size
        n_cols = w // grid_size
        all_cells = set((r, c) for r in range(n_rows) for c in range(n_cols))

        # Normal patches = all_cells - gt
        normal_patches = all_cells - gt

        # Compute detection stats
        has_defect = len(gt) > 0
        detected_any = len(predicted) > 0
        detected_all = gt.issubset(predicted) and len(gt) > 0
        detected_some = len(gt & predicted) > 0 and not detected_all
        detected_none = len(gt & predicted) == 0 and len(gt) > 0
        over_detected = len(predicted & normal_patches) > 0

        # Category assignment
        if not has_defect:
            if not detected_any:
                cat = 1  # No defect, nothing detected
            else:
                cat = 2  # No defect, but model detected some
        else:
            if detected_all:
                if not over_detected:
                    cat = 3  # All defects detected, no over-detection
                else:
                    cat = 4  # All defects detected, but over-detection
            elif detected_some:
                if not over_detected:
                    cat = 5  # Some defects detected, no over-detection
                else:
                    cat = 6  # Some defects detected, but over-detection
            elif detected_none:
                if not over_detected:
                    cat = 7  # No defects detected, no over-detection
                else:
                    cat = 8  # No defects detected, but over-detection
            else:
                # Should not happen, but fallback
                cat = 0

        categories[cat].append(os.path.basename(image_path))

    # Category descriptions
    category_descriptions = {
        1: "Image that has no defective patch and the model also didn't detect any defective patch.",
        2: "Image that has no defective patch but the model mistakenly detected few of the patches as defective.",
        3: "Image that has defective patches and the model detected every of them while not over detecting.",
        4: "Image that has defective patches and the model detected every of them but also over detected patches that are normal as defective.",
        5: "Image that has defective patches and the model detected some of them while not over detecting.",
        6: "Image that has defective patches and the model detected some of them but also over detected patches that are normal as defective.",
        7: "Image that has defective patches and the model detected none of them while not over detecting.",
        8: "Image that has defective patches and the model detected none of them but also over detected patches that are normal as defective."
    }

    # Output summary
    summary_lines = []
    for i in range(1, 9):
        imgs = categories[i]
        summary_lines.append(f"Category {i}: {len(imgs)} images")
        summary_lines.append(f"  {category_descriptions[i]}")
        for img in imgs:
            summary_lines.append(f"    {img}")
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    print(summary)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(summary)
        print(f"Summary saved to: {output_file}")

    return categories

def main():
    parser = argparse.ArgumentParser(description="Categorize images by detection/ground truth results")
    parser.add_argument("--annotation-dir", required=True, help="Directory containing annotation JSON files")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation JSON files")
    parser.add_argument("--output-file", help="File to save summary (optional)")
    args = parser.parse_args()

    categorize_images(args.annotation_dir, args.evaluation_dir, args.output_file)

if __name__ == "__main__":
    main()