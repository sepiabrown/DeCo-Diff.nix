import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import re
from collections import defaultdict
from utils import path_to_safe_filename

def compute_confusion_matrix_with_details(annotation_dir, evaluation_results_dir, output_dir=None):
    """
    Compute confusion matrix and track detailed information about false negatives.
    
    Args:
        annotation_dir: Directory containing annotation JSON files
        evaluation_results_dir: Directory containing evaluation JSON files
        output_dir: Directory to save detailed results (optional)
    
    Returns:
        dict: Confusion matrix results and false negative details
    """
    # Find all evaluation result files
    eval_files = glob.glob(os.path.join(evaluation_results_dir, '*__evaluation.json'))
    
    if not eval_files:
        print(f"No evaluation files found in {evaluation_results_dir}")
        return None
    
    print(f"Found {len(eval_files)} evaluation files")
    
    all_TP = all_FP = all_FN = all_TN = 0
    false_negatives = []  # Track false negative details
    false_positives = []  # Track false positive details
    true_positives = []   # Track true positive details
    
    for eval_file in eval_files:
        print(f"Processing: {os.path.basename(eval_file)}")
        
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        image_path = eval_data['image_path']
        predicted = set(tuple(x) for x in eval_data['defective_patches'])
        grid_size = eval_data['grid_size']
        
        # Look for annotation file with the correct naming convention
        annotation_filename = f"{path_to_safe_filename(image_path)}__annotations.json"
        annotation_file = os.path.join(annotation_dir, annotation_filename)
        
        if not os.path.exists(annotation_file):
            print(f"Warning: No annotation file found: {annotation_file}")
            continue
            
        with open(annotation_file, 'r') as f:
            anno_data = json.load(f)
        
        gt = set(tuple(x) for x in anno_data['defective_patches'])
        
        # Get image dimensions to calculate grid
        try:
            img = PILImage.open(image_path)
            h, w = img.height, img.width
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}")
            continue
            
        n_rows = h // grid_size
        n_cols = w // grid_size
        all_cells = set((r, c) for r in range(n_rows) for c in range(n_cols))
        
        # Process each cell
        for cell in all_cells:
            grid_row, grid_col = cell
            pred = cell in predicted
            truth = cell in gt
            
            # Calculate pixel coordinates for the patch
            pixel_x = grid_col * grid_size
            pixel_y = grid_row * grid_size
            
            patch_info = {
                'image_path': image_path,
                'grid_position': [grid_row, grid_col],
                'pixel_coordinates': [pixel_x, pixel_y, pixel_x + grid_size, pixel_y + grid_size],
                'patch_size': grid_size
            }
            
            if pred and truth:
                all_TP += 1
                true_positives.append(patch_info)
            elif pred and not truth:
                all_FP += 1
                false_positives.append(patch_info)
            elif not pred and truth:
                all_FN += 1
                false_negatives.append(patch_info)
            else:
                all_TN += 1
    
    # Calculate metrics
    total = all_TP + all_FP + all_FN + all_TN
    accuracy = (all_TP + all_TN) / total if total > 0 else 0
    precision = all_TP / (all_TP + all_FP) if (all_TP + all_FP) > 0 else 0
    recall = all_TP / (all_TP + all_FN) if (all_TP + all_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("CONFUSION MATRIX RESULTS")
    print("="*60)
    print(f"True Positives (TP): {all_TP}")
    print(f"False Positives (FP): {all_FP}")
    print(f"False Negatives (FN): {all_FN}")
    print(f"True Negatives (TN): {all_TN}")
    print(f"Total Patches: {total}")
    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Print false negative details
    print("\n" + "="*60)
    print(f"FALSE NEGATIVES ({len(false_negatives)} total)")
    print("="*60)
    
    if false_negatives:
        # Group by image file
        fn_by_image = defaultdict(list)
        for fn in false_negatives:
            image_name = os.path.basename(fn['image_path'])
            fn_by_image[image_name].append(fn)
        
        for image_name, patches in fn_by_image.items():
            print(f"\nImage: {image_name}")
            print(f"  False Negatives: {len(patches)}")
            for patch in patches:
                grid_row, grid_col = patch['grid_position']
                pixel_x, pixel_y, pixel_x2, pixel_y2 = patch['pixel_coordinates']
                print(f"    - Grid [{grid_row}, {grid_col}] at pixels ({pixel_x},{pixel_y})-({pixel_x2},{pixel_y2})")
    else:
        print("No false negatives found!")
    
    # Create confusion matrix visualization
    cm = np.array([[all_TN, all_FP], [all_FN, all_TP]])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Patch-level)', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Set labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Defective'], fontsize=12)
    plt.yticks(tick_marks, ['Normal', 'Defective'], fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1_score:.4f}'
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    # Save results
    results = {
        "confusion_matrix": {
            "TP": all_TP,
            "FP": all_FP,
            "FN": all_FN,
            "TN": all_TN,
            "total": total
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_positives": true_positives
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix plot
        cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix plot saved to: {cm_plot_path}")
        
        # Save detailed results
        results_file = os.path.join(output_dir, "detailed_confusion_matrix.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")
        
        # Save false negatives summary
        fn_summary_file = os.path.join(output_dir, "false_negatives_summary.txt")
        with open(fn_summary_file, 'w') as f:
            f.write("FALSE NEGATIVES SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total False Negatives: {len(false_negatives)}\n\n")
            
            if false_negatives:
                fn_by_image = defaultdict(list)
                for fn in false_negatives:
                    image_name = os.path.basename(fn['image_path'])
                    fn_by_image[image_name].append(fn)
                
                for image_name, patches in fn_by_image.items():
                    f.write(f"Image: {image_name}\n")
                    f.write(f"  False Negatives: {len(patches)}\n")
                    for patch in patches:
                        grid_row, grid_col = patch['grid_position']
                        pixel_x, pixel_y, pixel_x2, pixel_y2 = patch['pixel_coordinates']
                        f.write(f"    - Grid [{grid_row}, {grid_col}] at pixels ({pixel_x},{pixel_y})-({pixel_x2},{pixel_y2})\n")
                    f.write("\n")
        
        print(f"False negatives summary saved to: {fn_summary_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compute confusion matrix and track false negative patches")
    parser.add_argument("--annotation-dir", required=True, help="Directory containing annotation JSON files")
    parser.add_argument("--evaluation-dir", required=True, help="Directory containing evaluation JSON files")
    parser.add_argument("--output-dir", help="Directory to save detailed results (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotation_dir):
        print(f"Error: Annotation directory does not exist: {args.annotation_dir}")
        return
    
    if not os.path.exists(args.evaluation_dir):
        print(f"Error: Evaluation directory does not exist: {args.evaluation_dir}")
        return
    
    print(f"Annotation directory: {args.annotation_dir}")
    print(f"Evaluation directory: {args.evaluation_dir}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    
    results = compute_confusion_matrix_with_details(
        args.annotation_dir, 
        args.evaluation_dir, 
        args.output_dir
    )
    
    if results:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed!")

if __name__ == "__main__":
    main() 