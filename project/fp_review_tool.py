#!/usr/bin/env python3
"""
False Positive Review Tool for DeCo-Diff Fine-tuning

This tool helps with the fine-tuning workflow by:
1. Loading false positives from evaluation results
2. Allowing manual review of images
3. Updating the training dataset with selected normal images
4. Creating new split files for retraining

Usage:
    python fp_review_tool.py --input-dir path/to/evaluation_results --split-csv path/to/pcb-split.csv
"""

import os
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import shutil
import re

def natural_key(string):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', string)]
    
class FPReviewTool:
    def __init__(self, evaluation_dir: str, split_csv_path: str = None):
        self.evaluation_dir = evaluation_dir
        self.split_csv_path = split_csv_path
        self.fp_entries = []
        self.tn_entries = []
        self.selected_normal_images = set()
        self.existing_train_images = set()
        if split_csv_path:
            self.load_existing_training_data()
    
    def load_existing_training_data(self):
        """Load existing training images from split CSV"""
        with open(self.split_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == 'train' and row['label'] == 'normal':
                    self.existing_train_images.add(row['image'])
        print(f"Loaded {len(self.existing_train_images)} existing training images")
    
    def load_false_positives(self, max_anomaly_threshold: int = None):
        """Load false positive entries from evaluation results"""
        self.fp_entries = []
        self.tn_entries = []
        
        # Natural sort of JSON filenames
        filenames = [f for f in os.listdir(self.evaluation_dir) if f.endswith('.json')]
        filenames.sort(key=natural_key)
        
        for filename in filenames:
            file_path = os.path.join(self.evaluation_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                image_path = data.get('image_path', '')
                for patch in data.get('patch_analysis', []):
                    if patch.get('status') == 'FP':
                        anomaly_max = patch.get('anomaly_max')
                        grid_row = patch.get('grid_row')
                        grid_col = patch.get('grid_col')
                        
                        if (anomaly_max is not None and 
                            grid_row is not None and 
                            grid_col is not None and
                            (max_anomaly_threshold is None or anomaly_max <= max_anomaly_threshold)):
                            
                            self.fp_entries.append({
                                'image_path': image_path,
                                'grid_row': grid_row,
                                'grid_col': grid_col,
                                'anomaly_max': anomaly_max
                            })
                    if patch.get('status') == 'TN':
                        anomaly_max = patch.get('anomaly_max')
                        grid_row = patch.get('grid_row')
                        grid_col = patch.get('grid_col')
                        
                        if (anomaly_max is not None and 
                            grid_row is not None and 
                            grid_col is not None and
                            (max_anomaly_threshold is None or anomaly_max <= max_anomaly_threshold)):
                            
                            self.tn_entries.append({
                                'image_path': image_path,
                                'grid_row': grid_row,
                                'grid_col': grid_col,
                                'anomaly_max': anomaly_max
                            })
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
        
        # Sort by anomaly_max descending
        self.fp_entries.sort(key=lambda x: (-x['anomaly_max'], x['image_path']))
        self.fp_entries = self.fp_entries[:1000] + self.tn_entries
        self.fp_entries.sort(key=lambda x: (natural_key(x['image_path'])))
        print(f"Loaded {len(self.fp_entries)} false positive entries")
    
    def create_review_list(self, output_path: str = None, copy_images_dir: str = None):
        """Create a review list file with FP entries and optionally copy images for inspection"""
        if output_path is None:
            output_path = os.path.join(self.evaluation_dir, 'fp_review_list.txt')
        
        # Optionally copy images for inspection
        copied_images = set()
        if copy_images_dir:
            os.makedirs(copy_images_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# False Positive Review List\n")
            f.write("# Format: <image_path>:<grid_row>:<grid_col>:<anomaly_max>\n")
            f.write("# Mark with 'UNSELECT' at the end of line to drop from training data\n")
            f.write("# Example: path/to/image.png:0:0:255 UNSELECT\n\n")
            
            for entry in self.fp_entries:
                f.write(f"{entry['image_path']}:{entry['grid_row']}:{entry['grid_col']}:{entry['anomaly_max']}\n")
                # Copy image if requested and not already copied
                if copy_images_dir and entry['image_path'] not in copied_images:
                    try:
                        if os.path.exists(entry['image_path']):
                            shutil.copy2(entry['image_path'], copy_images_dir)
                            copied_images.add(entry['image_path'])
                    except Exception as e:
                        print(f"Warning: Could not copy {entry['image_path']}: {e}")
        
        if copy_images_dir:
            print(f"Copied {len(copied_images)} unique images to: {copy_images_dir}")
        print(f"Review list created: {output_path}")
        print("Edit this file and mark lines with 'UNSELECT' to drop images from training data")
        return output_path
    
    def load_selections_from_file(self, review_file_path: str):
        """Load selections from a manually edited review file"""
        self.selected_normal_images.clear()
        
        with open(review_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # If line ends with 'UNSELECT', skip it (do not add to selected images)
                if line.endswith('UNSELECT'):
                    continue
                # Otherwise, treat as selected
                parts = line.rsplit(':', 3)
                if len(parts) == 4:
                    image_path = parts[0]
                    self.selected_normal_images.add(image_path)
        
        print(f"Loaded {len(self.selected_normal_images)} selected normal images")
    
    def create_updated_split_csv(self, output_path: str = None, backup: bool = True):
        """Create updated split CSV with new training images"""
        if output_path is None:
            # Create backup of original
            if backup:
                backup_path = self.split_csv_path.replace('.csv', '_backup.csv')
                shutil.copy2(self.split_csv_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            output_path = self.split_csv_path.replace('.csv', '_updated.csv')
        
        # Read all existing data
        all_rows = []
        with open(self.split_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            all_rows = list(reader)
        
        # Add new training images
        new_train_count = 0
        newly_added_images = []
        for image_path in self.selected_normal_images:
            # Check if not already in training data
            if image_path not in self.existing_train_images:
                new_row = {
                    'object': 'pcb',
                    'split': 'train',
                    'label': 'normal',
                    'image': image_path,
                    'mask': '',
                    'category': 'good'
                }
                all_rows.append(new_row)
                new_train_count += 1
                newly_added_images.append(image_path)
        
        all_rows.sort(key=lambda x: (natural_key(x['image'])))
        # Write updated CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"Updated split CSV created: {output_path}")
        print(f"Added {new_train_count} new training images")
        #if newly_added_images:
        #    print("Newly added images:")
        #    for img in newly_added_images:
        #        print(img)
        return output_path
    
    def create_training_summary(self, output_path: str = None):
        """Create a summary of the fine-tuning changes"""
        if output_path is None:
            output_path = os.path.join(self.evaluation_dir, 'fine_tuning_summary.txt')
        
        with open(output_path, 'w') as f:
            f.write("Fine-tuning Summary\n")
            f.write("==================\n\n")
            f.write(f"Original training images: {len(self.existing_train_images)}\n")
            f.write(f"Selected normal images from FPs: {len(self.selected_normal_images)}\n")
            f.write(f"New training images added: {len(self.selected_normal_images - self.existing_train_images)}\n")
            f.write(f"Total training images after update: {len(self.existing_train_images) + len(self.selected_normal_images - self.existing_train_images)}\n\n")
            
            f.write("Selected Images:\n")
            f.write("---------------\n")
            for image_path in sorted(self.selected_normal_images):
                f.write(f"{image_path}\n")
        
        print(f"Summary created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="False Positive Review Tool for DeCo-Diff Fine-tuning")
    parser.add_argument('--input-dir', required=True, help='Directory containing evaluation JSON files')
    parser.add_argument('--split-csv', help='Path to the original split CSV file (required only for updating training set)')
    parser.add_argument('--max-anomaly-threshold', type=int, help='Maximum anomaly_max threshold to consider')
    parser.add_argument('--review-file', help='Path to manually edited review file')
    parser.add_argument('--output-split-csv', help='Output path for updated split CSV')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup of original CSV')
    parser.add_argument('--copy-images-dir', help='Directory to copy FP images for inspection (default: fp_images under input-dir)')
    
    args = parser.parse_args()
    
    if args.review_file:
        # For updating training set, split-csv is required
        if not args.split_csv:
            print("Error: --split-csv is required when updating the training set.")
            return
        # Re-initialize tool with split_csv and reload training data
        tool = FPReviewTool(args.input_dir, args.split_csv)
        tool.load_selections_from_file(args.review_file)
        # Create updated split CSV
        tool.create_updated_split_csv(args.output_split_csv, not args.no_backup)
        # Create summary
        tool.create_training_summary()
    else:
        tool = FPReviewTool(args.input_dir)
        tool.load_false_positives(args.max_anomaly_threshold)
        # Set default for copy_images_dir if not provided
        copy_images_dir = args.copy_images_dir or os.path.join(args.input_dir, 'fp_images')
        # Create new review list and optionally copy images
        review_file = tool.create_review_list(copy_images_dir=copy_images_dir)
        print(f"\nPlease edit {review_file} and mark lines with 'UNSELECT'")
        print("Then run this script again with --review-file {review_file} --split-csv path/to/pcb-split.csv")
        return

if __name__ == "__main__":
    main() 