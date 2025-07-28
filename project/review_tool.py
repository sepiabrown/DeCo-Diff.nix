#!/usr/bin/env python3
"""
Review Tool for DeCo-Diff Fine-tuning

This tool helps with the fine-tuning workflow by:
1. Allowing manual review of images
2. Updating the training dataset with selected normal images
3. Creating new split files for retraining

Usage:
    python review_tool.py --input-dir path/to/evaluation_results --split-csv path/to/pcb-split.csv
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
        self.tp_entries = []
        self.fn_entries = []
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
        """Load all patch entries from evaluation results"""
        self.fp_entries = []
        self.tp_entries = []
        self.fn_entries = []
        self.tn_entries = []
        self.grid_size = None  # Will be extracted from evaluation files
        
        # Natural sort of JSON filenames
        filenames = [f for f in os.listdir(self.evaluation_dir) if f.endswith('.json')]
        filenames.sort(key=natural_key)
        
        for filename in filenames:
            file_path = os.path.join(self.evaluation_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract grid_size from the first file we process
                if self.grid_size is None:
                    # Try to get grid_size from metadata first
                    if 'metadata' in data and 'grid_size' in data['metadata']:
                        self.grid_size = data['metadata']['grid_size']
                    # If not in metadata, try to infer from patch coordinates
                    elif 'patch_analysis' in data and data['patch_analysis']:
                        # Look for patch coordinates to infer grid size
                        for patch in data['patch_analysis']:
                            if 'pixel_coordinates' in patch:
                                coords = patch['pixel_coordinates']
                                if len(coords) >= 4:
                                    # Calculate grid size from pixel coordinates
                                    x1, y1, x2, y2 = coords[:4]
                                    if x2 > x1 and y2 > y1:
                                        grid_size_x = x2 - x1
                                        grid_size_y = y2 - y1
                                        if grid_size_x == grid_size_y:
                                            self.grid_size = grid_size_x
                                            break
                    # If still not found, use default
                    if self.grid_size is None:
                        self.grid_size = 128  # Default fallback
                        print(f"Warning: Could not determine grid_size from evaluation files, using default: {self.grid_size}")
                    else:
                        print(f"Extracted grid_size: {self.grid_size}")
                
                image_path = data.get('image_path', '')
                for patch in data.get('patch_analysis', []):
                    status = patch.get('status')
                    anomaly_max = patch.get('anomaly_max')
                    grid_row = patch.get('grid_row')
                    grid_col = patch.get('grid_col')
                    
                    if (anomaly_max is not None and 
                        grid_row is not None and 
                        grid_col is not None and
                        (max_anomaly_threshold is None or anomaly_max <= max_anomaly_threshold)):
                        
                        entry = {
                            'image_path': image_path,
                            'grid_row': grid_row,
                            'grid_col': grid_col,
                            'anomaly_max': anomaly_max
                        }
                        
                        if status == 'FP':
                            self.fp_entries.append(entry)
                        elif status == 'TP':
                            self.tp_entries.append(entry)
                        elif status == 'FN':
                            self.fn_entries.append(entry)
                        elif status == 'TN':
                            self.tn_entries.append(entry)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
        
        # Sort by anomaly_max descending for each type
        self.fp_entries.sort(key=lambda x: (-x['anomaly_max'], natural_key(x['image_path'])))
        self.tp_entries.sort(key=lambda x: (-x['anomaly_max'], natural_key(x['image_path'])))
        self.fn_entries.sort(key=lambda x: (-x['anomaly_max'], natural_key(x['image_path'])))
        self.tn_entries.sort(key=lambda x: (-x['anomaly_max'], natural_key(x['image_path'])))
        
        # Combine FP and TN for review (as before)
        self.fp_entries = self.fp_entries[:1000]
        
        print(f"Loaded {len(self.fp_entries)} total entries:")
        print(f"  FP: {len([e for e in self.fp_entries if e in self.fp_entries[:1000]])}")
        print(f"  TP: {len(self.tp_entries)}")
        print(f"  FN: {len(self.fn_entries)}")
        print(f"  TN: {len(self.tn_entries)}")
        print(f"  Grid size: {self.grid_size}")

    def create_review_list(self, output_path: str = None, copy_images_dir: str = None):
        """Create separate JSON review files for each patch type and organize images accordingly"""
        # Create separate directories for each type
        base_dir = os.path.dirname(self.evaluation_dir) if copy_images_dir else self.evaluation_dir
        
        fp_dir = os.path.join(base_dir, 'FP')
        tp_dir = os.path.join(base_dir, 'TP')
        fn_dir = os.path.join(base_dir, 'FN')
        tn_dir = os.path.join(base_dir, 'TN')
        
        if copy_images_dir:
            os.makedirs(fp_dir, exist_ok=True)
            os.makedirs(tp_dir, exist_ok=True)
            os.makedirs(fn_dir, exist_ok=True)
            os.makedirs(tn_dir, exist_ok=True)
            
            # Create images subdirectories
            fp_images_dir = os.path.join(fp_dir, 'images')
            tp_images_dir = os.path.join(tp_dir, 'images')
            fn_images_dir = os.path.join(fn_dir, 'images')
            tn_images_dir = os.path.join(tn_dir, 'images')
            
            os.makedirs(fp_images_dir, exist_ok=True)
            os.makedirs(tp_images_dir, exist_ok=True)
            os.makedirs(fn_images_dir, exist_ok=True)
            os.makedirs(tn_images_dir, exist_ok=True)
        
        # Track copied images for each type
        copied_images = {
            'FP': set(),
            'TP': set(),
            'FN': set(),
            'TN': set()
        }
        
        # Create separate JSON files for each type
        fp_review_data = {
            "metadata": {
                "evaluation_dir": self.evaluation_dir,
                "total_entries": len([e for e in self.fp_entries if e in self.fp_entries[:1000]]),
                "description": "False Positive review file. Mark entries with 'selected': false to exclude from training.",
                "grid_size": self.grid_size
            },
            "entries": []
        }
        
        tp_review_data = {
            "metadata": {
                "evaluation_dir": self.evaluation_dir,
                "total_entries": len(self.tp_entries),
                "description": "True Positive review file. Mark entries with 'selected': false to exclude from training.",
                "grid_size": self.grid_size
            },
            "entries": []
        }
        
        fn_review_data = {
            "metadata": {
                "evaluation_dir": self.evaluation_dir,
                "total_entries": len(self.fn_entries),
                "description": "False Negative review file. Mark entries with 'selected': false to exclude from training.",
                "grid_size": self.grid_size
            },
            "entries": []
        }
        
        tn_review_data = {
            "metadata": {
                "evaluation_dir": self.evaluation_dir,
                "total_entries": len(self.tn_entries),
                "description": "True Negative review file. Mark entries with 'selected': false to exclude from training.",
                "grid_size": self.grid_size
            },
            "entries": []
        }
        
        # Process FP entries
        for entry in self.fp_entries[:1000]:  # Only the first 1000 FP entries
            # Copy image to FP directory
            if copy_images_dir and entry['image_path'] not in copied_images['FP']:
                try:
                    if os.path.exists(entry['image_path']):
                        shutil.copy2(entry['image_path'], fp_images_dir)
                        copied_images['FP'].add(entry['image_path'])
                except Exception as e:
                    print(f"Warning: Could not copy {entry['image_path']}: {e}")
            
            # Add entry to FP JSON
            review_entry = {
                "image_path": entry['image_path'],
                "grid_row": entry['grid_row'],
                "grid_col": entry['grid_col'],
                "anomaly_max": entry['anomaly_max'],
                "status": "FP",
                "selected": True
            }
            fp_review_data["entries"].append(review_entry)
        
        # Process TP entries
        for entry in self.tp_entries:
            # Copy image to TP directory
            if copy_images_dir and entry['image_path'] not in copied_images['TP']:
                try:
                    if os.path.exists(entry['image_path']):
                        shutil.copy2(entry['image_path'], tp_images_dir)
                        copied_images['TP'].add(entry['image_path'])
                except Exception as e:
                    print(f"Warning: Could not copy {entry['image_path']}: {e}")
            
            # Add entry to TP JSON
            review_entry = {
                "image_path": entry['image_path'],
                "grid_row": entry['grid_row'],
                "grid_col": entry['grid_col'],
                "anomaly_max": entry['anomaly_max'],
                "status": "TP",
                "selected": True
            }
            tp_review_data["entries"].append(review_entry)
        
        # Process FN entries
        for entry in self.fn_entries:
            # Copy image to FN directory
            if copy_images_dir and entry['image_path'] not in copied_images['FN']:
                try:
                    if os.path.exists(entry['image_path']):
                        shutil.copy2(entry['image_path'], fn_images_dir)
                        copied_images['FN'].add(entry['image_path'])
                except Exception as e:
                    print(f"Warning: Could not copy {entry['image_path']}: {e}")
            
            # Add entry to FN JSON
            review_entry = {
                "image_path": entry['image_path'],
                "grid_row": entry['grid_row'],
                "grid_col": entry['grid_col'],
                "anomaly_max": entry['anomaly_max'],
                "status": "FN",
                "selected": True
            }
            fn_review_data["entries"].append(review_entry)
        
        # Process TN entries
        for entry in self.tn_entries:
            # Copy image to TN directory
            if copy_images_dir and entry['image_path'] not in copied_images['TN']:
                try:
                    if os.path.exists(entry['image_path']):
                        shutil.copy2(entry['image_path'], tn_images_dir)
                        copied_images['TN'].add(entry['image_path'])
                except Exception as e:
                    print(f"Warning: Could not copy {entry['image_path']}: {e}")
            
            # Add entry to TN JSON
            review_entry = {
                "image_path": entry['image_path'],
                "grid_row": entry['grid_row'],
                "grid_col": entry['grid_col'],
                "anomaly_max": entry['anomaly_max'],
                "status": "TN",
                "selected": True
            }
            tn_review_data["entries"].append(review_entry)
        
        # Write JSON files
        fp_json_path = os.path.join(fp_dir, 'fp_review_list.json')
        tp_json_path = os.path.join(tp_dir, 'tp_review_list.json')
        fn_json_path = os.path.join(fn_dir, 'fn_review_list.json')
        tn_json_path = os.path.join(tn_dir, 'tn_review_list.json')
        
        with open(fp_json_path, 'w') as f:
            json.dump(fp_review_data, f, indent=2)
        with open(tp_json_path, 'w') as f:
            json.dump(tp_review_data, f, indent=2)
        with open(fn_json_path, 'w') as f:
            json.dump(fn_review_data, f, indent=2)
        with open(tn_json_path, 'w') as f:
            json.dump(tn_review_data, f, indent=2)
        
        if copy_images_dir:
            print(f"Created organized structure:")
            print(f"  FP: {len(copied_images['FP'])} images -> {fp_dir}/images/")
            print(f"  TP: {len(copied_images['TP'])} images -> {tp_dir}/images/")
            print(f"  FN: {len(copied_images['FN'])} images -> {fn_dir}/images/")
            print(f"  TN: {len(copied_images['TN'])} images -> {tn_dir}/images/")
            print(f"  JSON files created in respective directories")
        
        print(f"Review JSON files created:")
        print(f"  FP: {fp_json_path}")
        print(f"  TP: {tp_json_path}")
        print(f"  FN: {fn_json_path}")
        print(f"  TN: {tn_json_path}")
        print("Edit these JSON files and set 'selected': false for entries you want to exclude from training")
        
        # Return the FP JSON path as default (for backward compatibility)
        return fp_json_path
    
    def load_selections_from_file(self, review_file_path: str):
        """Load selections from a JSON review file"""
        self.selected_normal_images.clear()
        
        with open(review_file_path, 'r') as f:
            review_data = json.load(f)
        
        for entry in review_data.get("entries", []):
            if entry.get("selected", True):  # Default to selected if not specified
                image_path = entry.get("image_path")
                if image_path:
                    self.selected_normal_images.add(image_path)
        
        print(f"Loaded {len(self.selected_normal_images)} selected normal images from JSON review file")
    
    def create_training_json(self, output_path: str = None):
        """Create JSON configuration for training that can be used directly with train_DeCo_Diff.py"""
        if output_path is None:
            output_path = os.path.join(self.evaluation_dir, 'training_config.json')
        
        # Create training configuration
        training_config = {
            "metadata": {
                "original_training_images": len(self.existing_train_images) if hasattr(self, 'existing_train_images') else 0,
                "selected_normal_images": len(self.selected_normal_images),
                "evaluation_dir": self.evaluation_dir,
                "description": "Training configuration generated from false positive review",
                "grid_size": self.grid_size
            },
            "splits": {
                "train": [],
                "val": [],
                "test": []
            }
        }
        
        # Add selected images to training split
        for image_path in sorted(self.selected_normal_images):
            # Find corresponding patches for this image
            patches = []
            for entry in self.fp_entries:
                if entry['image_path'] == image_path:
                    patches.append({
                        'grid_row': entry['grid_row'],
                        'grid_col': entry['grid_col'],
                        'anomaly_max': entry['anomaly_max'],
                        'status': 'FP' if entry in self.fp_entries[:1000] else 'TN'
                    })
            
            train_entry = {
                "object": "pcb",
                "label": "normal",
                "image": image_path,
                "mask": "",
                "category": "good"
            }
            
            if patches:
                train_entry["selected_patches"] = patches
            
            training_config["splits"]["train"].append(train_entry)
        
        with open(output_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        print(f"Training JSON created: {output_path}")
        print("You can use this directly with train_DeCo_Diff.py:")
        print(f'"split-json-path": "{output_path}"')
        return output_path

def main():
    parser = argparse.ArgumentParser(description="False Positive Review Tool for DeCo-Diff Fine-tuning")
    parser.add_argument('--input-dir', required=True, help='Directory containing evaluation JSON files')
    parser.add_argument('--max-anomaly-threshold', type=int, help='Maximum anomaly_max threshold to consider')
    parser.add_argument('--review-file', help='Path to manually edited JSON review file')
    parser.add_argument('--output-training-json', help='Output path for training JSON configuration')
    parser.add_argument('--copy-images-dir', help='Directory to copy FP images for inspection (default: fp_images under input-dir)')
    
    args = parser.parse_args()
    
    if args.review_file:
        # Load selections from JSON review file
        tool = FPReviewTool(args.input_dir)
        tool.load_false_positives(args.max_anomaly_threshold)
        tool.load_selections_from_file(args.review_file)
        # Create training JSON configuration
        tool.create_training_json(args.output_training_json)
    else:
        tool = FPReviewTool(args.input_dir)
        tool.load_false_positives(args.max_anomaly_threshold)
        # Set default for copy_images_dir if not provided
        copy_images_dir = args.copy_images_dir or os.path.join(args.input_dir, 'fp_images')
        # Create new review list and optionally copy images
        review_file = tool.create_review_list(copy_images_dir=copy_images_dir)
        print(f"\nPlease edit {review_file} and set 'selected': false for entries you want to exclude from training")
        print("Then run this script again with --review-file {review_file} --output-training-json training_config.json")
        return

if __name__ == "__main__":
    main() 