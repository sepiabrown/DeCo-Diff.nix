#!/usr/bin/env python3
"""
Automated Fine-tuning Workflow for DeCo-Diff

This script automates the fine-tuning process:
1. Run evaluation on current model
2. Generate FP review list
3. Apply automated or manual FP filtering
4. Update training dataset
5. Retrain model with new data

Usage:
    python automated_fine_tuning.py --config fine_tuning_config.json
"""

import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class AutomatedFineTuning:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_directories()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load fine-tuning configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_directories(self):
        """Setup working directories"""
        self.work_dir = Path(self.config.get('work_dir', 'fine_tuning_work'))
        self.work_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.work_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"Working directory: {self.run_dir}")
    
    def run_evaluation(self):
        """Run evaluation on current model"""
        print("Running evaluation...")
        
        eval_config = self.config['evaluation']
        cmd = [
            'python', 'evaluation_DeCo_Diff.py',
            '--input_json', eval_config['input_json'],
            '--model_path', eval_config['model_path'],
            '--output_dir', str(self.run_dir / 'evaluation_results'),
            '--batch_size', str(eval_config.get('batch_size', 1)),
            '--device', eval_config.get('device', 'cuda')
        ]
        
        if 'additional_args' in eval_config:
            for arg in eval_config['additional_args']:
                cmd.extend(arg.split())
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr}")
            return False
        
        print("Evaluation completed successfully")
        return True
    
    def generate_fp_review_list(self):
        """Generate FP review list from evaluation results"""
        print("Generating FP review list...")
        
        eval_results_dir = self.run_dir / 'evaluation_results'
        split_csv = self.config['data']['split_csv']
        
        cmd = [
            'python', 'fp_review_tool.py',
            '--input-dir', str(eval_results_dir),
            '--split-csv', split_csv
        ]
        
        if 'max_anomaly_threshold' in self.config['fp_filtering']:
            cmd.extend(['--max-anomaly-threshold', 
                       str(self.config['fp_filtering']['max_anomaly_threshold'])])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FP review list generation failed: {result.stderr}")
            return False
        
        print("FP review list generated")
        return True
    
    def apply_automated_filtering(self):
        """Apply automated filtering based on config"""
        print("Applying automated filtering...")
        
        fp_config = self.config['fp_filtering']
        review_file = self.run_dir / 'evaluation_results' / 'fp_review_list.txt'
        
        if not review_file.exists():
            print("Review file not found")
            return False
        
        # Read and filter entries
        filtered_entries = []
        with open(review_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(':')
                if len(parts) >= 4:
                    anomaly_max = int(parts[3])
                    
                    # Apply filters
                    if (anomaly_max <= fp_config.get('max_anomaly_threshold', 255) and
                        anomaly_max >= fp_config.get('min_anomaly_threshold', 0)):
                        filtered_entries.append(line + ' SELECT')
                    else:
                        filtered_entries.append(line)
        
        # Write filtered review file
        filtered_review_file = self.run_dir / 'filtered_fp_review_list.txt'
        with open(filtered_review_file, 'w') as f:
            f.write("# Automatically filtered FP review list\n")
            f.write("# Format: <image_path>:<grid_row>:<grid_col>:<anomaly_max>\n\n")
            for entry in filtered_entries:
                f.write(entry + '\n')
        
        print(f"Automated filtering completed: {filtered_review_file}")
        return str(filtered_review_file)
    
    def update_training_dataset(self, review_file: str):
        """Update training dataset with selected images"""
        print("Updating training dataset...")
        
        eval_results_dir = self.run_dir / 'evaluation_results'
        split_csv = self.config['data']['split_csv']
        updated_csv = self.run_dir / 'updated_split.csv'
        
        cmd = [
            'python', 'fp_review_tool.py',
            '--input-dir', str(eval_results_dir),
            '--split-csv', split_csv,
            '--review-file', review_file,
            '--output-split-csv', str(updated_csv),
            '--no-backup'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Dataset update failed: {result.stderr}")
            return False
        
        print(f"Training dataset updated: {updated_csv}")
        return str(updated_csv)
    
    def retrain_model(self, updated_split_csv: str):
        """Retrain model with updated dataset"""
        print("Starting model retraining...")
        
        train_config = self.config['retraining']
        
        # Update input JSON for training
        train_input_json = self.run_dir / 'train_input_updated.json'
        with open(train_config['input_json'], 'r') as f:
            train_input = json.load(f)
        
        # Update split CSV path
        train_input['split_csv'] = updated_split_csv
        
        with open(train_input_json, 'w') as f:
            json.dump(train_input, f, indent=2)
        
        # Run training
        cmd = [
            'python', 'train_DeCo_Diff.py',
            '--input_json', str(train_input_json),
            '--output_dir', str(self.run_dir / 'retrained_model'),
            '--batch_size', str(train_config.get('batch_size', 1)),
            '--learning_rate', str(train_config.get('learning_rate', 1e-4)),
            '--epochs', str(train_config.get('epochs', 100)),
            '--device', train_config.get('device', 'cuda')
        ]
        
        if 'additional_args' in train_config:
            for arg in train_config['additional_args']:
                cmd.extend(arg.split())
        
        print(f"Training command: {' '.join(cmd)}")
        
        # Note: Training might take a long time, so we just print the command
        # In practice, you might want to run this in a separate process
        print("Training command prepared. Run manually or implement background execution.")
        return True
    
    def run_complete_workflow(self):
        """Run the complete fine-tuning workflow"""
        print("Starting automated fine-tuning workflow...")
        
        steps = [
            ("Evaluation", self.run_evaluation),
            ("FP Review List Generation", self.generate_fp_review_list),
        ]
        
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            if not step_func():
                print(f"Workflow failed at: {step_name}")
                return False
        
        # Apply filtering
        if self.config['fp_filtering'].get('automated', False):
            print("\n--- Automated Filtering ---")
            review_file = self.apply_automated_filtering()
            if not review_file:
                print("Automated filtering failed")
                return False
        else:
            print("\n--- Manual Review Required ---")
            review_file = str(self.run_dir / 'evaluation_results' / 'fp_review_list.txt')
            print(f"Please manually edit: {review_file}")
            print("Then run: python automated_fine_tuning.py --config fine_tuning_config.json --continue")
            return True
        
        # Update dataset and retrain
        steps = [
            ("Dataset Update", lambda: self.update_training_dataset(review_file)),
            ("Model Retraining", lambda: self.retrain_model(str(self.run_dir / "updated_split.csv")))
        ]
        
        for step_name, step_func in steps:
            print(f"\n--- {step_name} ---")
            if not step_func():
                print(f"Workflow failed at: {step_name}")
                return False
        
        print("\n--- Fine-tuning workflow completed successfully ---")
        return True

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "work_dir": "fine_tuning_work",
        "evaluation": {
            "input_json": "input_json/eval_input.json",
            "model_path": "models/diffusion_pytorch_model.bin",
            "batch_size": 1,
            "device": "cuda",
            "additional_args": []
        },
        "fp_filtering": {
            "automated": True,
            "max_anomaly_threshold": 200,
            "min_anomaly_threshold": 50
        },
        "data": {
            "split_csv": "pcb_128/pcb-split.csv"
        },
        "retraining": {
            "input_json": "input_json/train_input.json",
            "batch_size": 1,
            "learning_rate": 1e-4,
            "epochs": 100,
            "device": "cuda",
            "additional_args": []
        }
    }
    
    with open('fine_tuning_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: fine_tuning_config.json")

def main():
    parser = argparse.ArgumentParser(description="Automated Fine-tuning Workflow for DeCo-Diff")
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--continue-workflow', action='store_true', help='Continue from manual review')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        return
    
    fine_tuning = AutomatedFineTuning(args.config)
    
    if args.continue_workflow:
        # Continue from manual review
        review_file = fine_tuning.run_dir / 'evaluation_results' / 'fp_review_list.txt'
        if not review_file.exists():
            print("Review file not found. Run the workflow first.")
            return
        
        updated_csv = fine_tuning.update_training_dataset(str(review_file))
        if updated_csv:
            fine_tuning.retrain_model(updated_csv)
    else:
        # Run complete workflow
        fine_tuning.run_complete_workflow()

if __name__ == "__main__":
    main() 