#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def collect_anomaly_max(input_dir):
    anomaly_max_values = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for patch in data.get('patch_analysis', []):
                    if patch.get('status') == 'FP':
                        anomaly_max = patch.get('anomaly_max')
                        if anomaly_max is not None:
                            anomaly_max_values.append(anomaly_max)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
    return anomaly_max_values

def plot_histogram(anomaly_max_values, output_path):
    if not anomaly_max_values:
        print("No anomaly_max values found for status 'FP'.")
        return
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_max_values, bins=30, color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('anomaly_max', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title("Histogram of anomaly_max for status 'FP'", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Draw histogram of anomaly_max for status 'FP' from evaluation JSON files.")
    parser.add_argument('--input-dir', required=True, help='Directory containing evaluation JSON files')
    parser.add_argument('--output-dir', help='Directory to save the histogram plot (defaults to input-dir)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'anomaly_max_histogram.png')

    anomaly_max_values = collect_anomaly_max(input_dir)
    plot_histogram(anomaly_max_values, output_path)

if __name__ == "__main__":
    main() 