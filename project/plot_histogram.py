#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def collect_data_by_status(input_dir, quantity='anomaly_max'):
    """Collect specified quantity values grouped by status combinations"""
    status_data = {
        'FP': [],
        'FP_TN': [],
        'FN': [],
        'FN_TP': []
    }
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for patch in data.get('patch_analysis', []):
                    status = patch.get('status')
                    value = patch.get(quantity)
                    
                    if value is not None:
                        if status == 'FP':
                            status_data['FP'].append(value)
                            status_data['FP_TN'].append(value)
                        elif status == 'TN':
                            status_data['FP_TN'].append(value)
                        elif status == 'FN':
                            status_data['FN'].append(value)
                            status_data['FN_TP'].append(value)
                        elif status == 'TP':
                            status_data['FN_TP'].append(value)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
    return status_data

def get_optimal_bins(values, method='auto'):
    """Calculate optimal number of bins for histogram"""
    if len(values) < 2:
        return 1
    
    # Convert to numpy array
    values = np.array(values)
    
    # Remove any infinite or NaN values
    values = values[np.isfinite(values)]
    
    if len(values) < 2:
        return 1
    
    # Try different binning methods
    if method == 'auto':
        # Use numpy's auto method
        bins = np.histogram_bin_edges(values, bins='auto')
        return len(bins) - 1
    elif method == 'sturges':
        # Sturges' formula
        return int(np.ceil(np.log2(len(values)) + 1))
    elif method == 'sqrt':
        # Square root rule
        return int(np.ceil(np.sqrt(len(values))))
    elif method == 'rice':
        # Rice rule
        return int(np.ceil(2 * len(values) ** (1/3)))
    elif method == 'fd':
        # Freedman-Diaconis rule
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(values) ** (1/3))
        if bin_width > 0:
            return int(np.ceil((values.max() - values.min()) / bin_width))
        else:
            return min(30, len(values))
    else:
        # Default to auto
        bins = np.histogram_bin_edges(values, bins='auto')
        return len(bins) - 1

def plot_histogram(values, output_path, title, quantity='anomaly_max', color='skyblue', bin_method='auto'):
    if not values:
        print(f"No {quantity} values found for {title}.")
        return
    
    # Calculate optimal number of bins
    optimal_bins = get_optimal_bins(values, bin_method)
    
    # Ensure we have a reasonable number of bins
    if optimal_bins < 5:
        optimal_bins = min(5, len(values))
    elif optimal_bins > 50:
        optimal_bins = 50
    
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=optimal_bins, color=color, edgecolor='black', alpha=0.8)
    plt.xlabel(quantity, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f"Histogram of {quantity} for {title} (bins: {optimal_bins})", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Histogram saved to: {output_path} (using {optimal_bins} bins)")

def plot_all_histograms(status_data, output_dir, quantity='anomaly_max', bin_method='auto'):
    """Plot histograms for all status combinations"""
    colors = {
        'FP': 'skyblue',
        'FP_TN': 'lightgreen',
        'FN': 'lightcoral',
        'FN_TP': 'gold'
    }
    
    titles = {
        'FP': 'status FP only',
        'FP_TN': 'status FP + TN',
        'FN': 'status FN only',
        'FN_TP': 'status FN + TP'
    }
    
    for status_key, values in status_data.items():
        if values:  # Only plot if there are values
            output_path = os.path.join(output_dir, f'{quantity}_histogram_{status_key}.png')
            plot_histogram(values, output_path, titles[status_key], quantity, colors[status_key], bin_method)

def main():
    parser = argparse.ArgumentParser(description="Draw histograms of specified quantity for different status combinations from evaluation JSON files.")
    parser.add_argument('--input-dir', required=True, help='Directory containing evaluation JSON files')
    parser.add_argument('--output-dir', help='Directory to save the histogram plots (defaults to input-dir)')
    parser.add_argument('--quantity', default='anomaly_max', help='Quantity to plot histogram for (default: anomaly_max)')
    parser.add_argument('--bin-method', default='auto', choices=['auto', 'sturges', 'sqrt', 'rice', 'fd'], 
                       help='Method for automatic bin selection (default: auto)')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    quantity = args.quantity
    bin_method = args.bin_method
    os.makedirs(output_dir, exist_ok=True)

    status_data = collect_data_by_status(input_dir, quantity)
    plot_all_histograms(status_data, output_dir, quantity, bin_method)

if __name__ == "__main__":
    main() 