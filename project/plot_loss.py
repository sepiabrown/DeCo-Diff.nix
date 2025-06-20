#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from matplotlib.ticker import LogLocator, LogFormatter, MultipleLocator

def parse_log_file(log_file_path, with_validation=False):
    """
    Parse the training log file to extract MSE loss values and their corresponding epochs.
    
    Args:
        log_file_path (str): Path to the log file
        with_validation (bool): Whether to also parse validation losses
        
    Returns:
        tuple: (epochs, losses, val_epochs, val_losses) where epochs/losses are training data
               and val_epochs/val_losses are validation data (empty if with_validation=False)
    """
    epochs = []
    losses = []
    val_epochs = []
    val_losses = []
    
    # Regular expressions to match loss entries and epoch beginnings
    loss_pattern = r'step=(\d+)\) MSE Loss: ([\d\.]+),'
    epoch_pattern = r'Beginning epoch (\d+)\.\.\.'
    val_loss_pattern = r'\[VAL\] Validation loss for checkpoint .*/(\d+)\.pt: ([\d\.]+)'
    
    current_epoch = 0
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Check for epoch changes first
                epoch_match = re.search(epoch_pattern, line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # Check for training loss values and use the most recent epoch
                loss_match = re.search(loss_pattern, line)
                if loss_match:
                    loss = float(loss_match.group(2))
                    epochs.append(current_epoch)
                    losses.append(loss)
                
                # Check for validation loss values if requested
                if with_validation:
                    val_match = re.search(val_loss_pattern, line)
                    if val_match:
                        val_epoch = int(val_match.group(1))
                        val_loss = float(val_match.group(2))
                        val_epochs.append(val_epoch)
                        val_losses.append(val_loss)
    
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return [], [], [], []
    except Exception as e:
        print(f"Error reading log file: {e}")
        return [], [], [], []
    
    return epochs, losses, val_epochs, val_losses

def plot_loss_graph(epochs, losses, val_epochs=None, val_losses=None, output_path=None, title="Training Loss", with_validation=False):
    """
    Plot the training loss graph vs epochs.
    
    Args:
        epochs (list): Epoch numbers for training
        losses (list): MSE loss values for training
        val_epochs (list): Epoch numbers for validation (optional)
        val_losses (list): Validation loss values (optional)
        output_path (str, optional): Path to save the plot. If None, displays the plot.
        title (str): Title for the plot
        with_validation (bool): Whether validation data is included
    """
    if not epochs or not losses:
        print("No data to plot.")
        return
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, losses, 'b-', linewidth=1.5, alpha=0.7, label='Training MSE Loss', marker='o', markersize=3)
    
    # Plot validation loss if available
    if with_validation and val_epochs and val_losses:
        plt.plot(val_epochs, val_losses, 'r-', linewidth=1.5, alpha=0.7, label='Validation Loss', marker='s', markersize=4)
    
    # Add a smoothed trend line for training
    if len(epochs) > 10:
        # Calculate moving average for smoother trend
        window_size = min(20, len(losses) // 5)  # Smaller window for epochs since they're typically fewer
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            # Adjust epochs to match the smoothed losses length
            start_idx = window_size // 2
            end_idx = start_idx + len(smoothed_losses)
            smoothed_epochs = epochs[start_idx:end_idx]
            plt.plot(smoothed_epochs, smoothed_losses, 'g-', linewidth=2, label=f'Training Moving Average (window={window_size})')
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    
    # Configure y-axis to show both major and minor tick labels
    ax = plt.gca()
    # Set major locator for powers of 10
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    # Set minor locator for intermediate values
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    # Use LogFormatter for both major and minor ticks
    ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(LogFormatter(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
    
    # Configure x-axis to show ticks at 5000 intervals
    ax.xaxis.set_major_locator(MultipleLocator(5000))
    
    # Make minor tick labels smaller and less prominent
    for label in ax.yaxis.get_minorticklabels():
        label.set_fontsize(8)
        label.set_alpha(0.7)
    
    plt.grid(True, alpha=0.3, which='both')  # Show grid for both major and minor ticks
    plt.legend()
    
    # Add some statistics as text
    min_loss = min(losses)
    max_loss = max(losses)
    final_loss = losses[-1]
    initial_loss = losses[0]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100
    
    stats_text = f'Training:\n  Initial: {initial_loss:.4f}\n  Min: {min_loss:.4f}\n  Max: {max_loss:.4f}\n  Final: {final_loss:.4f}\n  Total Epochs: {max(epochs)}\n  Improvement: {improvement:.1f}%'
    
    # Add validation stats if available
    if with_validation and val_losses:
        val_min = min(val_losses)
        val_final = val_losses[-1]
        stats_text += f'\n\nValidation:\n  Min: {val_min:.4f}\n  Final: {val_final:.4f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Parse training log and plot loss graph vs epochs')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('-o', '--output', help='Output path for the plot (optional)')
    parser.add_argument('-t', '--title', default='Training Loss vs Epochs', help='Title for the plot')
    parser.add_argument('-v', '--with-validation', action='store_true', help='Plot validation loss')
    
    args = parser.parse_args()
    
    # Parse the log file
    print(f"Parsing log file: {args.log_file}")
    epochs, losses, val_epochs, val_losses = parse_log_file(args.log_file, args.with_validation)
    
    if not epochs:
        print("No loss data found in the log file.")
        return
    
    print(f"Found {len(epochs)} training steps with loss values")
    print(f"Training epoch range: {min(epochs)} - {max(epochs)}")
    print(f"Training loss range: {min(losses):.4f} - {max(losses):.4f}")
    
    if args.with_validation:
        if val_epochs and val_losses:
            print(f"Found {len(val_epochs)} validation checkpoints")
            print(f"Validation epoch range: {min(val_epochs)} - {max(val_epochs)}")
            print(f"Validation loss range: {min(val_losses):.4f} - {max(val_losses):.4f}")
        else:
            print("No validation loss data found in the log file.")
    
    # Plot the graph
    plot_loss_graph(epochs, losses, val_epochs, val_losses, args.output, args.title, args.with_validation)

if __name__ == "__main__":
    main() 