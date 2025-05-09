import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss curves from logs")
    parser.add_argument("--history", type=int, choices=[0, 1], default=1,
                        help="Plot curves for history-enabled (1) or history-disabled (0) model")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value used in training")
    parser.add_argument("--checkpoint", type=int, default=500,
                        help="Maximum checkpoint to plot up to")
    parser.add_argument("--output_prefix", type=str, default="loss_comparison",
                        help="Prefix for output filenames")
    parser.add_argument("--compare", action="store_true",
                        help="Plot both history and no-history models on same graph")
    parser.add_argument("--combined", action="store_true",
                        help="Create a single plot with all losses instead of separate plots")
    return parser.parse_args()

def extract_losses_from_log(logfile):
    """Extract policy and value losses from a training log file"""
    policy_losses = []
    value_losses = []
    iterations = []
    
    with open(logfile, 'r') as f:
        for line in f:
            # Look for lines with loss information
            match = re.search(r'Iter (\d+)/\d+ \| Policy Loss: ([\d\.]+) \| Value Loss: ([\d\.]+)', line)
            if match:
                iter_num = int(match.group(1))
                policy_loss = float(match.group(2))
                value_loss = float(match.group(3))
                
                iterations.append(iter_num)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
    
    return iterations, policy_losses, value_losses

def smooth_curve(points, factor=0.8):
    """Apply exponential smoothing to a curve"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def generate_comparison_plots(history_data, no_history_data, alpha=1.0, max_iter=500, output_prefix="loss_comparison", combined=False):
    """Generate policy and value loss comparison plots"""
    if history_data and no_history_data:
        hist_iterations, hist_policy_losses, hist_value_losses = history_data
        no_hist_iterations, no_hist_policy_losses, no_hist_value_losses = no_history_data
        
        # Smooth the curves
        hist_policy_smooth = smooth_curve(hist_policy_losses)
        hist_value_smooth = smooth_curve(hist_value_losses)
        no_hist_policy_smooth = smooth_curve(no_hist_policy_losses)
        no_hist_value_smooth = smooth_curve(no_hist_value_losses)
        
        # Calculate averages of last 50 iterations
        hist_policy_avg = np.mean(hist_policy_losses[-50:])
        hist_value_avg = np.mean(hist_value_losses[-50:])
        no_hist_policy_avg = np.mean(no_hist_policy_losses[-50:])
        no_hist_value_avg = np.mean(no_hist_value_losses[-50:])
        
        if combined:
            # Create a single plot with all losses
            plt.figure(figsize=(12, 8))
            
            # Policy losses
            plt.plot(hist_iterations, hist_policy_smooth, 'b-', linewidth=2.5, label='Policy Loss (History)')
            plt.plot(no_hist_iterations, no_hist_policy_smooth, 'g-', linewidth=2.5, label='Policy Loss (No History)')
            
            # Value losses
            plt.plot(hist_iterations, hist_value_smooth, 'r-', linewidth=2.5, label='Value Loss (History)')
            plt.plot(no_hist_iterations, no_hist_value_smooth, 'm-', linewidth=2.5, label='Value Loss (No History)')
            
            # Averages
            plt.axhline(y=hist_policy_avg, color='b', linestyle='--', 
                       label=f'Avg Policy (History): {hist_policy_avg:.4f}')
            plt.axhline(y=hist_value_avg, color='r', linestyle='--',
                       label=f'Avg Value (History): {hist_value_avg:.4f}')
            plt.axhline(y=no_hist_policy_avg, color='g', linestyle='--',
                       label=f'Avg Policy (No Hist): {no_hist_policy_avg:.4f}')
            plt.axhline(y=no_hist_value_avg, color='m', linestyle='--',
                       label=f'Avg Value (No Hist): {no_hist_value_avg:.4f}')
            
            # Labels and styling
            plt.xlabel('Training Iteration', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Losses - History vs No History (α={alpha})', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=10)
            plt.ylim(bottom=0)
            
            # Save the plot
            filename = f"{output_prefix}_all_losses_alpha_{alpha}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Combined plot saved to {filename}")
            plt.close()
        
        else:  # Create separate plots for policy and value losses
            # Policy Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(hist_iterations, hist_policy_losses, 'b-', alpha=0.3)
            plt.plot(no_hist_iterations, no_hist_policy_losses, 'g-', alpha=0.3)
            plt.plot(hist_iterations, hist_policy_smooth, 'b-', linewidth=2.5, label='History Enabled')
            plt.plot(no_hist_iterations, no_hist_policy_smooth, 'g-', linewidth=2.5, label='History Disabled')
            
            plt.axhline(y=hist_policy_avg, color='b', linestyle='--', 
                       label=f'History Avg: {hist_policy_avg:.4f}')
            plt.axhline(y=no_hist_policy_avg, color='g', linestyle='--',
                       label=f'No History Avg: {no_hist_policy_avg:.4f}')
            
            # Calculate and display the difference
            diff = no_hist_policy_avg - hist_policy_avg
            plt.axhspan(hist_policy_avg, no_hist_policy_avg, color='lightblue', alpha=0.2)
            
            # Labels and styling
            plt.xlabel('Training Iteration', fontsize=12)
            plt.ylabel('Policy Loss', fontsize=12)
            plt.title(f'Policy Loss - History vs No History (α={alpha})\nDifference: {diff:.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=10)
            plt.ylim(bottom=0)
            
            # Save the plot
            filename = f"{output_prefix}_policy_loss_alpha_{alpha}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Policy loss comparison saved to {filename}")
            plt.close()
            
            # Value Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(hist_iterations, hist_value_losses, 'r-', alpha=0.3)
            plt.plot(no_hist_iterations, no_hist_value_losses, 'm-', alpha=0.3)
            plt.plot(hist_iterations, hist_value_smooth, 'r-', linewidth=2.5, label='History Enabled')
            plt.plot(no_hist_iterations, no_hist_value_smooth, 'm-', linewidth=2.5, label='History Disabled')
            
            plt.axhline(y=hist_value_avg, color='r', linestyle='--', 
                       label=f'History Avg: {hist_value_avg:.4f}')
            plt.axhline(y=no_hist_value_avg, color='m', linestyle='--',
                       label=f'No History Avg: {no_hist_value_avg:.4f}')
            
            # Calculate and display the difference
            v_diff = no_hist_value_avg - hist_value_avg
            plt.axhspan(hist_value_avg, no_hist_value_avg, color='mistyrose', alpha=0.2)
            
            # Labels and styling
            plt.xlabel('Training Iteration', fontsize=12)
            plt.ylabel('Value Loss', fontsize=12)
            plt.title(f'Value Loss - History vs No History (α={alpha})\nDifference: {v_diff:.4f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right', fontsize=10)
            plt.ylim(bottom=0)
            
            # Save the plot
            filename = f"{output_prefix}_value_loss_alpha_{alpha}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Value loss comparison saved to {filename}")
            plt.close()
    else:
        print("Missing data for either history or no-history model.")

def generate_single_model_plot(model_data, is_history, alpha=1.0, max_iter=500, output_prefix="loss"):
    """Generate a plot for a single model (either history or no-history)"""
    if not model_data:
        print("No data available for the model.")
        return
    
    iterations, policy_losses, value_losses = model_data
    
    # Smooth the curves
    policy_smooth = smooth_curve(policy_losses)
    value_smooth = smooth_curve(value_losses)
    
    # Calculate averages
    policy_avg = np.mean(policy_losses[-50:])
    value_avg = np.mean(value_losses[-50:])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot original and smoothed data
    plt.plot(iterations, policy_losses, 'b-', alpha=0.3, label='Policy Loss (raw)')
    plt.plot(iterations, value_losses, 'r-', alpha=0.3, label='Value Loss (raw)')
    plt.plot(iterations, policy_smooth, 'b-', linewidth=2, label='Policy Loss (smoothed)')
    plt.plot(iterations, value_smooth, 'r-', linewidth=2, label='Value Loss (smoothed)')
    
    # Add average lines
    plt.axhline(y=policy_avg, color='b', linestyle='--', 
               label=f'Avg Policy Loss: {policy_avg:.4f}')
    plt.axhline(y=value_avg, color='r', linestyle='--',
               label=f'Avg Value Loss: {value_avg:.4f}')
    
    # Labels and styling
    model_type = "History Enabled" if is_history else "History Disabled" 
    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Losses - {model_type} (α={alpha})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylim(bottom=0)
    
    # Save the plot
    history_tag = "history" if is_history else "no_history"
    filename = f"{output_prefix}_{history_tag}_alpha_{alpha}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

def find_log_file(history, alpha):
    """Find the log file for a given configuration"""
    # Try common log file locations and patterns
    potential_paths = [
        f"training_log_{5}_{bool(history)}_{alpha}.txt",  # Standard format with board size 5
        f"training_{bool(history)}_{alpha}.log",
        f"output_{bool(history)}_{alpha}.log",
        f"history_logs.txt" if history else "no_history_logs.txt",
        "training.log"
    ]
    
    # Check if any of these files exist
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # If no log file is found, prompt the user
    print(f"No log file found for history={bool(history)}, alpha={alpha}")
    user_path = input("Please enter the path to the log file: ")
    if os.path.exists(user_path):
        return user_path
    else:
        raise FileNotFoundError(f"Log file not found: {user_path}")

def main():
    args = parse_args()
    
    # Find log file for history-enabled model
    if args.compare or args.history == 1:
        try:
            history_log = find_log_file(1, args.alpha)
            history_data = extract_losses_from_log(history_log)
            print(f"Found history-enabled log: {history_log}")
        except FileNotFoundError:
            print("Could not find history-enabled log. Please run with --history 0 or provide a file.")
            history_data = None
    else:
        history_data = None
    
    # Find log file for history-disabled model
    if args.compare or args.history == 0:
        try:
            no_history_log = find_log_file(0, args.alpha)
            no_history_data = extract_losses_from_log(no_history_log)
            print(f"Found history-disabled log: {no_history_log}")
        except FileNotFoundError:
            print("Could not find history-disabled log. Please run with --history 1 or provide a file.")
            no_history_data = None
    else:
        no_history_data = None
    
    # Generate the appropriate plots
    if args.compare and history_data and no_history_data:
        generate_comparison_plots(history_data, no_history_data, args.alpha, args.checkpoint, args.output_prefix, args.combined)
    elif args.history == 1 and history_data:
        generate_single_model_plot(history_data, True, args.alpha, args.checkpoint, args.output_prefix)
    elif args.history == 0 and no_history_data:
        generate_single_model_plot(no_history_data, False, args.alpha, args.checkpoint, args.output_prefix)
    else:
        print("No data available to plot.")

if __name__ == "__main__":
    main() 