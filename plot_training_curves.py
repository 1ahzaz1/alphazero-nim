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
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: loss_curve_[history]_[alpha].png)")
    parser.add_argument("--compare", action="store_true",
                        help="Plot both history and no-history models on same graph")
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

def generate_loss_plot(history_enabled, history_disabled=None, alpha=1.0, max_iter=500, output=None):
    """Generate a plot of training losses"""
    plt.figure(figsize=(10, 6))
    
    # Plot history-enabled model if provided
    if history_enabled:
        iterations, policy_losses, value_losses = history_enabled
        plt.plot(iterations, policy_losses, 'b-', alpha=0.3, label='Policy Loss (History)')
        plt.plot(iterations, value_losses, 'r-', alpha=0.3, label='Value Loss (History)')
        
        # Add smoothed curves
        smoothed_policy = smooth_curve(policy_losses)
        smoothed_value = smooth_curve(value_losses)
        plt.plot(iterations, smoothed_policy, 'b-', linewidth=2, label='Policy Loss (History, smoothed)')
        plt.plot(iterations, smoothed_value, 'r-', linewidth=2, label='Value Loss (History, smoothed)')
        
        # Calculate average of last 50 iterations
        avg_policy = np.mean(policy_losses[-50:])
        avg_value = np.mean(value_losses[-50:])
        plt.axhline(y=avg_policy, color='b', linestyle='--', 
                   label=f'Avg Policy Loss: {avg_policy:.4f}')
        plt.axhline(y=avg_value, color='r', linestyle='--',
                   label=f'Avg Value Loss: {avg_value:.4f}')
    
    # Plot history-disabled model if provided (for comparison)
    if history_disabled:
        iterations_no, policy_losses_no, value_losses_no = history_disabled
        plt.plot(iterations_no, policy_losses_no, 'g-', alpha=0.3, label='Policy Loss (No History)')
        plt.plot(iterations_no, value_losses_no, 'm-', alpha=0.3, label='Value Loss (No History)')
        
        # Add smoothed curves
        smoothed_policy_no = smooth_curve(policy_losses_no)
        smoothed_value_no = smooth_curve(value_losses_no)
        plt.plot(iterations_no, smoothed_policy_no, 'g-', linewidth=2, 
                label='Policy Loss (No History, smoothed)')
        plt.plot(iterations_no, smoothed_value_no, 'm-', linewidth=2,
                label='Value Loss (No History, smoothed)')
        
        # Calculate average of last 50 iterations
        avg_policy_no = np.mean(policy_losses_no[-50:])
        avg_value_no = np.mean(value_losses_no[-50:])
        plt.axhline(y=avg_policy_no, color='g', linestyle='--',
                   label=f'Avg Policy Loss (No Hist): {avg_policy_no:.4f}')
        plt.axhline(y=avg_value_no, color='m', linestyle='--',
                   label=f'Avg Value Loss (No Hist): {avg_value_no:.4f}')
    
    # Set up plot details
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Losses (alpha={alpha})')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Set reasonable y-axis limits
    plt.ylim(bottom=0)
    
    # Save or show the plot
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output}")
    else:
        if history_disabled:
            default_filename = f"loss_curve_comparison_alpha_{alpha}.png"
        else:
            history_tag = "history" if history_enabled else "no_history"
            default_filename = f"loss_curve_{history_tag}_alpha_{alpha}.png"
        plt.savefig(default_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {default_filename}")
    
    plt.close()

def find_log_file(history, alpha):
    """Find the log file for a given configuration"""
    # Try common log file locations and patterns
    potential_paths = [
        f"training_log_{5}_{bool(history)}_{alpha}.txt",  # Standard format with board size 5
        f"training_{bool(history)}_{alpha}.log",
        f"output_{bool(history)}_{alpha}.log",
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
    
    # Generate the plot
    if args.compare:
        generate_loss_plot(history_data, no_history_data, args.alpha, args.checkpoint, args.output)
    elif args.history == 1 and history_data:
        generate_loss_plot(history_data, None, args.alpha, args.checkpoint, args.output)
    elif args.history == 0 and no_history_data:
        generate_loss_plot(None, no_history_data, args.alpha, args.checkpoint, args.output)
    else:
        print("No data available to plot.")

if __name__ == "__main__":
    main() 