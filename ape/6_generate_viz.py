import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.interpolate import make_interp_spline

def get_args():
    parser = argparse.ArgumentParser(description="Generate smoothed line charts of neuron distributions.")
    parser.add_argument("--files", nargs="+", required=True, 
                        help="List of .pt files containing the neuron_counts dictionaries.")
    parser.add_argument("--categories", nargs="+", required=True, 
                        help="Categories to plot (e.g., en_so tl_so). Must exist in the provided files.")
    parser.add_argument("--title", type=str, default="Bias Neuron Distribution", help="Chart title.")
    parser.add_argument("--out", type=str, default="distribution_chart.png", help="Output filename.")
    return parser.parse_args()

def smooth_data(x, y):
    """Generates a cubic spline for smooth visualization."""
    x_new = np.linspace(x.min(), x.max(), 500)
    spline = make_interp_spline(x, y, k=3)
    y_new = spline(x_new)
    # Clip to 0 since percentages cannot be negative
    return x_new, np.clip(y_new, 0, None)

def main():
    args = get_args()

    # 1. Load and Merge Data
    # We allow multiple files; we merge them into one master dict
    master_data = {}
    for f in args.files:
        try:
            data = torch.load(f, map_location='cpu')
            master_data.update(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # 2. Determine Number of Layers Automatically
    # Peek at the first category to see how many layers it has
    first_cat = list(master_data.keys())[0]
    num_layers = len(master_data[first_cat])
    # Note: dict keys are layer_idx (0-indexed), we map them to 1-indexed for the chart
    layers_x = np.arange(1, num_layers + 1)
    print(f"Detected {num_layers} layers.")

    # 3. Plotting Setup
    plt.figure(figsize=(12, 7))
    colors = plt.cm.get_cmap('tab10', len(args.categories))

    for i, cat in enumerate(args.categories):
        if cat not in master_data:
            print(f"Warning: Category '{cat}' not found in provided files. Skipping.")
            continue

        # Extract percentages: master_data[cat][layer_idx] returns (count, percentage)
        # Note: dict keys are layer_idx (0-indexed), we map them to 1-indexed for the chart
        y_values = [master_data[cat][l][1] for l in range(num_layers)]
        print(y_values)
        y_values = np.array(y_values)

        # Smooth the data
        x_smooth, y_smooth = smooth_data(layers_x, y_values)

        # Plot the smooth line and the original scatter points
        line_color = colors(i)
        plt.plot(x_smooth, y_smooth, color=line_color, linewidth=2, label=cat)
        plt.scatter(layers_x, y_values, color=line_color, s=20, zorder=5, alpha=0.7)

    # 4. Styling (Matches the Llama 3 Chart Style)
    plt.title(args.title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    
    plt.xticks(layers_x)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(frameon=True, loc='upper left')

    # Clean borders
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Chart saved to {args.out}")
    plt.show()

if __name__ == "__main__":
    main()