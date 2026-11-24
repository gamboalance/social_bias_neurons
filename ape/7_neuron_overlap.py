import argparse
import os
import torch

def load_neuron_data(file_paths):
    """Loads .pt files and extracts unique (layer, neuron) coordinate sets."""
    all_coordinate_sets = []
    all_keys = []

    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Error: The file '{path}' does not exist.")
        
        # Load the PyTorch file safely to CPU
        data_dict = torch.load(path, map_location=torch.device('cpu'))

        # compile dict keys
        for key, layer_dict in data_dict.items():
            all_keys.append(key)
            
            # Construct a unique (layer,neuron) coordinate set for this specific file
            coordinate_set = set()
            for layer, neurons in layer_dict.items():
                for neuron in neurons:
                    coordinate_set.add((int(layer), int(neuron)))
            
            all_coordinate_sets.append(coordinate_set)
            
    return all_keys, all_coordinate_sets

def compute_multi_overlap(coordinate_sets, method="union"):
    """
    Computes neuron overlap proportion.
    
    Methods:
      - 'union': Intersection divided by total unique elements across all sets.
      - 'top_k': Intersection divided by total elements in the first dictionary.
    """
    if not coordinate_sets:
        return 0.0
        
    # Calculate global intersection (neurons shared across ALL files)
    global_intersection = set.intersection(*coordinate_sets)
    
    # Determine denominator based on user choice
    if method == "top_k":
        denominator_val = len(coordinate_sets[0])
    else:  # Default to 'union'
        denominator_val = len(set.union(*coordinate_sets))
    
    if denominator_val == 0:
        return 0.0
        
    return len(global_intersection) / denominator_val

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the global unique neuron overlap proportion among multiple concept dictionaries."
    )
    parser.add_argument(
        "-i", "--inputs", 
        nargs="+", 
        required=True, 
        help="Space-separated paths to the input .pt files (e.g., file1.pt file2.pt file3.pt)"
    )
    parser.add_argument(
        "-d", "--denominator",
        choices=["union", "top_k"],
        default="union",
        help="The denominator strategy to use. 'union' calculates Jaccard similarity. "
             "'top_k' normalizes the intersection by the total size of the first file passed."
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of files being compared"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. Parse and extract coordinate profiles
        keys, coordinate_sets = load_neuron_data(args.inputs)
        
        # 2. Compute the multi-set overlap value
        overlap_val = compute_multi_overlap(coordinate_sets, method=args.denominator)
        
        # 3. Print results to standard output
        print("=" * 100)
        print("NEURON OVERLAP ANALYSIS " + args.description)
        print("=" * 100)
        print(f"Analyzed Keys:       {', '.join(keys)}")
        print(f"Denominator Method:  {args.denominator}")
        print(f"Computed Overlap:    {overlap_val:.6f}")
        print("=" * 100)
        
    except Exception as e:
        print(f"Analysis Failed: {str(e)}")

if __name__ == "__main__":
    main()