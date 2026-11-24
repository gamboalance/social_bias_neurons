import torch
import torch.nn.functional as F
import os
import argparse
from glob import glob

# Get arguments from user
def get_args():
    parser = argparse.ArgumentParser(description="Identify bias and/or lang neurons using activation probability entropy.")
    # Input can be a folder or a list of files
    parser.add_argument("paths", nargs="+", help="Directory path OR list of .pt activation files. Each file must pertain to \
                                                  a bias condition, a language condition, or a language x bias condition. ")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final identified neurons (.pt).")
    
    # Optional Customization Parameters
    parser.add_argument("--filter_rate", type=float, default=0.95, 
                        help="Threshold for discarding neurons that are inactive across all categories. Only the top neurons \
                              within the given percentage will be retained, and 1.00 minus the given will be discarded.")
    parser.add_argument("--top_rate", type=float, default=0.01, 
                        help="Percentage of neurons with lowest entropy to keep (default: 0.01). Lower entropy = more \
                              neurons specialization.")
    parser.add_argument("--activation_bar_ratio", type=float, default=0.95, 
                        help="Threshold to determine which neurons specialize in the given bias/language conditions. \
                              Only neurons with activation probabilities above the given ratio will be retained.")
    
    return parser.parse_args()

def identify_neurons(n, over_zero, num_layers, intermediate_size, cond_num, neurons_output, args):
    """
    Processes raw counts into activation probabilities and identifies neurons
    that are highly selective for specific categories based on entropy.

    n: tensor with length = number of conditions; each element pertains to number of instances per bias/language condition
    over_zero: tensor with shape [num_layers, num_neurons_per_layer, cond_num]; list of counter matrices tracking number of
               over-zero activations per neuron for each condition
    num_layers: number of model layers
    intermediate_size: number of neurons in each layer
    cond_num: number of language/bias conditions
    neuron_output: file name that will contain list of specialized neurons per condition, per layer
    """
    # Convert counts to probabilities/percentages by dividing over-zero activations
    # by total number of instances per condition
    # over_zero shape: [layer, intermediate, cond_num]
    # n shape: [cond_num]
    activation_probs = over_zero / n 
    
    # calculate entropy using Shannoon entropy formula
    normed_activation_probs = activation_probs / (activation_probs.sum(dim=-1, keepdim=True) + 1e-12)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    
    # Standard Entropy: -sum(p * log(p))
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), torch.tensor(0.0))
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    
    # Remove inactive neurons
    # flatten all activation probabilities into 1-dimensional list
    flattened_probs = activation_probs.flatten()
    # Find the threshold value at the filter_rate percentile (default: 95th perc)
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * args.filter_rate)).values.item()
    print(f"Global Probability Filter Threshold: {top_prob_value:.6f}")
    
    # Check if a neuron fires above threshold value for any of the lang/bias conditions
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    # Dismiss neuron if it doesn't fire above threshold value for any condition
    # Set to infinity so they are not selected in 'smallest' search below
    entropy[top_position == 0] = torch.inf  

    # Select neurons with lowest entropy (low entropy = high specialization)
    # flatten all entropy values into 1-dimensional list
    flattened_entropy = entropy.flatten()
    # Find the threshold value at the top_rate (default: top 1%)
    top_entropy_count = round(len(flattened_entropy) * args.top_rate)
    
    # Return flattend_entropy indices of entropy values within top 1%
    # We want the SMALLEST entropy (most selective), so largest=False
    _, index = flattened_entropy.topk(top_entropy_count, largest=False)

    # Based on flattened_entropy index, recover low-entropy neurons' layer number (row_index)
    # and neuron index (col_index). Layer number determined by dividing flattened_entropy 
    # index with length of original entropy matrix. Neuron index determined by getting the
    # remainder of this division process.
    row_index = torch.div(index, entropy.size(1), rounding_mode='floor')
    col_index = index % entropy.size(1)
    # Retrieve activation probabilities of neurons with low entropy
    selected_probs = activation_probs[row_index, col_index] # Shape: [N_selected, cond_num]

    # Assign low-entropy neurons to their specialized bias/language conditions based on activation bar
    # Determine threshold value for neuron activation based on activation_bar_ratio. Neuron will be considered
    # specialized on a condition if its activation exceeds the threshold value for this condition.
    selected_probs_transposed = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * args.activation_bar_ratio)).values.item()

    # Get condition indices and layer/neuron indices of specialized neurons
    cond_ids, selected_indices = torch.where(selected_probs_transposed > activation_bar)

    # Structuring Output: Nested list of layers per condition
    # Final structure is a nested list. 
    # First outer layer: list elements correspond to conditions.
    # Next layer: list elements correspond to model layers.
    # Innermost layer: list elements correspond to specialized neurons per layer.
    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    
    # Split the indices by lang/bias condition count to group them
    cond_counts = torch.bincount(cond_ids, minlength=cond_num).tolist()
    
    for cond_idx, split_indices in enumerate(selected_indices.split(cond_counts)):
        # Extract layer and neuron coordinates
        cond_coordinates = [tuple(row.tolist()) for row in merged_index[split_indices]]
        cond_coordinates.sort()
        
        # Group by layer index
        layer_buckets = [[] for _ in range(num_layers)]
        for l_idx, h_idx in cond_coordinates:
            layer_buckets[l_idx].append(h_idx)
            
        # Convert lists to tensors
        layer_buckets = [torch.tensor(h_list).long() for h_list in layer_buckets]
        final_indice.append(layer_buckets)

    # Save final list
    torch.save(final_indice, neurons_output)
    print(f"Successfully identified neurons for {len(final_indice)} categories.")
    print(f"Results saved to {neurons_output}")

def main():
    # get arguments from user
    args = get_args()
    
    # Handle flexible input paths
    # initialize list to contain activation files per language/bias condition
    all_files = []
    for path in args.paths:
        # if given input path is a folder, add all activation files within
        if os.path.isdir(path):
            all_files.extend(sorted(glob(os.path.join(path, "*.pt"))))
        # if given input path is a list of files, add each file to list
        elif os.path.isfile(path):
            all_files.append(path)

    # if there are no valid files, return an error
    if not all_files:
        print("Error: No activation files found.")
        return

    print(f"Loading {len(all_files)} files for analysis...")
    print(all_files)
    # initialize two lists
    # n_list: list of instances per condition (e.g., number of prompts or number of tokens)
    # over_zero_list: list of counter matrices (for over-zero activations) per condition
    n_list, over_zero_list = [], []

    # iterate through each activation file
    for f in all_files:
        # load and append n (instance count) and over_zero (activation counter matrix) to lists
        print(str(f))
        data = torch.load(f, map_location='cpu')
        n_list.append(data['n'])
        over_zero_list.append(data['over_zero'])

    # convert lists to tensors
    n_tensor = torch.tensor(n_list)
    over_zero_tensor = torch.stack(over_zero_list, dim=-1)
    # use shape of over_zero_tensor to determine number of model layers, 
    # number of neurons in each layer, and number of conditions
    num_layers, intermediate_size, cond_num = over_zero_tensor.size()

    # identify bias/language neurons given above inputs
    identify_neurons(
        n_tensor, over_zero_tensor, 
        num_layers, intermediate_size, cond_num, 
        args.output, args
    )

if __name__ == "__main__":
    main()