import torch
import argparse

def main():
    # Get user input
    parser = argparse.ArgumentParser(description="Filter bias neurons based on computed IGG scores.")
    parser.add_argument("--input", type=str, required=True, help="Data file path. Must be pt file containing avg IGG scores per neuron.")
    parser.add_argument("--mode", type=str, default="percent_max", choices=["percent_max", "percent_neurons","top_k"],
                        help="Threshold logic. percent_max: All neurons with IGG scores > max_score * k. \
                                               percent_neurons: All neurons with top k% IGG scores. \
                                               top_k: k neurons with top IGG scores." )
    parser.add_argument("--k", type=float, default=0.1, help="k to be used in filtering neurons.")
    parser.add_argument("--format", type=str, choices=["stats", "list"], default="list",
                        help="Output format: 'stats' (counts/percentages of neurons per layer)  \
                                              or 'list' (neuron index lists).")
    parser.add_argument("--dict_key", type=str, default="bias", help="lang and/or dim to describe nature of bias")
    parser.add_argument("--neurons_output", type=str, default="bias_neurons.pt", help="Output filename for indices of neurons that meet threshold.")

    args = parser.parse_args()

    # Load IGG scores per neuron
    layer_avg_attrs = torch.load(args.input,map_location=torch.device('cpu'))
    num_layers = len(layer_avg_attrs)

    # Identify Bias Neurons via Global Threshold
    bias_neurons_by_layer = []
    # Stack all IGG scores
    global_stack = torch.stack(layer_avg_attrs)

    # Percent_max mode: Retain neurons whose IGG score is above global_max * k
    if args.mode == "percent_max":
        # Find the maximum absolute attribution found across everything
        global_max_val = torch.max(torch.abs(global_stack))
        # Find threshold value by multiplying max with k
        threshold = args.k * global_max_val
        print(f"Threshold (Max-based): {threshold:.6f}")
        # List indices of neurons who meet the threshold value
        for layer_idx in range(num_layers):
            indices = torch.where(torch.abs(layer_avg_attrs[layer_idx]) >= threshold)[0]
            bias_neurons_by_layer.append(indices.cpu())

    # Percent_neuron mode: Return neurons whose IGG score is in top n%
    elif args.mode == "percent_neurons":
        # Find the value at the nth percentile
        flat_stack = torch.abs(global_stack).flatten()
        k = int(len(flat_stack) * args.k)
        threshold_val = torch.topk(flat_stack, k).values[-1]
        print(f"Threshold (Top {args.k*100}%): {threshold_val:.6f}")
        # List indices of neurons who meet the threshold value
        for layer_idx in range(num_layers):
            indices = torch.where(torch.abs(layer_avg_attrs[layer_idx]) >= threshold_val)[0]
            bias_neurons_by_layer.append(indices.cpu())

    # Top_k mode: Retain exactly k neurons with the highest absolute IGG scores across all layers
    elif args.mode == "top_k":
        # Flatten everything to find top k across the whole model
        flat_stack = torch.abs(global_stack).flatten()
        
        # Ensure k does not exceed total number of neurons
        k = min(int(args.k), len(flat_stack))
        
        # Get the global indices of the top k neurons
        # values are the attribution scores, indices are the flattened positions
        topk_res = torch.topk(flat_stack, k)
        global_indices = topk_res.indices
        threshold_val = topk_res.values[-1]
        
        print(f"Threshold (Top {k} neurons): {threshold_val:.6f}")

        # Map global indices back to (layer, neuron) coordinates
        # num_neurons_per_layer is the size of the last dimension
        num_neurons_per_layer = global_stack.shape[1]
        
        for layer_idx in range(num_layers):
            # Find which of the top_k global indices belong to this layer
            start_idx = layer_idx * num_neurons_per_layer
            end_idx = (layer_idx + 1) * num_neurons_per_layer
            
            # Filter global indices that fall within this layer's range
            layer_mask = (global_indices >= start_idx) & (global_indices < end_idx)
            layer_indices = global_indices[layer_mask] - start_idx
            
            # Sort for consistency and append
            bias_neurons_by_layer.append(torch.sort(layer_indices).values.cpu())

    # Convert to dictionary
    neurons_dict = {}

    # Get total number of specialized neurons for category
    total_bias_n = sum(len(layer_t) for layer_t in bias_neurons_by_layer)

    # Initialize dictionary that will contain neurons-per-layer data
    layer_data = {}
    
    # Iterate through every layer 
    for l_idx in range(num_layers):
        # Get list of neuron indices per layer
        neuron_indices = bias_neurons_by_layer[l_idx].tolist()

        # If output format is stats, count number of neurons per layer and
        # that layer's percentage for the total bias neurons. Add to dictionary
        if args.format == "stats":
            count = len(neuron_indices)
            perc = round((count / total_bias_n * 100), 2) if total_in_cat > 0 else 0.0
            layer_data[l_idx] = (count, perc)
        # If output format is list, add list of neuron indices to dictionary for the
        # category
        else:
            layer_data[l_idx] = neuron_indices

    # Add category dictionary to output dictionary
    neurons_dict[args.dict_key] = layer_data

    # Final Save
    torch.save(neurons_dict, args.neurons_output)
    print(f"Done. Saved neuron indices to {args.neurons_output}")

if __name__ == "__main__":
    main()
