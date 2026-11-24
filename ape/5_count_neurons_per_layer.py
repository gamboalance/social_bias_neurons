import torch
import argparse
import json
import os
from pprint import pprint

# Get input arguments
def get_args():
    parser = argparse.ArgumentParser(description="Count or list bias/language neurons per layer.")
    parser.add_argument("--input", type=str, required=True, help="Path to the identified neurons .pt file.")
    parser.add_argument("--format", type=str, choices=["stats", "list"], default="stats",
                        help="Output format: 'stats' (counts/percentages of neurons per layer)  \
                                              or 'list' (neuron index lists).")
    parser.add_argument("--mapping", type=str, help="Optional: JSON path or string mapping index to category name.")
    parser.add_argument("--output_file", type=str, help="Path to save the resulting dictionary (.pt).")
    return parser.parse_args()

def get_label(cat_idx, cat_mapping):
    """Helper to resolve category index to a category label. Returns cat label corresponding to
       index if mapping provided. Returns index if mapping not provided."""
    if cat_mapping:
        val = cat_mapping.get(str(cat_idx)) or cat_mapping.get(cat_idx)
        if val:
            return val
    return str(cat_idx)

def analyze_neurons(neurons_file, format_type, cat_mapping=None):
    # Load the nested list of neurons: [ [layer0_t, layer1_t, ...], [layer0_t, layer1_t, ...] ]
    # Outermost index: each element corrsponds to a category
    # Next-level index: each element corresponds to a layer
    # Innermost index: each element corresponds to a neuron
    neurons = torch.load(neurons_file, map_location='cpu')
    # Get number of categories and layers based on shape of neurons file
    num_categories = len(neurons)
    try:
        num_layers = len(neurons[0])
    except:
        first_key = list(neurons.keys())[0]
        num_layers = len(neurons[first_key])

    # Initialize output dictionary
    output_dict = {}

    # Iterate through each category in the list
    for cat_idx in range(num_categories):
        # Get category label based on cat_idx
        label = get_label(cat_idx, cat_mapping)
        # Get total number of specialized neurons for category
        try:
            total_in_cat = sum(len(layer_t) for layer_t in neurons[cat_idx])
        except:
            key = list(neurons.keys())[cat_idx]
            total_in_cat = sum([len(neurons[key][layer_t]) for layer_t in neurons[key]])

        # Initialize dictionary that will contain data for current category
        layer_data = {}
        # Iterate through every layer for current category
        for l_idx in range(num_layers):
            # Get list of neuron indices per layer
            try:
                neuron_indices = neurons[cat_idx][l_idx].tolist()
            except:
                neuron_indices = neurons[key][l_idx]

            # If output format is stats, count number of neurons per layer and
            # that layer's percentage for the category's neurons. Add to dictionary
            # for the category
            if format_type == "stats":
                count = len(neuron_indices)
                perc = round((count / total_in_cat * 100), 2) if total_in_cat > 0 else 0.0
                layer_data[l_idx] = (count, perc)
            # If output format is list, add list of neuron indices to dictionary for the
            # category
            else:
                layer_data[l_idx] = neuron_indices

        # Add category dictionary to output dictionary
        output_dict[label] = layer_data

    return output_dict

def main():
    # Get inputs
    args = get_args()
    
    # Resolve mapping from given mapping input
    mapping = None
    if args.mapping:
        # If mapping provided is a json file, load into dictionary
        if os.path.exists(args.mapping):
            with open(args.mapping, 'r') as f:
                mapping = json.load(f)
        else:
            try:
                # If mapping provided is a json string, load into dictionary
                mapping = json.loads(args.mapping)
            except json.JSONDecodeError:
                print("Error: Mapping is not a valid JSON string or file path.")
                return

    # Count or list neurons per categroy and layer
    data = analyze_neurons(args.input, args.format, mapping)

    # Output handling
    if args.output_file:
        torch.save(data, args.output_file)
        print(f"Dictionary saved to {args.output_file}")
    else:
        # Pretty print with formatting suited for nested dicts
        pprint(data, compact=False, sort_dicts=False)

if __name__ == "__main__":
    main()