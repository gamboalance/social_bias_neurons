import torch
import os
import argparse
from glob import glob

# Get arguments from user
def get_args():
    parser = argparse.ArgumentParser(description="Aggregate multiple activation files (pertaining to diff \
                                                  conditions) into one.")
    parser.add_argument("paths", nargs="+", help="Directory path containing activation files \
                                                  OR list of .pt activation files")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path for the consolidated output (e.g., total_activations.pt)")
    return parser.parse_args()

def compile_activations(activations_list, save_name):
    if not activations_list:
        print("No valid .pt files found.")
        return

    print(f"Compiling {len(activations_list)} files...")
    
    # Initialize with the first file, particular the variables for 
    # n (number of entries) and the activations counter matrix
    # map_location='cpu' is critical for avoiding GPU memory spikes
    first_dict = torch.load(activations_list[0], map_location='cpu')
    entries = first_dict['n']
    activations = first_dict['over_zero'].float()

    # Accumulate n and activations_counts across different files
    for file_path in activations_list[1:]:
        try:
            dic = torch.load(file_path, map_location='cpu')
            entries += dic['n']
            activations += dic['over_zero'].float()
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

    # Save the final dictionary
    final_dict = {
        'n': entries, 
        'over_zero': activations
    }

    torch.save(final_dict, save_name)
    print(f"Success! Total 'n': {entries}")
    print(f"Saved to: {save_name}")

if __name__ == "__main__":
    # Get arguments from user
    args = get_args()

    # Determine if input is a directory or a list of files
    files = []
    for path in args.paths:
        if os.path.isdir(path):
            # If path is a folder, find all .pt files inside
            found = glob(os.path.join(path, "*.pt"))
            # Add to list of files
            files.extend(found)
        elif os.path.isfile(path):
            # If path is a specific file, add to list of files
            # Do this for every input path listed in the arguments
            if path.endswith(".pt"):
                pfiles.append(path)
        else:
            print(f"Warning: Path not found or invalid: {path}")

    # Remove duplicates if any
    files = sorted(list(set(files)))

    # Compile activations into one output file
    compile_activations(files, args.output_file)