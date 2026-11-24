import os
import random
import argparse

def sample_files(input_folder, output_folder, sample_size=200):
    #Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Filter for .txt files in input_directory
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # If there are no txt files, exit
    if not files:
        print(f"No .txt files found in {input_folder}")
        return

    # Iterate through every txt file
    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        
        with open(input_path, 'r', encoding="utf-8") as f:
            # Use a list comprehension to strip and filter empty lines
            # Each list element will be a prompt
            prompts = [line.strip() for line in f if line.strip()]

        # If prompt list contains more data than given sample size, sample randomly
        if len(prompts) > sample_size:
            selected_prompts = random.sample(prompts, sample_size)
        # If prompt list has less elements than given sample size, retain original list
        else:
            selected_prompts = prompts

        # Construct output path
        output_path = os.path.join(output_folder, file_name)

        # Write sampled data into output path
        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in selected_prompts:
                f.write(f"{prompt}\n")

        print(f"Saved {len(selected_prompts)} prompts to: {output_path}")

if __name__ == "__main__":
    # Setup command line arguments for flexibility
    parser = argparse.ArgumentParser(description="Sample random lines from text files.")
    parser.add_argument("--input", type=str, default="data_probe_orig", help="Source folder")
    parser.add_argument("--output", type=str, default="data_probe_final", help="Destination folder")
    parser.add_argument("--size", type=int, default=200, help="Number of lines to sample")

    args = parser.parse_args()
    
    sample_files(args.input, args.output, args.size)