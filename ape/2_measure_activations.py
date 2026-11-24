import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm
import argparse

# Get arguments from user
def get_args():
    parser = argparse.ArgumentParser(description="Identify Language or Bias neurons in model.")
    parser.add_argument("--type", type=str, required=True, choices=["lang", "bias"], 
                        help="Type of neuron identification: language neurons (Use activations after each token) \
                        or bias neurons (Use activations after each complete prompt).")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model to be dissected.")
    parser.add_argument("--training_step", type=str, help="Model training step or checkpoint. For OLMO.")
    parser.add_argument("--input_dir", type=str, default="data_probe_final", help="Folder containing data/prompts.")
    parser.add_argument("--output_dir", type=str, default="activations_output", help="Folder for storing activation files.")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Access Token")
    return parser.parse_args()
        

def main():
    # Get arguments from user
    args = get_args()
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.token)
    if args.training_step:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            token=args.token, 
            revision=args.training_step,
            device_map="auto", 
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            token=args.token, 
            device_map="auto", 
            torch_dtype=torch.float16
        )

    if 'lama' in args.model_id: #llama
        # Determine number of layers
        num_layers = model.config.num_hidden_layers
        # Determine number of neurons in each layer
        intermediate_size = getattr(model.config, "intermediate_size", model.config.hidden_size * 4)
    elif 'wen' in args.model_id: #qwen4b
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
    elif 'lmo' in args.model_id: #olmo
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
    elif 'emma' in args.model_id: #gemma
        # Determine number of layers
        num_layers = model.config.text_config.num_hidden_layers
        # Determine number of neurons in each layer
        intermediate_size = model.config.text_config.intermediate_size
    elif 'loom' in args.model_id: #bloomz
        num_layers = getattr(model.config, "n_layer", model.config.num_hidden_layers)
        h_size = getattr(model.config, "hidden_size", model.config.hidden_size)
        intermediate_size = h_size * 4
    else:
        num_layers = model.config.num_hidden_layers
        intermediate_size = model.config.intermediate_size
        
    
    # Global counter matrix that the script will update
    # This counter will be re-zeroed for every file processed
    global over_zero
    # Create counter matrix with n_rows = num_layers, n_cols = no. of neurons per layer
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

    # Define hooks to be added to model layers based on neuron type to be identified
    def get_activation_hook(layer_idx, neuron_type, counter_tensor):
        def hook(module, input_, output):
            global over_zero
            # Llama-3 MLP act_fn output shape: [batch, seq_len, intermediate_size]

            # If identifying bias neurons, track activations after last token in prompt only
            # Reason: Bias is contained in entire prompt, not in each token; therefore, only
            #         last token activation is indicative of which neurons react to bias
            if neuron_type == "bias":
                # Only grab the last token activation
                # Shape becomes [batch, intermediate_size]
                act = output[:, -1, :]
                # Identify neurons with last token activations > 0
                # Sum across batch dimension
                # Sum across batch, ensure it's on the same device as our counter
                firing_counts = (act.float() > 0).sum(dim=0).to(torch.int32)
                #over_zero[layer_idx, :] += (act.float() > 0).sum(dim=0).to(torch.int32)

            # If identifying language neurons, track activations after every token in prompt
            # Reason: Each token is part of a language, so each token activation is indicative of 
            #         how model processes particular language
            else: 
                # Every token in the sequence counts
                # Sum across batch (0) and sequence (1) dimensions
                firing_counts = (output.float() > 0).sum(dim=(0, 1)).to(torch.int32)
                #over_zero[layer_idx, :] += (output.float() > 0).sum(dim=(0, 1)).to(torch.int32)
            
            # Perform the addition on the counter's device
            counter_tensor[layer_idx, :] += firing_counts.to(counter_tensor.device)
        return hook

    # Register Hooks for each layer
    print(f"Registering hooks for {args.type} neuron identification...")
    for i in range(num_layers):
        if 'lama' in args.model_id: #llama
            # Targeting the SiLU activation function in Llama-3 MLP
            target_layer = model.model.layers[i].mlp.act_fn
        elif '30B-A3B' in args.model_id: #qwen30b
            target_layer = model.model.layers[i].mlp.experts.act_fn
        elif 'wen' in args.model_id: #qwen4b
            target_layer = model.model.layers[i].mlp.act_fn
        elif 'lmo' in args.model_id: #olmo
            target_layer = model.model.layers[i].mlp.act_fn
        elif 'emma' in args.model_id: #gemma
            target_layer = model.model.language_model.layers[i].mlp.act_fn
        elif 'loom' in args.model_id: #bloomz
            target_layer = model.transformer.h[i].mlp.gelu_impl
        else:
            target_layer = model.model.layers[i].mlp.act_fn
        target_layer.register_forward_hook(get_activation_hook(i, args.type, over_zero))

    # Process Files
    # Make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create list of prompt files from input directory
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.txt')]

    # Iterate through every prompt file
    for file_name in files:
        file_path = os.path.join(args.input_dir, file_name)
        print(f"\nProcessing: {file_name}")
        
        # Reset counters for each file
        over_zero.zero_()
        instance_count = 0

        # Clean each prompt
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        # Iterate through each prompt in file
        for text in tqdm(prompts, desc=f"Inference ({args.type})"):
            # Tokenize the prompts into token id's
            inputs = tokenizer(text, return_tensors="pt").to('cuda')
            
            # Track n or instance_count calculation
            # This is needed because we need to be able to divide the number of activations
            # by the total number of instances. This will allow for percentage scores that can
            # be comparable across different prompt conditions with differing number of instances
            if args.type == "lang":
                instance_count += inputs['input_ids'].shape[1]
            else:
                instance_count += 1 # n = number of prompts for bias

            # Run model on input and count activations
            with torch.no_grad():
                model(**inputs)

        # Save Results
        save_data = {
            "n": instance_count,
            "over_zero": over_zero.to('cpu'),
            "type": args.type
        }
        
        out_name = f"activations.{file_name.replace('.txt', '.pt')}"
        save_path = os.path.join(args.output_dir, out_name)
        torch.save(save_data, save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()