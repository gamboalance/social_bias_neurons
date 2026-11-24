import torch
import pandas as pd
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# IGManager class that will handle hooks to be used in Calculating
# integrated gap gradients
class IGManager:
    def __init__(self, model_id, training_step, token=None):
        """Initialize the model and tokenizer with generalized settings."""
        self.model_id = model_id
        print(f"Loading model: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if training_step:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                token=token, 
                revision=training_step,
                device_map="auto", 
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=token,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        # increase memory savings
        self.model.gradient_checkpointing_enable()
        # intermediate activation layer
        self.intermediate_act = None
        # two modes: normal and attribution
        # normal mode: get activation weights for given prompt-answer combi
        # attribution mode: replace activation layer with scaled activation weights
        #                   to assess change in answer prob
        self.mode = "normal"
        # target_batch: batch of scaled activation weights to be tested
        self.target_batch = None

    def hook_fn(self, module, input, output):
        """Hook to capture clean activations or inject interpolated batches."""
        if self.mode == "normal":
            # output shape: [batch, sequence, hidden_size]
            # Capture last token activation for prompt
            self.intermediate_act = output[:, -1, :].detach()
        elif self.mode == "attribution":
            # Inject the batch of scaled activation weights during attribution calculation
            output[:, -1, :] = self.target_batch
        return output

    def compute_ig_attribution(self, prompt, answer1_str, answer2_str, layer_idx, num_steps=160, chunk_size=2):
        """Calculates Integrated Gradients attribution for a specific layer."""
        # Attempt to find the activation function layer dynamically
        if 'lama' in self.model_id: #llama
            # Targeting the SiLU activation function in Llama-3 MLP
            target_layer = self.model.model.layers[layer_idx].mlp.act_fn
        elif 'wen' in self.model_id: #qwen
            target_layer = self.model.model.layers[layer_idx].mlp.act_fn
        elif 'lmo' in self.model_id: #olmo
            target_layer = self.model.model.layers[layer_idx].mlp.act_fn
        elif 'emma' in self.model_id: #gemma
            target_layer = self.model.model.language_model.layers[layer_idx].mlp.act_fn
        elif 'loom' in self.model_id: #bloomz
            target_layer = self.model.transformer.h[layer_idx].mlp.gelu_impl
        else:
            target_layer = self.model.model.layers[layer_idx].mlp.act_fn

        # try:
        #     target_layer = self.model.model.layers[layer_idx].mlp.act_fn
        # except AttributeError:
        #     # Fallback for models like Gemma which may have different nested naming
        #     target_layer = self.model.language_model.model.layers[layer_idx].mlp.act_fn

        # Register hook on target layer
        handle = target_layer.register_forward_hook(self.hook_fn)

        # Convert prompt to token ids
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Baseline/Normal Pass to get baseline activation weights using the
        # given prompt
        self.mode = "normal"
        self.model(**inputs)

        # Retrieve baseline activation weights from IG manager
        original_val = self.intermediate_act
        # Determine step size between 0 and baseline weights using num_steps
        step_size = original_val / num_steps
        # initiate tensors to keep track of gradients computed per chunk
        total_grads = torch.zeros_like(original_val, dtype=torch.float32)
        # Switch to attribution calculation mode, 
        self.mode = "attribution"

        # Chunked Attribution Loop to Save VRAM
        for i in range(0, num_steps, chunk_size):
            # Determine actual chunk size (either the size that user set, or remaining
            # entries in last chunk)
            actual_chunk = min(chunk_size, num_steps - i)

            #print('Processing chunk: ' + str(range(i, i + actual_chunk)))
            
            # Prepare Interpolated Batch of scaled activation weights
            # We cat scaled versions of the original activation from 0 to 1
            scaled_weights = torch.cat([step_size * j for j in range(i, i + actual_chunk)], dim=0)
            scaled_weights = scaled_weights.to(torch.bfloat16).requires_grad_(True)

            # Injecting the model with the scaled attribution weights
            self.target_batch = scaled_weights
            # Generate input_ids n times, where n is chunk size
            batch_inputs = {k: v.repeat(actual_chunk, 1) for k, v in inputs.items()}

            # Attribution Pass to evaluate changes in probability given scaled
            # activation weights
            outputs = self.model(**batch_inputs)
            
            # Get token IDs for the target answers (ensure space for prefix tokenization)
            id1 = self.tokenizer.encode(answer1_str, add_special_tokens=False)[0]
            id2 = self.tokenizer.encode(answer2_str, add_special_tokens=False)[0]

            # Get next token probabilities for the attribution passes on the prompts
            probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
            # prob_gap: Probability(Biased Choice) - Probability(Unbiased Choice)
            prob_gap = probs[:, id1] - probs[:, id2]
            
            # calculate gradient/slope/rate-of-change of prob_gap with respect to
            # scaled activation weights
            grads = torch.autograd.grad(prob_gap.sum(), scaled_weights)[0]
            # Add gradients to tracker
            total_grads += grads.sum(dim=0, keepdim=True).to(torch.float32)

            # Cleanup immediately
            del outputs, probs, prob_gap, grads, scaled_weights
            torch.cuda.empty_cache()

        # Riemann approximation: multiply sum of gradients by step_size in order
        # to scale down the sum to singular unit of change while preserving
        # gradient differences across neurons
        attribution = (total_grads * step_size.to(torch.float32)).squeeze()
        #print("Total attribution computed.")
        
        handle.remove()
        return attribution


        # scaled_weights = torch.cat([step_size * i for i in range(num_steps)], dim=0)
        # scaled_weights = scaled_weights.to(torch.bfloat16).requires_grad_(True)

        # # Switch to attribution calculation mode, injecting the model with the
        # # scaled attribution weights
        # self.mode = "attribution"
        # self.target_batch = scaled_weights
        # Generate input_ids n times, where n is number of steps
        # batch_inputs = {k: v.repeat(num_steps, 1) for k, v in inputs.items()}

        # Attribution Pass to evaluate changes in probability given scaled
        # activation weights
        # outputs = self.model(**batch_inputs)

        # Get token IDs for the target answers (ensure space for prefix tokenization)
        # id1 = self.tokenizer.encode(answer1_str, add_special_tokens=False)[0]
        # id2 = self.tokenizer.encode(answer2_str, add_special_tokens=False)[0]

        # Get next token probabilities for the attribution passes on the prompts
        # probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        # # prob_gap: Probability(Biased Choice) - Probability(Unbiased Choice)
        # prob_gap = probs[:, id1] - probs[:, id2]

        # calculate gradient/slope/rate-of-change of prob_gap with respect to
        # scaled activation weights
        # grads = torch.autograd.grad(prob_gap.sum(), scaled_weights)[0]
        # Riemann approximation: multiply sum of gradients by step_size in order
        # to scale down the sum to singular unit of change while preserving
        # gradient differences across neurons
        # attribution = (grads.sum(dim=0) * step_size.squeeze()).to(torch.float32)

        # handle.remove()
        # return attribution

def load_data(file_path, prompt_col, biased_col, unbiased_col):
    """Utility to load dataset and resolve column names/indices."""
    # Load csv or json file
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide .csv or .json.")

    # Get names of prompt, biased, and unbiased columns
    cols = df.columns
    p_col = cols[int(prompt_col)] if prompt_col.isdigit() else prompt_col
    b_col = cols[int(biased_col)] if biased_col.isdigit() else biased_col
    u_col = cols[int(unbiased_col)] if unbiased_col.isdigit() else unbiased_col

    return df, p_col, b_col, u_col

def main():
    # Get user input
    parser = argparse.ArgumentParser(description="Identify bias neurons via Integrated Gap Gradients.")
    parser.add_argument("--input", type=str, required=True, help="Data file path. Must be csv or json.")
    parser.add_argument("--prompt_col", type=str, required=True, help="Prompt column name or index.")
    parser.add_argument("--biased_col", type=str, required=True, help="Biased answer column name or index.")
    parser.add_argument("--unbiased_col", type=str, required=True, help="Unbiased answer column name or index.")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID.")
    parser.add_argument("--training_step", type=str, default=None, help="Model training step or checkpoint. For OLMO.")
    parser.add_argument("--layers", type=int, help="No. of layers to process (auto-inferred if omitted).")
    parser.add_argument("--steps", type=int, default=160, help="Number of steps for scaling from 0 to orig activation scores in IGG calculation.")
    parser.add_argument("--chunk_size", type=int, default=2, help="Number of steps to process at a time, to prevent RAM crashes.")
    parser.add_argument("--token", type=str, help="HuggingFace token.")
    parser.add_argument("--neuron_scores_output", type=str, default="bias_neuron_scores.pt", help="Output filename for IGG scores per neuron.")

    args = parser.parse_args()

    # Initialize IG manager and number of layers
    manager = IGManager(args.model_id, args.training_step, token=args.token)
    #num_layers = args.layers if args.layers else manager.model.config.num_hidden_layers
    if 'lama' in args.model_id: #llama
        # Determine number of layers
        num_layers = args.layers if args.layers else manager.model.config.num_hidden_layers
        # Determine number of neurons in each layer
        intermediate_size = getattr(manager.model.config, "intermediate_size", manager.model.config.hidden_size * 4)
    elif 'wen' in args.model_id: #qwen
        num_layers = args.layers if args.layers else manager.model.config.num_hidden_layers
        intermediate_size = manager.model.config.intermediate_size
    elif 'lmo' in args.model_id: #olmo
        num_layers = args.layers if args.layers else manager.model.config.num_hidden_layers
        intermediate_size = manager.model.config.intermediate_size
    elif 'emma' in args.model_id: #gemma
        # Determine number of layers
        num_layers = args.layers if args.layers else manager.model.config.text_config.num_hidden_layers
        intermediate_size = manager.model.config.text_config.intermediate_size
    elif 'loom' in args.model_id: #bloomz
        num_layers = args.layers if args.layers else getattr(manager.model.config, "n_layer", manager.model.config.num_hidden_layers)
        h_size = getattr(manager.model.config, "hidden_size", manager.model.config.hidden_size)
        intermediate_size = h_size * 4
    else:
        num_layers = args.layers if args.layers else manager.model.config.num_hidden_layers
        intermediate_size = manager.model.config.intermediate_size

    # Load dataset and relevant columns
    df, p_col, b_col, u_col = load_data(args.input, args.prompt_col, args.biased_col, args.unbiased_col)
    # Load nested list that will store IGG scores for each layer and its neurons
    all_layer_data = [[] for _ in range(num_layers)]

    # Process Prompts
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating IGG"):
        prompt = row[p_col]
        ans1 = " " + str(row[b_col]) # Leading space for correct tokenization
        ans2 = " " + str(row[u_col])

        # Iterate through each layer
        for layer_idx in range(num_layers):
            #print("Working on layer " + str(layer_idx) + " of " + str(num_layers))
            try:
                # Compute IGG scores for each of the layer's neurons using the given prompt
                attr = manager.compute_ig_attribution(prompt, ans1, ans2, layer_idx, num_steps=args.steps, chunk_size=args.chunk_size)
                # Append the IGG scores for the layer and prompt to all_layer data
                all_layer_data[layer_idx].append(attr)
                print("Layer " + str(layer_idx) + " attribution added.")
            except Exception as e:
                print(f"Error at Layer {layer_idx}: {e}")

    # Aggregation
    # Initialize list for average IGG scores per layer
    layer_avg_attrs = []
    # Iterate through each layer
    for layer_idx in range(num_layers):
        if all_layer_data[layer_idx]:
            # Average the gaps across all prompts for this specific layer
            layer_avg_attrs.append(torch.stack(all_layer_data[layer_idx]).mean(dim=0))
        else:
            # If invalid data, generate zero matrix
            layer_avg_attrs.append(torch.zeros(intermediate_size))
            #layer_avg_attrs.append(torch.zeros(manager.model.config.intermediate_size))

    # Final Save
    torch.save(layer_avg_attrs, args.neuron_scores_output)
    print(f"Done. Saved neuron scores to {args.neuron_scores_output}")

if __name__ == "__main__":
    main()
