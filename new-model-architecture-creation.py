import math
import torch
import torch.nn as nn
from transformers import LlamaConfig
from quantization_utils import activation_quant, quantize_tensor_1_58bit
from llama_model import LlamaModel, BitLinear
from model_utils import count_parameters, verify_parameter_count
from tqdm import tqdm
import os
import time
import re
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description="Create a blank 1.58-bit quantised LLaMA model")
parser.add_argument("--params", required=True, help="Total parameter count, e.g. 750M or 1.5B")
parser.add_argument("--output_dir", default=None, help="Directory to save the model")
parser.add_argument("--e", action="store_true", help="Enable experimental quantization")
args = parser.parse_args()

def calculate_model_dimensions(num_params, recursion_depth=0, max_recursion_depth=5):
    min_hidden_size = 256
    max_hidden_size = 8192
    min_num_layers = 2
    max_num_layers = 128
    target_ratio = 0.995  # Increased target_ratio for more precise configurations

    def params_from_dimensions(hidden_size, num_layers):
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            num_attention_heads=max(1, hidden_size // 64),
            num_hidden_layers=num_layers,
            vocab_size=32000,
        )
        model = LlamaModel(config, experiment=args.e)
        return sum(p.numel() for p in model.parameters())

    best_config = None
    best_num_params = float('inf')

    # Define the range of hidden sizes and number of layers based on the target number of parameters
    if num_params < 1_000_000_000:  # For models smaller than 1B parameters
        max_hidden_size = 2048
        max_num_layers = 24
    elif num_params < 10_000_000_000:  # For models between 1B and 10B parameters
        max_hidden_size = 4096
        max_num_layers = 48
    else:  # For models larger than 10B parameters
        max_hidden_size = 8192
        max_num_layers = 128

    # Estimate initial values for hidden_size and num_layers based on num_params
    estimated_hidden_size = min(max_hidden_size, int(num_params ** 0.25))
    estimated_num_layers = min(max_num_layers, int(num_params / (estimated_hidden_size ** 2)))

    for num_layers in range(estimated_num_layers, min_num_layers - 1, -1):
        lower_hidden_size = min_hidden_size
        upper_hidden_size = max_hidden_size

        while lower_hidden_size <= upper_hidden_size:
            hidden_size = (lower_hidden_size + upper_hidden_size) // 2
            hidden_size = (hidden_size // 64) * 64  # Ensure hidden_size is divisible by 64
            total_params = params_from_dimensions(hidden_size, num_layers)

            if total_params < num_params:
                lower_hidden_size = hidden_size + 64
            else:
                upper_hidden_size = hidden_size - 64

            # Check if the current configuration is closer to the target than the best one so far
            if abs(total_params - num_params) < abs(best_num_params - num_params):
                best_config = LlamaConfig(
                    hidden_size=hidden_size,
                    intermediate_size=4 * hidden_size,
                    num_attention_heads=max(1, hidden_size // 64),
                    num_hidden_layers=num_layers,
                    vocab_size=32000,
                )
                best_num_params = total_params

            # Break the loop if a suitable configuration is found within the target ratio
            if abs(total_params - num_params) / num_params < (1 - target_ratio):
                break

    # Fallback strategy: If no suitable configuration is found, adjust the criteria and search again
    if best_config is None and recursion_depth < max_recursion_depth:
        print(f"No suitable configuration found. Adjusting the search criteria (recursion depth: {recursion_depth + 1}).")
        return calculate_model_dimensions(num_params * 1.1, recursion_depth + 1)  # Increase the target parameter count by 10% and search again
    elif best_config is None:
        raise ValueError(f"No suitable configuration found within the specified ranges after {max_recursion_depth} recursions.")

    return best_config

def parse_num_params(input_str):
    input_str = input_str.upper()
    if 'M' in input_str:
        num_params = float(input_str.replace('M', '')) * 1_000_000
    elif 'B' in input_str:
        num_params = float(input_str.replace('B', '')) * 1_000_000_000
    else:
        num_params = int(input_str)
    return int(num_params)

# Parse parameter count from command line
num_params = parse_num_params(args.params)
assert 300_000_000 <= num_params <= 200_000_000_000, "Number of parameters must be between 300M and 200B"

# Calculate the model dimensions based on the desired number of parameters
config = calculate_model_dimensions(num_params)
model = LlamaModel(config, experiment=args.e)
actual_params = count_parameters(model)
verify_parameter_count(actual_params, num_params)

# Move the model to the appropriate device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Ternarize the model weights using 1.58‑bit quantisation
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        q, scale = quantize_tensor_1_58bit(module.weight)
        module.weight = nn.Parameter(q.float() * scale)

# Quantize activations to 8 bits
def quantize_activations(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, BitLinear):
            orig_forward = module.forward

            def forward_wrapper(x, *args, orig_forward=orig_forward, **kwargs):
                out = orig_forward(x, *args, **kwargs)
                return activation_quant(out)

            module.forward = forward_wrapper
    return model

# Modify the calculate_model_size function
def calculate_model_size(model):
    def get_quant_size(tensor):
        # Bytes required when quantised to 1.58 bits plus shape metadata
        return math.ceil(tensor.numel() * 1.58 / 8) + tensor.dim() * 4

    model_size = 0
    for name, param in model.named_parameters():
        if args.e and ('embed_tokens' in name or 'norm' in name):
            model_size += get_quant_size(param)
        elif 'weight' in name and isinstance(param, torch.Tensor) and 'lm_head' not in name:
            model_size += get_quant_size(param)
        else:
            model_size += param.element_size() * param.numel()

    return model_size

model = quantize_activations(model)

print(f"Hidden size: {config.hidden_size}")
print(f"Number of attention heads: {config.num_attention_heads}")

model_size = calculate_model_size(model)
print(f"Model size: {model_size / (1024 * 1024):.2f} MB")

print("Saving ternarized and quantized model...")
save_dir = args.output_dir or f"llama_{num_params}_ternary_quantized_optimized"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)

print("Ternarized and quantized model saved at:", save_dir)
