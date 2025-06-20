import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from llama_model import LlamaModel
from quantization_utils import activation_norm_quant, gemm_lowbit
import time
import psutil

ACT2FN["llamamlp"] = lambda x: x * torch.sigmoid(x)

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.weight_scale = None

    def forward(self, x):
        w = self.weight  # a 1.58-bit weight tensor with shape [d, k]
        w_scale = self.weight_scale  # a full-precision weight scale tensor with shape [1]
        x_quant, x_scale = activation_norm_quant(x)
        y = gemm_lowbit(x_quant, w) / w_scale / x_scale
        return y


def load_quantized_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaModel.load_pretrained(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)

    start_time = time.time()
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    end_time = time.time()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generation_time = end_time - start_time
    num_tokens = len(output_ids[0])
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return generated_text

def evaluate_metrics(model, tokenizer, prompts, max_length=100):
    perplexities = []
    runtimes = []
    memory_usages = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.lm_head.weight.device)
        attention_mask = torch.ones_like(input_ids)

        start_time = time.time()
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
        )
        end_time = time.time()

        runtime = end_time - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 ** 2  # Convert bytes to MB

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        perplexity = calculate_perplexity(model, tokenizer, generated_text)

        perplexities.append(perplexity)
        runtimes.append(runtime)
        memory_usages.append(memory_usage)

    avg_perplexity = sum(perplexities) / len(perplexities)
    avg_runtime = sum(runtimes) / len(runtimes)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    print("Evaluation Metrics:")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    print(f"Average Runtime: {avg_runtime:.2f} seconds")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")

def calculate_perplexity(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.lm_head.weight.device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=torch.ones_like(input_ids))

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        perplexity = torch.exp(loss)

    return perplexity.item()


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with a quantised model")
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--prompt", action="append", help="Prompt text (may be repeated)")
    parser.add_argument("--prompt_file", help="File containing prompts, one per line")
    parser.add_argument("--max_length", type=int, default=100, help="Generation length")
    parser.add_argument("--eval", action="store_true", help="Evaluate metrics instead of generating")
    return parser


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.extend([l.strip() for l in f if l.strip()])
    return prompts


def run(model_path: str, prompts: list[str], max_length: int = 100, evaluate: bool = False) -> list[str] | None:
    model, tokenizer = load_quantized_model(model_path)
    if evaluate:
        evaluate_metrics(model, tokenizer, prompts, max_length=max_length)
        return None
    outputs = []
    for p in prompts:
        outputs.append(generate_text(model, tokenizer, p, max_length=max_length))
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    prompts = _load_prompts(args)
    if not prompts:
        parser.error("No prompts provided")
    run(args.model_path, prompts, max_length=args.max_length, evaluate=args.eval)


if __name__ == "__main__":
    main()
