import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load

def preprocess_data(data, tokenizer):
    input_ids = tokenizer.encode(data, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model_state_dict_path = os.path.join(model_path, "model.safetensors")
    with open(model_state_dict_path, "rb") as f:
        data = f.read()
    model_state_dict = load(data)

    adjusted_model_state_dict = {
        key.replace("model.", ""): value for key, value in model_state_dict.items()
    }
    model = AutoModelForCausalLM.from_config(config)

    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    expected = vocab_size * hidden_size
    actual = adjusted_model_state_dict["lm_head.weight"].numel()

    if actual != expected:
        loaded = adjusted_model_state_dict["lm_head.weight"]
        new = torch.zeros(vocab_size, hidden_size, dtype=loaded.dtype, device=loaded.device)
        n = min(actual, expected)
        new.view(-1)[:n] = loaded.view(-1)[:n]
        adjusted_model_state_dict["lm_head.weight"] = new
    else:
        adjusted_model_state_dict["lm_head.weight"] = adjusted_model_state_dict["lm_head.weight"].view(vocab_size, hidden_size)

    model.load_state_dict(adjusted_model_state_dict, strict=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

def run(model_path: str, text: str, max_length: int = 100) -> str:
    model, tokenizer = load_model(model_path)
    input_ids, attention_mask = preprocess_data(text, tokenizer)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    return generated_text


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a model and generate text using transformers")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max_length", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    run(args.model_path, args.text, max_length=args.max_length)


if __name__ == "__main__":
    main()
