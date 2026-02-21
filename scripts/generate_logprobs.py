"""
Generate per-model log-probabilities for all tokens in the vocabulary.

For each model listed in config/models.yaml, computes next-token
log-probabilities (log_softmax of last-token logits) for each prompt
in config/prompts.yaml and saves results to a CSV file in data/logprobs/.

Output format per model:
    token_id, token_name, token_decoded, best_ever_one, ..., terrible_time_please
    (3 metadata columns + 54 log-probability columns)

Usage:
    python scripts/generate_logprobs.py               # all models
    python scripts/generate_logprobs.py --model google/gemma-2-2b-it  # one model
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS matmuls

import argparse
import yaml
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).parent
REPO_ROOT = SCRIPT_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "data" / "logprobs"

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def safe_name(model_name):
    """Convert HuggingFace model ID to filesystem-safe name."""
    return model_name.replace("/", "--")


def tokenize_prompt(tokenizer, prompt, model_type):
    """Tokenize a prompt appropriately for the model type."""
    if model_type == "instruction-tuned":
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
            add_generation_prompt=True,
        )
    elif model_type == "pretrained":
        return tokenizer(prompt, return_tensors="pt")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def process_model(model_info, prompts, output_dir):
    """Generate and save log-probabilities for a single model."""
    name = model_info["name"]
    out_path = output_dir / f"{safe_name(name)}.csv"

    if out_path.exists():
        print(f"Skipping {name} — {out_path.name} already exists")
        return

    print(f"\nProcessing: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype_map[model_info["dtype"]],
        device_map="auto",
    ).eval()

    prompt_logprobs = {}

    for prompt_key, prompt_text in tqdm(prompts.items(), desc="Prompts", leave=False):
        inputs = tokenize_prompt(tokenizer, prompt_text, model_info["type"])
        device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)

        prompt_logprobs[prompt_key] = log_probs.cpu().numpy().tolist()

    # Build DataFrame with token metadata + prompt columns
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    token_names, token_ids = zip(*vocab_sorted)
    token_decoded = tokenizer.batch_decode(
        [[tid] for tid in token_ids], skip_special_tokens=False
    )

    df = pd.DataFrame({
        "token_id": token_ids,
        "token_name": token_names,
        "token_decoded": token_decoded,
    })
    for prompt_key, lp_values in prompt_logprobs.items():
        df[prompt_key] = [lp_values[tid] for tid in token_ids]

    df.to_csv(out_path, index=False)
    print(f"Saved {out_path.name}")

    del model, tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-model log-probability CSVs"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Process only this model (HuggingFace ID). Default: all models.",
    )
    args = parser.parse_args()

    # Determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    with open(CONFIG_DIR / "models.yaml", "r") as f:
        models = yaml.safe_load(f)
    with open(CONFIG_DIR / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found in config/models.yaml")
            return

    for model_info in models:
        process_model(model_info, prompts, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
