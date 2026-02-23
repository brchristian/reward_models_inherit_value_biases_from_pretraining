"""
Download pre-computed log-probability CSVs from HuggingFace Hub.

Fetches per-model CSVs into data/logprobs/ so that figure and table
scripts can run without a GPU.  Skips files that already exist locally.

Usage:
    python scripts/download_data.py                              # all models
    python scripts/download_data.py --model google/gemma-2-2b    # one model
"""

import argparse
import yaml
from pathlib import Path
from fetch_logprobs import fetch_missing_logprobs

SCRIPT_ROOT = Path(__file__).parent
REPO_ROOT = SCRIPT_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"
OUTPUT_DIR = REPO_ROOT / "data" / "logprobs"


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-computed logprob CSVs from HuggingFace Hub"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Download only this model (HuggingFace ID). Default: all models.",
    )
    args = parser.parse_args()

    with open(CONFIG_DIR / "models.yaml", "r") as f:
        models = yaml.safe_load(f)

    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"Model '{args.model}' not found in config/models.yaml")
            return

    fetch_missing_logprobs([m["name"] for m in models], OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
