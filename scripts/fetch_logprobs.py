"""
Shared helper: download missing logprob CSVs from HuggingFace Hub.

Used by download_data.py, generate_figure_2.py, generate_figure_3.py,
and generate_table_1.py to ensure data/logprobs/ is populated before
reading CSVs.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

HF_DATASET_REPO = "Oxford-HIPlab/iclr2026-lm-logprobs"


def safe_name(model_name):
    """Convert HuggingFace model ID to filesystem-safe name."""
    return model_name.replace("/", "--")


def fetch_missing_logprobs(model_ids, logprob_dir):
    """Download any logprob CSVs not already present in *logprob_dir*."""
    logprob_dir = Path(logprob_dir)
    logprob_dir.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        filename = f"{safe_name(model_id)}.csv"
        if not (logprob_dir / filename).exists():
            print(f"Downloading {filename} from HuggingFace Hub...")
            hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(logprob_dir),
            )
