"""
Generate Table 1: Top and bottom MWLR tokens for a single model pair and prompt.

Computes MWLR(token) = 0.5*(p_llama + p_gemma) * (logp_llama - logp_gemma)
for the "greatest_ever_one" prompt using Llama 3.2 3B Instruct vs Gemma 2 2B IT.

Outputs LaTeX for a two-panel table: Llama-preferred (top) and Gemma-preferred (bottom).

Usage:
    python tables/generate_table_1.py
"""

import os
import sys
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fetch_logprobs import fetch_missing_logprobs, safe_name

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
]
PROMPT_KEY = "greatest_ever_one"
N_SHOW = 15  # tokens to show on each side


def _is_cjk(ch):
    """Return True if ch is a CJK Unified Ideograph."""
    return unicodedata.category(ch) == "Lo" and "CJK" in unicodedata.name(ch, "")


def escape_latex(s):
    """Escape special LaTeX characters in token strings."""
    # First, escape real backslashes (before inserting LaTeX commands)
    s = s.replace("\\", "\\textbackslash ")
    # Now replace control characters with visible representations
    s = s.replace("\r\n", "\\textbackslash n\\textbackslash n")
    s = s.replace("\n", "\\textbackslash n")
    s = s.replace("\r", "\\textbackslash r")
    s = s.replace("\t", "\\textbackslash t")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    s = s.replace("_", "\\_")
    # Show leading spaces as underscores
    if s.startswith(" "):
        s = "\\_" + s[1:]
    # Wrap CJK characters
    result = []
    for ch in s:
        if _is_cjk(ch):
            result.append(f"\\begin{{CJK}}{{UTF8}}{{gbsn}}{ch}\\end{{CJK}}")
        else:
            result.append(ch)
    return "".join(result)


def compute_mwlr(logprob_llama, logprob_gemma):
    p_llama = np.exp(logprob_llama)
    p_gemma = np.exp(logprob_gemma)
    mixture_weight = 0.5 * (p_llama + p_gemma)
    log_ratio = logprob_llama - logprob_gemma
    return mixture_weight * log_ratio


def format_latex_table(top_df, bottom_df, n_total, llama_display, gemma_display, label_suffix):
    """Generate LaTeX for the two-panel table."""
    lines = []
    lines.append(r"\begin{table*}")
    lines.append(r"\centering\scriptsize")
    caption = (
        r"\caption{Optimal and pessimal response tokens for the prompt "
        r"``What, in one word, is the greatest thing ever?'', according to the "
        r"MWLR implicit-RM score. High-ranked tokens (left) are preferred by "
        f"{llama_display} and low-ranked tokens (right), by {gemma_display}.}}"
    )
    lines.append(caption)
    lines.append(f"\\label{{tab:mwlr_{label_suffix}}}")

    # Left panel: Llama-preferred (top MWLR)
    lines.append(r"\begin{minipage}{0.48\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-1em}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{rll}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Decoded & Score \\")
    lines.append(r"\midrule")
    for i, (_, row) in enumerate(top_df.iterrows()):
        rank = i + 1
        token = escape_latex(row["token_decoded"])
        score = f"{row['mwlr']:.5f}"
        lines.append(f"{rank} & \\texttt{{{token}}} & {score} \\\\")
    lines.append(r"... & ... & ... \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{minipage}")
    lines.append(r"\hfill")

    # Right panel: Gemma-preferred (bottom MWLR)
    lines.append(r"\begin{minipage}{0.48\textwidth}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-1em}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{rll}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Decoded & Score \\")
    lines.append(r"\midrule")
    lines.append(r"... & ... & ... \\")
    for i, (_, row) in enumerate(bottom_df.iterrows()):
        rank = n_total - len(bottom_df) + i + 1
        token = escape_latex(row["token_decoded"])
        score = f"{row['mwlr']:.5f}"
        lines.append(f"{rank:,} & \\texttt{{{token}}} & {score} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def load_display_names():
    """Load display names from config/models.yaml."""
    models_path = REPO_ROOT / "config" / "models.yaml"
    with open(models_path, "r") as f:
        models = yaml.safe_load(f)
    return {m["name"]: m["display_name"] for m in models if "display_name" in m}


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logprobs_dir = os.path.join(repo_root, "data", "logprobs")
    output_dir = os.path.join(repo_root, "tables", "output")
    os.makedirs(logprobs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Download logprobs if missing, then load Llama once
    all_models = [LLAMA_MODEL] + GEMMA_MODELS
    fetch_missing_logprobs(all_models, logprobs_dir)

    print(f"Loading {LLAMA_MODEL}...")
    llama_df = pd.read_csv(
        os.path.join(logprobs_dir, f"{safe_name(LLAMA_MODEL)}.csv"),
        usecols=["token_decoded", PROMPT_KEY],
    )
    llama_df = llama_df.dropna(subset=["token_decoded"])
    llama_df = llama_df.sort_values(PROMPT_KEY, ascending=False).drop_duplicates("token_decoded", keep="first")

    display_names = load_display_names()
    llama_display = display_names[LLAMA_MODEL]

    for gemma_model in GEMMA_MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {gemma_model}...")
        gemma_df = pd.read_csv(
            os.path.join(logprobs_dir, f"{safe_name(gemma_model)}.csv"),
            usecols=["token_decoded", PROMPT_KEY],
        )
        gemma_df = gemma_df.dropna(subset=["token_decoded"])
        gemma_df = gemma_df.sort_values(PROMPT_KEY, ascending=False).drop_duplicates("token_decoded", keep="first")

        # Join on token_decoded
        merged = pd.merge(
            llama_df.rename(columns={PROMPT_KEY: "logprob_llama"}),
            gemma_df.rename(columns={PROMPT_KEY: "logprob_gemma"}),
            on="token_decoded",
        )
        print(f"Common tokens: {len(merged):,}")

        # Compute MWLR
        merged["mwlr"] = compute_mwlr(merged["logprob_llama"].values, merged["logprob_gemma"].values)
        merged = merged.sort_values("mwlr", ascending=False).reset_index(drop=True)

        top = merged.head(N_SHOW)
        bottom = merged.tail(N_SHOW)

        # Print summary to stdout
        print(f"\nTop {N_SHOW} tokens (Llama-preferred):")
        for i, (_, row) in enumerate(top.iterrows()):
            print(f"  {i+1:>3d}  {row['token_decoded']:<20s}  {row['mwlr']:+.5f}")

        print(f"\nBottom {N_SHOW} tokens (Gemma-preferred):")
        for i, (_, row) in enumerate(bottom.iterrows()):
            rank = len(merged) - N_SHOW + i + 1
            print(f"  {rank:>6,d}  {row['token_decoded']:<20s}  {row['mwlr']:+.5f}")

        # Write LaTeX
        gemma_display = display_names[gemma_model]
        # e.g. "llama-3.2-3b_gemma-2-2b"
        label_suffix = (
            safe_name(LLAMA_MODEL).lower().replace("meta-llama--", "")
            + "_"
            + safe_name(gemma_model).lower().replace("google--", "")
        ).replace("-instruct", "").replace("-it", "")

        latex = format_latex_table(top, bottom, len(merged), llama_display, gemma_display, label_suffix)
        llama_short = LLAMA_MODEL.split("/")[-1]
        gemma_short = gemma_model.split("/")[-1]
        output_filename = f"MWLR_table_{llama_short}_to_{gemma_short}.tex"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
