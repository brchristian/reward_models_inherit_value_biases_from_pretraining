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
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fetch_logprobs import fetch_missing_logprobs, safe_name

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GEMMA_MODEL = "google/gemma-2-2b-it"
PROMPT_KEY = "greatest_ever_one"
N_SHOW = 15  # tokens to show on each side


def escape_latex(s):
    """Escape special LaTeX characters in token strings."""
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    # Show leading spaces as underscores
    if s.startswith(" "):
        s = "\\_" + s[1:]
    return s


def compute_mwlr(logprob_llama, logprob_gemma):
    p_llama = np.exp(logprob_llama)
    p_gemma = np.exp(logprob_gemma)
    mixture_weight = 0.5 * (p_llama + p_gemma)
    log_ratio = logprob_llama - logprob_gemma
    return mixture_weight * log_ratio


def format_latex_table(top_df, bottom_df, n_total):
    """Generate LaTeX for the two-panel table."""
    lines = []
    lines.append(r"\begin{table*}%")
    lines.append(r"\centering\scriptsize")
    lines.append(r"\caption{Optimal and pessimal response tokens for the prompt "
                 r"``What, in one word, is the greatest thing ever?'', according to the "
                 r"MWLR implicit-RM score. High-ranked tokens (left) are preferred by "
                 r"Llama 3.2 3B-Instruct and low-ranked tokens (right), by Gemma 2 IT 2B.%")
    lines.append(r"Underscores indicate whitespace.}%")
    lines.append(r"\label{tab:mwlr_llama-3.2-3b_gemma-2-2b}")

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


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logprobs_dir = os.path.join(repo_root, "data", "logprobs")
    output_dir = os.path.join(repo_root, "tables", "output")
    os.makedirs(logprobs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Download logprobs if missing, then load
    fetch_missing_logprobs([LLAMA_MODEL, GEMMA_MODEL], logprobs_dir)
    print(f"Loading {LLAMA_MODEL}...")
    llama_df = pd.read_csv(
        os.path.join(logprobs_dir, f"{safe_name(LLAMA_MODEL)}.csv"),
        usecols=["token_decoded", PROMPT_KEY],
    )
    llama_df = llama_df.dropna(subset=["token_decoded"])
    llama_df = llama_df.sort_values(PROMPT_KEY, ascending=False).drop_duplicates("token_decoded", keep="first")

    print(f"Loading {GEMMA_MODEL}...")
    gemma_df = pd.read_csv(
        os.path.join(logprobs_dir, f"{safe_name(GEMMA_MODEL)}.csv"),
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
    latex = format_latex_table(top, bottom, len(merged))
    output_path = os.path.join(output_dir, "table_1.tex")
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
