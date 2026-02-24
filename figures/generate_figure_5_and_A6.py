"""
Figure 5: Token rank changes between first and final training checkpoints.

Shows how the ranking of Big Two (Agency/Communion) tokens changes from
the first checkpoint to the final checkpoint of RM fine-tuning.
Tokens are filtered to the shared token intersection and Big Two dictionary,
grouped by normalised form (averaging scores across token variants),
and coloured by Agency/Communion category.

Replicates the approach in rminterp/analysis/figures.py:
  get_df_ranks_big2 → get_rank_changes_big2 → plot_dramatic_rank_change_tokens_big2
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data", "reward_model_training")
CORPORA_DIR = os.path.join(REPO_ROOT, "data", "corpora")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use Times New Roman to match other figures
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Colours matching rminterp/constants.py
COLORS_BIG2 = {
    "Agency": "#3CB371",
    "Communion": "#FFCE1B",
}


def get_df_ranks_big2(df, big2):
    """Filter to Big Two tokens, group by token_norm, average scores, and rank.

    Mirrors rminterp.analysis.figures.get_df_ranks_big2.
    """
    # Clean and normalise token text
    df = df.copy()
    df["token_norm"] = (
        df["token_decoded"]
        .str.encode("ascii", errors="ignore").str.decode("ascii")
        .str.strip()
        .str.lower()
    )
    df = df[df["token_norm"].str.len() > 0]

    # Inner join with Big Two nouns dictionary
    df = df.merge(big2, left_on="token_norm", right_on="token_decoded",
                  suffixes=("", "_dict"), how="inner")
    df = df.rename(columns={"Category": "big2"})

    # Identify checkpoint (step) columns
    step_cols = sorted(
        [c for c in df.columns if c.startswith("step-")],
        key=lambda x: int(x.split("-")[1]),
    )

    # Group by token_norm: average scores across token variants per checkpoint
    agg_dict = {col: "mean" for col in step_cols}
    agg_dict["big2"] = "first"
    grouped = df.groupby("token_norm").agg(agg_dict).reset_index()

    # Rank within each checkpoint (descending score = rank 1 is best)
    for col in step_cols:
        grouped[f"rank_{col}"] = grouped[col].rank(ascending=False)

    return grouped, step_cols


def get_rank_changes_big2(grouped, step_cols):
    """Compute rank change between first and final checkpoint.

    Positive rank_change = token moved UP in ranking (became more preferred).
    """
    first_rank_col = f"rank_{step_cols[0]}"
    final_rank_col = f"rank_{step_cols[-1]}"

    rank_changes = grouped[["token_norm", "big2"]].copy()
    rank_changes["rank_change"] = grouped[first_rank_col] - grouped[final_rank_col]
    return rank_changes.sort_values("rank_change", ascending=False)


def plot_dramatic_rank_change_tokens_big2(
    rank_changes, model_name, n_tokens=10, path_save=None
):
    """Horizontal bar plot of top rank movers, coloured by Agency/Communion.

    Matches rminterp.analysis.figures.plot_dramatic_rank_change_tokens_big2.
    """
    improved = rank_changes.head(n_tokens)
    worsened = rank_changes.tail(n_tokens)

    combined = pd.concat([
        improved.assign(category="Improved"),
        worsened.assign(category="Worsened"),
    ])

    plt.figure(figsize=(4, 5))
    ax = sns.barplot(
        data=combined,
        x="rank_change",
        y="token_norm",
        hue="big2",
        hue_order=sorted(rank_changes["big2"].unique()),
        palette=COLORS_BIG2,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        frameon=False,
        title=None,
        fontsize=10,
        handlelength=2,
    )

    # Side text annotations
    xlim = ax.get_xlim()
    zero_position = (0 - xlim[0]) / (xlim[1] - xlim[0])

    chunk_size = len(combined) // 2
    top_center = (chunk_size - 1) / 2
    bottom_center = chunk_size + (chunk_size - 1) / 2

    ax.text(
        zero_position - 0.1, top_center,
        "Preferred late in training",
        transform=ax.get_yaxis_transform(),
        ha="center", va="center", rotation=90, fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )
    ax.text(
        zero_position + 0.1, bottom_center,
        "Preferred early in training",
        transform=ax.get_yaxis_transform(),
        ha="center", va="center", rotation=90, fontsize=10,
        color="black",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    ax.set_xlabel("Rank Change")
    ax.set_ylabel("")

    # Left-justify y-axis labels (monospace, padded)
    ticks = ax.get_yticks()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, ha="left", fontfamily="monospace")
    longest = max((len(l) for l in labels), default=1)
    ax.tick_params(axis="y", pad=6.5 * longest)

    plt.tight_layout()

    if path_save is not None:
        plt.savefig(path_save, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved {path_save}")

    plt.close()


def main():
    # Load Big Two nouns dictionary (matches notebook pipeline)
    big2 = pd.read_csv(os.path.join(CORPORA_DIR, "dict_big2_nouns.csv"))

    # Load parquet files
    llama = pd.read_parquet(os.path.join(DATA_DIR, "greatest_ever_one_across_checkpoints_llama.parquet"))
    gemma = pd.read_parquet(os.path.join(DATA_DIR, "greatest_ever_one_across_checkpoints_gemma.parquet"))
    qwen = pd.read_parquet(os.path.join(DATA_DIR, "greatest_ever_one_across_checkpoints_qwen.parquet"))

    n_tokens = 10

    # Figure 5: Llama and Gemma
    for model_name, df in [("Llama", llama), ("Gemma", gemma)]:
        grouped, step_cols = get_df_ranks_big2(df, big2)
        print(f"{model_name}: {len(grouped)} unique Big Two token_norm entries "
              f"(checkpoints {step_cols[0]} → {step_cols[-1]})")

        rank_changes = get_rank_changes_big2(grouped, step_cols)

        output_file = os.path.join(OUTPUT_DIR, f"fig5_{model_name.lower()}.pdf")
        plot_dramatic_rank_change_tokens_big2(
            rank_changes,
            model_name=model_name,
            n_tokens=n_tokens,
            path_save=output_file,
        )

    # Figure A6: Qwen
    grouped, step_cols = get_df_ranks_big2(qwen, big2)
    print(f"Qwen: {len(grouped)} unique Big Two token_norm entries "
          f"(checkpoints {step_cols[0]} → {step_cols[-1]})")
    rank_changes = get_rank_changes_big2(grouped, step_cols)
    plot_dramatic_rank_change_tokens_big2(
        rank_changes,
        model_name="Qwen",
        n_tokens=n_tokens,
        path_save=os.path.join(OUTPUT_DIR, "figA6_qwen.pdf"),
    )


if __name__ == "__main__":
    main()
