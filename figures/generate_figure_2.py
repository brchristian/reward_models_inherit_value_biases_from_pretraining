"""Generate Figure 2: Violin plots of the log probs assigned to Agency and
Communion nouns by pretained and instruction-tuned versions of Gemma 2 2B and
Llama 3.2 3B."""

import os
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

SCRIPT_ROOT = Path(__file__).parent
REPO_ROOT = SCRIPT_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"

# Settings
parser = argparse.ArgumentParser()
parser.add_argument("--instruct", action="store_true", default=False,
                    help="Whether to plot instruction-tuned model (default: False)")
args = parser.parse_args()

if args.instruct:
    gemma_id = 'google/gemma-2-2b-it'
    gemma_name = 'Gemma 2 2B IT'
    llama_id = 'meta-llama/Llama-3.2-3B-Instruct'
    llama_name = 'LLama 3.2 3B Instruct'
    qwen_id = 'Qwen/Qwen2.5-3B-Instruct'
    qwen_name = 'Qwen 2.5 3B Instruct'
    y_text = (33, 30)
else:
    gemma_id = 'google/gemma-2-2b'
    gemma_name = 'Gemma 2 2B'
    llama_id = 'meta-llama/Llama-3.2-3B'
    llama_name = 'LLama 3.2 3B'
    qwen_id = 'Qwen/Qwen2.5-3B'
    qwen_name = 'Qwen 2.5 3B'
    y_text = (30, 25)

# Load prompts
with open(CONFIG_DIR / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

# Helper to convert model ID to CSV filename
def safe_name(model_id):
    return model_id.replace("/", "--")

# Load logprobs CSVs
logprob_dir = REPO_ROOT / "data" / "logprobs"
if args.instruct:
    gemma_file = logprob_dir / f"{safe_name(gemma_id)}.csv"
    llama_file = logprob_dir / f"{safe_name(llama_id)}.csv"
else:
    gemma_file = logprob_dir / f"{safe_name(gemma_id)}.csv"
    llama_file = logprob_dir / f"{safe_name(llama_id)}.csv"

gemma_lp = pd.read_csv(gemma_file)
llama_lp = pd.read_csv(llama_file)

# Load big 2 agency communion list, filter to tokens in both vocabularies
big2 = pd.read_csv(REPO_ROOT / "data" / "corpora" / "dict_big2_nouns.csv")
big2 = big2[
    big2.token_decoded.isin(gemma_lp.token_decoded) &
    big2.token_decoded.isin(llama_lp.token_decoded)
]

# For each prompt, get rank of logprob for each token in the big2 list
def is_good(key):
    return 'greatest' in key or 'best' in key or 'good' in key

dfs = []
for key in prompts:
    for model_name, lp_df in [(gemma_name, gemma_lp), (llama_name, llama_lp)]:
        df = pd.merge(big2, lp_df[['token_decoded', key]], on='token_decoded', how='inner')
        df['logprob'] = df[key]
        df['rank'] = df.logprob.rank(method='average', ascending=False)
        df['model'] = model_name
        df['prompt'] = key
        df['valence'] = 'good' if is_good(key) else 'bad'
        dfs.append(df)

big2_all_prompts = pd.concat(dfs)

# Compute median rank per (prompt, Category, model) — the unit of analysis
medians = big2_all_prompts.groupby(
    ['prompt', 'Category', 'model', 'valence']
).median(numeric_only=True).reset_index()

# Posthoc: Welch t-test, Gemma vs Llama within each (Category x valence) cell
cells = [('Agency', 'good'), ('Communion', 'good'),
         ('Agency', 'bad'), ('Communion', 'bad')]
p_values = []
for cat, val in cells:
    gemma_ranks = medians.query("Category==@cat & valence==@val & model==@gemma_name")['rank']
    llama_ranks = medians.query("Category==@cat & valence==@val & model==@llama_name")['rank']
    _, p = ttest_ind(gemma_ranks, llama_ranks, equal_var=False)
    p_values.append(p)

_, p_corrected = fdrcorrection(p_values)

def stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'n.s.'

sig_stars = [stars(p) for p in p_corrected]
print(f"FDR-corrected p-values: {[f'{p:.5f}' for p in p_corrected]}")
print(f"Significance: {list(zip([f'{c} {v}' for c, v in cells], sig_stars))}")

# Plot
sns.set_context('talk')
google_red = "#DB4437"
meta_blue = "#0064E0"
palette = sns.color_palette([google_red, meta_blue])

alpha=0.7
hue_order = [gemma_name, llama_name]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
to_plot_good = big2_all_prompts.query("valence=='good'").groupby(['prompt', 'Category', 'model']).median(numeric_only=True).reset_index()
sns.violinplot(data=to_plot_good, x='Category', y='rank', hue='model',
               ax=ax1, legend=False, split=True,
               hue_order=hue_order,
               palette=palette, alpha=alpha)
sns.stripplot(data=to_plot_good, x='Category', y='rank', hue='model',
            ax=ax1, dodge=True, 
            hue_order=hue_order,
            palette=palette, alpha=alpha, legend=False)
ax1.invert_yaxis()
ax1.set_ylabel("Median rank \n(positive prompts)")
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.xaxis.set_visible(False)
ax1.text(0, y_text[0], sig_stars[0], ha='center')
ax1.text(1, y_text[0], sig_stars[1], ha='center')

to_plot_bad = big2_all_prompts.query("valence=='bad'").groupby(['prompt', 'Category', 'model']).median(numeric_only=True).reset_index()
g = sns.violinplot(data=to_plot_bad, x='Category', y='rank', hue='model',
               ax=ax2, legend=True, split=True,
               hue_order=hue_order,
               palette=palette, alpha=alpha)
sns.stripplot(data=to_plot_bad, x='Category', y='rank', hue='model',
            ax=ax2, dodge=True, 
            hue_order=hue_order,
            palette=palette, alpha=alpha, legend=False)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.8, 1.5), title='Model')

ax2.invert_yaxis()
ax2.set_ylabel("Median rank \n(negative prompts)")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('Big2 Category')
ax2.text(0, y_text[1], sig_stars[2], ha='center')
ax2.text(1, y_text[1], sig_stars[3], ha='center')
# plt.tight_layout()

version = 'instruct' if args.instruct else 'pretrained'
plt.savefig(f'big2_logprob_rank_gemma_vs_llamma_{version}.png', dpi=300, bbox_inches='tight')
