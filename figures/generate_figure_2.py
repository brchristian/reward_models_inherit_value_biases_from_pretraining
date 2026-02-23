"""Generate Figure 2: Violin plots of the log probs assigned to Agency and
Communion nouns by pretained and instruction-tuned versions of Gemma 2 2B and
Llama 3.2 3B."""

import os
import Path
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from transformers import AutoTokenizer

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
    sig_stars = ['***', '***', '***', '***']
else:
    gemma_id = 'google/gemma-2-2b'
    gemma_name = 'Gemma 2 2B'
    llama_id = 'meta-llama/Llama-3.2-3B'
    llama_name = 'LLama 3.2 3B'
    qwen_id = 'Qwen/Qwen2.5-3B'
    qwen_name = 'Qwen 2.5 3B'
    y_text = (30, 25)
    sig_stars = ['**', '**', '**', '**']
    
# Load prompts
with open(CONFIG_DIR / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

# Get list of tokens shared by both tokenizers
tok_g = AutoTokenizer.from_pretrained(gemma_id)
tok_l = AutoTokenizer.from_pretrained(llama_id)
vocab_g = set(tok_g.decode(id) for id in tok_g.vocab.values())
vocab_l = set(tok_l.decode(id) for id in tok_l.vocab.values())
shared_tokens = list(vocab_g & vocab_l)

# Load big 2 agency communion list
big2 = pd.read_csv('../data/corpora/dict_big2_nouns.csv')
big2['in_vocab'] = big2.token_decoded.apply(lambda x: x in shared_tokens)
big2 = big2.query("in_vocab==True")

# Load logprobs for either pretrained or instruction tuned
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if args.instruct:
    llama_file = os.path.join(repo_root, 'data', 'logprobs', 'meta-llama--Llama-3.2-3B-Instruct.csv')
    gemma_file = os.path.join(repo_root, 'data', 'logprobs', 'google--gemma-2-2b-it.csv')
else:
    llama_file = os.path.join(repo_root, 'data', 'logprobs', 'meta-llama--Llama-3.2-3B.csv')
    gemma_file = os.path.join(repo_root, 'data', 'logprobs', 'google--gemma-2-2b.csv.csv')
try:
    gemma_lp = pd.read_csv(gemma_file)
    llama_lp = pd.read_csv(llama_file)
except FileNotFoundError:
    print(f"{gemma_file} or {llama_file} not found.")
    print("Ensure you run scripts/generate_logrpobs.py first.")

# For each prompt, get rank of logprob for each token in the big2 list
def is_good(key):
    if 'greatest' in key or 'best' in key or 'good' in key:
        return True
    else:
        return False
dfs = []
for key in prompts:
    for tok, lp in zip([tok_g, tok_l], [gemma_lp, llama_lp]):
        df = big2.copy()
        assert (df.token_decoded.apply(lambda x: tok(x)['input_ids']).apply(len) == 2).all()
        df['tok_idx'] = df.token_decoded.apply(lambda x: tok(x)['input_ids'][-1])
        df['logprob'] = df.tok_idx.apply(lambda x: lp[f'{key}_lp'][x])
        df['rank'] = df.logprob.rank(method='average', ascending=False)
        df['model'] = gemma_name
        df['prompt']= key
        df['valence'] = 'good' if is_good(key) else 'bad'
        
        dfs.append(df)
    
big2_all_prompts = pd.concat(dfs)

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
