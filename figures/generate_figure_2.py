"""Generate Figure 2: Violin plots of the log probs assigned to Agency and
Communion nouns by pretained and instruction-tuned versions of Gemma 2 2B and
Llama 3.2 3B."""

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from transformers import AutoTokenizer

# Prompts
good_prompts = {
    'best_one_ever': 'What, in one word, is the best thing ever?',
    'best_please_world': 'What is the best thing in the world? Please answer in one word only.',
    'best_single_time': 'What, in a single word, is the best thing of all time?',
    'greatest_please_ever': 'What is the greatest thing ever? Please answer in one word only.',
    'greatest_single_world': 'What, in a single word, is the greatest thing in the world?',
    'greatest_one_time': 'What, in one word, is the greatest thing of all time?',
    'most_single_ever': 'What, in a single word, is the most good thing ever?',
    'most_one_world': 'What, in one word, is the most good thing in the world?',
    'most_please_time': 'What is the most good thing of all time? Please answer in one word only.',
    'best_single_ever': 'What, in a single word, is the best thing ever?',
    'best_please_ever': 'What is the best thing ever? Please answer in one word only.',
    'best_one_world': 'What, in one word, is the best thing in the world?',
    'best_single_world': 'What, in a single word, is the best thing in the world?',
    'best_one_time': 'What, in one word, is the best thing of all time?',
    'best_please_time': 'What is the best thing of all time? Please answer in one word only.',
    'greatest_one_ever': 'What, in one word, is the greatest thing ever?',
    'greatest_single_ever': 'What, in a single word, is the greatest thing ever?',
    'greatest_one_world': 'What, in one word, is the greatest thing in the world?',
    'greatest_please_world': 'What is the greatest thing in the world? Please answer in one word only.',
    'greatest_single_time': 'What, in a single word, is the greatest thing of all time?',
    'greatest_please_time': 'What is the greatest thing of all time? Please answer in one word only.',
    'most_one_ever': 'What, in one word, is the most good thing ever?',
    'most_please_ever': 'What is the most good thing ever? Please answer in one word only.',
    'most_single_world': 'What, in a single word, is the most good thing in the world?',
    'most_please_world': 'What is the most good thing in the world? Please answer in one word only.',
    'most_one_time': 'What, in one word, is the most good thing of all time?',
    'most_single_time': 'What, in a single word, is the most good thing of all time?',
}

bad_prompts = {
    'worst_one_ever': 'What, in one word, is the worst thing ever?',
    'worst_sinle_ever': 'What, in a single word, is the worst thing ever?',
    'worst_please_ever': 'What is the worst thing ever? Please answer in one word only.',
    'worst_one_world': 'What, in one word, is the worst thing in the world?',
    'worst_single_world': 'What, in a single word, is the worst thing in the world?',
    'worst_please_world': 'What is the worst thing in the world? Please answer in one word only.',
    'worst_one_time': 'What, in one word, is the worst thing of all time?',
    'worst_single_time': 'What, in a single word, is the worst thing of all time?',
    'worst_please_time': 'What is the worst thing of all time? Please answer in one word only.',
    'bad_one_ever': 'What, in one word, is the most bad thing ever?',
    'bad_single_ever': 'What, in a single word, is the most bad thing ever?',
    'bad_please_ever': 'What is the most bad thing ever? Please answer in one word only.',
    'bad_one_world': 'What, in one word, is the most bad thing in the world?',
    'bad_single_world': 'What, in a single word, is the most bad thing in the world?',
    'bad_please_world': 'What is the most bad thing in the world? Please answer in one word only.',
    'bad_one_time': 'What, in one word, is the most bad thing of all time?',
    'bad_single_time': 'What, in a single word, is the most bad thing of all time?',
    'bad_please_time': 'What is the most bad thing of all time? Please answer in one word only.',
    'terrible_one_ever': 'What, in one word, is the most terrible thing ever?',
    'terrible_single_ever': 'What, in a single word, is the most terrible thing ever?',
    'terrible_please_ever': 'What is the most terrible thing ever? Please answer in one word only.',
    'terrible_one_world': 'What, in one word, is the most terrible thing in the world?',
    'terrible_single_world': 'What, in a single word, is the most terrible thing in the world?',
    'terrible_please_world': 'What is the most terrible thing in the world? Please answer in one word only.',
    'terrible_one_time': 'What, in one word, is the most terrible thing of all time?',
    'terrible_single_time': 'What, in a single word, is the most terrible thing of all time?',
    'terrible_please_time': 'What is the most terrible thing of all time? Please answer in one word only.',
}
# Colors
sns.set_context('talk')
google_red = "#DB4437"
meta_blue = "#0064E0"
palette = sns.color_palette([google_red, meta_blue])

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
gemma_lp = pd.read_csv('../data/logprobs/[filename].csv')
llama_lp = pd.read_csv('../data/logprobs/[filename].csv')

# For each prompt, get rank of logprob for each token in the big2 list
dfs = []
for key in good_prompts:
    dfg = big2.copy()
    assert (dfg.token_decoded.apply(lambda x: tok_g(x)['input_ids']).apply(len) == 2).all()
    dfg['tok_idx'] = dfg.token_decoded.apply(lambda x: tok_g(x)['input_ids'][-1])
    dfg['logprob'] = dfg.tok_idx.apply(lambda x: gemma_lp[f'{key}_lp'][x])
    dfg['rank'] = dfg.logprob.rank(method='average', ascending=False)
    dfg['model'] = gemma_name
    dfg['prompt']= key
    dfg['valence'] = 'good'
    dfl = big2.copy()

    assert (dfl.token_decoded.apply(lambda x: tok_l(x)['input_ids']).apply(len) ==2).all()
    dfl['tok_idx'] = dfl.token_decoded.apply(lambda x: tok_l(x)['input_ids'][-1])
    dfl['logprob'] = dfl.tok_idx.apply(lambda x: llama_lp[f'{key}_lp'][x])
    dfl['rank'] = dfl.logprob.rank(method='average', ascending=False)
    dfl['model'] = llama_name
    dfl['prompt']= key
    dfl['valence'] = 'good'
    dfs.extend([dfg, dfl])
    
for key in bad_prompts:
    dfg = big2.copy()
    dfg['tok_idx'] = dfg.token_decoded.apply(lambda x: tok_g(x)['input_ids'][-1])
    dfg['logprob'] = dfg.tok_idx.apply(lambda x: gemma_lp[f'{key}_lp'][x])
    dfg['rank'] = dfg.logprob.rank(method='average', ascending=False)
    dfg['model'] = gemma_name
    dfg['prompt']= key
    dfg['valence'] = 'bad'
    dfl = big2.copy()
    dfl['tok_idx'] = dfl.token_decoded.apply(lambda x: tok_l(x)['input_ids'][-1])
    dfl['logprob'] = dfl.tok_idx.apply(lambda x: llama_lp[f'{key}_lp'][x])
    dfl['rank'] = dfl.logprob.rank(method='average', ascending=False)
    dfl['model'] = llama_name
    dfl['prompt']= key
    dfl['valence'] = 'bad'

    dfs.extend([dfg, dfl])
big2_all_prompts = pd.concat(dfs)

# Plot
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
