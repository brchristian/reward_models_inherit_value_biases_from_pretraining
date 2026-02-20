#%%
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import seaborn as sns
from matplotlib import pyplot as plt
import gc
#%%
# login(insert hugging face login here)
instruct = False
overwrite = False
include_qwen = False

if instruct:
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
if include_qwen:
    tok_q = AutoTokenizer.from_pretrained(qwen_id)
    vocab_q = set(tok_q.decode(id) for id in tok_q.vocab.values())
    shared_tokens = list(vocab_g & vocab_l & vocab_q)
    idx_q = [ids[0] for ids in tok_q(shared_tokens).input_ids] # no extra start token
else:
    shared_tokens = list(vocab_g & vocab_l)
idx_g = [ids[1] for ids in tok_g(shared_tokens).input_ids]
idx_l = [ids[1] for ids in tok_l(shared_tokens).input_ids]

os.makedirs('logprobs/meta-llama', exist_ok=True)
os.makedirs('logprobs/google', exist_ok=True)
os.makedirs('logprobs/Qwen', exist_ok=True)
#%% 


# big 2 agency communion list
big2 = pd.read_csv('data/corpora/dict_big2_nouns.csv')

# big2_cap = big2.copy()
# big2['version'] = 'lowercase'
# big2_cap['version'] = 'firstcap'
# big2_cap.token_decoded = big2.token_decoded.str.capitalize()
# big2_allcap = big2.copy()
# big2_allcap.token_decoded = big2.token_decoded.str.upper()
# big2_allcap['version'] = 'uppercase'

# big2_nowhite = pd.concat([big2, big2_cap, big2_allcap])

# big2_leading = big2_nowhite.copy()
# big2_leading['version']= big2_leading['version'].apply(lambda x: x + '_leading_whitespace')
# big2_leading.token_decoded = big2_nowhite.token_decoded.apply(lambda x: ' ' + x)

# big2 = pd.concat([big2_nowhite, big2_leading])

# big2 = big2_allcap


# big2 = pd.concat([big2, big2_cap, big2_allcap, big2_leading])
# big2 = big2_leading

big2['in_vocab'] = big2.token_decoded.apply(lambda x: x in shared_tokens)
big2 = big2.query("in_vocab==True")
print(f'{len(big2)} tokens')

#%%
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

good_messages = {key:
    [{"role": "user", "content": good_prompts[key]}
     ] for key in good_prompts
}
bad_messages = {key:
    [{"role": "user", "content": bad_prompts[key]}
     ] for key in bad_prompts
}
all_messages = good_messages | bad_messages



#%% Tests
# inputs = tok_q.apply_chat_template(all_messages['best_one_ever'],
#                                    return_tensors="pt", return_dict=True,
#                                    add_generation_prompt=True)
#inputs2 = tok_l(all_messages['best_one_ever'][0]['content'], return_tensors="pt")
# print(tok_q.decode(inputs.input_ids[0]))
# print(tok_l.decode(inputs2.input_ids[0]))
#%%
def get_it_logprobs(model, tokenizer, message, mask=None):
    # Tokenize prompt
    if instruct:
        inputs = tokenizer.apply_chat_template(message, return_tensors="pt",
                                               return_dict=True,
                                               add_generation_prompt=True)
    else:
        inputs = tokenizer(message[0]['content'], return_tensors="pt")

    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids)
        if mask is not None:
            logits = outputs.logits[0, -1, mask]
        else:
            logits = outputs.logits[0, -1, :]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs_cpu = log_probs.cpu().numpy()
    return log_probs_cpu 
# %% Calculate Gemma IT logprobs
gemma_file = f'logprobs/{gemma_id}_logprobs.csv'
if os.path.exists(gemma_file) and not overwrite:
    gemma_lp = pd.read_csv(gemma_file)
else:
    gemma_model = AutoModelForCausalLM.from_pretrained(gemma_id)
    gemma_lp = pd.DataFrame()
    for key in all_messages:
        print(key)
        gemma_lp[f'{key}_lp'] = get_it_logprobs(gemma_model, tok_g, all_messages[key])
    gemma_lp.to_csv(gemma_file)
    del gemma_model
    gc.collect()

#%% Load presaved logprobs
# brian_gemma_lp = pd.read_parquet('gemma-logprobs/google--gemma-2-2b-it.parquet')
# %% Calculate LLama IT logrpobs
# todo load saved copies instead of calculating anew every time
llama_file = f'logprobs/{llama_id}_logprobs.csv'
if os.path.exists(llama_file) and not overwrite:
    llama_lp = pd.read_csv(llama_file)
else:
    llama_model = AutoModelForCausalLM.from_pretrained(llama_id)
    llama_lp = pd.DataFrame()
    for key in all_messages:
        print(key)
        llama_lp[f'{key}_lp'] = get_it_logprobs(llama_model, tok_l, all_messages[key])
    llama_lp.to_csv(llama_file)
    del llama_model
    gc.collect()
#%% Load presaved logprobs
# brian_llama_lp = pd.read_parquet('llama-logprobs/meta-llama--Llama-3.2-3B-Instruct.parquet')

# %% Calculate Qwen IT logrpobs
if include_qwen:
    qwen_file = f'logprobs/{qwen_id}_logprobs.csv'
    if os.path.exists(qwen_file) and not overwrite:
        qwen_lp = pd.read_csv(qwen_file)
    else:
        qwen_model = AutoModelForCausalLM.from_pretrained(qwen_id)
        qwen_lp = pd.DataFrame()
        for key in all_messages:
            print(key)
            qwen_lp[f'{key}_lp'] = get_it_logprobs(qwen_model, tok_q, all_messages[key])
        qwen_lp.to_csv(qwen_file)
        del qwen_model
        gc.collect()
# %% for each prompt, get rank of logprob for each token in the big2 list
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
    if include_qwen:
        dfq = big2.copy()
        assert (dfq.token_decoded.apply(lambda x: tok_q(x)['input_ids']).apply(len) == 1).all()
        dfq['tok_idx'] = dfq.token_decoded.apply(lambda x: tok_q(x)['input_ids'][-1])
        dfq['logprob'] = dfq.tok_idx.apply(lambda x: qwen_lp[f'{key}_lp'][x])
        dfq['rank'] = dfq.logprob.rank(method='average', ascending=False)
        dfq['model'] = qwen_name
        dfq['prompt']= key
        dfq['valence'] = 'good'
        dfs.extend([dfg, dfl, dfq])
    else:
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
    if include_qwen:
        dfq = big2.copy()
        dfq['tok_idx'] = dfq.token_decoded.apply(lambda x: tok_q(x)['input_ids'][-1])
        dfq['logprob'] = dfq.tok_idx.apply(lambda x: qwen_lp[f'{key}_lp'][x])
        dfq['rank'] = dfq.logprob.rank(method='average', ascending=False)
        dfq['model'] = qwen_name
        dfq['prompt']= key
        dfq['valence'] = 'bad'
        dfs.extend([dfg, dfl, dfq])
    else:
        dfs.extend([dfg, dfl])
big2_all_prompts = pd.concat(dfs)
# %% Plot, try to make it look sort of like Mira's
# Colors
sns.set_context('talk')
google_red = "#DB4437"
meta_blue = "#0064E0"
qwen_purple = "#5737CC"
# google_light = "#e36c62"
# meta_light = "#0082FB"
hue_order = [gemma_name, llama_name, qwen_name] if include_qwen else [gemma_name, llama_name]
palette = sns.color_palette([google_red, meta_blue, qwen_purple])
# light_pal = sns.color_palette([google_light, meta_light])
alpha=0.7
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
to_plot_good = big2_all_prompts.query("valence=='good'").groupby(['prompt', 'Category', 'model']).median(numeric_only=True).reset_index()
sns.violinplot(data=to_plot_good, x='Category', y='rank', hue='model',
               ax=ax1, legend=False, split=not include_qwen,
               hue_order=hue_order,
               palette=palette, alpha=alpha)
if not include_qwen:
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
               ax=ax2, legend=True, split=not include_qwen,
               hue_order=hue_order,
               palette=palette, alpha=alpha)
if not include_qwen:
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

version = 'instruct' if instruct else 'pretrained'
if include_qwen:
    plt.savefig(f'figures/output/big2_logprob_rank_qwen_vs_gemma_vs_llamma_{version}.png', dpi=300, bbox_inches='tight')
else:
    plt.savefig(f'figures/output/big2_logprob_rank_gemma_vs_llamma_{version}.png', dpi=300, bbox_inches='tight')






# %% 3-way ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data_to_analyse = big2_all_prompts.groupby(['prompt', 'Category', 'model', 'valence']).median(numeric_only=True).reset_index()
#perform three-way ANOVA
model = ols("""rank ~ C(Category) + C(valence) + C(model) +
               C(Category):C(valence) + C(Category):C(model) + C(valence):C(model) +
               C(Category):C(valence):C(model)""", data=data_to_analyse).fit()

sm.stats.anova_lm(model, typ=2)

#%% posthoc comparisons
data_to_analyse['group'] = data_to_analyse.apply(lambda x: f'{x.model}_{x.Category}_{x.valence}', axis=1)

comparisons = [(f'{llama_name}_Agency_good', f'{gemma_name}_Agency_good'),
               (f'{llama_name}_Communion_good', f'{gemma_name}_Communion_good'),
               (f'{llama_name}_Agency_bad', f'{gemma_name}_Agency_bad'),
               (f'{llama_name}_Communion_bad', f'{gemma_name}_Communion_bad')
               ]
ps = []
for x1, x2 in comparisons:
    t, p, deg = ttest_ind(data_to_analyse.query("group==@x1")['rank'], 
            data_to_analyse.query("group==@x2")['rank'],
            usevar='unequal')
    ps.append(p)

rej, p_cor = fdrcorrection(ps)
print([f'{p:.5f}' for p in p_cor])
# p_cor = array([0.00216544, 0.02842026, 0.00098616, 0.00997042]) # Instruct
# ['0.00000', '0.00002', '0.02151', '0.02463'] # pretrained
# tukey = pairwise_tukeyhsd(data_to_analyse['rank'], data_to_analyse['group'], alpha=0.05)
# print(tukey)



# %% ************* IMPLICIT REWARD SCORE *********

# Calculate Gemma IT logprobs for only overlapping tokens
gemma_df = pd.DataFrame()
gemma_model = AutoModelForCausalLM.from_pretrained(gemma_id)
prompt_key = 'greatest_one_ever'
# prompt_key = 'worst_one_ever'
gemma_df['logprob'] = get_it_logprobs(gemma_model, tok_g, all_messages[prompt_key], idx_g)
gemma_df['tok_idx'] = idx_g
gemma_df['decode'] = [tok_g.decode(idx) for idx in idx_g]
gemma_df['rank'] = gemma_df['logprob'].rank(ascending=False)
# gemma_df.sort_values('logprob', ascending=False).query("logprob<-20")
del gemma_model
gc.collect()

# %% Calculate LLama IT logrpobs for only overlapping tokens
llama_df = pd.DataFrame()
llama_model = AutoModelForCausalLM.from_pretrained(llama_id)
llama_df['logprob'] = get_it_logprobs(llama_model, tok_l, all_messages[prompt_key], idx_l)
llama_df['tok_idx'] = idx_l
llama_df['decode'] = [tok_l.decode(idx) for idx in idx_l]
llama_df['rank'] = llama_df['logprob'].rank(ascending=False)
# llama_df.sort_values('logprob', ascending=False).query("logprob<-20")
# llama_df.sort_values('logprob', ascending=False).query("rank>1134")
del llama_model
gc.collect()
# %% Verify normalization
gemma_df['logprob_like_gemma_mean'] = gemma_df['logprob']
gemma_df['logprob_like_gemma_max'] = gemma_df['logprob']# (gemma_lp - gemma_lp.mean()) / gemma_lp.std() * gemma_lp.std()
gemma_df['prob'] = gemma_df['logprob'].apply(np.exp)
# gemma_df['logprob_norm'] = gemma_df['logprob'] - gemma_df['logprob'].max()
gemma_df['model'] = 'gemma'

print(gemma_df['prob'].sum()) ## 1.0000012
llama_df['model'] = 'llama'
llama_df['prob'] = llama_df['logprob'].apply(np.exp)
llama_df['logprob12'] = np.maximum(llama_df['logprob'], -12)

# llama_norm['logprob_norm'] = llama_lp - llama_lp.max()
# llama_norm['logprob_like_gemma_max'] = (((llama_lp - llama_lp.max()) /llama_lp.std()) * gemma_lp.std()) + gemma_lp.max()
llama_df['logprob_like_gemma_mean'] = (((llama_df['logprob'] - llama_df['logprob'].mean()) /llama_df['logprob'].std()) * gemma_df['logprob'] .std()) + gemma_df['logprob'].mean()
llama_df['logprob_like_gemma_max'] = (((llama_df['logprob'] - llama_df['logprob'].max()) /llama_df['logprob'].std()) * gemma_df['logprob'] .std()) + gemma_df['logprob'].max()

print(llama_df['prob'].sum())  # 1.0000116
df_norm = pd.concat([gemma_df, llama_df])
sns.violinplot(data=df_norm, y='logprob', x='model')
# plt.plot([-0.5, 1.5], [-20, -20], '--')
# plt.plot([-0.5, 1.5], [-12, -12], '--')

#%%
def probMeanLogDiff(lp1, lp2):
    p1 = np.exp(lp1)
    p2 = np.exp(lp2)
    return ((p1 + p2) / 2) * (lp2 - lp1)
# %% Cap logprobs at -20
implicit_reward = pd.DataFrame()
gemma = gemma_df['logprob']
llama = llama_df['logprob']
gemma_df['logprob5'] = np.maximum(gemma, -5)
llama_df['logprob5'] = np.maximum(llama, -5)
gemma_df['logprob10'] = np.maximum(gemma, -10)
llama_df['logprob10'] = np.maximum(llama, -10)
gemma_df['logprob12'] = np.maximum(gemma, -12)
llama_df['logprob12'] = np.maximum(llama, -12)
gemma_df['logprob15'] = np.maximum(gemma, -15)
llama_df['logprob15'] = np.maximum(llama, -15)
gemma_df['logprob20'] = np.maximum(gemma, -20)
llama_df['logprob20'] = np.maximum(llama, -20)
gemma_df['logprob30'] = np.maximum(gemma, -30)
llama_df['logprob30'] = np.maximum(llama, -30)
# implicit_reward['score'] = llama_norm['logprob20'] - gemma_norm['logprob20']
implicit_reward['score_raw'] = llama - gemma
implicit_reward['risk_ratio'] = gemma/ llama 
implicit_reward['odds_ratio'] = (gemma/(1-gemma))/ (llama / (1-llama))
# implicit_reward['score5'] = llama_norm['logprob5'] - gemma_norm['logprob5']
implicit_reward['score10'] = llama_df['logprob10'] - gemma_df['logprob10']
implicit_reward['score15'] = llama_df['logprob15'] - gemma_df['logprob15']
implicit_reward['score20'] = llama_df['logprob20'] - gemma_df['logprob20']

implicit_reward['score_pmld'] = probMeanLogDiff(gemma, llama)
implicit_reward['score_mixed'] = llama_df['logprob12'] - gemma_df['logprob20']
# implicit_reward['score30'] = llama_norm['logprob30'] - gemma_norm['logprob30']
# implicit_reward['score_p'] = llama_norm['prob'] - gemma_norm['prob']
implicit_reward['decoded'] = shared_tokens
implicit_reward['gemma_lp'] = gemma
implicit_reward['llama_lp'] = llama
implicit_reward['gemma_tok_idx'] = idx_g
implicit_reward['llama_tok_idx'] = idx_l

#%%
measure = 'score_pmld'
implicit_reward['rank'] = implicit_reward[measure].rank(ascending=False)
sorted = implicit_reward.sort_values(measure, ascending=False)
sorted['decoded'] = sorted['decoded'].str.replace(' ', '_')
# print(sorted[['decoded', 'gemma_lp', 'llama_lp', 'gemma_tok_idx', 'llama_tok_idx', measure]].head(20))
# print(sorted[['decoded', 'gemma_lp', 'llama_lp', 'gemma_tok_idx', 'llama_tok_idx', measure]].tail(20))
print(sorted[['decoded', 'gemma_tok_idx', 'llama_tok_idx', 'rank', measure]].head(20))
print(sorted[['decoded', 'gemma_tok_idx', 'llama_tok_idx', 'rank', measure]].tail(20))


# %%
