"""Generate Figure 2: Violin plots of the log probs assigned to Agency and
Communion nouns by pretained and instruction-tuned versions of Gemma 2 2B and
Llama 3.2 3B."""

import os
import sys
from pathlib import Path
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from statsmodels.formula.api import ols

SCRIPT_ROOT = Path(__file__).parent
REPO_ROOT = SCRIPT_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fetch_logprobs import fetch_missing_logprobs, safe_name

DEFAULT_PAIRS = [
    ('google/gemma-2-2b', 'meta-llama/Llama-3.2-3B'),
    ('google/gemma-2-2b-it', 'meta-llama/Llama-3.2-3B-Instruct'),
]


def display_name(model_id, display_names):
    """Look up display name from config, falling back to deriving from the HF model ID."""
    return display_names.get(model_id, model_id.split("/")[-1].replace("-", " "))


def load_model_config():
    """Load config/models.yaml and return (family_map, display_names)."""
    models_path = CONFIG_DIR / "models.yaml"
    if not models_path.exists():
        return {}, {}
    with open(models_path, "r") as f:
        models = yaml.safe_load(f)
    family_map = {m['name']: m.get('family', '') for m in models}
    display_names = {m['name']: m['display_name'] for m in models if 'display_name' in m}
    return family_map, display_names


def load_family_colors():
    """Load config/family_colors.yaml and return {family: hex_color} mapping."""
    colors_path = CONFIG_DIR / "family_colors.yaml"
    if not colors_path.exists():
        return {}
    with open(colors_path, "r") as f:
        return yaml.safe_load(f)


def get_palette(model1_id, model2_id, family_map, family_colors):
    """Return a two-color palette based on model families, with positional fallback."""
    fam1 = family_map.get(model1_id, '')
    fam2 = family_map.get(model2_id, '')
    c1 = family_colors.get(fam1, '#DB4437')
    c2 = family_colors.get(fam2, '#0064E0')
    return sns.color_palette([c1, c2])


def is_good(key):
    return 'greatest' in key or 'best' in key or 'good' in key


def stars(p):
    if p < .001: return '***'
    if p < .01:  return '**'
    if p < .05:  return '*'
    return 'n.s.'


def generate_figure(model1_id, model2_id, prompts, family_map, family_colors, display_names):
    """Generate a single figure for a pair of models."""
    model1_name = display_name(model1_id, display_names)
    model2_name = display_name(model2_id, display_names)

    # Download logprobs CSVs if missing, then load
    logprob_dir = REPO_ROOT / "data" / "logprobs"
    fetch_missing_logprobs([model1_id, model2_id], logprob_dir)
    model1_lp = pd.read_csv(logprob_dir / f"{safe_name(model1_id)}.csv")
    model2_lp = pd.read_csv(logprob_dir / f"{safe_name(model2_id)}.csv")

    # Load big 2 agency communion list, filter to tokens in both vocabularies
    big2 = pd.read_csv(REPO_ROOT / "data" / "corpora" / "dict_big2_nouns.csv")
    big2 = big2[
        big2.token_decoded.isin(model1_lp.token_decoded) &
        big2.token_decoded.isin(model2_lp.token_decoded)
    ]

    # For each prompt, get rank of logprob for each token in the big2 list
    dfs = []
    for key in prompts:
        for model_name, lp_df in [(model1_name, model1_lp), (model2_name, model2_lp)]:
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

    # 3-way ANOVA (Type II): rank ~ Category * valence * model
    anova_model = ols('rank ~ C(Category) * C(valence) * C(model)', data=medians).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)
    print(f"\n--- 3-way ANOVA: {model1_name} vs {model2_name} ---")
    print(anova_table)

    # Posthoc: Welch t-test, model1 vs model2 within each (Category x valence) cell
    cells = [('Agency', 'good'), ('Communion', 'good'),
             ('Agency', 'bad'), ('Communion', 'bad')]
    p_values = []
    for cat, val in cells:
        m1_ranks = medians.query("Category==@cat & valence==@val & model==@model1_name")['rank']
        m2_ranks = medians.query("Category==@cat & valence==@val & model==@model2_name")['rank']
        _, p = ttest_ind(m1_ranks, m2_ranks, equal_var=False)
        p_values.append(p)

    _, p_corrected = fdrcorrection(p_values)
    sig_stars = [stars(p) for p in p_corrected]

    print(f"\n--- {model1_name} vs {model2_name} ---")
    print(f"FDR-corrected p-values: {[f'{p:.5f}' for p in p_corrected]}")
    print(f"Significance: {list(zip([f'{c} {v}' for c, v in cells], sig_stars))}")

    # Compute y_text positions from data
    to_plot_good = big2_all_prompts.query("valence=='good'").groupby(
        ['prompt', 'Category', 'model']).median(numeric_only=True).reset_index()
    to_plot_bad = big2_all_prompts.query("valence=='bad'").groupby(
        ['prompt', 'Category', 'model']).median(numeric_only=True).reset_index()
    y_text_good = to_plot_good['rank'].min() * 0.85
    y_text_bad = to_plot_bad['rank'].min() * 0.85

    # Plot
    sns.set_context('talk')
    palette = get_palette(model1_id, model2_id, family_map, family_colors)
    alpha = 0.7
    hue_order = [model1_name, model2_name]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    sns.violinplot(data=to_plot_good, x='Category', y='rank', hue='model',
                   ax=ax1, legend=False, split=True,
                   hue_order=hue_order,
                   palette=palette, alpha=alpha)
    sns.stripplot(data=to_plot_good, x='Category', y='rank', hue='model',
                  ax=ax1, dodge=True,
                  hue_order=hue_order,
                  palette=palette, alpha=alpha, legend=False)
    ax1.invert_yaxis()
    ax1.set_ylabel("Median Rank \n(positive prompts)")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_visible(False)
    ax1.text(0, y_text_good, sig_stars[0], ha='center')
    ax1.text(1, y_text_good, sig_stars[1], ha='center')

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
    ax2.set_ylabel("Median Rank \n(negative prompts)")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('Big-Two Category')
    ax2.text(0, y_text_bad, sig_stars[2], ha='center')
    ax2.text(1, y_text_bad, sig_stars[3], ha='center')

    # Save
    output_dir = SCRIPT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    filename = f"fig2_{safe_name(model1_id)}_vs_{safe_name(model2_id)}.pdf"
    outpath = output_dir / filename
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: violin plots comparing two models on Big2 Agency/Communion.")
    parser.add_argument("--model1", type=str, default=None,
                        help="HF model ID for the first model")
    parser.add_argument("--model2", type=str, default=None,
                        help="HF model ID for the second model")
    args = parser.parse_args()

    if (args.model1 is None) != (args.model2 is None):
        parser.error("Provide both --model1 and --model2, or neither (to use defaults).")

    if args.model1 and args.model2:
        pairs = [(args.model1, args.model2)]
    else:
        pairs = DEFAULT_PAIRS

    # Load shared config
    with open(CONFIG_DIR / "prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    family_map, display_names = load_model_config()
    family_colors = load_family_colors()

    for model1_id, model2_id in pairs:
        generate_figure(model1_id, model2_id, prompts, family_map, family_colors, display_names)


if __name__ == "__main__":
    main()
