from typing import Literal
import os
import json

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import gaussian_kde

import glob
from math import isnan

from matplotlib.patches import Rectangle



# ############################################
# HUMAN KDE
# ############################################
# Load Human dataset
h1 = pd.read_csv("human_data/ous_filtered.csv")
h2 = h1.copy()
h2["IB"] = (h2["IB1"] + h2["IB2"] + h2["IB3"] + h2["IB4"] + h2["IB5"]) / 5
h2["IH"] = (h2["IH1"] + h2["IH2"] + h2["IH3"] + h2["IH4"]) / 4
human_df = h2

def human_kde(human_df=h2, ax=None, alpha=1, colormap='Greys'):
    # Draw humans as KDE with IH on x-axis and IB on y-axis
    smoothness = 20
    ih_vals = np.linspace(1, 7, smoothness)  # Smoother grid for IH (x-axis)
    ib_vals = np.linspace(1, 7, smoothness)  # Smoother grid for IB (y-axis)
    ih_grid, ib_grid = np.mgrid[1:7:(smoothness*1j), 1:7:(smoothness*1j)]  # Switch order here
    positions = np.vstack([ih_grid.ravel(), ib_grid.ravel()])  # Switch order here
    values = np.vstack([human_df['IH'], human_df['IB']])  # Switch order here
    
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, ih_vals.shape + ib_vals.shape)  # Notice ih first, then ib
    
    if ax is None:
        # For imshow, extent is [left, right, bottom, top]
        plt.imshow(np.rot90(Z), cmap=colormap, extent=[1, 7, 1, 7], alpha=alpha)
    else:
        ax.imshow(np.rot90(Z), cmap=colormap, extent=[1, 7, 1, 7], alpha=alpha)


# ############################################
# CONVERGENCE FOR ROUND
# ############################################
def plot_rr_round(df, round=3):
    """
    Plot round robin answers at a specific round across all questions (convergence plot for single round)
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    
    round_df = df[df['round'] == round]
    # sns.set(style='whitegrid', font_scale=1.2)

    # Enforce consistent model order based on model_ensemble
    models = round_df['agent_name'].unique()
    models_shortname = [m.split("_")[1] for m in models] # for plotting

    # get all question_ids
    question_ids = round_df['question_id'].unique()
    n_questions = len(question_ids)

    fig, ax = plt.subplots(figsize=(n_questions * 1.25, len(models) * 3))

    answer_colors = {
        '1': '#960808ff',
        '2': '#ff3838ff',
        '3': '#ff9090ff',
        '4': 'lightgray',
        '5': '#90a6ffff',
        '6': '#3849ffff',
        '7': '#081d96ff',
        'No data': 'black',
    }
    
    for i, model in enumerate(models):
      for j, q_id in enumerate(question_ids):
            df_slice = round_df[(round_df['agent_name'] == model) & (round_df['question_id'] == q_id)]
            answer_float = df_slice['agent_answer'].iloc[0]
            label = 'No data' if isnan(answer_float) else str(int(answer_float))
            bg_color = answer_colors.get(label, 'lightgray')
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                             facecolor=bg_color, linewidth=2, alpha=0.7)
            ax.add_patch(rect)
            ax.text(j, i, label, ha='center', va='center', fontsize=12,
                    color='black' if label != "No data" else 'dimgray', weight='bold')

    ax.set_xticks(np.arange(n_questions))
    ax.set_xticklabels([f"Q{i}" for i in question_ids],
                       rotation=45, ha='right', fontsize=65)

    ax.set_yticks(np.arange(len(models_shortname)))
    ax.set_yticklabels(models_shortname, fontsize=65)  #  Now uses short model names

    ax.set_title(f"Round {round}", fontsize=65, pad=12)
    ax.set_xlim(-0.5, n_questions - 0.5)
    ax.set_ylim(-0.5, len(models) - 0.5)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    return fig, ax

