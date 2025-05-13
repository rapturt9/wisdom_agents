# ############################################
# IMPORTS
# ############################################
# from typing import Literal
# import os
# import json
# import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.patches import Rectangle

from scipy.stats import gaussian_kde

from math import isnan
from typing import List, Union, Dict, Optional, Any

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



# ############################################
# PLOT EACH QUESTION BY GROUPING (eg model or whatever)
# ############################################
def plot_by_question(
    data: pd.DataFrame,
    group_by: Union[str, List[str]] = 'model_name',
    question_id_col: str = 'question_num',
    category_col: str = 'category',
    value_col: str = 'mean',
    error_col: str = 'std',
    label_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,  # Add axis parameter for subplot support
    figsize: tuple = (15, 6),
    colormap: str = 'tab10',
    title: Optional[str] = 'Mean Values by Question and Group',
    grid_alpha: float = 0.3,
    marker_style: str = 'o',
    capsize: int = 3,
    rotate_xticks: int = 90,
    category_colors: Optional[Dict[str, str]] = None,
    category_order: Optional[List[str]] = None,
    use_sem: bool = False,
    sem_col: str = 'sem',
    show_legend: bool = True,
    legend_loc: str = 'best',
    y_label: str = 'Mean Value',
    x_label: str = 'Question',
    width_fill_percentage: float = 0.8,
    min_width: float = 0.05,
    max_width: float = 0.4,
    sort_by_category: bool = True,
    return_fig: bool = True  # Control what's returned
) -> Union[plt.Figure, plt.Axes]:
    """
    Creates a plot of mean values by question with error bars, grouped by specified columns.
    Questions are automatically sorted by category first, then by question identifier.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The DataFrame containing the aggregated data to plot
    group_by : str or list of str
        Column(s) to group by (e.g., 'model_name' or ['model_name', 'run_type'])
    question_id_col : str
        Column containing the question identifiers
    category_col : str
        Column that defines categories for coloring and grouping
    value_col : str
        Column containing the mean values to plot
    error_col : str
        Column containing standard deviation for error bars
    label_col : str, optional
        Column to use for labels in the legend. If None, will construct from group_by
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates a new figure and axes.
    figsize : tuple
        Figure size as (width, height) in inches (used only if ax is None)
    colormap : str
        Name of the matplotlib colormap to use
    title : str, optional
        Plot title
    grid_alpha : float
        Transparency for grid lines
    marker_style : str
        Marker style for data points
    capsize : int
        Size of caps on error bars
    rotate_xticks : int
        Rotation angle for x-tick labels
    category_colors : dict, optional
        Mapping of category values to colors
    category_order : list, optional
        Order of categories (e.g., ['IH', 'IB']). If None, sorts alphabetically.
    use_sem : bool
        If True, use standard error of mean instead of standard deviation
    sem_col : str
        Column name for standard error of mean (if use_sem is True)
    show_legend : bool
        Whether to display the legend
    legend_loc : str
        Location for the legend
    y_label : str
        Label for y-axis
    x_label : str
        Label for x-axis
    width_fill_percentage, min_width, max_width : float
        Parameters controlling the width calculation for group offsets
    sort_by_category : bool
        If True, sort questions by category first, then by question ID
    return_fig : bool
        If True, returns the figure object. If False, returns just the axes object.
        Only relevant when ax=None (creating a new figure).
        
    Returns:
    --------
    matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes object depending on return_fig parameter
    """
    # Create new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_figure = True
    else:
        fig = ax.figure
        created_figure = False
    
    # Ensure group_by is a list
    if isinstance(group_by, str):
        group_by = [group_by]
        
    # Create a label column if not specified
    if label_col is None:
        # Create a temporary label from the group_by columns
        data = data.copy()  # Avoid modifying the original DataFrame
        data['_temp_label_'] = data[group_by].apply(
            lambda row: '_'.join(str(row[col]) for col in group_by), axis=1
        )
        label_col = '_temp_label_'
    
    # Get unique categories
    if category_order is None:
        categories = sorted(data[category_col].unique())
    else:
        categories = category_order
    
    # Create a sorted list of questions
    if sort_by_category:
        # First sort by category, then by question ID within category
        sorted_questions = []
        for cat in categories:
            cat_data = data[data[category_col] == cat]
            cat_questions = sorted(cat_data[question_id_col].unique())
            sorted_questions.extend(cat_questions)
        # Remove duplicates while preserving order
        question_nums = []
        [question_nums.append(q) for q in sorted_questions if q not in question_nums]
    else:
        # Just sort by question ID
        question_nums = sorted(data[question_id_col].unique())
    
    # Get unique groups
    groups = sorted(data[label_col].unique())
    num_groups = len(groups)
    
    # Calculate width based on number of groups
    width = max(min(width_fill_percentage / num_groups, max_width), min_width)
    
    # Define colors for groups
    # if hasattr(plt.cm, 'get_cmap'):  # For backwards compatibility with older matplotlib
    #     group_cmap = plt.cm.get_cmap(colormap, num_groups)
    # else:  # For matplotlib 3.7+
    group_cmap = plt.colormaps[colormap].resampled(num_groups)
    group_colors = {group: group_cmap(i) for i, group in enumerate(groups)}
    
    # Default category colors if not provided
    if category_colors is None:
        # if hasattr(plt.cm, 'get_cmap'):  # For backwards compatibility with older matplotlib
        #     category_cmap = plt.cm.get_cmap('Set1', len(categories))
        # else:  # For matplotlib 3.7+
        category_cmap = plt.colormaps['Set1'].resampled(len(categories))

        category_colors = {cat: category_cmap(i) for i, cat in enumerate(categories)}
    
    # Create positions for the points (offset from question number index)
    # Note: We're using the index in question_nums, not the actual question ID value
    positions = {}
    for i, group in enumerate(groups):
        # This creates a slight offset for each group (-0.25, 0, 0.25 for 3 groups)
        offset = (i - (num_groups-1)/2) * width
        positions[group] = [i + offset for i in range(len(question_nums))]
    
    # Create a mapping from question ID to its position in the plot
    question_positions = {q: i for i, q in enumerate(question_nums)}
    
    # Plot the data for each group
    for i, group in enumerate(groups):
        # Get the data for this group
        group_data = data[data[label_col] == group]
        
        # Plot only the question numbers that exist for this group
        for q_num in question_nums:
            # Find data for this question and group
            q_data = group_data[group_data[question_id_col] == q_num]
            
            # Only plot if we have data
            if not q_data.empty:
                yerr_col = sem_col if use_sem else error_col
                
                # Get the position index for this question
                q_idx = question_positions[q_num]
                
                ax.errorbar(
                    positions[group][q_idx],  # Use the precomputed position
                    q_data[value_col].values[0],
                    yerr=q_data[yerr_col].values[0],
                    fmt=marker_style,  # Use circles as markers
                    color=group_colors[group],
                    label=group if q_idx == 0 else "",  # Only add to legend once
                    capsize=capsize,  # Add caps to error bars
                    markersize=6,
                    markeredgecolor='black',
                    markeredgewidth=0.5
                )
    
    # Set x-ticks at question positions (not the offset positions)
    ax.set_xticks(range(len(question_nums)))
    ax.set_xticklabels(question_nums, rotation=rotate_xticks)
    
    # Color the tick labels based on category
    for idx, question in enumerate(question_nums):
        tick = ax.get_xticklabels()[idx]
        
        # Find the category for this question
        q_data = data[data[question_id_col] == question]
        if not q_data.empty:
            q_category = q_data[category_col].iloc[0]
            tick.set_color(category_colors.get(q_category, 'black'))
    
    # Remove default grid
    ax.grid(False)
    
    # Add custom vertical grid lines between questions
    for i in range(len(question_nums) + 1):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=grid_alpha, zorder=0)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=grid_alpha+0.1)
    
    # Add category annotations
    y_min, y_max = ax.get_ylim()
    y_text = y_max * 0.95
    
    # Add category separators
    if sort_by_category:
        # Find boundaries between categories
        cat_boundaries = []
        prev_cat = None
        
        for idx, q in enumerate(question_nums):
            q_data = data[data[question_id_col] == q]
            if not q_data.empty:
                curr_cat = q_data[category_col].iloc[0]
                if prev_cat is not None and curr_cat != prev_cat:
                    cat_boundaries.append(idx - 0.5)  # Boundary is between indices
                prev_cat = curr_cat
        
        # Draw category separator lines
        for boundary in cat_boundaries:
            ax.axvline(x=boundary, color='black', linestyle='--', 
                      alpha=0.7, linewidth=1.5, zorder=5)
        
        # Add category labels in the middle of each category section
        cat_indices = {}
        for cat in categories:
            cat_indices[cat] = []
        
        # Collect indices for each category
        for idx, q in enumerate(question_nums):
            q_data = data[data[question_id_col] == q]
            if not q_data.empty:
                q_cat = q_data[category_col].iloc[0]
                cat_indices[q_cat].append(idx)
        
        # Add labels at the middle of each category section
        for cat, indices in cat_indices.items():
            if indices:  # If this category has questions
                ax.text(np.mean(indices), y_text, f'{cat} Category', 
                        color=category_colors[cat], ha='center', fontsize=12)
    
    # Add labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    
    if show_legend:
        ax.legend(loc=legend_loc)
    
    # Adjust layout if we created the figure
    if created_figure:
        plt.tight_layout()
        if return_fig:
            return fig
    
    # Return the axes by default if we didn't create a figure,
    # or if return_fig is False
    return ax