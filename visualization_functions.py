# ############################################
# IMPORTS
# ############################################
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

    # fig, ax = plt.subplots(figsize=(n_questions * 1.25, len(models) * 3))
    fig, ax = plt.subplots(figsize=(n_questions * .5, len(models)))

    # text_size = 65
    text_size = 20
    
    axtitle = f"Round {round}"
    if 'chat_type' in df.columns:

        if len(df['chat_type'].unique()) > 1:
            Warning('\n More than one chat type is asked to be plotted. Not plotting \n')
            return
        axtitle = axtitle + df['chat_type'].unique()[0]

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
                       rotation=45, ha='right', fontsize=text_size)

    ax.set_yticks(np.arange(len(models_shortname)))
    ax.set_yticklabels(models_shortname, fontsize=text_size)  #  Now uses short model names

    ax.set_title(axtitle, fontsize=text_size, pad=12)
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
    ax: Optional[plt.Axes] = None,
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
    return_fig: bool = True,
    # Parameters for consistent configuration styling
    match_inverted_colors: bool = True,
    inverted_indicator: str = 'inverted',
    # Parameters for legend positioning
    bbox_to_anchor: Optional[tuple] = None,
    ncol: int = 1
) -> Union[plt.Figure, plt.Axes]:
    """
    Creates a plot of mean values by question with error bars, grouped by specified columns.
    Questions are automatically sorted by category first, then by question identifier.
    
    Additional parameters:
    -----------
    match_inverted_colors: bool
        If True, pairs regular and inverted configurations with same colors but different fill
    inverted_indicator: str
        String that identifies inverted variants in label_col values
    bbox_to_anchor: tuple, optional
        Position to place the legend outside the axes
    ncol: int
        Number of columns in the legend
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
    
    # Extract base configurations for color matching if requested
    if match_inverted_colors:
        # Create mapping from label to base configuration
        data = data.copy()  # Avoid modifying the original
        
        # Define proper base config extraction
        def extract_base_config(label):
            base = label
            if 'ous_' in base:
                base = base.replace('ous_', '')
            if '_ring' in base:
                base = base.replace('_ring', '')
            if 'inverted_' in base:
                base = base.replace('inverted_', '')
            return base
        
        # Apply the extraction function
        data['_base_config_'] = data[label_col].apply(extract_base_config)
        data['_is_inverted_'] = data[label_col].apply(
            lambda x: inverted_indicator in x
        )
        
        # Get unique base configurations
        base_configs = sorted(set(data['_base_config_']))
        
        # Use a combination of colormaps for a wider range of distinct colors
        colors_tab10 = plt.cm.tab10.colors
        colors_tab20 = plt.cm.tab20.colors
        colors_tab20b = plt.cm.tab20b.colors
        
        # Combine and get unique colors
        all_colors = []
        all_colors.extend(colors_tab10)
        
        # Add more colors if needed, but ensure they're visually distinct
        if len(base_configs) > len(all_colors):
            for color in colors_tab20:
                if color not in all_colors:
                    all_colors.append(color)
        
        if len(base_configs) > len(all_colors):
            for color in colors_tab20b:
                if color not in all_colors:
                    all_colors.append(color)
        
        # Create color mapping for base configurations
        base_color_map = {}
        for i, config in enumerate(base_configs):
            base_color_map[config] = all_colors[i % len(all_colors)]
        
        # Create a map of labels to colors with proper base config matching
        group_colors = {}
        for label_name in groups:
            # Extract the base configuration directly
            base_config = extract_base_config(label_name)
            group_colors[label_name] = base_color_map[base_config]
    else:
        # Use original color mapping approach
        num_groups = len(groups)
        group_cmap = plt.colormaps[colormap].resampled(num_groups)
        group_colors = {group: group_cmap(i) for i, group in enumerate(groups)}
    
    # Default category colors if not provided
    if category_colors is None:
        category_cmap = plt.colormaps['Set1'].resampled(len(categories))
        category_colors = {cat: category_cmap(i) for i, cat in enumerate(categories)}
    
    # Calculate width based on number of groups
    num_groups = len(groups)
    width = max(min(width_fill_percentage / num_groups, max_width), min_width)
    
    # Create positions for the points (offset from question number index)
    
    positions = {}
    for i, group in enumerate(groups):
        # This creates a slight offset for each group (-0.25, 0, 0.25 for 3 groups)
        offset = (i - (num_groups-1)/2) * width
        positions[group] = [i + offset for i in range(len(question_nums))]
    
    # Create a mapping from question ID to its position in the plot
    question_positions = {q: i for i, q in enumerate(question_nums)}
    
    # Plot the data for each group
    for group in groups:
        # Get the data for this group
        group_data = data[data[label_col] == group]
        
        # Determine if this is an inverted configuration
        is_inverted = inverted_indicator in group if match_inverted_colors else False
        
        # Get color for this group
        color = group_colors[group]
        
        # Plot only the question numbers that exist for this group
        for q_num in question_nums:
            # Find data for this question and group
            q_data = group_data[group_data[question_id_col] == q_num]
            
            # Only plot if we have data
            if not q_data.empty:
                yerr_col = sem_col if use_sem else error_col
                
                # Get the position index for this question
                q_idx = question_positions[q_num]
                
                # Adjust marker appearance based on regular vs inverted status
                if match_inverted_colors and is_inverted:
                    # For inverted: outlined marker with no fill
                    ax.errorbar(
                        positions[group][q_idx],
                        q_data[value_col].values[0],
                        yerr=q_data[yerr_col].values[0],
                        fmt=marker_style,
                        color=color,                # Error bar color
                        markerfacecolor='white',    # White fill
                        markeredgecolor=color,      # Color outline
                        markeredgewidth=1.5,        # Thicker outline
                        label=group if q_idx == 0 else "",  # Only add to legend once
                        capsize=capsize,
                        markersize=6
                    )
                else:
                    # For regular: filled marker
                    ax.errorbar(
                        positions[group][q_idx],
                        q_data[value_col].values[0],
                        yerr=q_data[yerr_col].values[0],
                        fmt=marker_style,
                        color=color,                # Error bar color
                        markerfacecolor=color,      # Filled with color
                        markeredgecolor='black',    # Black outline for contrast
                        markeredgewidth=0.5,
                        label=group if q_idx == 0 else "",  # Only add to legend once
                        capsize=capsize,
                        markersize=6
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
    # remove extra spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add custom vertical grid lines between questions
    for i in range(len(question_nums) + 1):
        ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=grid_alpha, zorder=0)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=grid_alpha+0.1)
    
    # Add category annotations
    y_min, y_max = ax.get_ylim()
    y_text = y_max * 1.05
    
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
    
    # Sort the legend by base configuration for better organization
    if show_legend:
        handles, labels_list = ax.get_legend_handles_labels()
        
        if match_inverted_colors:
            # Group by base configuration
            base_configs_for_labels = []
            is_inverted_list = []
            
            for label in labels_list:
                is_inverted = inverted_indicator in label
                is_inverted_list.append(is_inverted)
                
                # Find the base configuration using the same extraction function
                base_config = extract_base_config(label)
                base_configs_for_labels.append(base_config)
            
            # Sort by base config first, then by inverted status (regular first, then inverted)
            sorted_indices = sorted(range(len(labels_list)), 
                                   key=lambda i: (base_configs_for_labels[i], is_inverted_list[i]))
            
            sorted_handles = [handles[i] for i in sorted_indices]
            sorted_labels = [labels_list[i] for i in sorted_indices]
            
            # Create legend with additional parameters if provided
            if bbox_to_anchor:
                ax.legend(sorted_handles, sorted_labels, 
                         loc=legend_loc, 
                         bbox_to_anchor=bbox_to_anchor, 
                         ncol=ncol)
            else:
                ax.legend(sorted_handles, sorted_labels, loc=legend_loc, ncol=ncol)
        else:
            # Create legend with additional parameters if provided
            if bbox_to_anchor:
                ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol)
            else:
                ax.legend(loc=legend_loc, ncol=ncol)
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)


    # Adjust layout if we created the figure
    if created_figure:
        plt.tight_layout()
        if return_fig:
            return fig
    
    # Return the axes by default if we didn't create a figure,
    # or if return_fig is False
    return ax


def plot_IH_v_IB(df_by_category, use_std=True, label='chat_type', ax_lims = [1,7], text_size = 12, base_colors = None):
    # Plot KDE with models on top
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot human KDE first using your existing function
    human_kde(human_df=h2, ax=ax, alpha=0.7)
    
    # Get unique base configurations
    df = df_by_category.copy()
    
    # Create new columns to help with color mapping
    df['base_config'] = df[label].apply(lambda x: x.lower().replace('ous_', '').replace('_ring', '').replace('inverted_', '').replace('ggb_','').replace('_inverted', ''))
    df['is_inverted'] = df[label].apply(lambda x: 'inverted' in x.lower())

    if base_colors is None:
        # Get unique base configurations
        base_configs = df['base_config'].unique()

        colors_tab10 = plt.cm.tab10.colors
        colors_tab20 = plt.cm.tab20.colors
        colors_tab20b = plt.cm.tab20b.colors
        
        # Combine and get unique colors
        all_colors = []
        all_colors.extend(colors_tab10)
        
        # Add more colors if needed, but ensure they're visually distinct
        if len(base_configs) > len(all_colors):
            for color in colors_tab20:
                if color not in all_colors:
                    all_colors.append(color)
        
        if len(base_configs) > len(all_colors):
            for color in colors_tab20b:
                if color not in all_colors:
                    all_colors.append(color)
        
        # Create color mapping for base configurations
        base_colors = {}
        for i, config in enumerate(base_configs):
            base_colors[config] = all_colors[i % len(all_colors)]
    
    # Create a map of labels to colors    
    labels_for_colors = df[label].unique()
    color_map = {}
    for label_name in labels_for_colors:
        # Extract base configuration
        base_config = label_name.lower().replace('ous_', '').replace('_ring', '').replace('inverted_', '').replace('ggb_','').replace('_inverted', '')
        # Get base color
        color_map[label_name] = base_colors[base_config]
    
    # Add color to the dataframe
    df['color'] = df[label].map(color_map)
    
    # Get unique labels
    labels = df[label].unique()

    # For each model, get data for both categories and plot
    for this_label in labels:
        # Get IH data for this model
        ih_data = df[(df[label] == this_label) & 
                     (df['category'] == 'IH')]
        
        # Get IB data for this model
        ib_data = df[(df[label] == this_label) & 
                     (df['category'] == 'IB')]
        
        # If we have both IH and IB data for this model
        if not ih_data.empty and not ib_data.empty:
            ih_mean = ih_data['mean'].values[0]
            if use_std:
                ih_sem = ih_data['std'].values[0]
            else: 
                ih_sem = ih_data['sem'].values[0]
            ib_mean = ib_data['mean'].values[0]
            if use_std:
                ib_sem = ib_data['std'].values[0]
            else:
                ib_sem = ib_data['sem'].values[0]
            
            # Get color from our mapping
            color = color_map[this_label]
            
            # Use the same marker outline color but different fill based on inverted status
            is_inverted = 'inverted' in this_label
            if is_inverted:
                # For inverted: outlined marker with no fill
                ax.errorbar(
                    ih_mean, ib_mean,
                    xerr=ih_sem, yerr=ib_sem, 
                    fmt='o', color=color, 
                    markerfacecolor='white',  # White fill
                    markeredgecolor=color,    # Color outline
                    markeredgewidth=1.5,      # Thicker outline
                    label=this_label, capsize=3
                )
            else:
                # For regular: filled marker
                ax.errorbar(
                    ih_mean, ib_mean,
                    xerr=ih_sem, yerr=ib_sem, 
                    fmt='o', color=color, 
                    markerfacecolor=color,    # Filled with color
                    markeredgecolor=color,
                    label=this_label, capsize=3
                )
    
    # Set axis labels and limits
    ax.set_xlabel('Insturmental Harm', fontsize = text_size)
    ax.set_ylabel('Impartial Beneficence', fontsize = text_size)
    ax.set_xlim(ax_lims[0], ax_lims[1])
    ax.set_ylim(ax_lims[0], ax_lims[1])
    ax.tick_params(axis='both', which='major', labelsize=text_size)  # Adjust size a
    
    # Sort legend for better organization
    handles, labels_list = ax.get_legend_handles_labels()
    
    # Group by base configuration
    label_base_configs = [l.replace('ous_', '').replace('_ring', '').replace('inverted_', '') for l in labels_list]
    is_inverted_list = ['inverted' in l for l in labels_list]
    
    # Sort by base config first, then by inverted status (regular first, then inverted)
    sorted_indices = sorted(range(len(labels_list)), 
                           key=lambda i: (label_base_configs[i], is_inverted_list[i]))
    
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels_list[i] for i in sorted_indices]
      
    # Create a more compact legend with multiple columns
    n_columns = 2
    if len(sorted_labels) > 8:
        n_columns = 3  # Use 3 columns for many items
        
    ax.legend(sorted_handles, sorted_labels,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15), 
              ncol=n_columns,
              frameon=True,
              fontsize=text_size)
    
    plt.tight_layout()
    plt.show()
    
    # Return both the figure and the dataframe with color information
    return fig, df

################################
# CLEANUP FOR PAPER FIGURES: 
################################

# FOR AIES: 
col_width = 3.3125 # inches
text_wdith = 7.0 # inches

def cleanup_IBvIH_plot(f,marker_size=4, font_size=9, legend_labels = None, col_width = col_width):
    ax = f.axes[0]  # Get the main axes
    # ax = axs[0]

    #markersize
    for line in ax.get_lines():
        line.set_markersize(marker_size)  # Adjust number to desired size

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ensure square axes with same scale
    ax.axis('square')

    # Set explicit limits to match your data (adjust these values as needed)
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)

    # Adjust figure size - increase height while maintaining width
    f.set_size_inches(col_width, col_width * 1.4)  # 1.3 ratio gives room for legend

    # Get handles and labels
    handles, old_labels = ax.get_legend_handles_labels()

    # Create new labels (if you haven't already)
    new_labels = []
    if legend_labels is None:
        for label in old_labels:
            if "GGB_inverted_" in label:
                model_name = label.replace("GGB_inverted_", "").capitalize()
                new_labels.append(f"{model_name} (inverted)")
            elif "GGB_" in label:
                model_name = label.replace("GGB_", "").capitalize()
                new_labels.append(model_name)
            elif 'inverted_' in label:
                model_name = label.replace("inverted_", "").capitalize()
                new_labels.append(f"{model_name} (inverted)")
            elif '_inverted' in label:
                model_name = label.replace("_inverted", "").capitalize()
                new_labels.append(f"{model_name} (inverted)")
            else:
                new_labels.append(label.capitalize())
    else:
        for label in old_labels:
            new_labels.append(legend_labels[label])
    
                                  

    # reorganize so that we have the inverted separate in the legend
    # For legend at the bottom 
    label_inds = [x for x, l in enumerate(new_labels) if 'inverted' in l]
    inverted_inds = [x for x,l in enumerate(new_labels) if 'inverted' not in l]
    new_handles = [handles[x] for x in inverted_inds + label_inds]
    new_new_labels = [new_labels[x] for x in inverted_inds + label_inds]

    ax.legend(new_handles, new_new_labels, 
                ncol=1, #2,                    # 2 columns as requested
                loc= 'center left',  #'upper center',        # Align by upper center
                bbox_to_anchor= (.85, 0.5), #(0.5, -0.16), # Position below axes (x=center, y=below)
                frameon=False, 
                fontsize=font_size)              # Keep the frame

    # Get current ticks from both axes
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Use the same set of ticks for both axes
    # (can choose either set, or merge them)
    all_ticks = [1,2,3,4,5,6,7]
    ax.set_xticks(all_ticks)
    ax.set_yticks(all_ticks)

    ax.set_xlabel('Insturmental Harm', fontsize = font_size)
    ax.set_ylabel('Impartial Beneficence', fontsize = font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size) 

    # Reduce whitespace by adjusting margins
    # f.subplots_adjust(left=0.1, right=0.95, top=1, bottom=0.3) # for bottom legend
    f.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1) # for rigth legend
    return f

#For help with colormaps:
def get_base_colors(df_in, ending_base = None):
    df = df_in.copy()
    df['base_config'] = df['label'].apply(lambda x: x.lower().replace('ous_', '').replace('_ring', '').replace('inverted_', '').replace('ggb_','').replace('_inverted', ''))
    base_labels = np.sort(df['base_config'].unique())

    if ending_base:
        is_ending_base = [ending_base in x for x in base_labels]
        base_labels = np.append(base_labels[np.invert(is_ending_base)], base_labels[is_ending_base])

    colors = ['darkred', 'darkorange','teal', 'olivedrab', 'deepskyblue', 'darkblue', 'deeppink']
    base_colors = dict(zip(base_labels, colors[:len(base_labels)]))
    return base_colors