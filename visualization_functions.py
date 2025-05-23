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
    
    # Input validation
    if data is None or data.empty:
        print(f"Warning: No data provided to plot_by_question")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or 'No Data Available')
            return fig if return_fig else ax
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return ax
    
    # Check required columns
    required_cols = [group_by, question_id_col, value_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'Missing columns: {missing_cols}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or 'Data Error')
            return fig if return_fig else ax
        else:
            ax.text(0.5, 0.5, f'Missing columns: {missing_cols}', ha='center', va='center', transform=ax.transAxes)
            return ax
    
    # Filter out rows with NaN values in critical columns
    data_clean = data.dropna(subset=[group_by, question_id_col, value_col])
    
    if data_clean.empty:
        print(f"Warning: No valid data after removing NaN values")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No valid data after filtering', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title or 'No Valid Data')
            return fig if return_fig else ax
        else:
            ax.text(0.5, 0.5, 'No valid data after filtering', ha='center', va='center', transform=ax.transAxes)
            return ax
    
    # Use label_col if provided, otherwise use group_by
    label_column = label_col if label_col else group_by
    
    # Get categories (create order if needed)
    if sort_by_category and category_col in data_clean.columns:
        if category_order:
            categories = category_order
        else:
            categories = sorted(data_clean[category_col].unique())
    else:
        categories = ['all']  # Single category if not sorting by category
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()
        created_fig = False
    
    # Prepare data for plotting
    plotting_data = []
    for _, row in data_clean.iterrows():
        plotting_data.append({
            'group': row[group_by],
            'question_id': row[question_id_col],
            'category': row.get(category_col, 'all'),
            'value': row[value_col],
            'error': row.get(error_col, 0) if not use_sem else row.get(sem_col, 0),
            'label': row.get(label_column, row[group_by])
        })
    
    plot_df = pd.DataFrame(plotting_data)
    
    # Sort questions by category then by question_id
    if sort_by_category and category_col in data_clean.columns:
        # Create a categorical ordering
        plot_df['category_cat'] = pd.Categorical(plot_df['category'], categories=categories, ordered=True)
        plot_df = plot_df.sort_values(['category_cat', 'question_id'])
    else:
        plot_df = plot_df.sort_values('question_id')
    
    # Get unique questions in order
    unique_questions = plot_df['question_id'].unique()
    x_positions = np.arange(len(unique_questions))
    
    # Get unique groups/labels
    groups = plot_df['group'].unique()
    labels = plot_df['label'].unique()
    
    # Calculate bar positions and widths
    n_groups = len(groups)
    total_width = width_fill_percentage
    bar_width = max(min_width, min(max_width, total_width / n_groups))
    
    # Create color scheme
    if match_inverted_colors and label_col:
        # Group labels by base configuration
        base_configs = {}
        for label in labels:
            base = label.lower().replace(inverted_indicator.lower(), '').strip('_')
            if base not in base_configs:
                base_configs[base] = []
            base_configs[base].append(label)
        
        # Assign colors to base configurations
        base_colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(base_configs)))
        color_map = {}
        for i, (base, label_list) in enumerate(base_configs.items()):
            for label in label_list:
                color_map[label] = base_colors[i]
    else:
        # Simple color mapping
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(groups)))
        color_map = {group: colors[i] for i, group in enumerate(groups)}
    
    # Plot bars for each group
    for i, group in enumerate(groups):
        group_data = plot_df[plot_df['group'] == group]
        
        # Get positions for this group's bars
        bar_positions = x_positions + (i - (n_groups - 1) / 2) * bar_width
        
        # Get values and errors for each question
        values = []
        errors = []
        for question in unique_questions:
            question_data = group_data[group_data['question_id'] == question]
            if not question_data.empty:
                values.append(question_data['value'].iloc[0])
                errors.append(question_data['error'].iloc[0])
            else:
                values.append(0)
                errors.append(0)
        
        # Get label and color
        label = group_data['label'].iloc[0] if not group_data.empty else group
        color = color_map.get(label, color_map.get(group, 'blue'))
        
        # Determine marker style for inverted configurations
        if match_inverted_colors and inverted_indicator.lower() in label.lower():
            markerfacecolor = 'white'
            markeredgecolor = color
            alpha = 0.7
        else:
            markerfacecolor = color
            markeredgecolor = color
            alpha = 1.0
        
        # Plot error bars
        ax.errorbar(bar_positions, values, yerr=errors,
                   fmt=marker_style, color=color,
                   markerfacecolor=markerfacecolor,
                   markeredgecolor=markeredgecolor,
                   alpha=alpha, capsize=capsize,
                   label=label)
    
    # Customize plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Q{q}" for q in unique_questions], rotation=rotate_xticks)
    ax.grid(True, alpha=grid_alpha)
    
    # Add legend
    if show_legend:
        if bbox_to_anchor:
            ax.legend(bbox_to_anchor=bbox_to_anchor, ncol=ncol, loc=legend_loc)
        else:
            ax.legend(ncol=ncol, loc=legend_loc)
    
    plt.tight_layout()
    
    return fig if (return_fig and created_fig) else ax


def plot_IH_v_IB(df_by_category, use_std=True, label='chat_type', ax_lims = [1,7], text_size = 12):
    # Plot KDE with models on top
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot human KDE first using your existing function
    human_kde(human_df=h2, ax=ax, alpha=0.7)
    
    # Get unique base configurations
    df = df_by_category.copy()
    
    # Create new columns to help with color mapping
    df['base_config'] = df[label].apply(lambda x: x.lower().replace('ous_', '').replace('_ring', '').replace('inverted_', '').replace('ggb_','').replace('_inverted', ''))
    df['is_inverted'] = df[label].apply(lambda x: 'inverted' in x.lower())
    
    # Get unique base configurations
    base_configs = df['base_config'].unique()

    # Use a combination of colormaps for a wider range of distinct colors
    # Start with tab10, then tab20, then tab20b for even more variations
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
    
    # Before coloring, change the order in base_configs
    gemini_bool =  ['gemini' in n.lower() for n in base_configs]
    mixed_bool = ['mixed' in n.lower() for n in base_configs]
    
    if not any(gemini_bool) and not any(mixed_bool) < 1:
        pass
    elif any(gemini_bool) and any(mixed_bool):
        Warning('Both Gemini and Mixed persent, not reordering coloring')
    elif any(gemini_bool):
        gemini_label = base_configs[gemini_bool]
        base_no_gemini = base_configs[np.invert(gemini_bool)]
        base_configs = np.append(base_no_gemini, gemini_label)
    elif any(mixed_bool):
        mix_label = base_configs[mixed_bool]
        base_no_mixed = base_configs[np.invert(mixed_bool)]
        base_configs = np.append(base_no_mixed, mix_label)
        
    # Create color mapping for base configurations
    base_colors = {}
    for i, config in enumerate(base_configs):
        base_colors[config] = all_colors[i % len(all_colors)]
    
    # Create a map of labels to colors
    # make sure gemini/hetero is at the end for consistent coloring
    
    color_map = {}
    for label_name in df[label].unique():
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

def cleanup_IBvIH_plot(f,marker_size=4, font_size=9, col_width = col_width):
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

    # If Gemini or Mixed is there - make it the last entry
    gemini_idx = [i for i,l in enumerate(old_labels) if 'gemini' in l.lower()]
    mixed_idx = [i for i,l in enumerate(old_labels) if 'mixed' in l.lower()]
    if len(gemini_idx) == 0 and len(mixed_idx) == 0:
        Warning('Not reordering legend. Both Gemini and Mixed Present')
    elif len(gemini_idx) > 0:
        for i, idx in enumerate(gemini_idx):
            idx = idx - i
            h = handles.pop(idx)
            l = old_labels.pop(idx)
            handles.append(h)
            old_labels.append(l)        
    elif len(mixed_idx) > 0:
       for i, idx in enumerate(mixed_idx):
            h = handles.pop(idx-i)
            l = handles.pop(idx-i)
            old_labels.append(h)
            old_labels.append(l)      


    # Create new labels (if you haven't already)
    new_labels = []
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





    # reorganize so that we have the inverted separate in the legend
    # For legend at the bottom 
    label_inds = [x for x, l in enumerate(new_labels) if 'inverted' in l]
    inverted_inds = [x for x,l in enumerate(new_labels) if 'inverted' not in l]
    new_handles = [handles[x] for x in inverted_inds + label_inds]
    new_new_labels = [new_labels[x] for x in inverted_inds + label_inds]

    ax.legend(new_handles, new_new_labels, 
                ncol=1, #2,                    # 2 columns as requested
                loc= 'center left',  #'upper center',        # Align by upper center
                bbox_to_anchor= (.9, 0.5), #(0.5, -0.16), # Position below axes (x=center, y=below)
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