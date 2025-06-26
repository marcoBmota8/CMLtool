import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.neighbors import KernelDensity

# TODO modify this function so that it takes a list of arrays and its corresponding list of string names
# It would make it easier to pass different sets of data of different lengths. The current approach of a dataframe
# as input requires to transform each array into a pandas series before make it a dataframe column so that NaN padding 
# is applied to all arrays. This is not necessary if we just pass a list of arrays.

def kde_rugplot_multivar(
    df,
    kernel='gaussian', 
    bandwidth=0.75,
    ext_perc=10,
    figsize=(8, 6),
    kde_alpha=1,
    kde_linewidth=2,
    rug_height=0.5,
    rug_alpha=0.33,
    rug_lw=1.5,
    palette=None,
    ylabel='KDE Density',
    xlabel=None,
    legend_fontsize=15,
    label_fontsize=18,
    kde_ylabel_fontsize=22,
    sharex=True,
    show=False
):
    """
    Plots KDEs for each column in the dataframe on a shared axis,
    and a rugplot for each variable on a separate horizontal axis.
    Supports 'gaussian' and 'tophat' (Kirchhoff) kernels.
    """
    n_vars = df.shape[1]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_vars + 1, 1, height_ratios=[8] + [0.25]*n_vars, hspace=0)
    ax_kde = fig.add_subplot(gs[0])

    if palette is None:
        palette = sns.color_palette(n_colors=n_vars)
        palette = {col: palette[i] for i, col in enumerate(df.columns)}

    data_min = df.min().min()
    data_max = df.max().max()
    if ext_perc is not None:
        data_range = data_max - data_min
        data_min = data_min - ext_perc/100 * data_range
        data_max = data_max + ext_perc/100 * data_range
        
    for i, col in enumerate(df.columns):
        color = palette.get(col, None)
        data = df[col].dropna().values[:, np.newaxis]
        x_grid = np.linspace(data_min, data_max, 500)[:, np.newaxis]

        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
        log_density = kde.score_samples(x_grid)
        ax_kde.plot(
            x_grid[:, 0], np.exp(log_density),
            label=col,
            color=color,
            alpha=kde_alpha,
            linewidth=kde_linewidth,
        )

        ax_rug = fig.add_subplot(gs[i+1], sharex=ax_kde if sharex else None)
        sns.rugplot(
            df[col].dropna(),
            ax=ax_rug,
            color=color,
            height=rug_height,
            alpha=rug_alpha,
            lw=rug_lw,
            clip_on=False
        )
        ax_rug.set_yticks([])
        ax_rug.set_ylabel('')
        ax_rug.tick_params(axis='y', left=False, labelleft=False)
        ax_rug.spines['top'].set_visible(False)
        ax_rug.spines['right'].set_visible(True)
        ax_rug.spines['left'].set_visible(True)
        
        if i == 0:
            ax_rug.spines['top'].set_visible(True)
        else:
            ax_rug.spines['top'].set_visible(False)
        
        if i == n_vars - 1:
            ax_rug.spines['bottom'].set_visible(True)
        else:
            ax_rug.spines['bottom'].set_visible(False)

        if i < n_vars-1:
            ax_rug.tick_params(axis='x', which='both', labelbottom=False, bottom=False, labelleft=False, left=False)
            ax_rug.set_xlabel('')
        else:
            ax_rug.tick_params(axis='x', which='both', labelsize=label_fontsize)
            ax_rug.tick_params(axis='y', which='both', labelbottom=False, bottom=False, labelleft=False, left=False)
            ax_rug.set_xlabel(xlabel, fontsize=label_fontsize)

    ax_kde.set_ylim([0, ax_kde.get_ylim()[1]])
    ax_kde.tick_params(axis='both', which='both', bottom=False, labelbottom=False, labelsize=label_fontsize)
    ax_kde.set_ylabel(ylabel, fontsize=kde_ylabel_fontsize)
    ax_kde.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig
