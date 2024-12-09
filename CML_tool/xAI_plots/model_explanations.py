
import logging 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity


def plot_shap_ridgelines(shaps, data_rep, all_features_names, features_name_map, title, xmax, bandwidth, signs=None, npoints=None,
                         top:int=10, overlap: float=0.5, figsize:tuple=(4,2), return_all=False, color_kde='C0', color_rugs='C1'):
    
    # check if passed shaps are absolute
    if  all(all(val >= 0 for val in row) for row in shaps):
        logging.info(f'All shapley values are positive. It is highly likely what was passed is abs(SHAP).')

    if isinstance(all_features_names, list):
        all_features_names = np.array(all_features_names)
    
    assert isinstance(shaps, np.ndarray), "shapley values array must be a NumPy array."    
    assert shaps.ndim == 2, "Shapley values array must be a 2D array."
    assert isinstance(all_features_names, np.ndarray), "`all_features_names` must be a NumPy array or list of strings."
    
    if npoints is None:
        npoints=int(xmax/bandwidth)
    
    shap_means = np.mean(shaps, axis=0) # Mean shap value for each feature
    top_idx = list(np.argsort(abs(shap_means))[::-1][:top]) # The indices for the top features based on absolute mean shapley value
    
    X_top = shaps[:,top_idx].T # Shap values for the top features

    if data_rep.lower() == 'signatures':
        labels = [f'[{x}]: '+features_name_map.get(x, 'Unknown') for x in all_features_names[top_idx]]
    elif data_rep.lower() == 'channels':
        labels = [features_name_map.get(x,'Unknown') if x in all_features_names[top_idx] else x for x in all_features_names[top_idx]]
    else:
        raise ValueError(f'Data representation `{data_rep}` is not supported.')
    
    # Add signs
    if signs is not None:
        signs_labels = ['+' if s > 0 else '-' if s < 0 else '0' for s in signs[top_idx]] # Whether the mean shap across patients is protective or inducive
        labels = [l+ ' ' + '(' + signs_labels[i] + ')' for i,l in enumerate(labels)]
        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=top, ncols=1)

    x_grid = np.linspace(0.0, xmax, npoints)

    axes = []
    curves = []

    for i, row in enumerate(X_top):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(row[:, None])
        logprob = kde.score_samples(x_grid[:,None])
        curve = np.exp(logprob)
        curves.append(curve)
        ax = fig.add_subplot(gs[i:i+1, :])
        axes.append(ax)

        ax.fill_between(x_grid, curve, color=color_kde, alpha=0.25, linewidth=0)
        ax.plot(x_grid, curve, color=color_kde, linewidth=1.0, alpha=0.75)
        ax.scatter(row, np.zeros(len(row)), marker=3, color=color_rugs, alpha=0.15)

        ax.set_xlim(0.0, xmax)
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_yticks([])
        ax.set_yticklabels([])
        
        if i+1 < top:
            ax.set_xticks([])
            ax.set_xticklabels([])

        else:
            ax.set_xlabel(r'Mean Absolute Shapley Value')
            

        ax.text(0.0, 1.0, labels[i],
                horizontalalignment='right',
                verticalalignment='center')
        
        plt.tick_params(axis="x", which="both", top=False)


    ymin = np.inf
    ymax = -np.inf
    for ax in axes:
        _ymin, _ymax = ax.get_ylim()
        ymin = min(_ymin, ymin)
        ymax = max(_ymax, ymax)

    for ax in axes:
        ax.set_ylim([ymin, ymax])

    gs.update(hspace=-overlap)
    gs.update(top=1)
    fig.suptitle(title)
    
    if return_all:
        return fig, axes, curves
    else:
        return fig