
import logging 

import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

from CML_tool.ML_Utils import compute_1dkde_curve

def plot_shap_ridgelines(shaps, data_rep, all_features_names, features_name_map, title, xmax, bandwidth, signs=None, npoints=None,
                         top:int=10, min_shap:float=1e-5, overlap: float=0.5, figsize:tuple=(4,2), fontsize:float=12, return_all=False, color_kde='C0', color_rugs='C1'):
    
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
    try:
        top=np.minimum(top, np.argwhere(np.sort(abs(shap_means))[::-1]>=min_shap).flatten()[-1]) 
    except:
        pass
    top=np.maximum(top,10) # ensure minimum 10 features
    top_idx = list(np.argsort(abs(shap_means))[::-1][:top]) # The indices for the top features based on absolute mean shapley value
    
    X_top = shaps[:,top_idx].T # Shap values for the top features

    if data_rep.lower() == 'signatures':
        labels = [f'[{x}]:'+features_name_map.get(x, 'Unknown') for x in all_features_names[top_idx]]
    elif data_rep.lower() == 'channels':
        # Returns strings with replacements according to the mapping otherwise retunrs original (map input string)
        labels = [reduce(lambda x, kv: x.replace(*kv), features_name_map.items(), label) for label in all_features_names[top_idx]]
    else:
        raise ValueError(f'Data representation `{data_rep}` is not supported.')
    
    # Add signs if provided
    if signs is not None:
        signs_labels = ['+' if s > 0 else '--' if s < 0 else '0' for s in signs[top_idx]] # Whether the mean shap across patients is protective or inducive
        labels = [l+' (' + signs_labels[i] + ')' for i,l in enumerate(labels)]
        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=top, ncols=1)

    x_grid = np.linspace(0.0, xmax, npoints)

    axes = []
    curves = []

    for i, row in enumerate(X_top):
        
        # Compute KDE curve
        curve = compute_1dkde_curve(
            x=row[:,None],
            x_grid=x_grid[:,None],
            bandwidth=bandwidth,
            kernel='gaussian'
        )
        curves.append(curve)
        
        # Plotting
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
            ax.set_xlabel(r'Mean Absolute Shapley Value', fontsize=fontsize)
            ax.tick_params(axis="x", which="both", top=False, labelsize=fontsize)

        ax.text(-0.025,0.1, labels[i],
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize,
                transform=ax.transAxes)
        
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
    
def plot_comparing_shap_ridgelines(shaps_ref, shaps_comp, data_rep, all_features_names, features_name_map, 
                                   title, xmax, bandwidth, label_ref, label_comp,
                                   features_chars=None,
                                   return_all=False,
                                   order:list=None, npoints=None, top:int=10, min_shap:float=1e-5, overlap: float=0.5,
                                   figsize:tuple=(4,2), fontsize:float=12, 
                                   color_kde_ref='royalblue', color_rugs_ref='slateblue', color_kde_comp='darkviolet', color_rugs_comp='mediumorchid'):
    
    '''
    This function plots ridgeline plots for the same variables for two datasets.
    The first (shaps_ref) is taken as reference. If no order is passed as, then its data dictates 
    which top variables are plotted through mean sorting. if an or
    The second (shaps_comp) is used to compare agains the reference.
    The data distribution (rugplot and KDE) of both are plotted on the same axis for the same variables. 
    
    Important: It assumes that the dimensionality for both `shaps_ref` and `shaps_comp` are the same and are ordered in the same way.
    '''

    if isinstance(all_features_names, list):
        all_features_names = np.array(all_features_names)
    
    assert (isinstance(shaps_ref, np.ndarray) & isinstance(shaps_comp, np.ndarray)), "shapley values array must be a NumPy array."    
    assert ((shaps_ref.ndim == 2) & (shaps_comp.ndim == 2)), "Shapley values array must be a 2D array."
    assert isinstance(all_features_names, np.ndarray), "`all_features_names` must be a NumPy array or list of strings."
    assert np.shape(all_features_names)==np.shape(features_chars) or features_chars is None, f"`all_features_names` and `features_chars` must have the same shape. {all_features_names.shape} != {np.shape(features_chars)}"
    assert isinstance(order, list) or order is None, 'The order indices `order` must be a list of integers. Probably an array was passed.'
    
    if npoints is None:
        npoints=int(xmax/bandwidth)
    
    shap_means = np.mean(shaps_ref, axis=0) # Mean shap value for each feature in the reference dataset
    try:
        top=np.minimum(top, np.argwhere(np.sort(abs(shap_means))[::-1]>=min_shap).flatten()[-1]) 
    except:
        pass
    top=np.maximum(top,10) # ensure minimum 10 features
    top_idx = list(np.argsort(abs(shap_means))[::-1][:top]) if order is None else order[:top] # The indices for the top features based on absolute mean shapley value
    
    X_top_ref = shaps_ref[:,top_idx].T # Reference shap values for the top features
    X_top_comp = shaps_comp[:,top_idx].T # Comparison shap values for the top features

    if data_rep.lower() == 'signatures':
        labels = [f'[{x}]:'+features_name_map.get(x, 'Unknown') for x in all_features_names[top_idx]]
    elif data_rep.lower() == 'channels':
        # Returns strings with replacements according to the mapping otherwise retunrs original (map input string)
        labels = [reduce(lambda x, kv: x.replace(*kv), features_name_map.items(), l) for l in all_features_names[top_idx]]
    else:
        raise ValueError(f'Data representation `{data_rep}` is not supported.')
    
    # If passed, add feature characteristics to text labels
    if features_chars is not None:
        labels = [l+f' ({(np.round(c,decimals=3))})' for l,c in zip(labels, features_chars[top_idx])]
        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=top, ncols=1)

    x_grid = np.linspace(0.0, xmax, npoints)

    axes = []
    curves_ref = []
    curves_comp = []

    for i, (row_ref, row_comp) in enumerate(zip(X_top_ref, X_top_comp)):
        # KDE reference
        curve_ref = compute_1dkde_curve(
            x=row_ref[:,None],
            x_grid=x_grid[:,None],
            bandwidth=bandwidth,
            kernel='gaussian'
        )
        curves_ref.append(curve_ref)
        
        # KDE comparison
        curve_comp = compute_1dkde_curve(
            x=row_comp[:,None],
            x_grid=x_grid[:,None],
            bandwidth=bandwidth,
            kernel='gaussian'
        )
        curves_comp.append(curve_comp)
        
        ax = fig.add_subplot(gs[i:i+1, :])
        axes.append(ax)
        
        # Plot KDE for X_top
        ax.fill_between(x_grid, curve_ref, color=color_kde_ref, alpha=0.25, linewidth=0)
        ax.plot(x_grid, curve_ref, color=color_kde_ref, linewidth=1.0, alpha=0.75, label=label_ref)
        ax.scatter(row_ref, np.zeros(len(row_ref)), marker=3, color=color_rugs_ref, alpha=0.15)
        
        # Plot KDE for X_comp
        ax.fill_between(x_grid, curve_comp, color=color_kde_comp, alpha=0.25, linewidth=0)
        ax.plot(x_grid, curve_comp, color=color_kde_comp, linewidth=1.0, alpha=0.75, label=label_comp)
        ax.scatter(row_comp, np.zeros(len(row_comp)), marker=3, color=color_rugs_comp, alpha=0.15)
        
        # Rest of the formatting remains the same
        ax.set_xlim(0.0, xmax)
        ax.patch.set_alpha(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        if i + 1 < top:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r'Mean Absolute Shapley Value', fontsize=fontsize)
            ax.tick_params(axis="x", which="both", top=False, labelsize=fontsize)
            
        if i == 0:
            ax.legend(prop=dict(size=fontsize))
        
        ax.text(-0.025,0.1, labels[i],
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=fontsize,
                transform=ax.transAxes)

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
        return fig, axes, curves_ref, curves_comp
    else:
        return fig                         
        