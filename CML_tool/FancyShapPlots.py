# %%
import logging
 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy import stats as st
from scipy.ndimage import gaussian_filter1d

from shap.plots.colors._colorconv import lab2rgb, lch2lab

from CML_tool.ML_Utils import compute_empirical_ci


def fit_univariate_spline(x, y, resolution=1000, s=None, k=3, spline='smoothing'):
    """
    Fit a univariate spline to the data and 
    return the fitted spline and predicted values.
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray  
        Dependent variable
    resolution : int, default=100
        Number of points for prediction
    s : float, default=None
        Smoothing factor (default is None, which means no smoothing)
    k : int, default=3
        Degree of the spline (default is cubic)
    spline : bool, default='smoothing'
        Whether to use a `smoothing` spline (default) or an `overfitted` PCHIP spline (only available if k is 3, i.e. cubic splines (default)).
    
    Returns:
    --------
    spline : UnivariateSpline
        Fitted spline object    
    x_pred : np.ndarray
        Predicted x values
    y_pred : np.ndarray
        Predicted y values
    """ 
    
    # Check if x and y are 1D arrays
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    # Check if x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    # Check for NaN values
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("x and y must not contain NaN values.")
    # Check for infinite values
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        raise ValueError("x and y must not contain infinite values.")
    
    
    # Sort the data
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
        
    non_nan = ~np.isnan(x_sorted) & ~np.isnan(y_sorted)
    if sum(non_nan) > 3:  # Need at least 3 points for spline
        try:
            # Fit a smoothing spline
            if s is None:
                s = len(x_sorted[non_nan]) * 0.1
            if spline=='overfitted':
                if k!=3:
                    logging.info(f"Passed k={k}. Defaulting to cubic spline (k=3) for PCHIP interpolation.")
                # Aggregate repeated x values for PchipInterpolator
                unique_x = np.unique(x_sorted[non_nan])
                y_mean = [np.mean(y_sorted[non_nan][x_sorted[non_nan] == x]) for x in unique_x]
                spline = PchipInterpolator(unique_x, y_mean, extrapolate=True)            
            elif spline=='smoothing':  
                # Smoothing spline
                spline = UnivariateSpline(x_sorted[non_nan], y_sorted[non_nan], s=s, k=k)
            else:
                raise ValueError("Invalid spline type. Use 'smoothing' with k in [1,5] or 'overfitted' with k=3.")
            
            x_pred = np.linspace(min(x_sorted[non_nan]), max(x_sorted[non_nan]), resolution)
            y_pred = spline(x_pred)
        except Exception as e:
            print(f"Could not fit spline: {e}")
    else:
        raise ValueError("Not enough non-NaN points to fit a spline.")

    return spline, x_pred, y_pred
    

def scatter_plot(
    explanation,
    feature_idx,
    feature_names,
    labels=None,
    labels_dict=None,
    xlabel='default',
    ylabel='default',
    ax=None,
    spline_type='smoothing',
    sigma=None,
    show_scatter=True,
    show_reg=True,
    show_ci=True,
    ci_type='bootstrap',
    significance_level=0.05,
    n_bootstrap=1000,
    show_hist=True,
    hist_bins=30,
    show=False,
    **kwargs
):
    """
    Create an enhanced SHAP dependence plot with histograms and regression line using a SHAP Explainer.
    
    Parameters:
    -----------
    explanation : shap.Explaination
        SHAP Explainer object (e.g., TreeExplainer, DeepExplainer)
    feature_idx : int
        Feature index
    feature_names : list
        List of feature names
    labels : np.array or list, default=None
        Binary labels for the data points
        If None, all scatter data points are shown with the same specs (color, alpha and marker)
    labels_dict : dict, default=None
        Dictionary mapping labels to the plotting specs of each unique value
        The dictionary should contain the following keys:
            - 'color': Color of the points and histograms
            - 'alpha': Transparency level
            - 'marker_size': Size of the points
            - 'marker': Marker style
            - 'label': String for the label class legend
            - 'edgecolor': Edge color of the points (optional)        
        If None, the default specs are used for all points. 
    xlabel : str, default='default'
        X-axis label. If None is passed no label is shown.
        If no string is passed the feature name is used (default).
    ylabel : str, default='default'
        Y-axis label. If None is passed no label is shown. 
        If no string is passed the default label (P(Y): Shapley value) is used (default).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created and returned.
        If provided, the function will not create a new figure and return the axes.
    spline_type: str, default='smoothing'
        Type of spline to fit ('smoothing' or 'overfitted')
    show_scatter : bool, default=True
        Whether to show scatter points
    show_reg : bool, default=True
        Whether to show regression line
    show_ci : bool, default=True
        Whether to show confidence intervals
    ci_type : str, default='bootstrap'
        Type of confidence interval to show ('prediction_error' or 'bootstrap')
    sigma : float, default=None
        Standard deviation for Gaussian smoothing (if None, no smoothing is applied)
        Used to smooth the regression line and confidence intervals for
        spline_type='overfitted' and ci_type='bootstrap'. Otherwise ignored.
    significance_level : float, default=0.05
        Significance level for confidence intervals
    n_bootstrap : int, default=1000
        Number of bootstrap samples for confidence intervals.
        Only used if ci_type is 'bootstrap'.
    show_hist : bool, default=True
        Whether to show histograms
    hist_bins : int, default=30
        Number of bins for histograms
    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
    """

    shap_values = explanation.values[:, feature_idx] if explanation.values.ndim == 2 else explanation.values[:, feature_idx, 1]
    feature_vals = explanation.data[:, feature_idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (7,7)))
        own_fig = True
    else:
        own_fig = False

    # Main scatter plot
    if show_scatter:
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(
                    feature_vals[mask], shap_values[mask], 
                    alpha=labels_dict[label].get('alpha', 0.5), 
                    edgecolor=labels_dict[label].get('edgecolor', 'none'),
                    color=labels_dict[label]['color'], 
                    s=labels_dict[label]['marker_size'],
                    marker=labels_dict[label].get('marker', '.'),
                    label=labels_dict[label]['label'],
                )
        else:
            ax.scatter(
                feature_vals, shap_values, alpha=kwargs.get('alpha', 0.5), edgecolor='none',
                color=kwargs.get('color_scatter', 'blue'), s=kwargs.get('marker_size', 8)
            )
        xlims, ylims = ax.get_xlim(), ax.get_ylim() # Get the axis limits
        ax.vlines(x=0, ymin=ylims[0], ymax=ylims[1], colors='gray', linestyles='dashed', lw=1, alpha=0.25)
        ax.hlines(y=0, xmin=xlims[0], xmax=xlims[1], colors='gray', linestyles='dashed', lw=1, alpha=0.25)
        ax.tick_params(axis='both', which='both', labelsize=kwargs.get('fontsize', 16))

    # Regression line and confidence intervals
    if show_reg:
        s = kwargs.get('s', len(feature_vals) * 0.1)
        k = kwargs.get('k', 3)
        resolution = kwargs.get('resolution', 3 * len(feature_vals))
        spline, x_spline_pred, y_pred = fit_univariate_spline(
            x=feature_vals, y=shap_values, s=s, k=k, resolution=resolution, spline=spline_type
        )
        
        # Calculate and plot confidence intervals
        if show_ci:
            if ci_type == 'prediction_error':  # Use single spline fit prediction error (Not recommended for most cases, unless one wants to overfit to the data)
                # Sort the data
                sorted_idx = np.argsort(feature_vals)
                x_sorted = feature_vals[sorted_idx]
                y_sorted = shap_values[sorted_idx]
                non_nan = ~np.isnan(x_sorted) & ~np.isnan(y_sorted)
                # Compute confidence intervals
                residuals = y_sorted[non_nan] - spline(x_sorted[non_nan])
                sigma = np.std(residuals)
                ci = st.norm.ppf(1-significance_level) * sigma  # 95% confidence interval
                low_ci_lim, upper_ci_lim = y_pred - ci, y_pred + ci
                
            elif ci_type == 'bootstrap': # Bootstrap approach (Recommended for better trend approximation and accurate confidence intervals)
                idx_boot = np.sort(np.random.choice(len(feature_vals), size= (n_bootstrap ,len(feature_vals)), replace=True))
                data_boot = [(feature_vals[idx], shap_values[idx]) for idx in idx_boot]
                X_splines = np.vstack([fit_univariate_spline(x, y, s=s, k=k, resolution=resolution, spline=spline_type)[2] for (x,y) in data_boot])
                X_splines = X_splines[~np.isnan(X_splines).any(axis=1)] # drop rows with NaN values
                low_ci_lim, upper_ci_lim = zip(*compute_empirical_ci(
                    X=X_splines,
                    alpha=0.05,
                    type='quantile'
                ))
                del y_pred
                y_pred = np.median(X_splines, axis=0) # Median of all bootstrap samples
                
                if spline_type=='overfitted' and sigma is not None:
                    low_ci_lim = gaussian_filter1d(low_ci_lim, sigma=sigma)
                    upper_ci_lim = gaussian_filter1d(upper_ci_lim, sigma=sigma)
                    y_pred = gaussian_filter1d(y_pred, sigma=sigma)
                
            else:
                raise ValueError("Invalid ci_type. Use `prediction_error` or `bootstrap`.")
            # Plot confidence intervals
            ax.fill_between(x_spline_pred, low_ci_lim, upper_ci_lim, color=kwargs.get('color_ci','lightblue'), alpha=0.15)
        
        ax.plot(
            x_spline_pred, y_pred,
            color=kwargs.get('color_reg', 'C0'), alpha=0.33, linewidth=kwargs.get('linewidth', 2)
        )

    # Histograms (optional, can be improved for subplots)
    if show_hist:
        # Set zero padding between histograms and main axis
        ax_histx = ax.inset_axes([0, 1.0, 1, 0.18], sharex=ax)
        ax_histy = ax.inset_axes([1.0, 0, 0.18, 1], sharey=ax)
        
        if labels is not None:
            for label in unique_labels:
                mask = labels == label
                ax_histx.hist(feature_vals[mask], bins=hist_bins, alpha=0.6, color=labels_dict[label]['color'])
                ax_histy.hist(shap_values[mask], bins=hist_bins, alpha=0.6, orientation='horizontal', color=labels_dict[label]['color'])
        else:
            ax_histx.hist(feature_vals, bins=hist_bins, alpha=1, color=kwargs.get('color_hist', 'C0'))
            ax_histy.hist(shap_values, bins=hist_bins, alpha=1, orientation='horizontal', color=kwargs.get('color_hist', 'C0'))

        # Hide ticks and ticklabels for both histogram axes
        ax_histx.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )
        ax_histy.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False, labelright=False, labeltop=False
        )

    # Labels
    if isinstance(xlabel, str) and xlabel != 'default': 
        ax.set_xlabel(xlabel, fontsize=kwargs.get('fontsize', 18))
    elif xlabel is None:
        pass
    else:
        feature_name = feature_names[feature_idx]
        ax.set_xlabel(r"$P(X)$: " + feature_name, fontsize=kwargs.get('fontsize', 18))
    if isinstance(ylabel, str) and ylabel != 'default':
        ax.set_ylabel(ylabel, fontsize=kwargs.get('fontsize', 18))
    elif ylabel is None:
        pass
    else:
        ax.set_ylabel(r"$P(Y|X): $ Shapley value", fontsize=kwargs.get('fontsize', 18))

    if show:
        plt.show()
        
    if own_fig:
        return fig, ax
    else:
        return ax

# Generate a custom colormap for shap (scatter mainly) plots
def lch2rgb(x):
    return lab2rgb(lch2lab([[x]]))[0][0]

def tighten_gradient_transition(colors, n_points, yellow_rgb, side='left'):
    n = len(colors)
    if side == 'left':
        start_idx = n - n_points
        for i in range(start_idx, n):
            alpha = (i - start_idx) / n_points
            colors[i] = (1 - alpha) * colors[i] + alpha * yellow_rgb
    else:  # right side
        for i in range(n_points):
            alpha = i / n_points
            colors[i] = (1 - alpha) * yellow_rgb + alpha * colors[i]
    return colors

def create_red_yellow_blue_colormap(center_width=0.01, nsteps=300, gradient_fraction=0.1):
    # Define your Lch colors as in your code
    l_mid = 40.0
    blue_lch= [35, 60.0, 4.6588]
    red_lch = [35, 80.0, 0.35470565 + 2 * np.pi]
    gray_lch = [55.0, 0.0, 0.0]
    blue_rgb = lch2rgb(blue_lch)
    red_rgb = lch2rgb(red_lch)
    gray_rgb = lch2rgb(gray_lch)
    yellow_rgb = np.array([0.95, 0.95, 0.5])  # yellow

    l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps // 2)) + list(np.linspace(l_mid, red_lch[0], nsteps // 2))
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
    blue_to_red_rgb = np.array([lch2rgb([l, c, h]) for l, c, h in zip(l_vals, c_vals, h_vals)])

    center_start = 0.5 - center_width / 2
    center_end = 0.5 + center_width / 2

    n_left = int(nsteps * center_start)
    n_center = max(1, int(nsteps * center_width))
    n_right = int(nsteps * (1 - center_end))

    left_positions = np.linspace(0, center_start, n_left)
    left_colors = blue_to_red_rgb[:n_left]
    left_gradient_points = max(1, int(n_left * gradient_fraction))
    left_colors = tighten_gradient_transition(left_colors, left_gradient_points, yellow_rgb, 'left')

    center_positions = np.linspace(center_start, center_end, n_center)
    center_colors = np.tile(yellow_rgb, (n_center, 1))

    right_positions = np.linspace(center_end, 1, n_right)
    right_colors = blue_to_red_rgb[-n_right:]
    right_gradient_points = max(1, int(n_right * gradient_fraction))
    right_colors = tighten_gradient_transition(right_colors, right_gradient_points, yellow_rgb, 'right')

    positions = np.concatenate([left_positions, center_positions, right_positions])
    colors = np.vstack([left_colors, center_colors, right_colors])

    cmap = LinearSegmentedColormap.from_list("red_yellow_blue", list(zip(positions, colors)))
    cmap.set_bad(gray_rgb.tolist(), 1.0)
    cmap.set_over(gray_rgb.tolist(), 1.0)
    cmap.set_under(gray_rgb.tolist(), 1.0)

    return cmap

# custom colormap
red_blue_yellow_tight = create_red_yellow_blue_colormap(center_width=0.01, nsteps=256, gradient_fraction=0.1)
       
