# %%
import logging
 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy import stats as st
from scipy.ndimage import gaussian_filter1d

from CML_tool.ML_Utils import compute_empirical_ci


def scatter_plot(explanation, feature_idx, feature_names, spline_type='smoothing', sigma=None,
                   show_scatter=True, show_reg=True, show_ci=True, 
                   ci_type='bootstrap', significance_level=0.05, n_bootstrap=1000,
                   show_hist=True, hist_bins=30, alpha=0.6, show=False, **kwargs):
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
    alpha : float, default=0.6
        Transparency level
    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` (default) allows the plot
        to be customized further after it has been created.
    """
    
    shap_values = explanation.values[:, feature_idx] if explanation.values.ndim == 2 else explanation.values[:, feature_idx, 1]    
    feature_vals = explanation.data[:, feature_idx]
    
    # Set up figure layout with GridSpec
    fig = plt.figure(figsize=kwargs.get('figsize', (8, 8)))
    gs = gridspec.GridSpec(6, 6, figure=fig, wspace=0, hspace=0)
    
    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[1:, :5])
    
    # Histograms
    ax_hist_x = fig.add_subplot(gs[0, 0:5], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs[1:, 5], sharey=ax_scatter)
    
    # Turn off histogram tick labels
    ax_hist_x.tick_params(axis='x', labelbottom=False)
    ax_hist_y.tick_params(axis='y', labelleft=False)
    
    
# Add regression line with confidence intervals
    if show_reg:
        # Fit spline and predict values
        s=kwargs.get('s', len(feature_vals)*0.1)
        k=kwargs.get('k', 3)
        resolution=kwargs.get('resolution', 3*len(feature_vals))
        spline, x_spline_pred, y_pred = fit_univariate_spline(
            x=feature_vals,
            y=shap_values,
            s=s,
            k=k,
            resolution=resolution,
            spline=spline_type
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
            ax_scatter.fill_between(x_spline_pred, low_ci_lim, upper_ci_lim, color=kwargs.get('color_ci','lightblue'), alpha=0.25)

        # Plot the fitted spline    
        ax_scatter.plot(x_spline_pred, y_pred, color=kwargs.get('color_reg', 'C0'), alpha=0.5, linewidth=kwargs.get('linewidth', 2))
        
    # Draw scatter plot
    if show_scatter:
        ax_scatter.scatter(feature_vals, shap_values, alpha=alpha, edgecolor='none',
                           color=kwargs.get('color_scatter', 'blue'), s=kwargs.get('marker_size', 8)
                           )
        # Set ticklabels size
        ax_scatter.tick_params(axis='both', which='both', labelsize=kwargs.get('fontsize', 16))
            
    # Add histograms
    if show_hist:
        ax_hist_x.hist(feature_vals, bins=hist_bins, alpha=1, density=True, color=kwargs.get('color_hist', 'C0'))
        ax_hist_y.hist(shap_values, bins=hist_bins, alpha=1, orientation='horizontal', density=True, color=kwargs.get('color_hist', 'C0'))
                
        # Remove ticks and labels from histograms
        ax_hist_x.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=True, top=False, left=False, right=False)
        ax_hist_y.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=True, right=False)

        # Remove spines between plots
        ax_hist_x.spines['bottom'].set_visible(False)
    
    # Labels and titles
    feature_name = feature_names[feature_idx]
    ax_scatter.set_xlabel(r"$P(X)$: "+ feature_name, fontsize=kwargs.get('fontsize', 18))
    ax_scatter.set_ylabel(r"$P(Y|X): $ Shapley value", fontsize=kwargs.get('fontsize', 18))
    
    if show:
        plt.show()
    
    return fig, (ax_scatter, ax_hist_x, ax_hist_y)


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
    

