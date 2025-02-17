# %%
import numpy as np

from CML_tool.ML_Utils import compute_empirical_ci
from mct import _compute_single_calibration, _resample_calibration

def compute_ici_results(probs, labels, resolution=0.01, n_bootstrap=1000, alpha=0.05, ci_type='pivot'):

    x_min = max((0, np.amin(probs) - resolution))
    x_max = min((1, np.amax(probs) + resolution))

    # Estimate using the full data
    pivot_ici = _compute_single_calibration(
        x_values=np.arange(x_min, x_max, step=resolution),
        probs=probs, # Model predicted probabilities
        actual=labels, # Binary labels
        kernel='gaussian',
        bandwidth=0.01
        ).ici

    # Bootstrapped samples
    ici_samples = np.array(_resample_calibration(
        num_iterations=n_bootstrap,
        x_values=np.arange(x_min, x_max, step=resolution),
        probs=probs, # Model predicted probabilities
        actual=labels, # Binary labels
        kernel='gaussian',
        bandwidth=0.01,
        n_jobs=24
        ).ici
                        )
    
    # Confidence intervals
    ici_ci = tuple(0 if num < 0 else num for num in compute_empirical_ci(
        X=ici_samples.reshape(-1,1),
        pivot=pivot_ici,
        alpha=alpha,
        type=ci_type
        )[0]
          )
    
    return {
        'estimate':pivot_ici,
        'samples':ici_samples,
        'ci':ici_ci
    }
# %%
