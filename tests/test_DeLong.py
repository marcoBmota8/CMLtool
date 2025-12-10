
import math
from collections import defaultdict

import numpy as np

from CML_tool.DeLong import auc_ci

def simulate_ci_widths(n, noise_std, ci_type, repeats=30, pos_rate=0.5, alpha=0.05):
    widths = []
    for _ in range(repeats):
        n_pos = int(math.floor(n * pos_rate))
        n_neg = n - n_pos
        # ground truth balanced by pos_rate
        gt = np.array([1]*n_pos + [0]*n_neg)
        # base scores: positives centered at 0.9, negatives at 0.1
        base = np.concatenate([np.ones(n_pos)*0.9, np.zeros(n_neg)*0.1])
        preds = base + np.random.normal(0, noise_std, size=n)
        preds = np.clip(preds, 0.0, 1.0)
        auc, var, ci = auc_ci(alpha=alpha, ground_truth=gt, predictions=preds, ci_type=ci_type)
        w = ci[1]-ci[0]
        if w is not None and np.isfinite(w):
            widths.append(w)
    return np.array(widths)

# Settings to explore (adjust as needed)
sample_sizes = [50, 100, 200, 500, 1000, 10000]
noise_levels = [0.05, 0.1, 0.25, 0.5, 0.75]
repeats = 30

print("\nSimulation: mean CI width (alpha=0.05) over repeats")
for ci_type in ['logistic', 'wald']:
    print(f'---------------- CONFIDENCE INTERVAL TYPE: {ci_type.upper()} ---------------')
    for n in sample_sizes:
        for noise in noise_levels:
            widths = simulate_ci_widths(n=n, noise_std=noise, repeats=repeats, ci_type=ci_type, alpha=0.05)
            if widths.size == 0:
                print(f"n={n:5d}, noise={noise:.3f}: no valid CI widths (check auc_ci return format)")
                continue
            mean_w = widths.mean()
            std_w = widths.std(ddof=1) if widths.size > 1 else 0.0
            print(f"n={n:5d}, noise={noise:.3f}: mean_width={mean_w:.4f}, std={std_w:.4f}, valid={widths.size}/{repeats}")

