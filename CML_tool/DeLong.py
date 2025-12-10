# %%
import logging

import numpy as np
import scipy.stats 
import scipy.special
import xgboost as xgb 
from typing import Tuple

from .ML_Utils import contains_val_CI

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_midrank(x):
    """Computes midranks.

    Args: 
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_midrank_weight(x, sample_weight):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong test computation.
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n #covariance matrix of AUCs to compare
    return aucs, delongcov

def calc_pvalue(aucs, sigma_sq, ci_type='wald', alpha=0.05):
    """Computes and returns the AUC difference
    test statistic (U-statistic difference)
    and the p-value log(10) of the test

    Args:
       aucs (numpy.array): AUCs to compare
       sigma_sq (numpy.array): AUC DeLong covariance matrix
       ci_type (str): Type of confidence interval to compute. 
            - "wald": Wald confidence interval of the AUC difference. (Default)
            - "logistic": Logistic confidence interval of the AUC difference
       alpha(float): sinificance level. (Default=0.05)
    Returns:
        AUC_diff_signed (float): Difference in AUCs with sign (auc1-auc2)
        pvalue: p-value of the DeLong test
        significance (bool): Whether or not the test is significant at the significance level requested. 
        ci (tuple): Confidence interval (as specified by the user) of the AUC difference.
            -"wald": Wald confidence interval of the AUC difference.
                Must use if looking whether or not the confidence interval of the difference crosses 0
            -"logistic": Logistic confidence interval of the AUC difference 
                Important! Do not use for looking to whether or not the CI crosses 0, it will never do so as it 
                confines values in [0,1].
        ci_crossing (bool): Whether or not the confidence interval crosses zero or not.
    """
        
    # Test  
    l = np.array([[1, -1]])
    sigma_diff = np.sqrt(np.dot(np.dot(l, sigma_sq), l.T)) #std of the AUC difference
    auc_diff = abs(np.diff(aucs[::-1]))  # AUC difference absolute value
    auc_diff_signed = np.diff(aucs[::-1]) # AUC difference
    z = auc_diff/np.maximum(sigma_diff, 1e-10) #U-statistic
    
    # p-value
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    pvalue = 10**log10_p_value[0][0]

   # Confidence interval of the AUC difference.
    if ci_type == 'wald':
       ci_low_lim, ci_up_lim = np.ravel(wald_delong_ci(alpha = alpha, theta = auc_diff, var = sigma_diff**2))
    elif ci_type == 'logistic':
       ci_low_lim, ci_up_lim = np.ravel(delong_logistic_ci(alpha = alpha, theta = auc_diff, var = sigma_diff**2))
    else:
         raise ValueError("ci_type must be 'wald' or 'logistic'. Got: %s" % type)
    # Significance
    if pvalue<alpha:
        significance = True
    else:
        significance = False
    
    # Confidence interval crossing zero
    if ci_type == 'logistic':
        logging.warning('Indicator for AUC difference confidence interval crossing 0 is not valid for logistic confidence interval. Use wald confidence interval instead.')
        
    ci_crossing = contains_val_CI((ci_low_lim, ci_up_lim),0)

    return auc_diff_signed[0], pvalue, significance, (ci_low_lim,ci_up_lim), ci_crossing

def delong_logistic_ci(alpha,theta,var):
    """

    Since the AUC is restricted to [0,1], Pepe et al 2003 has argued that an asymmetric confidence
    interval within (0,1) should be preferred. Using a logistic transformation, the limits are... (Quin and Hotilovac 2008)

    This function used DeLong estimator for the variance
    and the logi function to ensure confidence interval bounds fall within [0,1]
    The Delta method is used to approximate the transformation which is 
    reversed via expit function into the original scale.
    """
    if (theta == 1) and (var==0):
        up_lim = 1.0
        low_lim = 1.0
    elif (theta ==0) and (var==0):
        up_lim = 0.0
        low_lim = 0.0
    else:
        low_exp = scipy.special.logit(theta) - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(var)/(theta*(1-theta)) # Quin and Hotilovac 2008
        up_exp = scipy.special.logit(theta) + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(var)/(theta*(1-theta))  # Quin and Hotilovac 2008
        up_lim = scipy.special.expit(up_exp)
        low_lim = scipy.special.expit(low_exp)

    return  (low_lim, up_lim)

def wald_delong_ci (alpha,theta, var):
    """
    Wald type confidence interval
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(var) # Cho et al. 2018
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(var) # Cho et al. 2018

    return (low_lim, up_lim)

def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    #assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single set of predictions.

    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1

    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def delong_roc_test(ground_truth, predictions_one, predictions_two, ci_type='wald', alpha=0.05):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different.

    Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        ci_type (str): Type of confidence interval to compute.        
            - "wald": Wald confidence interval of the AUC difference. (Default)
            - "logistic": Logistic confidence interval of the AUC difference
        alpha(float): sinificance level. (Default=0.05)
        
    Returns:
        AUC_diff_signed (float): Difference in AUCs with sign (auc1-auc2)
        pvalue: p-value of the DeLong test
        significance (bool): Whether or not the test is significant at the significance level requested. 
        ci (tuple): Confidence interval (as specified by the user) of the AUC difference.
            -"wald": Wald confidence interval of the AUC difference.
                Must use if looking whether or not the confidence interval of the difference crosses 0
            -"logistic": Logistic confidence interval of the AUC difference 
                Important! Do not use for looking to whether or not the CI crosses 0, it will never do so as it 
                confines values in [0,1].
        ci_crossing (bool): Whether or not the confidence interval crosses zero or not.

    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov, ci_type=ci_type, alpha=alpha)


def auc_ci(ground_truth,predictions,alpha=0.05, ci_type='wald'):
    """Calculate DeLong AUROC and provide
    both logistic and wald confidence intervals.
    
    Args:
        -ground_truth: array of true labels
        -predictions: array of 
        -alpha: Delong test significance level (Default=0.05)
    
    Returns (in order):
        -Mean value of the DeLong estimation of the area under the receiving operating characteristic curve (AUROC): float
        -Variance: float
        -Confidence interval: tuple

    """
    auc, variance = delong_roc_variance(ground_truth,predictions)
    if ci_type == 'wald':
        low_lim_ci, up_lim_ci = np.ravel(wald_delong_ci(alpha = alpha,theta = auc,var = variance))
    elif ci_type == 'logistic':
        low_lim_ci, up_lim_ci = np.ravel(delong_logistic_ci(alpha = alpha,theta = auc,var = variance))
    else:
        raise ValueError("ci_type must be 'wald' or 'logistic'. Got: %s" % type)
        
    return auc, variance, (low_lim_ci, up_lim_ci)


def fit_method_delong_auroc(predt: np.ndarray, dataset: xgb.DMatrix) -> Tuple[str, float]:
    """
    Custom evaluation metric function to pass to XGBoost.fit('eval_metric'=...)
    for an XGBoostClassifier model. 

    Args:
    -predt: predicted probabilities array
    -dataset xgboost DMatrix hosting data matrix (X) and labels (y)
    """
    dl_auc = delong_roc_variance(
        predictions=predt,
        ground_truth=dataset.get_label()
    )[0]
    return 'DeLong_AUROC', float(dl_auc)



