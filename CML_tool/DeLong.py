# %%
import numpy as np
import scipy.stats 
import scipy.special
import xgboost as xgb
from typing import Tuple

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
    delongcov = sx / m + sy / n #covariance matrix of both AUC1 & AUC2
    return aucs, delongcov

def calc_pvalue(aucs, sigma_sq, alpha, printing = False):
    """Computes and returns the AUC difference
    test statistic (U-statistic difference)
    and the p-value log(10) of the test

    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariance matrix
    Returns:
       log10(pvalue)
       test statistic
       Wald_CI
       Logistic_logit_CI
    """
    #Individual AUCs
    # I
    up_lim_AUC1, low_lim_AUC1 = np.ravel(DL_logit_CI(alpha = alpha, theta=aucs[0], Var=sigma_sq[0,0]))
    if printing == True:
        print('AUC 1 = ', aucs[0], 'variance = ',sigma_sq[0,0],
        str(int((1-alpha)*100)),'% CI:[',low_lim_AUC1,',',up_lim_AUC1,'] \n')

    #II
    up_lim_AUC2, low_lim_AUC2 = np.ravel(DL_logit_CI(alpha = alpha, theta=aucs[1], Var=sigma_sq[1,1]))
    if printing == True:
        print('AUC 2 = ', aucs[1], 'variance = ',sigma_sq[1,1],
        str(int((1-alpha)*100)),'% CI:[',low_lim_AUC2,',',up_lim_AUC2,']\n')
        
    #Test  
    l = np.array([[1, -1]])
    sigma_diff = np.sqrt(np.dot(np.dot(l, sigma_sq), l.T)) #std of the AUC difference
    AUC_diff = abs(np.diff(aucs))  # AUC difference absolute value
    AUC_diff_signed = np.diff(aucs) # AUC difference
    z = AUC_diff/sigma_diff #U-statistic
    
    #P-value
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    pvalue = 10**log10_p_value[0][0]

   #CI of the AUC difference.
    wald_up_lim, wald_low_lim = np.ravel(Wald_type_DL_CI(alpha = alpha, theta = AUC_diff, Var = sigma_diff**2))
    logit_up_lim, logit_low_lim = np.ravel(DL_logit_CI(alpha = alpha, theta = AUC_diff, Var = sigma_diff**2))
    
    if printing == True:
        print('DeLong test results: log10(p-value)= ', log10_p_value[0][0], '(p-value = ',pvalue,'), AUC difference = ',AUC_diff, 
        str(int((1-alpha)*100)),'% wald type CI:[',wald_up_lim,',',wald_low_lim,'], logit CI:[', logit_low_lim,',',logit_up_lim,'].')

    if pvalue<alpha:
        significance = True
        if printing==True:
            print('\n Significant')
    else:
        significance = False
        if printing==True:
            print('\n Not significant')

    return aucs[0], aucs[1], AUC_diff_signed[0], pvalue, significance, wald_low_lim, wald_up_lim, logit_low_lim, logit_up_lim

def DL_logit_CI(alpha,theta,Var):
    """

    Since the AUC is restricted to [0,1], Pepe et al 2003 has argued that an asymmetric confidence
    interval within (0,1) should be preferred. Using a logistic transformation, the limits are... (Quin and Hotilovac 2008)

    This function used DeLong estimator for the variance
    and the logi function to ensure CI bounds fall within [0,1]
    The Delta method is used to approximate the transformation which is 
    reversed via expit function into the original scale.
    """
    if (theta == 1) and (Var==0):
        up_lim = 1.0
        low_lim = 1.0
    elif (theta ==0) and (Var==0):
        up_lim = 0.0
        low_lim = 0.0
    else:
        low_exp = scipy.special.logit(theta) - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta)) # Quin and Hotilovac 2008
        up_exp = scipy.special.logit(theta) + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta))  # Quin and Hotilovac 2008
        up_lim = scipy.special.expit(up_exp)
        low_lim = scipy.special.expit(low_exp)

    return  np.ravel(low_lim), np.ravel(up_lim)

def Wald_type_DL_CI (alpha,theta, Var):
    """
    Wald type CI
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var) # Cho et al. 2018
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var) # Cho et al. 2018

    return np.ravel(low_lim),np.ravel(up_lim)

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

def delong_roc_test(ground_truth, predictions_one, predictions_two,alpha, printing=False):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different.

    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1

    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov, alpha=alpha, printing = printing)


def AUC_CI(ground_truth,predictions,alpha, printing = False):
    """Calculate DeLong AUROC and provide
    both logit and wald confidence intervals.
    
    Args:
        -ground_truth: array of true labels
        -predictions: array of 
        -alpha: Delong test significance level
        -printing: whether to print the result or not (bool) 
    
    Returns (in order):
        -Mean value of the DeLong estimation of the area under the receiving operating characteristic curve (AUROC)
        -Variance
        -Wald confidence interval lower limit
        -Wald confidence interval upper limit
        -Logit confidence interval lower limit
        -Logit confidence interval upper limit

    """
    AUC, variance = delong_roc_variance(ground_truth,predictions)
    low_lim_wald, up_lim_wald = np.ravel(Wald_type_DL_CI(alpha = alpha,theta = AUC,Var = variance))
    low_lim_logit, up_lim_logit= np.ravel(DL_logit_CI(alpha = alpha,theta = AUC,Var = variance))
    if printing == True:    
        print('AUC = ', AUC, 'variance = ',variance, '\n',
              'Wald type:', str(int((1-alpha)*100)),'% CI:[',low_lim_wald,',',up_lim_wald,'] \n'
              'Logit type:', str(int((1-alpha)*100)),'% CI:[',low_lim_logit,',',up_lim_logit,'] \n')

    return AUC, variance, low_lim_wald, up_lim_wald, low_lim_logit, up_lim_logit

def DeLong_AUROC(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom evaluation metric function to pass to XGBoost binary classifiers models as 
    hyperparameter 'eval_metric'. 

    Args:
    -y_true: true labels array
    -y_pred: predicted probabilities array
    """
    DL_AUC = delong_roc_variance(
        predictions=y_pred,
        ground_truth=y_true
    )[0]
    return float(DL_AUC)


def fit_method_DeLong_AUROC(predt: np.ndarray, dataset: xgb.DMatrix) -> Tuple[str, float]:
    """
    Custom evaluation metric function to pass to XGBoost.fit('eval_metric'=...)
    for an XGBoostClassifier model. 

    Args:
    -predt: predicted probabilities array
    -dataset xgboost DMatrix hosting data matrix (X) and labels (y)
    """
    DL_AUC = delong_roc_variance(
        predictions=predt,
        ground_truth=dataset.get_label()
    )[0]
    return 'DeLong_AUROC', float(DL_AUC)


#%%
if __name__=='__main__':

    # Perfect case
    probs = np.array(45*[1,1,1,1,0,1,0,0,1,1,1,1,1,1])
    gt = np.array(45*[1,1,1,1,0,1,0,0,1,1,1,1,1,1])
    AUC_CI(alpha=0.05, ground_truth=gt,predictions=probs,printing=True)

    
    #Comparison
    probs1 = np.array(45*[0.5,0.6,0.9,0.1,0.001,0.67,0.87,0.35,0.75,0.5,0.5,0.4,0.6,0.7])
    probs2 = np.array(45*[0.45,0.2,0.99,0.001,0.25,0.8,0.4,0.9,0.7,0.5,0.5,0.4,0.6,0.7])
    delong_roc_test(gt,probs1,probs2,alpha = 0.05, printing=True)

    AUC_CI(alpha = 0.05,ground_truth=gt,predictions=probs1,printing = True)
    AUC_CI(alpha = 0.05,ground_truth=gt,predictions=probs2,printing = True)

# %%



