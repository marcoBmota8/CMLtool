# %%
import pandas as pd
import os
import numpy as np
import scipy.stats 
import scipy.special
import pickle

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
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
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
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
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

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
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

def calc_pvalue(aucs, sigma, alpha, printing = False):
    """Computes and returns the AUC difference
    test statistic (U-statistic difference)
    and the p-value log(10) of the test

    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
       test statistic
       CI

    """
    #Individual AUCs
    # I
    up_lim_AUC1, low_lim_AUC1 = np.ravel(DL_logit_CI(alpha = alpha, theta=aucs[0], Var=sigma[0,0]))
    if printing == True:
        print('AUC 1 = ', aucs[0], 'variance = ',sigma[0,0],
        str(int((1-alpha)*100)),'% CI:[',low_lim_AUC1,',',up_lim_AUC1,'] \n')

    #II
    up_lim_AUC2, low_lim_AUC2 = np.ravel(DL_logit_CI(alpha = alpha, theta=aucs[1], Var=sigma[1,1]))
    if printing == True:
        print('AUC 2 = ', aucs[1], 'variance = ',sigma[1,1],
        str(int((1-alpha)*100)),'% CI:[',low_lim_AUC2,',',up_lim_AUC2,']\n')
    #AUC difference   
    l = np.array([[1, -1]])
    sigma_diff = np.sqrt(np.dot(np.dot(l, sigma), l.T)) #std of the AUC difference
    AUC_diff = abs(np.diff(aucs))  # AUC difference absolute
    AUC_diff_signed = np.diff(aucs) # AUC difference
    z = AUC_diff/sigma_diff #normally distributed 
    #P-value
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    pvalue = 10**log10_p_value[0][0]

   #CI
    up_lim, low_lim = np.ravel(DL_logit_CI(alpha = alpha, theta = AUC_diff, Var = sigma_diff))
    if printing == True:
        print('DeLong test results: log10(p-value)= ', log10_p_value[0][0], '(p-value = ',pvalue,'), AUC difference = ',AUC_diff, 
        str(int((1-alpha)*100)),'% CI:[',up_lim,',',low_lim,']')

    if pvalue<alpha:
        significance = True
        if printing==True:
            print('\n Significant')
    else:
        significance = False
        if printing==True:
            print('\n Not significant')

    return aucs[0], aucs[1], AUC_diff_signed[0], pvalue, significance

def DL_logit_CI(alpha,theta,Var):
    """

    Since the AUC is restricted to [0,1], Pepe et al 2003 has argued that an asymmetric confidence
    interval within (0,1) should be preferred. Using a logistic transformation, th e limits are... (Qin et al 2008)

    DeLong estimator 
    for the variance
    and the logit function to 
    ensure CI bounds
    fall within [0,1] (Delta method)
    The transformation is 
    reversed via expit
    function into
    the original scale.
    """

    low_exp = scipy.special.logit(theta) - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta))
    up_exp = scipy.special.logit(theta) + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta))
    up_lim = scipy.special.expit(up_exp)
    low_lim = scipy.special.expit(low_exp)

    return  np.ravel(low_lim), np.ravel(up_lim)

def Wald_type_DL_CI (alpha,theta, Var):
    """
    Wald type CI
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)

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
    AUC, variance = delong_roc_variance(ground_truth,predictions)
    low_lim_wald, up_lim_wald = np.ravel(Wald_type_DL_CI(alpha = alpha,theta = AUC,Var = variance))
    low_lim_logit, up_lim_logit= np.ravel(DL_logit_CI(alpha = alpha,theta = AUC,Var = variance))
    if printing == True:    
        print('AUC = ', AUC, 'variance = ',variance, '\n',
              'Wald type:', str(int((1-alpha)*100)),'% CI:[',low_lim_wald,',',up_lim_wald,'] \n'
              'Logit type:', str(int((1-alpha)*100)),'% CI:[',low_lim_logit,',',up_lim_logit,'] \n')

    return AUC, variance, low_lim_wald, up_lim_wald, low_lim_logit, up_lim_logit
#%%
if __name__=='__main__':

    df = pd.read_csv('~/Rota_I/Models_last_record_timepoint/ICA/Probabilities predictions/CV_probs_train_sample.csv')
    df = df.iloc[np.where(df['Ground truth']!='Unknown')[0],:]
    df.set_index('Patient ID', inplace = True)
    df.drop(columns = ['Unnamed: 0','Ground truth', 'Probability Excluded'], inplace = True)
    df.rename(columns = {'Predicted label': 'Phenotypes label', 'Probability Included': 'Phenotypes SLE probability'}, inplace = True)

    df_raw = pd.read_csv('~/Rota_I/Models_last_record_timepoint/Raw/Probabilities predictions/CV_probs_train_sample.csv')
    df_raw = df_raw.iloc[np.where(df_raw['Ground truth']!='Unknown')[0],:]
    df_raw.set_index('Patient ID', inplace = True)
    df_raw.drop(columns = ['Unnamed: 0', 'Probability Excluded'], inplace = True)
    df_raw.rename(columns = {'Predicted label': 'Raw variables label', 'Probability Included': 'Raw variables SLE probability'}, inplace = True)
    print(df_raw.shape)

    df_merge = pd.merge(df_raw,df,on='Patient ID' )
    
    delong_roc_test(df_merge['Ground truth'].values,
        df_merge['Phenotypes SLE probability'].values,
        df_merge['Raw variables SLE probability'],
        alpha = 0.05)

    AUC_CI(alpha = 0.05,
           ground_truth=df_merge['Ground truth'].values,
           predictions= df_merge['Phenotypes SLE probability'].values,
           printing = True)

# %%



