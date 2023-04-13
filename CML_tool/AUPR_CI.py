
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
import scipy.special
import pickle


from sklearn.metrics import average_precision_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('science')

def variance_stand_approx(theta_hat, n):
    """
    Binomial distribution approximation
    via normal distribution
    """
    return theta_hat*(1-theta_hat)/n


def calc_pvalue(aucs, sigma, alpha):
    """Computes and returns the AUC difference
    test statistic (U-statistic difference)
    and the p-value log(10) of the test

    Args:
       aucs: 1D array of AUCs
       sigma: AUC variance
    Returns:
       log10(pvalue)
       test statistic
       CI

    """
    #Individual AUCs
    # I
    up_lim_AUC1, low_lim_AUC1 = np.ravel(plain_AUCPR_CI(alpha = alpha, theta=aucs[0], Var=sigma[0]))
    print('AUC 1 = ', aucs[0], 'variance = ',sigma[0],
    str(int((1-alpha)*100)),'% CI:[',low_lim_AUC1,',',up_lim_AUC1,'] \n')

    #II
    up_lim_AUC2, low_lim_AUC2 = np.ravel(plain_AUCPR_CI(alpha = alpha, theta=aucs[1], Var=sigma[1]))
    print('AUC 2 = ', aucs[1], 'variance = ',sigma[1],
    str(int((1-alpha)*100)),'% CI:[',low_lim_AUC2,',',up_lim_AUC2,']\n')
    #AUC difference   
    l = np.array([[1, -1]])
    sigma_diff = abs(sigma[0]-sigma[1]) #std of the AUC difference
    AUC_diff = np.abs(np.diff(aucs)) #Absolute AUC difference
    z = AUC_diff/sigma_diff #normally distributed

    #P-value (normally distributed random variables)
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) #calculate p-value while changing its logarithmic base (e->10)
    pvalue = 10**log10_p_value[0]

   #CI
    up_lim, low_lim = np.ravel(AUCPR_logit_CI(alpha = alpha, theta = AUC_diff, Var = sigma_diff))

    print('Z-test test results: log10(p-value)= ', log10_p_value[0], '(p-value = ',pvalue,'), AUC difference = ',AUC_diff, 
    str(int((1-alpha)*100)),'% CI:[',up_lim,',',low_lim,']')

    if pvalue<alpha:
      print('\n Significant')
    else:
      print('\n Not significant')

    return 

def AUCPR_logit_CI(alpha,theta,Var):
    """
    It transforms the binomial
    approximation variance
    into the
    Delta method 
    variance of AUCPR

    Logit function used to 
    ensure CI bounds
    fall within [0,1].

    The transformation is 
    reversed via expit
    function into
    the original scale.
    """

    low_exp = scipy.special.logit(theta) + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta))
    up_exp = scipy.special.logit(theta) - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)/(theta*(1-theta))
    up_lim = scipy.special.expit(up_exp)
    low_lim = scipy.special.expit(low_exp)

    return np.ravel(up_lim), np.ravel(low_lim)

def plain_AUCPR_CI (alpha,theta, Var):
    """
    Naive CI -> Needs
    to use the 
    binomial- normal
    approximation variance
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(Var)

    return np.ravel(up_lim), np.ravel(low_lim)

def AUCPR_comparison_test(ground_truth,probs1, probs2, alpha):
    aucs = []
    var = []
    aucs.append(average_precision_score(ground_truth,probs1))
    var.append(variance_stand_approx(aucs[0],len(probs1)))
    aucs.append(average_precision_score(ground_truth,probs2)) 
    var.append(variance_stand_approx(aucs[1],len(probs2)))
    return calc_pvalue(aucs = aucs, sigma = var, alpha = alpha)

def AUC_CI(ground_truth,predictions,alpha):
    """
    Gives CI for a
    set of prediction
    probabilities
    and the corresponding
    ground truth 
    labels
    """
    AUCPR = average_precision_score(y_true=ground_truth, y_score=predictions)
    variance = variance_stand_approx(theta_hat=AUCPR, n=len(ground_truth))
    low_lim, up_lim = np.ravel(AUCPR_logit_CI(alpha = alpha,theta = AUCPR,Var = variance))
    #print('AUC = ', AUCPR, 'variance = ',variance,
    #str(int((1-alpha)*100)),'% CI:[',low_lim,',',up_lim,'] \n')

    return AUCPR, variance, low_lim, up_lim




