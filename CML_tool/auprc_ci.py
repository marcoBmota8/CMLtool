# %%
import warnings

import numpy as np
import scipy.stats 
import scipy.special
from sklearn.metrics import average_precision_score

warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
def calc_pvalue(aucs, sigma_sq, alpha,n):
    """Computes the AUCPR difference test (Z-statistic difference) giving a p-value and whether
    the difference is significant or not. The test is based on the binomial-normal approximation.
    It also returns a Wald confidence interval for the difference in AUCPRs.
    
        Args:
            aucs (list): List of two AUCPRs.
            sigma_sq (list): List of two variances. # Unused
            alpha (float): Significance level.
            n (int): Number of positive samples.
        Returns:
            pvalue (float): P-value for the test.
            ci (tuple): Confidence interval for the difference in AUCPRs.
            significance (bool): Whether the difference is significant or not.

    """
    #test statistic to compare two binomial distributions (does not account for correlation as DeLong's test does for AUROC)
    aucpr_diff = aucs[0]-aucs[1]
    theta_hat = (aucs[0]+ aucs[1])/2
    z = abs(aucpr_diff)/np.sqrt(theta_hat*(1-theta_hat)*2/n) # Binomial normal aproximation test statistic

    #P-value
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) #calculate p-value while changing its logarithmic base (e->10)
    pvalue = 10**log10_p_value

   # Confidence interval
    up_lim, low_lim = binomial_difference_wald_ci(theta1=aucs[0],theta2=aucs[1],n1=n,n2=n,alpha=alpha)

    if pvalue<alpha:
        significance = True
    else:
        significance = False
        
    return 10**log10_p_value, (up_lim, low_lim), significance

def binomial_difference_wald_ci(theta1,theta2,n1,n2,alpha):
    theta_diff = theta1-theta2
    standard_error = np.sqrt(theta1*(1-theta1)/n1 + theta2*(1-theta2)/n2)
    z = scipy.stats.norm.ppf(1-alpha/2) * standard_error
    return theta_diff - z, theta_diff + z

def auprc_logit_ci(alpha,theta,n):
    """
    Assymmetric logit confidence interval.
    Makes use of the logistic tranformation to ensure that the confidence
    intverval bounds fall within [0,1]. Variance is estimated via the 
    biomial-normal approximation and the delta method (Boyd et al. 2013). 
    """
    
    up_exp = scipy.special.logit(theta) + scipy.stats.norm.ppf(1-alpha/2)/np.sqrt(n*theta*(1-theta)) # Boyd et al. 2013
    low_exp = scipy.special.logit(theta) - scipy.stats.norm.ppf(1-alpha/2)/np.sqrt(n*theta*(1-theta)) # Boyd et al. 2013
    up_lim = scipy.special.expit(up_exp)
    low_lim = scipy.special.expit(low_exp)
    # Handle the extreme cases where theta = 0 or theta = 1
    if np.isnan(low_lim): 
        low_lim = theta
    if np.isnan(up_lim):
        up_lim = theta
        
    return np.ravel(low_lim), np.ravel(up_lim)

def wald_auprc_ci (alpha,theta, n):
    """
    Binomial Wald confidence interval detailed in Boyd et al. 2013.
    Makes use of the binomial- normal approximation variance
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(theta*(1-theta)/n) # Boyd et al. 2013
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(theta*(1-theta)/n) # Boyd et al. 2013

    return np.ravel(low_lim), np.ravel(up_lim)

def auprc_comparison_test(ground_truth,probs1, probs2, alpha):
    assert np.unique(ground_truth).tolist() == [0,1], 'Ground truth labels must be binary'
    n = sum(ground_truth)
    aucs = []
    var = []
    aucs.append(average_precision_score(ground_truth,probs1))
    var.append(aucs[0]*(1-aucs[0]))
    aucs.append(average_precision_score(ground_truth,probs2)) 
    var.append(aucs[1]*(1-aucs[1]))
    return calc_pvalue(aucs = aucs, sigma_sq = var, alpha = alpha, n = n)

def auc_ci(ground_truth,predictions, ci_type='wald', alpha=0.05):
    """
    Computes the AUC and its confidence interval.
    
        Args:
            ground_truth (array): True labels.
            predictions (array): Predicted probabilities.
            ci_type (str): Type of confidence interval. 'logistic' or 'wald'. Default is 'wald'.
            alpha (float): Significance level. Default is 0.05.
        Returns:
            aucpr (float): AUC value.
            variance (float): Variance of the AUC.
            ci (tuple): Confidence interval for the AUC.
    """
    assert np.unique(ground_truth).tolist() == [0,1], 'Ground truth labels must be binary'
    n = sum(ground_truth)
    aucpr = average_precision_score(y_true=ground_truth, y_score=predictions)
    variance = aucpr*(1-aucpr)
    if ci_type == 'logistic':
        ci_low_lim, ci_up_lim = np.ravel(auprc_logit_ci(alpha = alpha,theta = aucpr, n = n))
    elif ci_type == 'wald':
        ci_low_lim, ci_up_lim = np.ravel(wald_auprc_ci(alpha = alpha,theta = aucpr, n = n))
        if ci_low_lim < 0:
            ci_low_lim = 0
        if ci_up_lim > 1:
            ci_up_lim = 1
    else:
        raise ValueError("ci_type must be either 'logistic' or 'wald'")
    return aucpr, variance, (ci_low_lim, ci_up_lim)
