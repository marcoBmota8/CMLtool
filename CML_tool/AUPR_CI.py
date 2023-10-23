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
       aucs: 1D array of AUCs
       sigma_sq: AUC variance
    Returns:
       log10(pvalue)
       test statistic
       CI

    """
    #Individual AUCs
    # I
    up_lim_AUC1, low_lim_AUC1 = np.ravel(Wald_AUCPR_CI(alpha = alpha, theta=aucs[0], n = n))
    print('AUC 1 = ', aucs[0], 'variance = ',sigma_sq[0],
    str(int((1-alpha)*100)),'% CI:[',low_lim_AUC1,',',up_lim_AUC1,'] \n')

    #II
    up_lim_AUC2, low_lim_AUC2 = np.ravel(Wald_AUCPR_CI(alpha = alpha, theta=aucs[1], n = n))
    print('AUC 2 = ', aucs[1], 'variance = ',sigma_sq[1],
    str(int((1-alpha)*100)),'% CI:[',low_lim_AUC2,',',up_lim_AUC2,']\n')
    
    #TEST
    #test statistic to compare two binomial distributions (does not account for correlation as DeLong's test does for AUROC)
    aucpr_diff = aucs[0]-aucs[1]
    theta_hat = (aucs[0]+ aucs[1])/2
    z = abs(aucpr_diff)/np.sqrt(theta_hat*(1-theta_hat)*2/n) # Binomial normal aproximation test statistic

    #P-value
    log10_p_value = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) #calculate p-value while changing its logarithmic base (e->10)
    pvalue = 10**log10_p_value

   #CI
    up_lim, low_lim = binomial_difference_Wald_CI(theta1=aucs[0],theta2=aucs[1],n1=n,n2=n,alpha=alpha)

    print('Z-test test results: log10(p-value)= ', log10_p_value, '(p-value = ',pvalue,'), AUC difference = ',aucpr_diff, 
    str(int((1-alpha)*100)),'% CI:[',up_lim,',',low_lim,']')

    if pvalue<alpha:
      print('\n Significant')
    else:
      print('\n Not significant')


def binomial_difference_Wald_CI(theta1,theta2,n1,n2,alpha):
    theta_diff = theta1-theta2
    standard_error = np.sqrt(theta1*(1-theta1)/n1 + theta2*(1-theta2)/n2)
    z = scipy.stats.norm.ppf(1-alpha/2) * standard_error
    return theta_diff - z, theta_diff + z

def AUCPR_logit_CI(alpha,theta,n):
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

def Wald_AUCPR_CI (alpha,theta, n):
    """
    Binomial Wald confidence interval detailed in Boyd et al. 2013.
    Makes use of the binomial- normal approximation variance
    """
    up_lim = theta + scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(theta*(1-theta)/n) # Boyd et al. 2013
    low_lim = theta - scipy.stats.norm.ppf(1-alpha/2)*np.sqrt(theta*(1-theta)/n) # Boyd et al. 2013

    return np.ravel(low_lim), np.ravel(up_lim)

def AUCPR_comparison_test(ground_truth,probs1, probs2, alpha):
    assert np.unique(ground_truth).tolist() == [0,1], 'Ground truth labels must be binary'
    n = sum(ground_truth)
    aucs = []
    var = []
    aucs.append(average_precision_score(ground_truth,probs1))
    var.append(aucs[0]*(1-aucs[0]))
    aucs.append(average_precision_score(ground_truth,probs2)) 
    var.append(aucs[1]*(1-aucs[1]))
    return calc_pvalue(aucs = aucs, sigma_sq = var, alpha = alpha, n = n)

def AUC_CI(ground_truth,predictions,alpha):
    """
    Gives CI for a
    set of prediction
    probabilities
    and the corresponding
    ground truth 
    labels
    """
    assert np.unique(ground_truth).tolist() == [0,1], 'Ground truth labels must be binary'
    n = sum(ground_truth)
    aucpr = average_precision_score(y_true=ground_truth, y_score=predictions)
    variance = aucpr*(1-aucpr)
    logit_low_lim, logit_up_lim = np.ravel(AUCPR_logit_CI(alpha = alpha,theta = aucpr, n = n))
    wald_low_lim, wald_up_lim = np.ravel(Wald_AUCPR_CI(alpha = alpha,theta = aucpr, n = n))

    return aucpr, variance, wald_low_lim, wald_up_lim, logit_low_lim, logit_up_lim


#%%
if __name__=='__main__':

    # Perfect case
    probs = np.array([1,1,1,1,0,1,0,0,1,1,1,1,1,1])
    gt = np.array([1,1,1,1,0,1,0,0,1,1,1,1,1,1])
    print(AUC_CI(alpha=0.05, ground_truth=gt,predictions=probs))

    
    #Comparison
    probs1 = np.array([0.5,0.6,0.9,0.1,0.001,0.67,0.87,0.35,0.75,0.5,0.5,0.4,0.6,0.7])
    probs2 = np.array([0.45,0.2,0.99,0.001,0.25,0.8,0.4,0.9,0.7,0.5,0.5,0.4,0.6,0.7])
    print(AUCPR_comparison_test(gt,probs1,probs2,alpha = 0.05))

    print(AUC_CI(alpha = 0.05,ground_truth=gt,predictions=probs1))
    print(AUC_CI(alpha = 0.05,ground_truth=gt,predictions=probs2))
# %%
