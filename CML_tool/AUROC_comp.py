import numpy as np
from scipy.stats import norm

from CML_tool.DeLong import delong_roc_variance, Wald_type_DL_CI, DL_logistic_CI

class AurocStats:
    '''
    This class implements several methods to report statistics
    for uncorrelated AUROC results.
    It does however use the Delong method to compute the variance (hence SE) of each AUROC
    since it was proven to be more accurate than the exponential approximation by Hanley 
    and McNeil 1982 (Hajian-Tilaki et al. 2002).
    
    NOTE: For a comparison of correlated AUROC use the DeLong functionality of CML_tool.
    '''
    def __init__(self,
                probs1:np.array=None,
                labels1:np.array=None,
                probs2:np.array=None,
                labels2:np.array=None):
        
        self._probs1=probs1
        self._probs2=probs2
        self._labels1=labels1
        self._labels2=labels2
        
        # Compute Aurocs and their standard errors (via DeLong method)
        self._auroc1, var1=delong_roc_variance(
            ground_truth=self._labels1,
            predictions=self._probs1
        )
        self._auroc2, var2=delong_roc_variance(
            ground_truth=self._labels2,
            predictions=self._probs2
        )
        
        self._se1 = np.sqrt(var1)
        self._se2 = np.sqrt(var2)

    @property
    def probs1(self):
        return self._probs1

    @property
    def probs2(self):
        return self._probs2

    @property
    def labels1(self):
        return self._labels1

    @property
    def labels2(self):
        return self._labels2
    
    @property
    def auroc1(self):
        return self._auroc1

    @property
    def auroc2(self):
        return self._auroc2

    @property
    def se1(self):
        return self._se1

    @property
    def se2(self):
        return self._se2
    
    def comparison_uncorrelated_aurocs(self, alpha:float=0.05):
        '''
        Uses the theory from McNeil and Hanley 1983 and 1984 to compare two
        AUROCs on uncorrelated data.
        Computes the critical ratio and compares it to the normal distribution z-statistic 
        to determine significance given a significance level (alpha).
        
        Args:
            -alpha(float): Significance level (default:0.05)
        Returns:
            -(bool) whether or not the difference in AUROC is significant.
            -ci (tuple): Confidence interval.
            - p_value (float).
        '''
        se_diff = np.sqrt(self._se1**2+self._se2**2) # standard error on the AUROC difference
        auroc_diff = abs(self._auroc1-self._auroc2) # absolute difference
        critical_ratio = auroc_diff/se_diff # U-statistic
        p_value = 2*(1-norm.cdf(critical_ratio)) # p-value on the difference
        z = norm.ppf(1-alpha/2) # z-statistic
        wald_ci = (Wald_type_DL_CI(alpha=alpha, theta=auroc_diff, Var=se_diff**2))# Wald confidence interval
        logistic_ci = (DL_logistic_CI(alpha=alpha, theta=auroc_diff, Var=se_diff**2)) # Logistic confidence interval
        if critical_ratio>=z:
            return True, p_value, wald_ci, logistic_ci 
        else:
            return False, p_value,wald_ci, logistic_ci