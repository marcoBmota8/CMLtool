import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

from CML_tool.DeLong import delong_roc_variance

def odds_ratio_from_DF(df, treatment, diagnosis):
    '''
    DF columns, 
    treatment & diagnosis, 
    need to be
    one-hot encoded
    '''
    matrix = df.groupby([treatment,diagnosis]).size()
    print(matrix)
    print('Odds ratio: ', matrix[1,1]*matrix[0,0]/(matrix[0,1]*matrix[1,0]))
    
def binary_classifier_metrics(threshold, y_true,probas):
    ''''
    Compute accuracy,sensitivity, specificity,ppv,npv,f1_score
    Returned in that order. 
    '''
        #Computing metrics
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=(probas>threshold).astype(int),
        labels= np.unique(y_true)
        )
    total1=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(cm[0,0]+cm[1,1])/total1
    sensitivity = cm[0,0]/np.maximum(1e-15,(cm[0,0]+cm[0,1]))
    specificity = cm[1,1]/np.maximum(1e-15,(cm[1,0]+cm[1,1]))  
    ppv = cm[0,0]/np.maximum(1e-15,(cm[0,0]+cm[1,0]))
    npv = cm[1,1]/np.maximum(1e-15,(cm[1,1]+cm[0,1]))
    f1_score = 2*sensitivity*ppv/np.maximum(1e-15,(sensitivity+ppv))

    return accuracy, sensitivity, specificity,ppv,npv,f1_score

def overlap_CI(CI1, CI2): 
    '''
    Returns whether two confidence intervals overlap or not.
    
    Args:
        -CI1: Frist confidence interval (tuple)
        -CI2: Second confidence interval (tuple)
    Returns:
        Boolean flag
    '''
    flag = False
    
    l1, u1 = CI1
    l2, u2 = CI2

    if (l1 <= u2) and (l2 <= u1):
        flag = True
    return flag

def contains_val_CI(CI, val):
    '''
    Returns whether a confidence intervals contains a value or not.
    Assumes that confidence interval limits are ordered in increasing order,
    i.e. ci[0]<=ci[1].
    
    Args:
        -CI (tuple): Frist confidence interval (
        -val (float): value to check for
    Returns:
        Boolean flag stating whether or not the value is within the CI or not.
    '''
    if CI[0] < val < CI[1]:
        return True 
    elif (CI[0] == val) and (CI[1] == val):
        return True
    else:
        return False

def ci_proportion(numerator, denominator,alpha):
    '''
    This is the recommended method to obtain the confidence interval of
    a proportion (such as PPV, sensitivity, specificity, etc) in `Statistics with confidence`
    by Altman, Machin, Bryant and  Gardner 2000. The theory is depicted in chapter 6 pages 46-47.
    '''
    z = norm.ppf(alpha/2)
    p=numerator/denominator
    q=1-p
    
    A = 2*numerator+z**2
    B = z*np.sqrt((z**2*4*numerator*q))
    C = 2*(denominator+z**2)
    
    return ((A-B)/C, (A+B)/C)

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
            - p_value (float).
        '''
        se_diff = np.sqrt(self._se1**2+self._se2**2)
        critical_ratio = abs(self._auroc1-self._auroc2)/se_diff
        p_value = 2*(1-norm.cdf(critical_ratio))
        z = norm.ppf(alpha/2)
        if critical_ratio>=z:
            return True, p_value
        else:
            return False, p_value


    