import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

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

def compute_empirical_ci(X: np.array, alpha:float=0.05, type:str='pivot', bootstrap_repeats_pivot: int=5000):
        '''
        Compute confidence intervals of a data matrix with (repetition/bootstrapped)
        samples across several dimensions.
               
        Args:
            -X (numpy.array): Data matrix with measurements (rows) and dimensions (columns).
            -alpha (float): significance level. (Default:0.05)
            -type (str): Type of empirical confidence interval to return
                *'quantile': Empirical distribution quantiles.
                *'pivot': Pivot-based CI (Default).
            -bootstrap_repeats_pivot (int): How many bootstrapped samples use to estimate the pivot
                for a pivot-based confidence interval.

        Returns:
            -ci_list (list): List of tuples of the form (CI_lower,CI_upper), one per passed dimension in X.
                
        ''' 
        
        # Calculate confidence intervals
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        
        if type == 'quantile':
            # Compute quantiles
            lower_bound = np.percentile(X, lower_percentile, axis=0)
            upper_bound = np.percentile(X, upper_percentile, axis=0)
        elif type == 'pivot':
            # Compute pivot 
            bootstrap_means = np.array([np.mean(X[np.random.choice(X.shape[0], X.shape[0], replace=True)],axis=0) for _ in range(bootstrap_repeats_pivot)])
            bootstrap_lower_quant = np.percentile(bootstrap_means, lower_percentile, axis=0)
            bootstrap_upper_quant = np.percentile(bootstrap_means, upper_percentile, axis=0)   
                     
            mean_X = np.mean(X, axis=0)
            lower_bound = 2*mean_X-bootstrap_upper_quant
            upper_bound = 2*mean_X-bootstrap_lower_quant
        else:
            raise ValueError (f'{type} is not a valid confidence interval type.')
        
        return [(lower_bound[i], upper_bound[i]) for i in range(np.shape(X)[0])]

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
    elif ((CI[0] == val) and (CI[1] != val)) or ((CI[1] == val) and (CI[0] != val)):
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

