import numpy as np

from scipy.stats import norm, skew, skewtest
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity 

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

def compute_empirical_ci(X: np.array, pivot:np.array=None, alpha:float=0.05, type:str='pivot'):
        '''
        Compute confidence intervals of a data matrix in each of its dimensions (i.e. columns).
        This can be done either on the data matrix itself or in a bootstrapped sample of it.
               
        Args:
            -X (numpy.array or list of numpy.array): It can be a 1)data matrix with measurements (rows) to compute CIs across each dimension (columns); or 
                2)list of measurement matrices to compute CIs across instances (e.g. patients) (rows) and dimensions (columns).
                In essence, if a 2D array is passed it assumes rows are the samples whereas if a list is passed each matrix is considered a differente sample.
            -pivot(numpy.array): If X contains the results of a function applied to bootrsapped
                samples of the original observations, `external_pivot` must provide the result of that same 
                function on the original observations. (Default: None)
            -alpha (float): significance level. (Default:0.05)
            -type (str): Type of empirical confidence interval to return
                *'quantile': Empirical distribution quantiles. (Default)
                *'pivot': Pivot-based confidence interval. Requires a `pivot` as input.
    
        Returns:
            -ci_list (list): List of tuples of the form (CI_lower, CI_upper), where
                if X is a 2D array: there is one tuple per passed dimension each containing two floats
                if X is a list of 2D array: there is one tuple per instance (e.g. patient) each containing two 1D arrays 
                of length equal to the number of dimensions.
                
        ''' 
        
        # Calculate confidence intervals
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
                    
        # Compute quantiles
        lower_bound = np.percentile(X, lower_percentile, axis=0)
        upper_bound = np.percentile(X, upper_percentile, axis=0)
        
        if type == 'quantile':
            return [(lower_bound[i], upper_bound[i]) for i in range(np.shape(X)[1])]
        elif type == 'pivot': 
            lower_bound_pivot = 2*pivot-upper_bound
            upper_bound_pivot = 2*pivot-lower_bound
            return [(lower_bound_pivot[i], upper_bound_pivot[i]) for i in range(np.shape(X)[1])]
        else:
            raise ValueError (f'{type} is not a valid confidence interval type.')
        
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

def bootstrap_matrix(matrix, n_bootstraps=1000, random_state=None):
    """
    Perform bootstrap sampling on matrix rows and stack results along a third axis
    to form the resulting matrix
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix to be bootstrapped
    n_bootstraps : int, optional (default=1000)
        Number of bootstrap samples to generate
    random_state : int, optional
        Seed for reproducibility
    
    Returns:
    --------
    numpy.ndarray
        Bootstrapped matrices stacked along a new third axis
    """
    # Create an array to hold bootstrapped samples
    bootstrapped_matrices = np.zeros((*matrix.shape, n_bootstraps), 
                                     dtype=matrix.dtype)
    
    # Perform bootstrap sampling
    for i in range(n_bootstraps):
        # Resample rows with replacement
        bootstrapped_matrices[:,:,i] = resample(matrix, 
                                            replace=True, 
                                            n_samples=matrix.shape[0],
                                            random_state=random_state)
    
    return bootstrapped_matrices

def compute_skewness(X:np.array, indices:np.array=slice(None), significant_filter:bool=True, alpha:float=0.05):
    '''
    Compute adjusted (bias corrected) Fisher-Pearson's coefficients of skewness in a data matrix.
    This function assumes that the rows of X are instances and columns are variables.
    It computes skewness on the rows of X indicated by the array `indices` for each column. 
    By default it computes it over the full matrix X.
    
    Args:
     - X (numpy.array): Data matris of shape n_instances * p_dimensions.
     - indices (numpy.array): Row indices across which to compute skewness.
     - significant_filter (bool): Whether or not replace by zero skeness values
        that are not statistically different from zero (normal distribution 
        skewness). (Default: True)
     - alpha (float): two-sided test significance level (Default: 0.05) 
         
    
    Output:
     - skewness_coefficients of shape (p_dimensions,)
     
     '''
    assert len(X.shape)==2, 'Input array X must be 2-dimensional.'
    
    fp_coeff = skew(
        a = X[indices, :],
        nan_policy='omit',
        bias=False,
        keepdims=False,
        axis=0
        )
    
    if significant_filter:
        pvalues =  skewtest(
            a = X,
            axis=0,
            nan_policy='omit',
            alternative='two-sided',
            keepdims=False
            )[1]
        
        fp_coeff = [val if (pvalues[i]>=alpha) else 0 for i, val in enumerate(fp_coeff)]
    
    return fp_coeff

def compute_1dkde_curve(x, x_grid, bandwidth, kernel='gaussian'):
    
    assert x_grid.ndim == 1, f"Wrong `x_grid` shape, got {x_grid.shape}"    
    assert x.ndim == 1, f"Wrong `x` shape, got {x.shape}"    
    
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    kde.fit(x[:,None])
    logprob = kde.score_samples(x_grid[:,None])
    return np.exp(logprob)