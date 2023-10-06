import numpy as np
from scipy import stats
from .Utils import contains_val_CI

class GelmanScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.constant_indices = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.constant_indices = np.where(self.std < 1e-8)[0]

    def transform(self, X):
        if self.mean is None or self.std is None or self.constant_indices is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")
        
        # Avoid dividing by std for constant variables
        std_divisor = np.where(self.std >= 1e-8, 2*self.std, 1.0) # we dont use !=0 to account for floating-point numerical precision
        
        return (X - self.mean) / std_divisor

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def Grubb_test(data, val_outlier = 0.0, alpha = 0.05):
    ''''
    Two-sided Grubb's test to determine whether a value is 
    an outlier in a data sample. It assumes the data is 
    normally distributed.

    Args:
        - data (numpy.array): data sample
        - val_outlier (float): value to evaluate as an outlier. (Default: 0.0)
        - alpha (float): test significance level. (Default:0.05)

    Returns:
        boolean indicating significance (True: val_outlier is an outlier) or
        not (False: val_outlier is not an outlier).
    '''
    # Calculate sample mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Calculate the test statistic
    test_statistic = (np.abs(val_outlier - mean)) / np.maximum(1e-20,std_dev)

    # Calculate the critical value from the t-distribution
    n = data.shape[0]
    t_squared = np.square(stats.t.ppf(1 - alpha / (2 * n), n - 2))
    critical_value = ((n-1)/np.sqrt(n))*np.sqrt(t_squared/(n-2+t_squared))

    # Check if the test statistic exceeds the critical value
    return test_statistic > critical_value

def coef_bootstrapped_CI(model,X,y,alpha=0.05,n_bootstrap=100,nonzero_flag = False):
    '''
    Compute confidence intervals based on bootstrapping distribution
    alpha and 1-alpha quantiles.
    Can provide a flag for nonzero coefficients based on 
    its confidence interval including 0 or not.

    Args:
        -model (sklearn object): model instance 
        -X (numpy.arrray): training data 
        -y (numpy.array): labels
        -alpha (float): test significance level. (Default:0.05)
        -n_bootstrap (int): number of bootstrapped repetitions (Default:100)
        -nonzero_flag (bool): whether or not nonzero flags should be returned. 
            (Default=True)

    Returns:
        -confidence_intervals(numpy.array): array of tuples of the form (beta mean,CI_lower,CI_upper)
        - flags (numpy.array): array of boolean flags should be
            returned marking which coefficients are statistically nonzero. Statsitically different
            from zero at 1-alpha confidence is obtained by checking whether the confidence intervals include 
            zero or not. Only returned if nonzero_flag = True.
    ''' 

    #data shape
    n,p = X.shape

    #initialize the coefficient matrix
    coefs = np.array([]).reshape(0,X.shape[1])

    # Perform bootstrapping
    for _ in range(n_bootstrap):
        # Generate a random bootstrap sample with replacement
        bootstrap_sample = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrap, y_bootstrap = X[bootstrap_sample,:],y[bootstrap_sample]

        #train model
        model.fit(X_bootstrap,y_bootstrap)

        #append to the coefficient matrix
        coefs = np.vstack((coefs, model.coef_.ravel()))
    
    # Calculate confidence intervals
    lower_percentile = 100*alpha/2
    upper_percentile = 100-lower_percentile
    lower_bound = np.percentile(coefs, lower_percentile, axis=0)
    upper_bound = np.percentile(coefs, upper_percentile, axis=0)

    # list of tuples one per coefficient
    confidence_intervals = [(np.mean(coefs, axis = 0),lower_bound[i], upper_bound[i]) for i in range(p)]

    if nonzero_flag:
        return confidence_intervals, np.array([not contains_val_CI(CI = (lower_bound[i], upper_bound[i]), val = 0) for i in range(p)])
    else:
        return confidence_intervals


