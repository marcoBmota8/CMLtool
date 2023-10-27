import numpy as np
from scipy import stats
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from Utils import contains_val_CI
from ShapUtils import calculate_shap_values, CI_shap

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
    
class BootstrapLinearModel:
    
    def __init__(self,model,X,y,n_bootstrap=1000):
        '''
        Args:
            -n_bootstrap (float or int): Number of bootsrapped samples to run
            -model (sklearn object): model instance 
            -X (numpy.arrray): training data 
            -y (numpy.array): training labels
        '''
        self.n_bootstrap = n_bootstrap
        self.model = model
        self.X = X
        self.y = y
        
    def coef_CIs(self,alpha=0.05,nonzero_flag = False):
        '''
        Compute confidence intervals and p-value based on 
        bootstrapping distribution alpha and 1-alpha quantiles.
        Can provide a flag for nonzero coefficients based on 
        its confidence interval including 0 or not.

        Args:
            -alpha (float): test significance level. (Default:0.05)
            -n_bootstrap (int): number of bootstrapped repetitions (Default:1000)
            -nonzero_flag (bool): whether or not nonzero flags should be returned. 
                (Default=True)

        Returns:
            -confidence_intervals_E (numpy.array): Bootstrap quantiles CI, array of tuples of the form (beta mean,CI_lower,CI_upper)
            -confidence_intervals_P (numpy.array): Bootstrap pivot-based CI (PREFERRED),array of tuples of the form (beta mean,CI_lower,CI_upper)
            - flags (numpy.array): array of boolean flags should be
                returned marking which coefficients are statistically nonzero. Statsitically different
                from zero at 1-alpha confidence is obtained by checking whether the confidence intervals include 
                zero or not. Only returned if nonzero_flag = True.
        ''' 

        #data shape
        n,p = self.X.shape

        #initialize the coefficient matrix
        coefs = np.array([]).reshape(0,p)

        # Perform bootstrapping
        for _ in range(self.n_bootstrap):
            # Generate a random bootstrap sample with replacement
            bootstrap_sample = np.random.choice(n, size=n, replace=True)
            X_bootstrap, y_bootstrap = self.X[bootstrap_sample,:],self.y[bootstrap_sample]

            #train model
            model_boot = clone(self.model)
            model_boot.fit(X_bootstrap,y_bootstrap)

            #append to the coefficient matrix
            coefs = np.vstack((coefs, model_boot.coef_.ravel()))
        
        # Calculate confidence intervals
        #Bootstrap quantiles CI
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        lower_bound_E = np.percentile(coefs, lower_percentile, axis=0)
        upper_bound_E = np.percentile(coefs, upper_percentile, axis=0)
        
        # Pivot based bootstrapp CI
        estimate = np.mean(coefs, axis=0)
        lower_bound_P = 2*estimate-upper_bound_E
        upper_bound_P = 2*estimate-lower_bound_E
        
        # list of tuples one per coefficient
        confidence_intervals_E = [(estimate,lower_bound_E[i], upper_bound_E[i]) for i in range(p)]
        confidence_intervals_P = [(estimate,lower_bound_P[i], upper_bound_P[i]) for i in range(p)]

        if nonzero_flag:
            return confidence_intervals_E,\
                confidence_intervals_P,\
                np.array([not contains_val_CI(CI = (lower_bound_E[i], upper_bound_E[i]), val = 0) for i in range(p)]), \
                np.array([not contains_val_CI(CI = (lower_bound_P[i], upper_bound_P[i]), val = 0) for i in range(p)])
        else:
            return confidence_intervals_E, confidence_intervals_P
    

def calc_pvalues_LogisticRegression(model, X):
    if isinstance(model, LogisticRegression):
        pass
    else:
        raise ValueError('Passed model object is not an sklearn LogisticRegression')
    
    if model.penalty != 'none':
        warnings.warn("p-values for penalized logistic regression coeffients are not accurate. \
                        Care must be taken when interpreting the reported values.\
                        User is advised to not use the results")
    
    probs = model.predict_proba(X)
    n = len(probs)
    
    if model.fit_intercept:
        x_full = np.matrix(np.insert(np.array(X), 0, 1, axis = 1))
        m = len(model.coef_[0]) + 1
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
    else:
        x_full = np.matrix(X)
        m = len(model.coef_[0])
        coefs = model.coef_[0]
        
    D = np.diag(probs[:,1]*probs[:,0])
    H = x_full.T@D@x_full # compute the Hessian at the evaluated coefficients
    vcov = np.linalg.inv(np.matrix(H)) # covariance matrix of the coefficients
    se = np.sqrt(np.diag(vcov)) # standard error
    t1 =  coefs/se  # t-statistic
    p_values = (1 - stats.norm.cdf(abs(t1))) * 2 # p-values (interncept, beta_1, ..., beta_m)
    
    return p_values


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




