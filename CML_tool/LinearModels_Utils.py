# %%
import warnings

import numpy as np
import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from joblib import Parallel, delayed

from Utils import contains_val_CI
from ShapUtils import calculate_shap_values

# %%
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


class GelmanScaler:
    """
    Implementation of the Gelman 2008 scaler. It enables to scale contnuous and binary variables
    to the same scale on the contrary to standard scaler. 
    """
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
    """
    A class that takes a linear model and implements several methods
    to make inference on its parameters based on bootstrapping.
    """
    
    def __init__(self,model):
        '''
        Args:
            -model (sklearn object): model instance 
        '''
        self._model = model #immutable attribute
    
    @property
    def model(self):
        return self._model
    
    @staticmethod
    def get_test_data(cls,X,train_idx):
        return np.setdiff1d(range(X.shape[0]), train_idx)        

    def fit(self, X:np.array,y:np.array,n_bootstrap:int=1000):
        '''
        Args:
            -n_bootstrap (float or int): Number of bootsrapped samples to run
            -X (numpy.arrray): training data 
            -y (numpy.array): training labels
            
        ------------ Notes -------------
        
        Generate the bootstrapping coefficients and data indices.
        '''
        self.n_bootstrap = n_bootstrap
        self.X = X
        self.y = y
        self.n, self.p = X.shape
        
        #initialize the coefficient, intercepts and indices matrices
        self.coefs = np.array([]).reshape(0,self.p)
        self.intercepts = np.array([]).reshape(0,1)
        self.train_idx = []
        
        # Perform bootstrapping
        for _ in range(self.n_bootstrap):
            # Generate a random bootstrap sample with replacement
            bootstrap_sample = np.random.choice(self.n, size=self.n, replace=True)
            X_bootstrap, y_bootstrap = self.X[bootstrap_sample,:],self.y[bootstrap_sample]

            # Train model
            model_boot = clone(self.model)
            model_boot.fit(X_bootstrap,y_bootstrap)

            # Stack to the coefficient and interncepts matrix
            self.coefs = np.vstack((self.coefs, model_boot.coef_.ravel()))
            if model_boot.fit_intercept:
                self.intercepts = np.vstack((self.intercepts, model_boot.interncept_))
            # Append the boostrap sample instance (rows) indices 
            self.train_idx.append(bootstrap_sample)

    
    def coef_CIs(self, alpha:float=0.05, nonzero_flag:bool=False):
        '''
        Compute confidence intervals and point estimate based on 
        bootstrapping distribution alpha and 1-alpha quantiles. 
        Can provide a flag for nonzero coefficients based on 
        its confidence interval including 0 or not.

        Args:
            -alpha (float): test significance level. (Default:0.05)
            -nonzero_flag (bool): whether or not nonzero flags should be returned. (Default=True)

        Returns:
            -mean
            -confidence_intervals_E (numpy.array): Bootstrap quantiles CI, array of tuples of the form (beta mean,CI_lower,CI_upper)
            -confidence_intervals_P (numpy.array): Bootstrap pivot-based CI (PREFERRED),array of tuples of the form (beta mean,CI_lower,CI_upper)
            - flags (numpy.array): array of boolean flags should be
                returned marking which coefficients are statistically nonzero. Statsitically different
                from zero at 1-alpha confidence is obtained by checking whether the confidence intervals include 
                zero or not. Only returned if nonzero_flag = True.
        ''' 
        
        # Calculate confidence intervals
        #Bootstrap quantiles CI
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        lower_bound_E = np.percentile(self.coefs, lower_percentile, axis=0)
        upper_bound_E = np.percentile(self.coefs, upper_percentile, axis=0)
        
        # Pivot based bootstrapp CI
        estimate = np.mean(self.coefs, axis=0)
        lower_bound_P = 2*estimate-upper_bound_E
        upper_bound_P = 2*estimate-lower_bound_E
        
        # list of tuples one per coefficient
        confidence_intervals_E = [(estimate,lower_bound_E[i], upper_bound_E[i]) for i in range(self.p)]
        confidence_intervals_P = [(estimate,lower_bound_P[i], upper_bound_P[i]) for i in range(self.p)]

        if nonzero_flag:
            return estimate, confidence_intervals_E,\
                confidence_intervals_P,\
                np.array([not contains_val_CI(CI = (lower_bound_E[i], upper_bound_E[i]), val = 0) for i in range(self.p)]), \
                np.array([not contains_val_CI(CI = (lower_bound_P[i], upper_bound_P[i]), val = 0) for i in range(self.p)])
        else:
            return estimate, confidence_intervals_E, confidence_intervals_P
        
    def explain_shap(self, explainer_type:str='linear', link_function:str='logit',
                     feature_perturbation:str='interventional', exact_masking:str='independent',
                     alpha:float=0.05, n_jobs:int=1):
        '''
        Compute shapley values and their confidence intervals from the bootstrapped linear models.
        Relies on CML_tools.ShapUtils which in turns relies on Lundberg's 'shap' package.
        
        Args:
            -explainer_type (str): 'linear' or 'exact'. Linear SHAP (model-specific) is preferred to Exact 
            (model-agnostic). (Default: 'linear') #TODO implement kernelSHAP.
            -link_function (str): 'identity' (no-op link function) or 'logit' (useful with classification models
                so that each feature contribution to the probability outcome can be expressed in log-odds) 
                (Default: 'identity')
            -feature_perturbation (str): Only relevant for 'linear' explainer.
            'interventional' or 'observational'. (Default: 'interventional').

                +In the observational scenario ,which is the originally proposed by Lundberg et al 2017,
                we stay "true to the data" (Chen et al. 2020). We compute the full/observational
                conditional expectation so that the model is always evaluated in datapoints within 
                the true data manifold of the problem. This approach respects the correlations between 
                features allowing correlated variables to share credit for the prediction (i.e. SHAP value).
                This option uses the Impute masker.

                +The interventional scenario is an approximation of the observational/full conditional
                expectation. It assumes features are independent so it disregards any correlation among features.
                We stay  "true to the model" (Chen et al, 2020). HOWEVER, from a causal perspective 
                it is the correct way to compute the marginal contribution of a feature to a model prediction 
                as it is analogus to the expectation computed using Pearl's do-operator (Janzing et al. 2020).
                All dependence structures are broken and so it uncovers how the model would behave 
                if we intervened and changed some of the inputs. Shapley values are calculated so that 
                credit for a prediction is only given to the features the model actually uses. The benefit of
                this approach is that it will never give credit to features that are not used by the model but that
                are correlated with (at least one) that are. This option uses the Independent masker.
                
            -exact_masking (str): Only relevant to 'exact' explainer: 'independent' or 'correlation'. Whether to 
                consider features independent (computes SHAP values as the unconditional expectation via 
                Shapley sampling values method enumeratign all coealitions. This is t is the correct way to 
                compute the marginal contribution of a feature to a model prediction from a causal perspective 
                (Janzing et al. 2020)) or enforce a hierarchical structure among predictors based on correlation 
                (computes Owen values) when masking. (Default: 'independent') 
            -alpha (float): significance level.
            -n_jobs (int): number of threats/workers to use for parallel computation.
            
        Returns:
            - point estimate (np.array): mean SHAP values for each feature.
            - lower bounds (np.array): confidence interval lower bounds for each feature.
            - upper bounds (np.array): confidence interval upper bounds for each feature.
        '''
        
        for i in range(self.n_bootstrap):
            
            shap_values_samples = Parallel(n_jobs=n_jobs)(
                delayed(calculate_shap_values)(
                        model = (self.coefs[i,:], self.intercepts[i]),
                        background_data = self.X[self.train_idx[i],:],
                        training_outcome = self.y[self.train_idx[i]],
                        test_data=self.get_test_data(self.X,self.train_idx[i]),
                        explainer_type=explainer_type,
                        link_function=link_function,
                        feature_perturbation=feature_perturbation,
                        exact_masking=exact_masking
                    ) for i in tqdm(list(range(self.n_bootstrap)))
                                                          )
    

        # Calculate the mean of the Shapley values as the point estimate
        estimate = np.mean(shap_values_samples, axis=0)

        # Calculate confidence intervals
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        lower_bound = np.percentile(shap_values_samples, lower_percentile, axis=0)
        upper_bound = np.percentile(shap_values_samples, upper_percentile, axis=0)

        
        return estimate, lower_bound, upper_bound
# %%








