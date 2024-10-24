# %%
import warnings
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from joblib import Parallel, delayed

from CML_tool.Utils import twoarrays_2_tupledf
from CML_tool.ML_Utils import contains_val_CI
from CML_tool.ShapUtils import calculate_shap_values


# %%
def replace_values_in_coef_array(values, positions, out_array_length, fill_value = 0.0):
    out_array = np.full((1,out_array_length),fill_value)
    out_array[0,positions] = values
    return out_array

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
    '''
    Compute the p-value for the regression coefficients of
    a Logistic regression model under the null hypothesis H0: beta_i != 0.
    '''
    if isinstance(model, LogisticRegression):
        pass
    else:
        raise ValueError('Passed model object is not an sklearn LogisticRegression')
    
    if model.penalty != 'none':
        warnings.warn("p-values for penalized logistic regression coeffients are not accurate. \
                        Care must be taken when interpreting the reported values.\
                        User is advised to not use the results.")
    
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
    p_values = (1 - stats.norm.cdf(abs(t1))) * 2 # p-values (intercept, beta_1, ..., beta_m)
    
    return p_values
    
    
class BootstrapLinearModel:
    """
    A class that takes a linear model and implements several methods
    to make inference on its parameters based on bootstrapping.
    """
    
    def __init__(self,model):
        '''
        Args:
            -model (sklearn object): fitted model instance 
        '''
        # Set the coefficients estimates from the full training data (coef_hat)
        if not (hasattr(model, "coef_")):
            raise TypeError("""Model object passed is not fitted. Fit it to the full
                            training data before bulding its BootstrapLinearModel object.""")
        elif hasattr(model, "coef_"):
            self._coef_hat = model.coef_.ravel()
            if hasattr(model, "intercept_"):
                self._intercept_hat = model.intercept_
            else:
                self._intercept_hat = np.full(1,np.nan)
            
        self._model = model 
    
    @property
    def model(self):
        return self._model
    
    @property
    def coef_hat(self):
        return self._coef_hat
    
    @property
    def intercept_hat(self):
        return self._intercept_hat
    
    @staticmethod
    def get_test_data(X,train_idx):
        return X[np.setdiff1d(range(X.shape[0]), train_idx),:]        

    def fit(self, X:np.array,y:np.array,X_HOS:np.array,n_bootstrap:int=1000, n_jobs:int=1):
        '''
        Args:
            -X (numpy.arrray): training data 
            -y (numpy.array): training labels
            -X_HOS (numpy.array): Hold out data
            -n_bootstrap (float or int): Number of bootsrapped samples to run
  
        ------------ Notes -------------
        
        Generate the bootstrapping coefficients and data indices.
        '''
        self.n_bootstrap = n_bootstrap
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.n, self.p = self.X.shape
        self.X_HOS = X_HOS.values if isinstance(X_HOS, pd.DataFrame) else X

        #initialize the coefficient, intercepts and indices matrices
        self.coefs = np.array([]).reshape(0,self.p)
        self.intercepts = np.array([]).reshape(0,1)
        self.train_idx = []

        # Define a helper function for parallelization
        def fit_bootstrap():
            # Generate a random bootstrap sample with replacement
            bootstrap_sample = np.random.choice(self.n, size=self.n, replace=True)
            X_bootstrap, y_bootstrap = self.X[bootstrap_sample,:],self.y[bootstrap_sample]

            # Train model
            model_boot = clone(self.model)
            model_boot.fit(X_bootstrap,y_bootstrap)

            # Return the results
            return model_boot.coef_.ravel(), model_boot.intercept_ if hasattr(model_boot, "intercept_") else None, bootstrap_sample

        # Perform bootstrapping in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(fit_bootstrap)() for _ in tqdm(range(self.n_bootstrap)))

        # Collect the results
        for coef, intercept, bootstrap_sample in results:
            # Stack to the coefficient and intercepts matrix
            self.coefs = np.vstack((self.coefs, coef))
            if intercept is not None:
                self.intercepts = np.vstack((self.intercepts, intercept))

            # Append the boostrap sample instance (rows) indices 
            self.train_idx.append(bootstrap_sample)

    
    def coef_CI(self, alpha:float=0.05, nonzero_flag:bool=True):
        '''
        Compute confidence intervals and point estimate based on 
        bootstrapping distribution alpha and 1-alpha quantiles. 
        Can provide a flag for nonzero coefficients based on 
        its confidence interval not including 0.
               
        Args:
            -alpha (float): test significance level. (Default:0.05)
            -nonzero_flag (bool): whether or not nonzero flags should be returned. (Default=True)

        Returns:
            -Dataframe with columns:
                *'theta_mean': coefficient point estimate (mean)
                *'quantile_ci': Bootstrap quantiles CI (empirical CIs), array of tuples of the form (beta mean,CI_lower,CI_upper)
                *'pivot_ci': Bootstrap pivot-based CI (PREFERRED),array of tuples of the form (beta mean,CI_lower,CI_upper)
                * 'nonzero_quant' and 'nonzero_pivot' (optional): arrays of boolean flags marking which coefficients are statistically nonzero. 
                Statsitically different from zero at 1-alpha confidence is obtained by checking whether the confidence intervals include 
                zero or not. Only returned if nonzero_flag=True.
        ''' 
        
        # Calculate confidence intervals
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        
        if hasattr(self._model, "intercept_"):
            betas_star = np.hstack((self.intercepts,self.coefs))
            betas_hat = np.hstack((self._intercept_hat,self._coef_hat))
            logging.warning(msg='Careful, first elements refer to the intercept.')
        else :
            betas_star = self.coefs
            betas_hat = self._coef_hat
        
        #Bootstrap quantiles CI (empirical confidence intervals)
        lower_bound_E = np.percentile(betas_star, lower_percentile, axis=0)
        upper_bound_E = np.percentile(betas_star, upper_percentile, axis=0)
        
        # Pivot based bootstrapp CI
        lower_bound_P = 2*betas_hat-upper_bound_E
        upper_bound_P = 2*betas_hat-lower_bound_E
        
        # list of tuples one per coefficient
        confidence_intervals_E = [(lower_bound_E[i], upper_bound_E[i]) for i in range(np.shape(betas_hat)[0])]
        confidence_intervals_P = [(lower_bound_P[i], upper_bound_P[i]) for i in range(np.shape(betas_hat)[0])]
        
        # Bootstrap coefficient point estimate
        estimate = np.mean(betas_star, axis=0)

        if nonzero_flag: #Preferred
            df = pd.DataFrame((estimate, \
                confidence_intervals_E,\
                confidence_intervals_P,\
                np.array([not contains_val_CI(CI = (lower_bound_E[i], upper_bound_E[i]), val = 0) for i in range(np.shape(betas_hat)[0])]), \
                np.array([not contains_val_CI(CI = (lower_bound_P[i], upper_bound_P[i]), val = 0) for i in range(np.shape(betas_hat)[0])]))).T
            return df.set_axis(['theta_mean','quantile_ci','pivot_ci','nonzero_quant','nonzero_pivot'], axis = 1)
        else:
            df = pd.DataFrame((estimate, confidence_intervals_E, confidence_intervals_P)).T
            return df.set_axis(['theta_mean','quantile_ci','pivot_ci'], axis = 1)
            
        
    def shap_CI(self, explainer_type:str='linear', link_function:str='identity',
                     feature_perturbation:str='interventional', exact_masking:str='independent',
                     alpha:float=0.05, n_jobs:int=1):
        '''
        Compute shapley values and their confidence intervals from the bootstrapped linear models.
        Relies on CML_tools.ShapUtils which in turns relies on Lundberg's 'shap' package.
        
        Args:
            -explainer_type (str): 'linear' or 'exact'. Linear SHAP (model-specific) is preferred to Exact 
            (model-agnostic). (Default: 'linear') #TODO implement kernelSHAP.
            -link_function (str): 'identity' (no-op link function) or 'logit' (useful with classification models
                so that each feature contribution to the probability outcome can be expressed in log-odds instead of 
                probability units.) (Default: 'identity')
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
                are correlated with (at least one) feature that is. This option uses the Independent masker.
                
            -exact_masking (str): Only relevant to 'exact' explainer: 'independent' or 'correlation'. Whether to 
                consider features independent (computes SHAP values as the unconditional expectation via 
                Shapley sampling values method enumeratign all coealitions. This is t is the correct way to 
                compute the marginal contribution of a feature to a model prediction from a causal perspective 
                (Janzing et al. 2020)) or enforce a hierarchical structure among predictors based on correlation 
                (computes Owen values) when masking. (Default: 'independent') 
            -alpha (float): significance level.
            -n_jobs (int): number of threats/workers to use for parallel computation.
            
        Returns:
            - shap_values_samples (numpy.array): Bootstrapped Shapley values samples.
            - shap_dict (dict):
                * 'shap_mean' (np.array): point estimate (mean) SHAP values for each feature.
                * 'quantile_ci' (pandas.DataFrame): Bootstrap quantiles CI (empirical CIs), each element is a tuple (CI_lower,CI_upper).
                * 'pivot_ci' (pandas.DataFrame): Bootstrap pivot-based CI (PREFERRED), each element is a tuple (CI_lower,CI_upper).
            - feature_importance (pandas.DataFrame): feature importance is measure based on |SHAP|. 
                Rows:
                * fimp: Mean |SHAP|
                * 'quantile_ci': Bootstrap quantiles CI (empirical CIs), each element is a tuple (CI_lower,CI_upper).
                * 'pivot_ci': Bootstrap pivot-based CI (PREFERRED), each element is a tuple (CI_lower,CI_upper).
                 
        '''
        
        # Intercept
        if hasattr(self._model, "intercept_"):   
            intercepts = self.intercepts
            intercept_hat = self._intercept_hat
        else:
            intercepts = np.zeros(self.n_bootstrap)
            intercept_hat = np.zeros(1)
            
        # Bootstrapped samples SHAP values
        shap_values_samples = np.array(Parallel(n_jobs=n_jobs)(
            delayed(calculate_shap_values)(
                model = (self.coefs[i,:], intercepts[i]),
                background_data = self.X[self.train_idx[i],:],
                training_outcome = self.y[self.train_idx[i]],
                test_data=self.X_HOS,
                explainer_type=explainer_type,
                link_function=link_function,
                feature_perturbation=feature_perturbation,
                exact_masking=exact_masking,
                pretrained=True
            ) for i in tqdm(
                list(range(self.n_bootstrap))
                )
        ))
                
        # SHAP values for the full model
        shap_hat = np.array(calculate_shap_values(
                model = (self._coef_hat, intercept_hat),
                background_data = self.X,
                training_outcome = self.y,
                test_data=self.X_HOS,
                explainer_type=explainer_type,
                link_function=link_function,
                feature_perturbation=feature_perturbation,
                exact_masking=exact_masking,
                pretrained=True
        ))
        
        # Calculate the mean of the Shapley values as the point estimate
        estimate = np.mean(shap_values_samples, axis=0)

        # Calculate confidence intervals
        lower_percentile = 100*alpha/2
        upper_percentile = 100-lower_percentile
        
        #Bootstrap quantiles CI (empirical confidence intervals)
        lower_bound_E = np.percentile(shap_values_samples, lower_percentile, axis=0)
        upper_bound_E = np.percentile(shap_values_samples, upper_percentile, axis=0)
        
        # Pivot based bootstrapp CI
        lower_bound_P = 2*shap_hat-upper_bound_E
        upper_bound_P = 2*shap_hat-lower_bound_E
        
        # Hold out set SHAP values dataframes
        shap_dict = {
            'shap_mean': estimate,
            'quantile_ci': twoarrays_2_tupledf(lower_bound_E, upper_bound_E),
            'pivot_ci': twoarrays_2_tupledf(lower_bound_P, upper_bound_P)
        }
        
        # Feature importance is based on mean |SHAP| value
        fimp = np.mean(abs(shap_values_samples), axis=1)
        fimp_hat = np.mean(abs(shap_hat), axis=0)
        
        # Bootstrap quantiles CI (empirical confidence intervals)
        fimp_lower_bound_E = np.percentile(fimp, lower_percentile, axis=0)
        fimp_upper_bound_E = np.percentile(fimp, upper_percentile, axis=0)
        fimp_ci_E = [(fimp_lower_bound_E[i], fimp_upper_bound_E[i]) for i in range(self.p)]
        
        # Pivot based bootstrapp CI
        fimp_lower_bound_P = np.maximum(2*fimp_hat-fimp_upper_bound_E,0) # enforce positivity TODO: Is there a better way to enforce positivity in this CI?
        fimp_upper_bound_P = np.maximum(2*fimp_hat-fimp_lower_bound_E,0) # idem
        fimp_ci_P = [(fimp_lower_bound_P[i], fimp_upper_bound_P[i]) for i in range(self.p)]

        # Bootstrap feature importance point estimate
        fimp_estimate = np.mean(fimp, axis=0)
        
        # Feature importance stats
        feature_importance = (pd.DataFrame(
            (fimp_estimate, fimp_ci_E, fimp_ci_P)).T).set_axis(['fimp_mean','quantile_ci','pivot_ci'], axis = 1)
        
        return shap_values_samples, shap_dict, feature_importance
        
