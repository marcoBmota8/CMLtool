import numpy as np
import pandas as pd

class GelmanScaler:
    """
    Implementation of the Gelman 2008 scaler. It enables to scale continuous and binary variables
    to the same scale on the contrary to standard scaler. It scales continuous variables but leaves 
    binary ones untouched.
    """
    def __init__(self, eps=1e-8):
        self._mean = None
        self._stdev = None
        self._constant_indices = None
        self._binary_indices = None
        self._log_indices = None
        self._eps = eps
    
    @property
    def mean_(self):
        return self._mean

    @property
    def stdev_(self):
        return self._stdev

    @property
    def constant_indices(self):
        return self._constant_indices
    
    @property
    def binary_indices_(self):
        return self._binary_indices
    
    @property
    def log_indices_(self):
        return self._log_indices
    
    @property
    def eps_(self):
        return self._eps
    
    @ property
    def std_divisor_(self):
        return self._std_divisor
        
    @staticmethod
    def get_binary_indices(X):
        if isinstance(X, pd.DataFrame):
            data=X.values.astype(float)
        elif isinstance(X, np.ndarray):
            data=X.astype(float)
        else:
            raise NotImplementedError("Passed data type not supported.")
        return [i for i in range(data.shape[1]) if set(np.unique(data[:,i])).issubset({0.0,1.0}) or set(np.unique(data[:,i])).issubset({True,False})]
        
    def fit(self, X, log_indices:list):
        self._log_indices=log_indices
        if self.log_indices_ is not None:
       
            if isinstance(X, pd.DataFrame):
                X = X.applymap(float)  # Correct DataFrames dtype to perform operations with float Series
                X.loc[:,self.log_indices_] = np.log(X.loc[:, self.log_indices_]+self.eps_)
            elif isinstance(X, np.ndarray):
                X[:,self.log_indices_] = np.log(X[:, self.log_indices_]+self.eps_)
            else:
                raise ValueError('Unsuported input matrix data type.')
        self._mean = np.mean(X, axis=0)
        self._stdev = np.std(X, axis=0)
        self._constant_indices = np.where(self._stdev < self.eps_)[0]
        self._binary_indices = self.get_binary_indices(X)
        self._std_divisor = np.where(self.stdev_ >= self.eps_, 2*self.stdev_, 1.0) # Avoid dividing by 0 for constant variables
        
        return self
            
    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() before transform().")
        
        # Correct DataFrames dtype to perform operations with float Series
        if isinstance(X, pd.DataFrame):
            X = X.applymap(float)
    
        
        # Keep binary as binary
        if isinstance(X, pd.DataFrame):
            if self.log_indices_ is not None: # Apply logarithm transformation to log indices
                X.loc[:, self.log_indices_] = np.log(X.loc[:, self.log_indices_] + self.eps_)
            # Rescaling and mean centering
            X_scaled = (X - self.mean_) / self.std_divisor_
            X_scaled.iloc[:,self.binary_indices_] = X.iloc[:,self.binary_indices_] 
        elif isinstance(X, np.ndarray):
            if self.log_indices_ is not None: # Apply logarithm transformation to log indices
                X[:, self.log_indices_] = np.log(X[:, self.log_indices_] + self.eps_)
            # Rescaling and mean centering
            X_scaled = (X - self.mean_) / self.std_divisor_
            X_scaled[:,self.binary_indices_] = X[:,self.binary_indices_] 
        else:
            raise NotImplementedError("Passed data type not supported.")

        return X_scaled

    def fit_transform(self, X, log_indices):
        self.fit(X,log_indices)
        return self.transform(X)
    
    def inverse_transform(self, X):
        assert np.array_equal(self.get_binary_indices(X), self.binary_indices_), \
        'Fitted data binary variables are different than passed data for inverse tranform.'
        
        # Reverse the standard scaling for all columns
        X_reverted = X * self.std_divisor_ + self.mean_
        
        if isinstance(X, pd.DataFrame):        
            X_reverted.loc[:, self.log_indices_] =  np.power(10, X_reverted.loc[:,self.log_indices_]) - self.eps_ # Return log transformed columns to its orginal scale
            X_reverted.iloc[:,self.binary_indices_] = X.iloc[:,self.binary_indices_] 
    elif isinstance(X, np.ndarray):
            X_reverted[:, self.log_indices_] =  np.power(10, X_reverted[:,self.log_indices_]) - self.eps_ # Return log transformed columns to its orginal scale
            X_reverted[:,self.binary_indices_] = X[:,self.binary_indices_] 
        else:
            raise NotImplementedError("Passed data type not supported.")

        return X_reverted
