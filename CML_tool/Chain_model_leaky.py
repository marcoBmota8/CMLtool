# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
import numpy as np
from aenet import AdaptiveElasticNet

# %%
class Chain_model:
    def __init__(self,X_full , y_full, penalty = 'elasticnet', C_EN = 1, C_L2 = 1, C_coupling = False, l1_ratio = 0.5,gamma = 1, alpha = 1,
        solver='saga',max_iter = 4000, warm_start = False, tol = 1e-4, eps_coef = 1e-21, positive_tol = 1e-15):
        #params
        self.C_EN = C_EN # Inverse of the strenght of the regularization (naive EN)
        self.C_L2 = C_L2
        self.C_coupling = C_coupling # Whether or not to use the C of EN for the L2 pass
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.alpha = alpha #This is the strangeth of the regularization (Aenet loss function formulation if different than naive EN)
        self.eps_coef = eps_coef # Small constant to prevent zero division in adaptive weights
        self.positive_tol = positive_tol # Numerical optimization (cvxpy) may return slightly negative coefs.
        #If coef > -positive_tol, ignore this and forcively sets negative coef to zero. Otherwise, raise ValueError.
        self.max_iter = max_iter
        self.tol = tol 
        self.solver = solver
        self.warm_start = warm_start
        self.X_full = X_full
        self.y_full = y_full
        self.penalty = penalty

    def fit(self,X,y):
        '''
        Requires X to be a dataframe
        and y a Series with 1-2-1 correlation 
        order to X instances (label <-> data)

        Make sure both are float (64 or 32) datatype,
        int may give issues
        '''
        #fit the feature selection model
        if self.penalty == 'elasticnet':
            self.EN = LogisticRegression(penalty = 'elasticnet', C = self.C_EN,
                l1_ratio = self.l1_ratio, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif self.penalty=='adanet':
            self.EN = AdaptiveElasticNet(alpha = self.alpha, l1_ratio = self.l1_ratio, gamma = self.gamma, 
                max_iter = self.max_iter, tol = self.tol, eps_coef = self.eps_coef, solver = 'OSQP', fit_intercept = True)
        else:
            raise TypeError("Wrong feature selection model input string: 'elasticnet' or 'adanet'.")
        self.EN.fit(self.X_full,self.y_full)

        #Select non-zero parameters
        features_names = X.columns.values
        self.sel_features = features_names[np.where(abs(self.EN.coef_[0]) > 0)[0]]
        #Fit parameter estimation model
        if self.C_coupling:
            self.L2_model = LogisticRegression(penalty = 'l2', C = self.C_L2, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif not self.C_coupling:
            self.L2_model = LogisticRegression(penalty = 'l2', C = self.C_EN, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        else:
            raise ValueError('Wrong C_coupling specification.')
        #Handle the exception of not selecting any feature
        if len(self.sel_features)==0:
            self.X_sel = np.zeros(X.shape)
            self.L2_model.fit(self.X_sel,y)
            #self.L2_model.intercept_ = np.zeros(1)
        else:
            self.X_sel = X.loc[:,self.sel_features]
            self.L2_model.fit(self.X_sel,y)

    def get_params(self,deep = True):
        return {'C_EN':self.C_EN, 'C_L2':self.C_L2, 'gamma':self.gamma, 'alpha':self.alpha, 'l1_ratio': self.l1_ratio,
             'solver': self.solver, 'warm_start': self.warm_start,'eps_coef':self.eps_coef, 'positive_tol':self.positive_tol,
             'tol':self.tol, 'max_iter':self.max_iter,'penalty':self.penalty,
             'X_full': self.X_full, 'y_full': self.y_full}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict_proba(self,X):
        if len(self.sel_features)==0: # Handle exception of not selecting any feature
            pass
        elif self.X_sel.shape != X.shape and len(X.shape)>1:
            X = X.loc[:,self.sel_features]
        elif self.X_sel.shape != X.shape and len(X.shape)==1:
            X = X.loc[self.sel_features].to_frame().T
        return self.L2_model.predict_proba(X)
    
    def decision_function(self,X):
        if len(self.sel_features)==0: #Handle the exception of not selecting any feature
            pass
        elif self.X_sel.shape != X.shape and len(X.shape)>1:
            X_selected = X.loc[:,self.sel_features]
        elif self.X_sel.shape != X.shape and len(X.shape)==1:
            X_selected = X.loc[self.sel_features].to_frame().T
        return self.L2_model.decision_function(X_selected)
    
    def classes_(self):
        return self.L2_model.classes_

    def score_EN(self, X_test, y_test):
        #score based on feature selection and EN (naive(double-shrinkage) or Adaptive) estimation 
        return roc_auc_score(y_score=self.EN.predict_proba(X_test)[:,1], y_true=y_test)\

    def EN_pred_prob(self, X):
        return self.EN.predict_proba(X)[:,1]

    def score(self, X_test, y_test):
        #Score based on final model estimation
        if len(self.sel_features)==0: #Handle the exception of not selecting any feature
            pass
        elif self.X_sel.shape != X_test.shape and len(X_test.shape)>1:
            X_test = X_test.loc[:,self.sel_features]
        elif self.X_sel.shape != X_test.shape and len(X_test.shape)==1:
            X_test = X_test.loc[self.sel_features].to_frame().T
        return roc_auc_score(y_score=self.L2_model.predict_proba(X_test)[:,1], y_true=y_test)

    def coef_(self):
        coefficients = self.L2_model.coef_
        return coefficients


# %%
