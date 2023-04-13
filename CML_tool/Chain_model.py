# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 
import numpy as np
from aenet import AdaptiveElasticNet

# %%
class Chain_model:
    def __init__(self, penalty = 'elasticnet', C_EN = 1, C_L2 = 1, l1_ratio = 0.5,gamma = 1, alpha = 1, C_coupling = False,
        solver='saga',max_iter = 4000, warm_start = False, tol = 1e-4, eps_coef = 1e-21, positive_tol = 1e-15):
        #params
        self.C_EN = C_EN # Inverse of the strenght of the regularization (naive EN)
        self.C_L2 = C_L2
        self.C_coupling = C_coupling # Whether or not to use the C of EN for the L2 pass
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.alpha = alpha #This is the strength of the regularization (Aenet loss function formulation if different than naive EN)
        self.eps_coef = eps_coef # Small constant to prevent zero division in adaptive weights and C_L2
        self.positive_tol = positive_tol # Numerical optimization (cvxpy) may return slightly negative coefs.
        #If coef > -positive_tol, ignore this and forcively sets negative coef to zero. Otherwise, raise ValueError.
        self.max_iter = max_iter
        self.tol = tol 
        self.solver = solver
        self.warm_start = warm_start
        self.penalty = penalty

    def fit(self,X,y):
        #fit the feature selection model
        if self.penalty == 'elasticnet':
            self.FS = LogisticRegression(penalty = 'elasticnet', C = self.C_EN,
                l1_ratio = self.l1_ratio, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif self.penalty=='adanet':
            self.FS = AdaptiveElasticNet(alpha = self.alpha, l1_ratio = self.l1_ratio, gamma = self.gamma, 
                max_iter = self.max_iter, tol = self.tol, eps_coef = self.eps_coef, solver = 'OSQP', fit_intercept = True)
        elif self.penalty == 'l1':
            self.FS = LogisticRegression(penalty = 'l1', C = self.C_EN, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol) 
        else:
            raise TypeError("Wrong feature selection model input string: 'elasticnet', 'adanet' or 'l1' (LASSO).")
        #Set the classes that the model deals with
        self.classes_ = np.unique(y)
        #fit the feature selection model
        self.FS.fit(X,y)
        #Select non-zero parameters
        features_names = X.columns.values
        self.sel_features = features_names[np.where(abs(self.FS.coef_.ravel()) > 0)] #Ordered by name
        #Fit parameter estimation model
        if (self.C_coupling == True) and (self.penalty == 'elasticnet'):
            self.L2_model = LogisticRegression(penalty = 'l2', C = self.C_EN/(np.max([self.eps_coef,1-self.l1_ratio])), solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif (self.C_coupling == True) and (self.penalty == 'adanet'):
            self.L2_model = LogisticRegression(penalty = 'l2', C = 1/(np.max([self.eps_coef,(1-self.l1_ratio)*self.alpha])), solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif (self.C_coupling == True) and (self.penalty == 'l1'):
            self.L2_model = LogisticRegression(penalty = 'l2', C = self.C_EN, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        elif self.C_coupling == False:
            self.L2_model = LogisticRegression(penalty = 'l2', C = self.C_L2, solver=self.solver,
                max_iter = self.max_iter, warm_start= self.warm_start, tol = self.tol)
        else:
            raise ValueError('Wrong C_coupling specification.')

        #Handle the exception of not selecting any feature
        if len(self.sel_features)==0:
            self.X_sel = np.zeros(X.shape)
            self.L2_model.fit(self.X_sel,y)
            self.L2_model.intercept_ = np.zeros(1)
        else:
            self.X_sel = X.loc[:,self.sel_features]
            self.L2_model.fit(self.X_sel,y)

        #Set the L2 pass coefficients
        self.coef_ = self.L2_model.coef_[0]
        #Set the coefficients along the zeroed coefficients 
        self.coef_zeros_ = np.zeros(len(self.FS.coef_[0]))
        if np.sum(abs(self.FS.coef_))!=0:
            self.coef_zeros_[np.where(self.FS.coef_!=0)[1]] = self.coef_


        return self

    def get_params(self,deep = True):
        return {'C_EN':self.C_EN, 'C_L2':self.C_L2, 'C_coupling': self.C_coupling, 'gamma':self.gamma, 'alpha':self.alpha, 'l1_ratio': self.l1_ratio,
             'solver': self.solver, 'warm_start': self.warm_start,'eps_coef':self.eps_coef, 'positive_tol':self.positive_tol,
             'tol':self.tol, 'max_iter':self.max_iter,'penalty':self.penalty,
             }

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

    def score_FS(self, X_test, y_test):
        #score based on feature selection algorithm
        return roc_auc_score(y_score=self.FS.predict_proba(X_test)[:,1], y_true=y_test)\

    def FS_pred_prob(self, X):
        return self.FS.predict_proba(X)

    def score(self, X_test, y_test):
        #AUROC Score based on final model estimation
        if len(self.sel_features)==0: #Handle the exception of not selecting any feature
            pass
        elif self.X_sel.shape != X_test.shape and len(X_test.shape)>1:
            X_test = X_test.loc[:,self.sel_features]
        elif self.X_sel.shape != X_test.shape and len(X_test.shape)==1:
            X_test = X_test.loc[self.sel_features].to_frame().T
        return roc_auc_score(y_score=self.L2_model.predict_proba(X_test)[:,1], y_true=y_test)
    
    def selected_features_ordered(self, how):
        '''Return features selected ordered by coefficient absolute value (absolute) or plain value (signed)'''
        if (how == 'absolute') and (len(self.sel_features)>0):
            return self.sel_features[np.argsort(self.coef_)][::-1]
        elif (how == 'signed') and (len(self.sel_features)>0):
            return self.sel_features[np.argsort(abs(self.coef_))][::-1]
        elif (len(self.sel_features)==0):
            return self.sel_features
        else:
            raise ValueError('Wrong "how" string passed. Use "absolute" or "signed" instead.')
        return
            
# %%
