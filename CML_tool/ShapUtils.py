# %%
import warnings
import logging

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from shap.maskers import Independent, Partition 
from shap.explainers import Linear, Exact, Tree, GPUTree
from shap.links import logit, identity

from CML_tool.ML_Utils import compute_empirical_ci, bootstrap_matrix


# %% 
def calculate_shap_values(
        model,
        background_data,
        training_outcome,
        test_data,
        pretrained = False,
        explainer_type = None,
        link_function = None,
        feature_perturbation = 'interventional',
        n_samples = 1000,
        max_samples = 1000
        ):
    '''
    Compute Shapley values using some of the model-agnostic and model-specific approaches 
    described in Lundberg et al. (2017) given a trained model, the training (background) 
    data and the test data to make predictions and calculate shap values on.
    
    Args:
        -model: model instance (sklearn object) or tuple (coefficients, intercept) for linear models (this requires
        that explainer_type = 'linear').
        -background_data: training data used to compute its mean and covariance which
        in turn are used to compute conditional expectations (either observational or 
            interventional). -> numpy.array 
        -training_outcome: labels or target values for the training instances. -> numpy.array
        -test_data: Data for which predictions shapley values are calculated. -> numpy.array
        -pretrained: Boolean indicating whether the passed model object is already trained on the
            background data or not. If False, model is fitted to the background data passed (Default: False) -> bool
        -explainer_type: 'linear', 'exact', 'tree', 'treeGPU'. Linear SHAP (linear models) and TreeExplainer (Lundberg et al. 2020) (tree-based models) 
            are model-specific. Exact is model-agnostic. -> str 
            #TODO implement the fast model agnostic kernelSHAP (Linear LIME + Shapley values) 
        -link_function: The link function used to map between the output units of the model to the SHAP value units.
            'identity' (no-op link function, for binary classification this keeps Shapley values as probability) or 
            'logit' (with binary classification models this option expresses each feature Shapley value as log-odds) 
            (Default: 'identity') -> str
        -feature_perturbation: 'interventional' or 'observational'. Whether to 
            consider features 'interventional' (computes SHAP values as the unconditional expectation via 
            Shapley sampling values method enumerating all coalitions. This is the correct way to 
            compute the marginal contribution of a feature to a model prediction from a causal perspective 
            (Janzing et al. 2020)) or enforce a hierarchical structure among predictors based on 'correlation' 
            (computes Owen values) when masking. (Default: 'interventional') -> str

            +In the observational scenario ,which is the originally proposed by Lundberg et al 2017,
              we stay "true to the data" (Chen et al. 2020). We compute the full/observational
              conditional expectation so that the model is always evaluated in datapoints within 
              the true data manifold of the problem. This approach respects the correlations between 
              features allowing correlated variables to share credit for the prediction (i.e. SHAP value).
              For linear models this option uses the Impute masker.

            +The interventional scenario is an approximation of the full/observational conditional
              expectation using marginal expectation instead.
              It assumes features are independent so it disregards any correlation among features.
              We stay  "true to the model" (Chen et al, 2020). HOWEVER, from a causal perspective 
              it is the correct way to compute the marginal contribution of a feature to a model prediction 
              as it is analogus to the expectation computed using Pearl's do-operator (Janzing et al. 2020).
              All dependence structures are broken and so it uncovers how the model would behave 
              if we intervened and changed some of the inputs. Shapley values are calculated so that 
              credit for a prediction is only given to the features the model actually uses. The benefit of
              this approach is that it will never give credit to features that are not used by the model but that
              are correlated with (at least one) that are. This option uses the Independent masker.

        -n_samples: Only useful for feature_perturbation = 'observational'. Number of samples to use when estimating
        the transformation matrix used to account for feature correlations. (Default:1000) -> int
        -max_samples: The maximum number of samples to use from the passed background data in the independent masker. 
        (Default:1000) -> int
        
    Returns:
        -Shapley values as a numpy array of the same shape as test_data.

    '''

    # get the link function callable
    if link_function == 'identity':
        link_function_func = identity
    elif link_function == 'logit':
        link_function_func = logit
    else:
        raise ValueError("""Wrong link function. Select between identity and logit carefully depending on your model.
                         Note: for logistic regression identity gives probability and logot log-odds.
                         """)
    
    # Train or use pretrained model
    if not pretrained:
        model.fit(background_data,training_outcome)
    elif pretrained:
        pass
    else:
        raise ValueError('Model pretrainined status mispeified.')

    #Get prediction method from model and model type
    if hasattr(model, 'predict_proba'): # Classifiers
        prediction_function = model.predict_proba
        model_type = 'classifier'
    elif not hasattr(model, 'predict_proba') & hasattr(model, 'predict'): # Regressors
        prediction_function = model.predict
        model_type = 'regressor'
    else:
        raise ValueError(f"Model does not have either predict or predict_proba method.\
                        Check that the model object you are passing is correct.")
    
    # Define the masker based on the feature perturbation
    if feature_perturbation == 'interventional':

        masker = Independent(
            data = background_data,
            max_samples = max_samples
        )
        
    elif feature_perturbation == 'observational':

        masker = Partition(
            data = background_data,
            clustering = 'correlation',
            max_samples = max_samples
        )
        #TODO consider including masker Impute when missing data exists(?)

    else:
        raise ValueError('Invalid option. Choose between "interventional" or "observational".')

    # define the explainer
    # TODO implement the rest of the explainers (KernelSHAP, etc)
    if explainer_type == 'linear':

        explainer = Linear(
            model = model,
            masker = masker,
            feature_perturbation = "correlation_dependent" if feature_perturbation == 'observational' else feature_perturbation,
            link = link_function_func,
            nsamples = n_samples,
            disable_completion_bar = True
        )
        
        # Warnings
        if (model_type=="classifier") & (link_function=='identity'):
            warnings.warn(
                f"""WARNING: Selected '{link_function}' as link function with a classifier model and LinearExplainer.
                LinearExplainer ignores the passed link function and always uses identity pre logistic transformation.
                Hence, shapley values are computed in log-odds space.
                For marginal probability contributions apply a logistic function to the provided output.
                """, UserWarning)

        if feature_perturbation == 'observational':
            warnings.warn(
                """WARNING: Feature perturbation is set to observational with LinearExplainer.
                The LinearExplainer uses the Impute masker intead of the Partition masker.
                          """, UserWarning)
        
        # Compute Shapley values
        shap_values = explainer.shap_values(test_data)
        
    elif explainer_type == 'exact':
        
        explainer = Exact(
            model = prediction_function,
            masker = masker,
            link = link_function_func,
            linearize_link = True,
        )
        
        # Compute Shapley values
        if model_type == 'classifier':
            shap_values = explainer(test_data).values[...,1] # Positive class shapley values
        else:
            # TODO: support exact explainer for regression models.
            raise ValueError('Regressors currently not supported')
    
    elif (explainer_type == 'tree') or (explainer_type == 'treeGPU'):
        
        # get model output
        if feature_perturbation=='observational':
            model_output='raw'
            feature_perturbation = 'tree_path_dependent'
            warnings.warn("""WARNING: Passed background data is not being used. 
                          TreeSHAP with 'observational' (i.e. 'tree_path_dependent') feature perturbation
                          uses the trees pathways followed by training samples during learning 
                          to obtain the backgroun distribution and compute observational expectations.
                          """, UserWarning)
            warnings.warn(f"""WARNING: Passed link_function: \'{link_function}\' is ignored.
                          'tree_path_dependent' feature perturbation forces 'raw' as the output format being 
                          explained by Shapley values. This is the output from the decission trees. 
                          For most sklearn classifiers expressed in probability, but for XGBoost classifier
                          this is log-odds ratio (i.e. marginal contribution of the feature in log-odds units). 
                          Note that post-computation transformation (e.g. log-odds -> prob via a logistic) cannot be done 
                          exactly for Tree SHAP values.
                          """, UserWarning)      
                      
        if feature_perturbation=="interventional":
            if (model_type == 'classifier') & (link_function == 'logit') :
                raise NotImplementedError ("""Log-odds (logit link) are not supported unless it is the \'raw\' output 
                                        of the individual decission trees as in XGBoostClassifier.
                                        """) #TODO: We want Shapley values expressed in log-odds in this case
            elif (model_type == 'classifier') & (link_function == 'identity'):
                model_output = 'probability'
            elif model_type == 'regressor':
                model_output = 'raw'
            else:
                raise ValueError('Faulty model object was passed.')
        
        if explainer_type == 'tree':
            explainer = Tree(
                model=model,
                data=background_data if feature_perturbation == 'interventional' else None,
                masker=masker,
                feature_perturbation=feature_perturbation,
                model_output=model_output,
                silent=True
            )
        elif explainer_type == 'treeGPU':
            explainer = GPUTree(
                model=model,
                data=background_data if feature_perturbation == 'interventional' else None,
                masker=masker,
                feature_perturbation=feature_perturbation,
                model_output=model_output,
                silent=True
            )
            
        warnings.warn(f"WARNING: Provided Shapley values are in {explainer.model.tree_output} units.")
        
        # Compute Shapley values
        shap_values = np.array(explainer.shap_values(test_data))[1,...]
        
        # Return the shapley values for positive class in binary classification problems

    else:
        raise ValueError(f'Explainer_type: {explainer_type} currently not supported.')

    #compute and return shapley values
        
    return shap_values

def CI_shap(
        model,
        background_data,
        training_outcome,
        test_data,
        test_outcomes,
        randomness_distortion="bootstrapping_test_set",
        n_jobs = 1,
        MC_repeats = 1000,
        alpha = 0.05,
        explainer_type = None,
        link_function = 'identity',
        feature_perturbation = 'interventional',
        n_samples = 1000,
        max_samples = 1000,
        ci_type=None,
        return_samples=False,
        return_mav=True,
        
        ):
    '''
    Compute empirical variability and confidence intervals of Shapley values. 
    This function makes use of calculate_shap_values to compute each Shapley value sample and then 
    calculates the bootstrapped confidence intervals.

    -Args:
        -model: model instance -> sklearn object
         -background_data: training data used to compute its mean and covariance which
        in turn are used to compute conditional expectations (either observational or 
            interventional). -> numpy.array 
        -training_outcome: labels or target values for the training instances. -> numpy.array
        -test_data: Data for which predictions shapley values are calculated. -> numpy.array
        -test_outcome: labels or target values for the test instances. -> numpy.array
        - randomness_distortion: Whether to compute CIs based on boostrapped samples of the training data ("bootstrapping_train_data"),
            different randon seeds of the model during training on the full datatset ("seeds"),
            or bootstrapped reapeats of the test set ("bootstrapping_test_set")-> str (Default:"bootstrapping_test_set")
        -n_jobs: number of threads to use durign parallel computation of the MonteCarlo samples. (Default: 1) -> int
        -MC_repeats: MonteCarlo simulations to estimate the Shapley values distribution.
            (Default:1000) -> int
        -alpha: confidence level. (Default: 0.05) -> float
        -explainer_type: 'linear', 'exact' or 'tree'. Linear SHAP (linear models) and TreeExplainer (Lundberg et al. 2020) (tree-based models) 
            are model-specific. Exact is model-agnostic.) -> str 
        -link_function: The link function used to map between the output units of the model to the SHAP value units.
            'identity' (no-op link function, for binary classification this keeps Shapley values as probability) or 
            'logit' (with binary classification models this option expresses each feature Shapley value as log-odds) 
            (Default: 'identity') -> str
        -feature_perturbation: 'interventional' or 'observational'. For explanation see calculate_shap_values docstring.
            (Default: 'interventional') -> str
        -n_samples: Only useful for feature_perturbation = 'observational'. Number of samples to use when estimating the transformation matrix used 
            to account for feature correlations. (Default:1000) -> int
        -max_samples: The maximum number of samples to use from the passed background data in the independent masker. (Default:1000) -> int
        -ci_type: What confidence interval to compute from the empirical distribution: `pivot` or `quantile` based. Default is to not return neither (Default: None)
        -return_samples: Whether or not to return the full samples matrix.(Default: False).
        -return_mav: Whather or not to return mean absolute value shap values across the set instances, e.g. patients. (Default: True)
        
    -Returns:
        - point estimate (np.array): mean SHAP values for each feature.
        - lower bounds (np.array): confidence interval lower bounds for each feature.
        - upper bounds (np.array): confidence interval upper bounds for each feature.
        - shap_values_samples (list): List with all shapley values matrices used to compute CIs. Only returned if `return_samples=True`. 

    '''
    
    assert return_mav!=return_samples, 'Cannot return both mean absolute values and the full samples matrix'

    assert isinstance(background_data, np.ndarray), '`background_data` must be a numpy array.'
    assert isinstance(training_outcome, np.ndarray), '`training_outcome` must be a numpy array.'
    assert isinstance(test_data, np.ndarray), '`test_data` must be a numpy array.'
    
    assert np.array_equal(test_outcomes, test_outcomes.astype(bool).astype(float)), '`test_outcomes` array is not binary.'

    if (ci_type is not None) or (randomness_distortion == 'bootstrapping_test_set'):
        # Calculate point estimates Shapley values on test data
        point_estimates = calculate_shap_values(
                model = model,
                background_data = background_data,
                training_outcome = training_outcome,
                pretrained=False,
                test_data=test_data,
                explainer_type=explainer_type,
                link_function=link_function,
                feature_perturbation=feature_perturbation,
                n_samples=n_samples,
                max_samples=max_samples
                )
    
    if randomness_distortion == 'train_data_bootstrapping':
        # Estimate confidence intervals through Monte Carlo sampling via bootstrapped samples of the training dataset
        shap_values_samples = Parallel(n_jobs=n_jobs)(delayed(calculate_shap_values)(
            model = model,
            background_data = background_data[idx,:],
            training_outcome = training_outcome[idx],
            pretrained=False,
            test_data=test_data,
            explainer_type=explainer_type,
            link_function=link_function,
            feature_perturbation=feature_perturbation,
            n_samples=n_samples,
            max_samples=max_samples
            ) for idx in tqdm([
                np.random.choice(background_data.shape[0], size=background_data.shape[0], replace=True) for _ in range(MC_repeats)
                ])
            )
        # Transform samples into an array
        shap_values_samples = np.stack(shap_values_samples, axis=-1)
        
    elif randomness_distortion == 'seeds':
        shap_values_samples = Parallel(n_jobs=n_jobs)(delayed(calculate_shap_values)(
            model = model.set_params(**{'random_state':seed}),
            background_data = background_data,
            training_outcome = training_outcome,
            pretrained=False,
            test_data=test_data,
            explainer_type=explainer_type,
            link_function=link_function,
            feature_perturbation=feature_perturbation,
            n_samples=n_samples,
            max_samples=max_samples
            ) for seed in tqdm(np.random.choice(range(MC_repeats), size=MC_repeats, replace=False))
            )
        # Transform samples into an array
        shap_values_samples = np.stack(shap_values_samples, axis=-1)
        
    elif randomness_distortion == 'bootstrapping_test_set':
        shap_values_samples = bootstrap_matrix(
            matrix=point_estimates,
            n_bootstraps=MC_repeats,
            random_state=None)
    
    else:
        raise ValueError(f"`{randomness_distortion}` is not a supported option as `randomness_distortion` input.")
    
    if ci_type is not None:
        # Compute confidence intervals bounds 
        lower_bounds, upper_bounds = zip(*compute_empirical_ci(
            X=shap_values_samples,
            pivot=point_estimates, # Only used when ci_type='pivot'
            alpha=alpha,
            type=ci_type
        ))

    if return_mav:
        return {
            'point_estimates' : point_estimates,
            'shaps': np.mean(abs(shap_values_samples), axis=0).T
        }
    elif return_samples:
        if ci_type is None:
            return {
            'shap_samples': shap_values_samples,
            'point_estimates': point_estimates
        }
        else: 
            return {
                'shaps_samples':shap_values_samples,
                'shaps_point_estimates':point_estimates,
                'ci_lower_bounds':np.array(lower_bounds),
                'ci_upper_bounds':np.array(upper_bounds)
                }
    else:
        logging.info('Neither Shapley sampels matrix nor mean absoliute values were requested. Thus defaulting to returning the full samples matrix and point estimates...')
        return {
            'shap_samples': shap_values_samples,
            'point_estimates': point_estimates
        }