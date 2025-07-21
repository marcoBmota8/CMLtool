# %%
import warnings
import logging

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from shap.maskers import Independent, Partition, Impute
from shap.explainers import Linear, Exact, Tree, GPUTree
from shap.links import logit, identity

from CML_tool.ML_Utils import compute_empirical_ci, bootstrap_matrix


def calculate_shap_values(
        model,
        background_data,
        training_outcome,
        test_data,
        pretrained = False,
        explainer_type = None,
        link_function = None,
        feature_perturbation = 'interventional_independent',
        n_samples = 1000,
        max_samples = 1000,
        retrieve_explainer = True,
        retrieve_interactions = False,
        **kwargs
        
        ):
    '''
    Compute Shapley values using some of the model-agnostic and model-specific approaches 
    described in Lundberg et al. (2017) given a trained model, the training (background) 
    data and the test data to make predictions and calculate shap values on.
    
    Args:
        -model: model instance (sklearn object) or tuple (coefficients, intercept) for linear models (this requires
        that explainer_type = 'linear').
        -background_data: training data used to compute its mean and covariance which
        in turn are used to compute conditional expectations -> numpy.array 
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
        -feature_perturbation: 'interventional_independent', 'interventional_correlation', or 'observational'. Whether to 
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

        -n_samples: Only useful for feature_perturbation = 'observational' in the linear explainer. Number of samples to use when estimating
            the transformation matrix used to account for feature correlations. (Default:1000) -> int
        -max_samples: The maximum number of samples to use from the passed background data in the independent masker. 
            Use `None` to use the full dataset.
            (Default:1000) -> int or None
        -retrieve_explainer: Whether to return the explainer object used to compute the Shapley values.
            (Default: True) -> bool
        -retrieve_interactions: Whether to return the interactions matrix instead of shapley values.
            This is only possible for TreeExplainer and GPUTreeExplainer.
            (Default: False) -> bool
        
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
    if feature_perturbation == 'interventional_independent':

        masker = Independent(
            data = background_data,
            max_samples = max_samples if max_samples is not None else background_data.shape[0]
        )
        
    elif feature_perturbation == 'interventional_correlation':

        masker = Partition(
            data = background_data,
            clustering = 'correlation',
            max_samples = max_samples if max_samples is not None else background_data.shape[0]
        )
    elif feature_perturbation == 'observational':
        masker = Impute(
            data = background_data
        )
        
    else:
        raise ValueError(f'Invalid option {feature_perturbation}. Choose between "interventional_independent", "interventional_correlation" or "observational".')

    # define the explainer
    # TODO implement the rest of the explainers (KernelSHAP, etc)
    if explainer_type == 'linear':

        explainer = Linear(
            model = model,
            masker = masker if 'interventional' in feature_perturbation else masker,
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
                      
        if "interventional" in feature_perturbation:
            if retrieve_interactions:
                raise NotImplementedError(""" 
                                          TreeExplainer Shapley interactions can only be computed using the `observational`
                                          feature perturbation (i.e. 'tree_path_dependent') which estimates conditional expectations
                                          based on the trained model strcuture and disregards passed background data.
                                          """
                                          )
            if (model_type == 'classifier') & (link_function == 'logit') :
                raise NotImplementedError ("""
                                           Log-odds (logit link) are not supported unless it is the \'raw\' output 
                                        of the individual decission trees as in XGBoostClassifier.
                                        """
                                        ) #TODO: We want Shapley values expressed in log-odds in this case
            elif (model_type == 'classifier') & (link_function == 'identity'):
                model_output = 'probability'
            elif model_type == 'regressor':
                model_output = 'raw'
            else:
                raise ValueError('Faulty model object was passed.')
        
        if explainer_type == 'tree':
            explainer = Tree(
                model=model,
                data=background_data if 'interventional' in feature_perturbation else None, 
                feature_perturbation='interventional' if 'interventional' in feature_perturbation else 'tree_path_dependent',
                model_output=model_output,
                silent=True
            )
        elif explainer_type == 'treeGPU':
            explainer = GPUTree(
                model=model,
                data=background_data if 'interventional' in feature_perturbation else None,
                feature_perturbation='interventional' if 'interventional' in feature_perturbation else 'tree_path_dependent',
                model_output=model_output,
                silent=True
            )
            
        warnings.warn(f"WARNING: Provided Shapley values are in {explainer.model.tree_output} units.")
        
        if retrieve_interactions:
            shap_interactions = np.array(explainer.shap_interaction_values(test_data))[1,...] # interactions for positive class in binary classification problems
            warnings.warn(f"""WARNING: 
                          TreeExplainer Shapley interaction values can only be computed using the observational feature perturbation (conditional expectations).
                          Hence, in general, the sum across features wont add up to the shapley values computed using the interventional feature perturbation (marginal expectactions),
                          but they will if computed using the same feature perturbation (expectation)
                """, UserWarning)   
            if retrieve_explainer:
                return shap_interactions, explainer
            else:
                return shap_interactions
        else:
            shap_values = np.array(explainer.shap_values(test_data))[1,...] # Return the shapley values for positive class in binary classification problems
            if retrieve_explainer:
                return shap_values, explainer
            else:
                return shap_values
    else:
        raise ValueError(f'Explainer_type: {explainer_type} currently not supported.')
    
    
    if retrieve_explainer:
        return shap_values, explainer
    else:   
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
        feature_perturbation = 'interventional_independent',
        n_samples = 1000,
        max_samples = 1000,
        ci_type=None,
        return_samples=False,
        return_agg=True,
        retrieve_interactions=False,
        **kwargs
        ):
    '''
    Compute empirical variability and confidence intervals of Shapley values. 
    This function makes use of calculate_shap_values to compute each Shapley value sample and then 
    calculates the bootstrapped confidence intervals.

    -Args:
        -model: model instance -> sklearn object
         -background_data: training data used to compute its mean and covariance which
        in turn are used to compute conditional expectations -> numpy.array 
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
        -feature_perturbation: 'interventional_independent', 'interventional_correlation' or 'observational'. For explanation see calculate_shap_values docstring.
            (Default: 'interventional_independent') -> str
        -n_samples: Only useful for feature_perturbation = 'observational' in linear explainer. Number of samples to use when estimating the transformation matrix used 
            to account for feature correlations. (Default:1000) -> int
        -max_samples: The maximum number of samples to use from the passed background data in the independent masker. 
            Use `None` to use the full dataset. (Default:1000) -> int or None
        -ci_type: What confidence interval to compute from the empirical distribution: `pivot` or `quantile` based. Default is to not return neither (Default: None)
        -return_samples: Whether or not to return the full samples matrix.(Default: False).
        -return_agg: Whather or not to return aggregations of shapley values (mav) or shapley interaction values (main and total indirect effetcs) across the set instances,
            e.g. patients. (Default: True)
        -retrieve_interactions: Whether to return the interactions matrix instead of shapley values. Only possible for explainer_type='tree'.
        
    -Returns:
        - point estimate (np.array): mean SHAP values for each feature.
        - lower bounds (np.array): confidence interval lower bounds for each feature.
        - upper bounds (np.array): confidence interval upper bounds for each feature.
        - shap_values_samples (list): List with all shapley values matrices used to compute CIs. Only returned if `return_samples=True`. 

    '''
    
    assert return_mav!=return_samples, 'Cannot return both mean absolute values and the full samples matrix'
    if 'interventional' in feature_perturbation:
        assert isinstance(background_data, np.ndarray), '`background_data` must be a numpy array.'
        assert isinstance(training_outcome, np.ndarray), '`training_outcome` must be a numpy array.'
    
    assert isinstance(test_data, np.ndarray), '`test_data` must be a numpy array.'
    
    assert np.array_equal(test_outcomes, test_outcomes.astype(bool).astype(float)), '`test_outcomes` array is not binary. This function is only meant for binary classification problems.'
    
    # Calculate point estimates Shapley values on test data
    point_estimates, explainer = calculate_shap_values(
            model = model,
            background_data = background_data,
            training_outcome = training_outcome,
            pretrained=False,
            test_data=test_data,
            explainer_type=explainer_type,
            link_function=link_function,
            feature_perturbation=feature_perturbation,
            n_samples=n_samples,
            max_samples=max_samples,
            retrieve_explainer=True,
            retrieve_interactions=retrieve_interactions
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
            max_samples=max_samples,
            retrieve_explainer=False,
            retrieve_interactions=retrieve_interactions
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
            max_samples=max_samples,
            retrieve_explainer=False,
            retrieve_interactions=retrieve_interactions
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

    if return_agg:
        if not retrieve_interactions:
            # Return mean absolute values of the Shapley samples
            return {
                'point_estimates' : point_estimates,
                'explainer': explainer,
                'shaps': np.mean(abs(shap_values_samples), axis=0).T
            }
        elif retrieve_interactions:
            # Return direct and indirect effects
            main_effects, indirect_effects = get_main_interaction_effects(
                shap_int=shap_values_samples,
                total_split_effects=True,
                mav_indirect=False
            )
            return {
                'point_estimates' : point_estimates,
                'direct_effects': main_effects, 
                'indirect_effects': indirect_effects, 
                'explainer': explainer,
            }
    elif return_samples:
        if ci_type is None:
            return {
            'shap_samples': shap_values_samples,
            'explainer': explainer,
            'point_estimates': point_estimates
        }
        else: 
            return {
                'shaps_samples':shap_values_samples,
                'shaps_point_estimates':point_estimates,
                'explainer': explainer,
                'ci_lower_bounds':np.array(lower_bounds),
                'ci_upper_bounds':np.array(upper_bounds)
                }
    else:
        logging.info('Neither Shapley samples matrix nor mean absoliute values were requested. Thus defaulting to returning the full samples matrix, explainer and point estimates...')
        return {
            'shap_samples': shap_values_samples,
            'point_estimates': point_estimates,
            'explainer': explainer
        }
        
def get_main_interaction_effects(shap_int:np.ndarray, total_split_effects:bool=True, mav_indirect:bool=False):
    
    n,p,p = shap_int.shape
    
    # direct effects
    main_effects = np.diagonal(shap_int, axis1=1, axis2=2)
    # indirect effects
    mask = ~np.eye(p, dtype=bool)
    off_diagonals = shap_int[:, mask].reshape(n,p-1,p, order='F') # Fortran order (rows reading slower than columns) to avoid mixing off diagonal elements from different features in the reduced matrix
    
    if total_split_effects: # Get the total indirect effects per feature and instance considering signs
        ind_effects=off_diagonals.sum(axis=1)
        return main_effects, ind_effects
    elif mav_indirect: # Get the mean absolute value of the indirect effects per feature and instance
        ind_effects = np.mean(abs(off_diagonals), axis=1)
        return main_effects, ind_effects
    else:
        raise ValueError('''
                         Either total_split_effects or mav_indirect must be True. 
                         Alternatively, set them both to False if no aggregation is desired. This option returns the separation into main and indirect effects but is highly memory hungry.
                         '''
                         )
        