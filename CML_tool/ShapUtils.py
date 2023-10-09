# 
# %%
from shap.maskers import Impute, Independent, Partition
from shap.explainers import Linear, Exact
from shap.links import logit, identity
from joblib import Parallel, delayed
import logging
import numpy as np
from tqdm import tqdm

# %% 
def calculate_shap_values(
        model,
        background_data,
        training_outcome,
        test_data,
        pretrained = False,
        explainer_type = None,
        link_function = 'identity',
        feature_perturbation = 'interventional',
        exact_masking = 'independent',
        algorithm = 'auto',
        n_samples = 1000,
        max_samples = 1000
        ):
    '''
    Compute Shapley values using Lundberg et al. (2017) KernelSHAP approximation
    given a trained model, the training (background) data and the test data
    to make predictions and calculate shap values on.
    
    Args:
        -model: model instance -> sklearn object
        -background_data: training data used to compute its mean and covariance which
        in turn are used to compute conditional expectations (either observational or 
            interventional). -> numpy.array 
        -training_outcome: labels or target values for the training instances. -> numpy.array
        -test_data: Data for which predictions shapley values are calculated. -> numpy.array
        -pretrained: Boolean indicating whether the passed model object is already trained on the
            background data or not (Default: False) -> boolean
        -explainer_type: 'linear', 'exact'. Linear can only be used with linear models such as 
         logistic regression. Exact is model agnostic. (Default: 'exact') -> str
        -link_function: 'identity' (no-op link function) or 'logit' (useful with classification models
            so that each feature contribution to the probability outcome can be expressed in log-odds) 
            (Default: 'identity') -> str
        -exact_masking: Only relevant to 'exact' explainer: 'independent' or 'correlation'. Whether to 
            consider features independently (computes Shapley values) or enforce a hierarchical structure among
            predictors based on correlation (computes Owen values). (Default: 'independent') -> str
        -feature_perturbation: Only relevant for 'linear' explainer.
            'interventional' or 'observational'. (Default: 'interventional') -> str

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
        
        -algorithm: The algorithm used to estimate the Shapley values. 
        There are many different algorithms that can be used to estimate the Shapley values 
        (and the related value for constrained games), each of these algorithms have various tradeoffs and 
        are preferable in different situations. By default the “auto” options attempts to make the best choice
        given the passed model and masker, but this choice can always be overridden by passing the name of a
        specific algorithm. Options are “auto”, “permutation”, “partition”, “tree”, or “linear”
        The type of algorithm used will determine what type of subclass object is returned by this constructor,
        and you can also build those subclasses directly if you prefer or need more fine grained control over their
        options. (Default:'auto') -> str

        -n_samples: Only useful for feature_perturbation = 'observational'. Number of samples to use when estimating the transformation matrix used 
        to account for feature correlations. LinearExplainer uses sampling to estimate a transform 
        that can then be applied to explain any prediction of the model. (Default:1000) -> int

        -max_samples: The maximum number of samples to use from the passed background data in the independent masker.
        If data has more than max_samples then shap.utils.sample is used to subsample the dataset. 
        The number of samples coming out of the masker (to be integrated over) matches the number of
        samples in the background dataset. 
        This means larger background dataset cause longer runtimes. 
        Normally about 1, 10, 100, or 1000 background samples are reasonable choices. (Default:1000) -> int
        
    Returns:
        -Shapley values as a numpy array of the same shape as test_data.

    '''

    # get the link function callable
    if link_function == 'identity':
        link_function = identity
    elif link_function == 'logit':
        link_function = logit
    else:
        raise ValueError('Wrong link function. Selected between identity and logit')
    
    # Train or use pretrained model
    if not pretrained:
        model.fit(background_data,training_outcome)
    elif pretrained:
        pass
    else:
        raise ValueError('Model pretrainined status mispeified.')

    # define the explainer
    if explainer_type == 'linear':

        if feature_perturbation == 'observational':
            masker = Impute(
                data = background_data,
                method = 'linear'
            )
            feature_perturbation_string = 'correlation_dependent'

            explainer = Linear(
                model = model,
                masker = masker,
                feature_perturbation = feature_perturbation_string,
                link = link_function,
                nsamples = n_samples,
                disable_completion_bar = True
            )

        elif feature_perturbation == 'interventional':
            masker = Independent(
                data = background_data,
                max_samples = max_samples
            )
            feature_perturbation_string = 'interventional'

            explainer = Linear(
                model = model,
                masker = masker,
                feature_perturbation = feature_perturbation_string,
                link = link_function,
                nsamples = n_samples,
                disable_completion_bar = True
            )
        
        else:
            raise ValueError('Invalid option. Choose between "interventional" or "observational".')

        # Compute Shapley values from explainer
        shap_values = explainer.shap_values(test_data)

    elif explainer_type == 'exact':

        #get prediction method from model
        if hasattr(model, 'predict_proba'):
            prediction_function = model.predict_proba
        elif hasattr(model, 'predict'):
            prediction_function = model.predict
        else:
            raise ValueError('Model does not have either predict or predict_proba method.')
        
        if exact_masking == 'independent':

            masker = Independent(
                data = background_data,
                max_samples = max_samples
            )

            explainer = Exact(
                model = prediction_function,
                masker = masker,
                link = link_function,
                linearize_link = True,
            )

        elif exact_masking == 'correlation':

            masker = Partition(
                data = background_data,
                clustering = 'correlation',
                max_samples = max_samples
            )

            explainer = Exact(
                model = prediction_function,
                masker = masker,
                link = link_function,
                linearize_link = False #TODO with the correlation structure and a "logit" link linearizing throws an error
            )
        
        else:
            raise ValueError('Invalid option. Choose between "independent" or "correlation".')

        # Compute Shapley values from explainer
        shap_values = explainer(test_data).values[:,:,1]        

    else:
        raise ValueError('Explainer_type not specified. Please select among "linear" or "exact".')

    #compute and return shapley values
    return shap_values


def CI_shap(
        model,
        background_data,
        training_outcome,
        test_data,
        n_jobs = 1,
        MC_repeats = 1000,
        alpha = 0.05,
        explainer_type = None,
        link_function = 'identity',
        feature_perturbation = 'interventional',
        exact_masking = 'independent',
        algorithm = 'auto',
        n_samples = 1000,
        max_samples = 1000
        ):
    '''
    Compute variability (confidence intervals) in Shapley values via Monte Carlo sampling 
    using bootstrapped samples from training data. Each boostrapped sample is used to train a model
    and serves as background data for each shapley values computation. It makes use of the function 
    calculate_shap_values to make such computation.

    -Args:
        -model: model instance -> sklearn object
        -background_data: training data used to compute its mean and covariance which
            in turn are used to compute conditional expectations (either observational or 
            interventional). -> numpy.array
        -training_outcome: labels or target values for the training instances. -> numpy.array
        -test_data: Data for which predictions shapley values are calculated. -> numpy.array
        -n_jobs: number of threads to use durign parallel computation of the MonteCarlo samples. (Default: 1) -> int
        -MC_repeats: MonteCarlo simulation drawings to estimate the Shapley values distribution.
        (Default:1000) -> int
        -alpha: confidence level. (Default: 0.05) -> float
        -explainer_type: 'linear', 'exact'. Linear can only be used with linear models such as 
            logistic regression. Exact is model agnostic. (Default: 'exact') -> str
        -link_function: 'identity' (no-op link function) or 'logit' (useful with classification models
            so that each feature contribution to the probability outcome can be expressed in log-odds) 
            (Default: 'identity') -> str
        -feature_perturbation: Only relevant for 'linear' explainer. 'interventional' or 'observational'.
            (Default: 'interventional') -> str
        -exact_masking: Only relevant to 'exact' explainer: 'independent' or 'correlation'. Whether to 
            consider features independently (computes Shapley values) or enforce a hierarchical structure among
            predictors based on correlation (computes Owen values). (Default: 'independent') -> str
        -algorithm: The algorithm used to estimate the Shapley values. 
            There are many different algorithms that can be used to estimate the Shapley values 
            (and the related value for constrained games), each of these algorithms have various tradeoffs and 
            are preferable in different situations. By default the “auto” options attempts to make the best choice
            given the passed model and masker, but this choice can always be overridden by passing the name of a
            specific algorithm. Options are “auto”, “permutation”, “partition”, “tree”, or “linear”
            The type of algorithm used will determine what type of subclass object is returned by this constructor,
            and you can also build those subclasses directly if you prefer or need more fine grained control over their
            options. (Default:'auto') -> str
        -n_samples: Only useful for feature_perturbation = 'observational'. Number of samples to use when estimating the transformation matrix used 
            to account for feature correlations. LinearExplainer uses sampling to estimate a transform 
            that can then be applied to explain any prediction of the model. (Default:1000) -> int
        -max_samples: The maximum number of samples to use from the passed background data in the independent masker.
            If data has more than max_samples then shap.utils.sample is used to subsample the dataset. 
            The number of samples coming out of the masker (to be integrated over) matches the number of
            samples in the background dataset. 
            This means larger background dataset cause longer runtimes. 
            Normally about 1, 10, 100, or 1000 background samples are reasonable choices. (Default:1000) -> int
        
    -Returns:
        - 3-tuple containing the shapley values point estimate, lower and upper bounds of 
        the cofidence interval matrices.
    '''

    # Estimate confidence intervals through Monte Carlo sampling via bootstrapped samples of the training dataset
    shap_values_samples = Parallel(n_jobs=n_jobs)(delayed(calculate_shap_values)(
        model = model,
        background_data = background_data[idx,:],
        training_outcome = training_outcome[idx],
        test_data=test_data,
        explainer_type=explainer_type,
        link_function=link_function,
        feature_perturbation=feature_perturbation,
        exact_masking=exact_masking,
        algorithm=algorithm,
        n_samples=n_samples,
        max_samples=max_samples
        ) for idx in tqdm([
            np.random.choice(background_data.shape[0], size=background_data.shape[0], replace=True) for _ in range(MC_repeats)
            ])
        )
    
    # sample_idx=np.random.choice(background_data.shape[0], size=background_data.shape[0], replace=True), #bootstrapped repeats for the MC sampling

    # Calculate the mean of the Shapley values as the point estimate
    mean_shap_values = np.mean(shap_values_samples, axis=0)

    # Calculate confidence intervals
    lower_percentile = 100*alpha/2
    upper_percentile = 100-lower_percentile
    lower_bound = np.percentile(shap_values_samples, lower_percentile, axis=0)
    upper_bound = np.percentile(shap_values_samples, upper_percentile, axis=0)

    # Create a tuple of point estimate, lower bound, and upper bound
    shap_summary = (mean_shap_values, lower_bound, upper_bound)     
    
    return shap_summary
