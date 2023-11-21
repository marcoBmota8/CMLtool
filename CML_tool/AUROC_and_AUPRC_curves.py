# %%
import os 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from concurrent.futures import ThreadPoolExecutor

# %%
def compute_auroc_sample(sample_idx, data, labels):
    fpr, tpr, _ = roc_curve(labels[sample_idx], data[sample_idx])
    return tpr, fpr

def plot_AUROC(labels:np.array, predictions:np.array, figsize:tuple, style:str = None, color:str="b", color_ci:str="dodgerblue", n_boot_iters:float or int=5000, alpha:float=0.05, n_jobs:int=1, ax:plt.axis=None, fig:plt.figure=None):
    """ Plots the AUROC curve for a given set of labels and predictions.
        Confidence 
    Args:
        - labels (np.array): The true labels of the data.
        - predictions (np.array): The predicted probabilities of the data.
        - figsize (tuple): The size of the figure.
        - style (string): Matplotlib to plot the figure with. Currently supporting 'science' besides the default.
        - n_boot_iters (float or integer): Number of bootstrapped repeats to perform to find the confidence intervals and mean curve. (Default=5000)
        - alpha (float): Significance level. (Default=0.05)
        - n_jobs (int): Number of treaths/workers to use. (Default=1)
        -fig: Matplotlib figure object. Needed if one wants to plot on top of an existing figure.
        -ax: Axis of the figure object. Needed if one wants to plot on top of an existing figure.
        
    Returns:
        -fig: Matplotlib figure object.
        -ax: Axis of the figure object.
    """
    
    # Parallelized computation of AUROC samples with a specific number of threads
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        tprs, fprs = zip(*list(
            executor.map(
                compute_auroc_sample,
                [np.random.choice(
                    np.arange(len(predictions)),
                    len(predictions), 
                    replace=True
                    ) for _ in range(n_boot_iters)
                ],
                [predictions] * n_boot_iters, 
                [labels] * n_boot_iters
            )
        ))
    
    interp_fpr = np.linspace(0, 1, 100)
    interp_tprs = np.empty((0,len(interp_fpr)))
    
    for tpr, fpr in zip(tprs,fprs):
        interp_tprs = np.vstack([interp_tprs,np.interp(interp_fpr,fpr,tpr)])
    interp_tprs[:,0] = np.zeros(n_boot_iters)
    interp_tprs[:,-1] = np.ones(n_boot_iters)

    # Compute the mean AUROC based on the mean interpolated tpr
    interp_tpr_estimate = np.mean(interp_tprs, axis = 0)
    auroc_estimate = auc(interp_fpr, interp_tpr_estimate)
    
    # Confidence interval percentiles
    lower_percentile = 100*alpha/2
    upper_percentile =100*(1-alpha/2)
    
    # Confidence intervals on sensitivity (tpr)
    lower_ci_tpr = np.percentile(interp_tprs, lower_percentile, axis = 0)
    upper_ci_tpr = np.percentile(interp_tprs, upper_percentile, axis = 0)
    
    # Compute AUROC confidence intervals
    lower_ci = np.maximum(auc(interp_fpr, lower_ci_tpr),0)
    upper_ci = np.minimum(auc(interp_fpr,upper_ci_tpr),1)
        
    # Plotting
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label=r"Chance: %0.3f" % float(1/2), alpha=0.8)
    ax.plot(interp_fpr, interp_tpr_estimate, color=color,label=r"AUROC: %0.3f" % auroc_estimate,lw=2,alpha=1,)
    if style == 'science':
        plt.style.use("science")
        ax.fill_between(interp_fpr, lower_ci_tpr, upper_ci_tpr, color=color_ci, alpha=0.3, label=r"%d%s CI: [%0.3f, %0.3f]" % (int((1-alpha)*100),'\%', lower_ci, upper_ci))
    else:
        ax.fill_between(interp_fpr, lower_ci_tpr, upper_ci_tpr, color=color_ci, alpha=0.3, label=r"%d%%CI: [%0.3f, %0.3f]" % (int((1-alpha)*100), lower_ci, upper_ci))
    ax.set(
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
    )
    ax.legend(loc="lower right", fontsize=18)
    ax.set_xlabel('1-Specificity',fontsize=18)
    ax.set_ylabel('Sensitivity',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    
    return fig, ax
   
def compute_AUPRC_sample(sample_idx, data, labels):
    precision, recall, _ = precision_recall_curve(labels[sample_idx], data[sample_idx])
    return precision, recall 
    
def plot_AUPRC(labels:np.array, predictions:np.array, figsize:tuple, style:str, color:str="b", color_ci:str="dodgerblue", n_boot_iters:float or int=5000, alpha:float=0.05, n_jobs:int=1, ax:plt.axis=None, fig:plt.figure=None):
    """ Plots the AUPRC curve for a given set of labels and predictions.
        Confidence 
    Args:
        - labels (np.array): The true labels of the data.
        - predictions (np.array): The predicted probabilities of the data.
        - figsize (tuple): The size of the figure.
        - style (string): Matplotlib to plot the figure with. Currently supporting 'science' besides the default.
        - n_boot_iters (float or integer): Number of bootstrapped repeats to perform to find the confidence intervals and mean curve. (Default=5000)
        - alpha (float): Significance level. (Default=0.05)
        - n_jobs (int): Number of treaths/workers to use. (Default=1)
        -fig: Matplotlib figure object. Needed if one wants to plot on top of an existing figure.
        -ax: Axis of the figure object. Needed if one wants to plot on top of an existing figure.
        
    Returns:
        -fig: Matplotlib figure object.
        -ax: Axis of the figure object.
    """
    
    # Compute the chance for the problem at hand
    assert np.array_equal(np.unique(labels),np.array([0,1])), "Please binarize labels array into '0' and '1' before inputting."
    num_pos = len(labels[labels==1])
    num_totals = len(labels)
    chance_level = num_pos/num_totals
    
    # Parallelized computation of AUROC samples with a specific number of threads
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        precisions, recalls = zip(*list(
            executor.map(
                compute_AUPRC_sample,
                [np.random.choice(
                    np.arange(len(predictions)),
                    len(predictions), 
                    replace=True
                    ) for _ in range(n_boot_iters)
                ],
                [predictions] * n_boot_iters, 
                [labels] * n_boot_iters
            )
        ))
    
    interp_recalls = np.linspace(0,1,100)
    interp_precisions = np.empty((0,len(interp_recalls)))
    
    for precision, recall in zip(precisions,recalls):
        interp_precisions = np.vstack([interp_precisions,np.interp(interp_recalls,recall[::-1],precision[::-1])])
    interp_precisions[:,0] = np.ones(n_boot_iters)
    interp_precisions[:,-1] = chance_level*np.ones(n_boot_iters)

    # Compute the mean AUPRC
    interp_precission_estimate = np.mean(interp_precisions, axis = 0)
    AUPRC_estimate = auc(interp_recalls, interp_precission_estimate)

    # Confidence interval percentiles
    lower_percentile = 100*alpha/2
    upper_percentile =100*(1-alpha/2)
    
    # Confidence intervals on sensitivity (tpr)
    lower_ci_precission = np.percentile(interp_precisions, lower_percentile, axis = 0)
    upper_ci_precission = np.percentile(interp_precisions, upper_percentile, axis = 0)
    
    # Compute AUROC confidence intervals
    lower_ci = np.maximum(auc(interp_recalls,lower_ci_precission),0)
    upper_ci = np.minimum(auc(interp_recalls,upper_ci_precission),1)
        
    # Plotting
    if ax==None:
        fig, ax = plt.subplots(figsize=figsize) 
        ax.axhline(y=chance_level, xmin=0,xmax=1, linestyle="--", lw=2, color="r", label=r"Chance: %0.3f" % chance_level, alpha=0.8)
    ax.plot(interp_recalls, interp_precission_estimate, color=color,label=r"AUPRC: %0.3f" % AUPRC_estimate,lw=2,alpha=1,)
    if style == 'science':
        plt.style.use("science")
        ax.fill_between(interp_recalls, lower_ci_precission, upper_ci_precission, color=color_ci, alpha=0.3, label=r"%d%s CI: [%0.3f, %0.3f]" % (int((1-alpha)*100),'\%', lower_ci, upper_ci))
    else:
        ax.fill_between(interp_recalls, lower_ci_precission, upper_ci_precission, color=color_ci, alpha=0.3, label=r"%d%%CI: [%0.3f, %0.3f]" % (int((1-alpha)*100), lower_ci, upper_ci))
    ax.set(
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
    )
    ax.legend(loc="lower right", fontsize=18)
    ax.set_xlabel('Recall',fontsize=18)
    ax.set_ylabel('Precision',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    
    return fig, ax

# %%
if __name__ == "__main__":
    preds = np.random.choice([0.1,0.2,0.22,0.345,0.9,0.99],replace = True,size = (100))
    labels = np.random.choice([0,1], replace = True, size = (100))
    
    fig_test, ax_test = plot_AUROC(
           predictions=preds,
           labels=labels,
           style='science',
           figsize=(7,7),
           n_boot_iters=100,
           alpha = 0.05,
           n_jobs = 24,
           )
    
    plot_AUROC(
           predictions=preds,
           labels=labels,
           style='science',
           figsize=(7,7),
           n_boot_iters=1000,
           alpha = 0.05,
           n_jobs = 24,
           color='firebrick',
           color_ci='sienna',
           ax=ax_test,
           fig=fig_test
    )
    # %%
    plot_AUPRC(predictions=preds,
            labels=labels,
            style='science',
            figsize=(7,7),
            n_boot_iters=5000,
            alpha = 0.05,
            n_jobs = 24
            )
# %%
