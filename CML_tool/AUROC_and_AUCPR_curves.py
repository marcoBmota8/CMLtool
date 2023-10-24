# %%
import os 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from concurrent.futures import ThreadPoolExecutor

# %%
def compute_auroc_sample(sample_idx, data, labels):
    fpr, tpr, _ = roc_curve(labels[sample_idx], data[sample_idx])
    return tpr, fpr

def plot_AUROC(labels:np.array, predictions:np.array, figsize:tuple, n_boot_iters:float or int=5000, alpha:float=0.05, n_jobs:int=1):
    """ Plots the AUROC curve for a given set of labels and predictions.
        Confidence 
    Args:
        labels (np.array): The true labels of the data.
        predictions (np.array): The predicted probabilities of the data.
        figsize (tuple): The size of the figure.
        path (str): The path to save the figure.
        filename (str): The name of the figure.
        n_samples (float): The number of samples to use for bootstrapping.
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
    interp_aucs = []
    
    for tpr, fpr in zip(tprs,fprs):
        interp_tprs = np.vstack([interp_tprs,np.interp(interp_fpr,fpr,tpr)])
        interp_aucs.append(auc(interp_fpr,interp_tprs[-1,:]))
    interp_tprs[:,0] = np.zeros(n_boot_iters)

    # Compute the mean AUROC
    interp_tpr_estimate = np.mean(interp_tprs, axis = 0)
    auroc_estimate = np.mean(interp_aucs)
    
    # Confidence interval percentiles
    lower_percentile = 100*alpha/2
    upper_percentile =100*(1-alpha/2)
    
    # Confidence intervals on sensitivity (tpr)
    lower_ci_tpr = np.percentile(interp_tprs, lower_percentile, axis = 0)
    upper_ci_tpr = np.percentile(interp_tprs, upper_percentile, axis = 0)
    
    # Compute AUROC confidence intervals
    lower_ci = np.percentile(interp_aucs, lower_percentile)
    upper_ci = np.percentile(interp_aucs, upper_percentile)
        
    # Plotting
    fig, ax = plt.subplots(figsize=figsize) 
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax.plot(interp_fpr, interp_tpr_estimate, color="b",label=r"AUROC: %0.3f" % auroc_estimate,lw=2,alpha=1,)
    ax.fill_between(interp_fpr, lower_ci_tpr, upper_ci_tpr, color="dodgerblue", alpha=0.3, label=r"%d%%CI: [%0.3f, %0.3f]" % (int((1-alpha)*100), lower_ci, upper_ci))

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right", fontsize=18)
    ax.set_xlabel('1-Specificity',fontsize=18)
    ax.set_ylabel('Sensitivity',fontsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)


# def AUCPR(model,X,y,positive_label,path,filename):
#     #Plot recall-precision curve with 10-fold CV

#     cv = StratifiedKFold(n_splits=10,shuffle = True)
#     precisions = []
#     APs = []
#     mean_recall = np.linspace(0, 1, 100)

#     fig, ax = plt.subplots(figsize = (8,8))
#     for i, (train, test) in enumerate(cv.split(X,y)):
#         model.fit(X.iloc[train,:],y[train])
#         viz = plot_precision_recall_curve(
#             model,
#             X.iloc[test,:],
#             y[test],
#             name="PRC fold {}".format(i),
#             alpha=0.3,
#             lw=1,
#             ax=ax,
#         )
#         interp_precision = np.interp(mean_recall, viz.recall[::-1], viz.precision[::-1])    
#         interp_precision[0] = 1.0
#         precisions.append(interp_precision)
#         APs.append(viz.average_precision)

#     #calculate the chance for the problem at hand
#     num_pos = len(y[y==positive_label])
#     num_totals = len(y)
#     chance_level = num_pos/num_totals
#     ax.axhline(y=chance_level, xmin=0.57,xmax=1, linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

#     mean_precision = np.mean(precisions, axis=0)
#     mean_precision[-1] = 0.0
#     mean_AP = np.mean(APs)
#     std_auc = np.std(APs)
#     ax.plot(mean_recall,mean_precision,color="b",label=r"Mean PR (AP = %0.2f $\pm$ %0.2f)" % (mean_AP, std_auc),lw=2,alpha=0.8)

#     std_precision = np.std(precisions, axis=0)
#     precision_upper = np.minimum(mean_precision + std_precision, 1)
#     precision_lower = np.maximum(mean_precision - std_precision, 0)
#     ax.fill_between(mean_recall, precision_lower, precision_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.",)

#     ax.set(
#         xlim=[-0.05, 1.05],
#         ylim=[-0.05, 1.05],
#         )

#     ax.legend(loc="lower left", fontsize = 18)
#     ax.set_xlabel('Recall',fontsize = 18)
#     ax.set_ylabel('Precision',fontsize = 18)
#     ax.xaxis.set_tick_params(labelsize=18)
#     ax.yaxis.set_tick_params(labelsize=18)
#     fig.savefig(path+'AUCPR_all_folds_'+filename+'.pdf', dpi = 300)

#     #Mean PR curve
#     fig2, ax2 = plt.subplots(figsize = (6,6))

#     ax2.axhline(y=chance_level, xmin=0,xmax=1, linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
#     ax2.plot(mean_recall,mean_precision,color="b",label=r"Mean PR" "\n" r"(AP = %0.2f $\pm$ %0.2f)" % (mean_AP, std_auc),lw=2,alpha=0.8)
#     ax2.fill_between(mean_recall, precision_lower, precision_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.",)
#     ax2.set(
#         xlim=[-0.05, 1.05],
#         ylim=[-0.05, 1.05],
#         )

#     ax2.legend(loc="lower left", fontsize = 16)
#     ax2.set_xlabel('Recall',fontsize = 18)
#     ax2.set_ylabel('Precision',fontsize = 18)
#     ax2.xaxis.set_tick_params(labelsize=16)
#     ax2.yaxis.set_tick_params(labelsize=16)
#     fig2.savefig(path+'AUCPR_only_mean_'+filename+'.pdf', dpi = 300)

#     return fig, ax, fig2, ax2

# %%
if __name__ == "__main__":
    preds = np.random.random(size = (200))
    labels = np.random.choice([0,1], replace = True, size = (200))
    plot_AUROC(predictions=preds,
           labels=labels,
           figsize=(10,10),
           n_boot_iters=1000,
           alpha = 0.05,
           n_jobs = 24)
# %%
