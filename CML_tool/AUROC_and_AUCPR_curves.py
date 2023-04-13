# %%
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

## TODO change both plotting functions so that they operate with predictions and 
#   1. AUROC use DeLong method for CIs.
#   2. AUPR uses the delta method for CIs.
# %%
def AUROC(model,X,y,path,filename):
# Classification and ROC analysis
    cv = StratifiedKFold(n_splits=10, shuffle = True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize = (8,8))
    for i, (train, test) in enumerate(cv.split(X, y,)):

        model.fit(X.iloc[train,:], y[train])
        viz = plot_roc_curve(
            model,
            X.iloc[test,:],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr,mean_tpr,color="b",label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),lw=2,alpha=0.8,)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower,tprs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right", fontsize = 18)
    ax.set_xlabel('1-Specificity',fontsize = 18)
    ax.set_ylabel('Sensitivity',fontsize = 18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    fig.savefig(path+'AUROC_all_folds_'+filename+'.pdf', dpi = 300)

    #ROC only mean
    fig2, ax2 = plt.subplots(figsize = (6,6))

    ax2.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax2.plot(mean_fpr,mean_tpr,color="b",label=r"Mean ROC" "\n" r"(AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),lw=2,alpha=0.8,)
    ax2.fill_between(mean_fpr, tprs_lower,tprs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.")
    ax2.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax2.legend(loc="lower right", fontsize = 16)
    ax2.set_xlabel('1-Specificity',fontsize = 18)
    ax2.set_ylabel('Sensitivity',fontsize = 18)
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    fig2.savefig(path+'AUROC_only_mean_'+filename+'.pdf', dpi = 300)

    return fig, ax, fig2, ax2

#############################################

def AUCPR(model,X,y,positive_label,path,filename):
    #Plot recall-precision curve with 10-fold CV

    cv = StratifiedKFold(n_splits=10,shuffle = True)
    precisions = []
    APs = []
    mean_recall = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize = (8,8))
    for i, (train, test) in enumerate(cv.split(X,y)):
        model.fit(X.iloc[train,:],y[train])
        viz = plot_precision_recall_curve(
            model,
            X.iloc[test,:],
            y[test],
            name="PRC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_precision = np.interp(mean_recall, viz.recall[::-1], viz.precision[::-1])    
        interp_precision[0] = 1.0
        precisions.append(interp_precision)
        APs.append(viz.average_precision)

    #calculate the chance for the problem at hand
    num_pos = len(y[y==positive_label])
    num_totals = len(y)
    chance_level = num_pos/num_totals
    ax.axhline(y=chance_level, xmin=0.57,xmax=1, linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_precision = np.mean(precisions, axis=0)
    mean_precision[-1] = 0.0
    mean_AP = np.mean(APs)
    std_auc = np.std(APs)
    ax.plot(mean_recall,mean_precision,color="b",label=r"Mean PR (AP = %0.2f $\pm$ %0.2f)" % (mean_AP, std_auc),lw=2,alpha=0.8)

    std_precision = np.std(precisions, axis=0)
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)
    ax.fill_between(mean_recall, precision_lower, precision_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.",)

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        )

    ax.legend(loc="lower left", fontsize = 18)
    ax.set_xlabel('Recall',fontsize = 18)
    ax.set_ylabel('Precision',fontsize = 18)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    fig.savefig(path+'AUCPR_all_folds_'+filename+'.pdf', dpi = 300)

    #Mean PR curve
    fig2, ax2 = plt.subplots(figsize = (6,6))

    ax2.axhline(y=chance_level, xmin=0,xmax=1, linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax2.plot(mean_recall,mean_precision,color="b",label=r"Mean PR" "\n" r"(AP = %0.2f $\pm$ %0.2f)" % (mean_AP, std_auc),lw=2,alpha=0.8)
    ax2.fill_between(mean_recall, precision_lower, precision_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.",)
    ax2.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        )

    ax2.legend(loc="lower left", fontsize = 16)
    ax2.set_xlabel('Recall',fontsize = 18)
    ax2.set_ylabel('Precision',fontsize = 18)
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)
    fig2.savefig(path+'AUCPR_only_mean_'+filename+'.pdf', dpi = 300)

    return fig, ax, fig2, ax2