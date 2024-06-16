import numpy as np 

from CML_tool.ML_Utils import binary_classifier_metrics
import CML_tool.DeLong as DL 

def classifier_performance_and_save_model(model,X_train,y_train,X_HOS,y_HOS,threshold_resolution=0.01, final_params=None, **kwargs):
    
    # Set best model
    if final_params is not None:
        model.set_params(**final_params)

    # Fit best model
    model.fit(
        X_train, 
        y_train
        )

    #Compute HOS probabilities
    HOS_all_probs = model.predict_proba(X_HOS)[:,1]

    # Compute DeLong AUROC
    AUC,var,(low_lim_wald, up_lim_wald), (low_lim_logit, up_lim_logit) = DL.AUC_CI(
    ground_truth=y_HOS,
    predictions=HOS_all_probs,
    alpha = 0.05
    )

    # Thresholds to computes metrics at
    thres = np.linspace(0,1,int(1+1/threshold_resolution))
    
    #Compute metrics for each threshold
    results = []
    for val in thres:
        results.append(binary_classifier_metrics(
            y_true=y_HOS,
            probas=HOS_all_probs,
            threshold=val
        ))

    performance_dict = {
        'DeLong_AUC': (AUC, low_lim_logit, up_lim_logit),
        'thresholds':thres,
        'accuracy':list(list(zip(*results))[0]),
        'sensitivity':list(list(zip(*results))[1]),
        'specificity':list(list(zip(*results))[2]),
        'PPV':list(list(zip(*results))[3]),
        'NPV':list(list(zip(*results))[4]),
        'f1-score':list(list(zip(*results))[5]),
        'HOS_probs': HOS_all_probs,
        'true_HOS_labels': y_HOS,
        'final_model': model
    }
    
    return performance_dict