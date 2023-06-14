import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, mean_squared_error

def compute_metrics(true_model_coefs, model_coefs):
    """
    Compute metrics based on coefficient values of a sparse model compared to the true model.

    Parameters:
        true_model_coefs (array): Coefficient values from the true model (ground truth).
        model_coefs (array): Coefficient values from the model.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    # Create binary labels for the coefficient values (nonzero vs zero)
    true_labels = np.array([1 if coef != 0 else 0 for coef in true_model_coefs])
    model_labels = np.array([1 if coef != 0 else 0 for coef in model_coefs])

    # Compute confusion matrix
    cm_model = confusion_matrix(true_labels, model_labels)

    # non zero coefficients
    non_zero_mask = model_labels>0

    # Compute metrics
    tn_model, fp_model, fn_model, tp_model = cm_model.ravel()
    fpr_model = fp_model / (fp_model + tn_model)
    fnr_model = fn_model / (fn_model + tp_model)
    accuracy_model = accuracy_score(true_labels, model_labels)
    precision_model = precision_score(true_labels, model_labels)
    f1_score_model = f1_score(true_labels, model_labels)
    mse_model = mean_squared_error(true_model_coefs, model_coefs)
    if sum(non_zero_mask)>0:
        mse_model_TN = mean_squared_error(true_model_coefs[non_zero_mask],model_coefs[non_zero_mask])
    else:
        mse_model_TN = np.nan

    # Create a dictionary to store the computed metrics
    metrics = {
        'FPR': fpr_model,
        'FNR': fnr_model,
        'Accuracy': accuracy_model,
        'Precision': precision_model,
        'F1-score': f1_score_model,
        'All coefficients RMSE': np.sqrt(mse_model),
        'True model RMSE': np.sqrt(mse_model_TN)
    }

    return metrics