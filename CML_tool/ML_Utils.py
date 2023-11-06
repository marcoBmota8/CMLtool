import numpy as np
from sklearn.metrics import confusion_matrix


def odds_ratio_from_DF(df, treatment, diagnosis):
    '''
    DF columns, 
    treatment & diagnosis, 
    need to be
    one-hot encoded
    '''
    matrix = df.groupby([treatment,diagnosis]).size()
    print(matrix)
    print('Odds ratio: ', matrix[1,1]*matrix[0,0]/(matrix[0,1]*matrix[1,0]))
    
def binary_classifier_metrics(threshold, y_true,probas):
    ''''
    Compute accuracy,sensitivity, specificity,ppv,npv,f1_score
    Returned in that order. 
    '''
        #Computing metrics
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=(probas>threshold).astype(int),
        labels= np.unique(y_true)
        )
    total1=sum(sum(cm))
    #####from confusion matrix calculate accuracy
    accuracy=(cm[0,0]+cm[1,1])/total1
    sensitivity = cm[0,0]/np.maximum(1e-15,(cm[0,0]+cm[0,1]))
    specificity = cm[1,1]/np.maximum(1e-15,(cm[1,0]+cm[1,1]))  
    ppv = cm[0,0]/np.maximum(1e-15,(cm[0,0]+cm[1,0]))
    npv = cm[1,1]/np.maximum(1e-15,(cm[1,1]+cm[0,1]))
    f1_score = 2*sensitivity*ppv/np.maximum(1e-15,(sensitivity+ppv))

    return accuracy, sensitivity, specificity,ppv,npv,f1_score

def ICI_calculation(prob_samples,labels,positive_label,resolution = 0.01,bandwidth = 0.05, **kde_args):
    import mct
    '''Assumes a gaussian kernel
    and that prob_samples are passed in the same order as labels
    '''
    x_min = max((0, np.amin(prob_samples) - resolution))
    x_max = min((1, np.amax(prob_samples) + resolution))
    x_values = np.arange(x_min, x_max, step= resolution)
    positives_probs = prob_samples[labels==positive_label]

    all_intensity = mct._compute_intensity(x_values=x_values, probs=prob_samples,
                                       kernel = 'gaussian',bandwidth=bandwidth,
                                       **kde_args)
    pos_intensity = mct._compute_intensity(x_values = x_values, probs =positives_probs,
                                           kernel='gaussian', bandwidth = bandwidth,
                                           **kde_args)

    ICI = mct.compute_ici(orig=x_values, calibrated=pos_intensity/all_intensity,all_intensity=all_intensity)
    return ICI

def overlap_CI(CI1, CI2): 
    '''
    Returns whether two confidence intervals overlap or not.
    
    Args:
        -CI1: Frist confidence interval (tuple)
        -CI2: Second confidence interval (tuple)
    Returns:
        Boolean flag
    '''
    flag = False
    
    l1, u1 = CI1
    l2, u2 = CI2

    if (l1 <= u2) and (l2 <= u1):
        flag = True
    return flag

def contains_val_CI(CI, val):
    '''
    Returns whether a confidence intervals contains a value or not.
    
    Args:
        -CI (tuple): Frist confidence interval (
        -val (float): value to check for
    Returns:
        Boolean flag
    '''
    if CI[0] < val < CI[1]:
        return True 
    elif (CI[0] == val) and (CI[1] == val):
        return True
    else:
        return False