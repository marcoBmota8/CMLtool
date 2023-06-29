#Marco Barbero mota
#Started February 2023
#Utils for Computational Medicine Lab pipelines

import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import confusion_matrix

def write_pickle(object, path, filename):
    """
    Function to write a python object as pickle into a path with a filename
    """
    if not filename.endswith(".pkl"):
        filename = filename+".pkl"

    with open(os.path.join(path,filename), "wb") as file:
        pickle.dump(object, file)
        
def read_pickle(path, filename):
    if not filename.endswith(".pkl"):
        filename = filename+".pkl"
        
    with open(os.path.join(path,filename), "rb") as file:
        return pickle.load(file)


def save_dataframe_to_csv(df, filename, metadata=None, sep=',', header=True, index=True, encoding='utf-8'):
    """
    Save a pandas DataFrame to a CSV file with optional metadata.
    If the file already exists, a version number is appended to the filename.
    Args:
        -param df: the DataFrame to save
        -param filename: the filename to save the CSV file as
        -param metadata: optional metadata dictionary to save as a text file with the same name as the CSV file
        -param sep: the delimiter to use when saving the CSV file
        -param header: whether to include the column names in the CSV file
        -param index: whether to include the row index in the CSV file
        -param encoding: the character encoding to use when saving the CSV file
    """
    if os.path.isfile(filename):
        # File already exists, append a version number
        base_filename, ext = os.path.splitext(filename)
        i = 2
        while os.path.isfile(f"{base_filename}_v{i}{ext}"):
            i += 1
        versioned_filename = f"{base_filename}_v{i}{ext}"
        if metadata:
            metadata_filename = f"{base_filename}_v{i}.txt"
            with open(metadata_filename, "w") as f:
                json.dump(metadata, f, indent=4)
    else:
        # File does not exist yet, save as given filename
        versioned_filename = filename
        if metadata:
            metadata_filename = f"{os.path.splitext(filename)[0]}.txt"
            with open(metadata_filename, "w") as f:
                json.dump(metadata, f, indent=4)
    df.to_csv(versioned_filename, sep=sep, header=header, index=index, encoding=encoding)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def flatten(t):
    return [item for sublist in t for item in sublist]

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def replace_values_in_coef_array(values, positions, out_array_length, fill_value = 0.0):
    out_array = np.full((1,out_array_length),fill_value)
    out_array[0,positions] = values
    return out_array

def look_up_description(df,description):
    '''
    Use this to find what 
    index in meta_df 
    matches with a certain
    string in its description
    '''
    results = df[df['description'].str.contains(description)]
    print(results)
    return results

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
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])  
    ppv = cm[0,0]/(cm[0,0]+cm[1,0])
    npv = cm[1,1]/(cm[1,1]+cm[0,1])
    f1_score = 2*sensitivity*ppv/(sensitivity+ppv)

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
