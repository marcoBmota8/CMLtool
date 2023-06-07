#Marco Barbero mota
#Started February 2023
#Utils for phenotype/raw variables processing

import numpy as np
import os
import pandas as pd

import os
import pandas as pd
import json


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
