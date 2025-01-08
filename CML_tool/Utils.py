import os
import logging

import pandas as pd
import numpy as np
import json
import pickle
import tqdm
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def chunked_dataframe(df, chunk_size):
    for i in range(0, len(df)-chunk_size, chunk_size):
        yield df.iloc[i:i + chunk_size]

def write_pickle(object, path, filename):
    """
    Function to write a python object as pickle into a path with a filename
    """
    if not filename.endswith(".pkl"):
        filename = filename+".pkl"

    #Try to save the file at once (may run into memory issues for large files)
    try:
        with open(os.path.join(path,filename), "wb") as file:
            pickle.dump(object, file)

    #If there is an error thrown save and the file is a DataFrame save it in chunks
    except:
        if isinstance(object, pd.DataFrame):
            with open(os.path.join(path,filename), 'wb') as file:
                #calculate a decent chunk size: ~10% of its length
                chunk_size = int(0.1 * len(object))
                for chunk in chunked_dataframe(object, chunk_size):
                    pickle.dump(chunk, file)
        else:
            raise ValueError(
                "File couldn't be pickle directly and neither in chunks (DataFrame). Revise the write_pickle function for your current use.")        
        
def read_pickle(path, filename, chunk_size=1024, silent=True):
    if not filename.endswith(".pkl"):
        filename = filename+".pkl"
    file_path = os.path.join(path, filename)
    file_size = os.path.getsize(file_path)
    if not silent:
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading pickle file")

    buffer = bytearray()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            if not silent:
                progress_bar.update(len(chunk))
    if not silent:
        progress_bar.close()
    data = pickle.loads(buffer)
    return data


def write_json(json_obj:dict=None, path:str=None, filename:str=None):
    
    if (path is not None) and (json_obj is not None) and (filename is not None):
        extension = filename.rsplit(".", 1)[-1]
        if  '.' in filename and extension != 'json': # make sure correct extension in filename
            logging.INFO(f'Found filename={filename} with extension={extension}...')
            filename = os.path.splitext(filename)[0]+'.json'
            logging.INFO(f'Using {filename}...')
            
        if os.path.exists(os.path.join(path,filename+".json")): # Check if object exists
            raise FileExistsError(f"{filename}.json already exists, erase it before procceding if you want to overide it.")
            
        else: # Save the object
            os.makedirs(path, exist_ok=True) # Ensure that the host folder exists
            logging.info(msg = f"Saving dictionary object as {filename}.json at {path} ..." )
            with open(os.path.join(path,filename+'.json'), "w") as f:
                json.dump(json_obj, f, indent=4)
                
    else:
        raise ValueError('No filename, path or object passed.')

def read_json(path: str = None, filename: str = None) -> dict:
    if path is not None and filename is not None:
        # Ensure the filename has a .json extension
        extension = filename.rsplit(".", 1)[-1]
        if '.' in filename and extension != 'json':
            
            logging.info(f'Found filename={filename} with extension={extension}...')
            filename = os.path.splitext(filename)[0] + '.json'
            logging.info(f'Using {filename}...')
        elif '.' not in filename:
            filename += '.json'

        full_path = os.path.join(path, filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {full_path} does not exist.")

        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except:
            logging.error(f"Failed to decode JSON from {full_path}. Ensure it's a valid JSON file.")
            raise 
    else:
        raise ValueError('No filename or path provided.')    

def save_df_w_metadata(df:pd.DataFrame, path:str=None, filename:str=None, metadata:dict=None, sep=',', header=True, index=True, encoding='utf-8'):
    """
    Save a pandas DataFrame to a CSV file with optional metadata.
    Args:
        -param df: the DataFrame to save
        -param filename: the filename to save the CSV file as
        -param metadata: optional metadata dictionary to save as a json file with the same name as the CSV file
        -param sep: the delimiter to use when saving the CSV file
        -param header: whether to include the column names in the CSV file
        -param index: whether to include the row index in the CSV file
        -param encoding: the character encoding to use when saving the CSV file
    """
    filename = os.path.splitext(filename)[0]
    
    if (path is not None) and (filename is not None):
    
        if os.path.exists(os.path.join(path,filename+".csv")): # Check if DataFrame exits
            logging.info(msg = f"{filename}.csv already exists. Neither the df nor metadata were saved.")
            
        else: # Save the dataframe and, if passed, its metadata
            os.makedirs(path, exist_ok=True) # Ensure that the host folder exists
                
            # Save the DataFrame
            logging.info(msg = f"Saving DataDrame as {filename}.csv at {path} ..." )
            df.to_csv(path_or_buf=os.path.join(path,filename+".csv"))
            
            if metadata is not None:            
                # Save metadata. 
                # It overides if a matedata file already exists for the DataFrame to avoid matching confusion with residual files from previous runs
                metadata_filename = f"{filename}_metadata.json"        
                logging.info(msg = f"Saving metadata for as {metadata_filename} at {path} ..." )
                with open(os.path.join(path,metadata_filename), "w") as f:
                    json.dump(metadata, f, indent=4)
            else: 
                logging.info('No metadata passed, so none was saved.')
                
    else:
        raise ValueError('No path and/or filename passed.')

def oredered_features(selected_features,coefs, how):
    '''
    Args:
        -Selected_features: List of the tuples/strings of the selected features
        -coefs: array with the coefficient values for the selected features. 
            It is assumed that it is passed in the same order
        -how: method for how the features are ordered according to their coef
            values. "signed" (taking into account the coefficient sign) or 
            "absolute" (absolute value). (Default: "absolute)
    '''
    if (how == 'absolute') and (len(selected_features)>0):
        return selected_features[np.argsort(coefs)][::-1]
    elif (how == 'signed') and (len(selected_features)>0):
        return selected_features[np.argsort(abs(coefs))][::-1]
    elif (len(selected_features)==0):
        return selected_features
    else:
        raise ValueError('Wrong "how" string passed. Use "absolute" or "signed" instead.')

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

def round_to_resolution(x, resolution, direction):
    """
    Round an array to the desired resolution.
    for example:
    >>> x = np.array([0.22, -0.39, 1.02, -3.28])
    >>> round_to_resolution(x, 0.25, 'round')
    >>> np.array([0.25, -0.50, 1.00, -3.50])
    >>> round_to_resolution(x, 0.25, 'floor')
    >>> np.array([0.00, -0.50, 1.00, -3.50])
    >>> round_to_resolution(x, 0.25, 'absolute-ceil')
    >>> np.array([0.25, -0.5, 1.25, -3.50])
    """
    sign = np.sign(x)
    if direction =='ceil':
        return np.ceil(x/resolution) * resolution
    elif direction =='floor':
        return np.floor(x/resolution) * resolution
    elif direction =='absolute-ceil':
        return sign*np.ceil(abs(x)/resolution) * resolution
    elif direction =='absolute-floor':
        return  sign*np.floor(abs(x)/resolution) * resolution
    elif direction =='round':
        return np.round(x/resolution) * resolution
    elif direction =='absolute-round':
        return sign*np.round(abs(x)/resolution) * resolution        
    else:
        raise ValueError('Rounding direction is misspecified. Choose among "ceil", "absolute-ceil", "floor", "absolute-floor" and "round".')

def round_up_sig_figs(number, sig_figs):
    if number == 0:
        return 0
    
    # Handle negative numbers
    sign = 1 if number > 0 else -1
    number = abs(number)
    
    # Calculate the magnitude (power of 10) 
    magnitude = math.floor(math.log10(number))
    
    # Scale the number to make the first sig_figs digits 
    scaled = number / (10 ** (magnitude - sig_figs + 1))
    
    # Round up the scaled number 
    rounded_scaled = math.ceil(scaled)
    
    # Scale back down
    result = rounded_scaled * (10 ** (magnitude - sig_figs + 1))
    
    # Reapply the original sign
    return sign * result
    
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

def twoarrays_2_tupledf(array1:np.array,array2:np.array):
    # Create a new array to store tuples
    combined_array = np.empty(array1.shape, dtype=tuple)

    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            combined_array[i, j] = (array1[i, j], array2[i, j])
            
    return pd.DataFrame(combined_array)

def math_w_tuple(t:tuple, op:str, num:float):
    '''
    Function that allow to multiply or divide a tuple by a contstant.
    ''' 
    if op == 'divide':
        return tuple(elem/num for elem in t)
    elif op == 'multiply':
        return tuple(elem*num for elem in t)
    elif op == 'sum':
        return tuple(elem+num for elem in t)
    elif op == 'susbtract':
        return tuple(elem-num for elem in t)
    else:
        raise NotImplementedError('Possible operations: "divide", "multiply","sum" and "substract".')
