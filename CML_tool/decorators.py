import os
import logging

import pandas as pd
import numpy as np
import json
import pickle

from matplotlib.figure import Figure

from CML_tool.Utils import write_pickle, read_pickle

# Configure the logging module
logging.basicConfig(level=logging.INFO)

#######################
##### Decorators ######
#######################

def file_based_cacheing(path:str=None, filename:str=None, extension_desired:str='.pkl'):
    """
    File based cacheing with support for multiple formats including NPZ. For several arrays as NPZ
    it returns a dictionary with the key:array pairs and a boolean indicating whether the decorated function
    was cached or not. The latter is useful for long data pipelines with chained sections/checkpoints where the output of one
    is the input of the next one.
    
    If path, filename and/or extension argument are passed to the function instance
    the decorator decorates those are used instead of the values in the decorator. If no path and/or 
    filename is specified in either instance, an error is thrown.
    
        1. It attempts to read the file and return the object.
        2. If fails:
            1. It runs the function it decorates.
            2. Saves the result to the path+file_name. 
                In the process creates the nested folder tree needed 
                if it doesnt exists already.
            3. Returns the result object.
    
    If the user-specified saving fails, the decorator default behavior is
    to override the user-specified file_name by removing the specified extension 
    if any, and append pkl at the end.
    It then saves the data/python object.

    Parameters:
        path (str): Directory path for caching
        filename (str): Name of the cache file
        extension_desired (str): File extension ('.pkl', '.csv', '.json', '.npy', '.npz')
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # If path, filename and extension argument are passed to the function instance take those instead of decorator values
            path_fn = kwargs.get('path', path)
            filename_string = kwargs.get('filename', filename)
            if filename_string is not None:
                filename_fn = os.path.splitext(filename_string)[0]
            else:
                filename_fn=None
            extension_desired_fn = kwargs.get('extension_desired', extension_desired)
            cached = False # flag indicating whether the file was cached or not            
            if (path_fn is not None) and (filename_fn is not None):
        
                try: # Try to retrieve the object
                    if 'pkl' in extension_desired_fn:
                        obj_var = read_pickle(path=path_fn, filename=filename_fn+".pkl")
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                        cached = True
                    elif 'cvs' in extension_desired_fn:
                        obj_var = pd.read_csv(filepath = os.path.join(path_fn,filename_fn+".csv"))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                        cached = True
                    elif 'json' in extension_desired_fn:
                        with open(os.path.join(path_fn, filename_fn+".json"),'r') as openfile:
                            obj_var = json.load(openfile)
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                        cached = True
                    elif 'npy' in extension_desired_fn:
                        obj_var = np.load(os.path.join(path_fn, filename_fn+".npy"))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                        cached = True
                    elif 'npz' in extension_desired_fn:
                        try:
                            with np.load(os.path.join(path_fn, filename_fn+".npz")) as data: # Load NPZ file - note that this returns a dict-like object
                                if len(data.files) == 1: # If there's only one array, return it directly
                                    obj_var = data[data.files[0]]
                                else: # Otherwise return the dict object 
                                    obj_var = {k: data[k] for k in data.files}
                            logging.info(msg = "Function "+func.__name__+" CACHED.")
                        except:
                            obj_var = read_pickle(path=path_fn, filename=filename_fn+".pkl")
                            logging.info(msg=f"No NPZ file {filename_fn+'.npz'} was found but a PICKLE file with that name was found and retrieved.")
                            logging.info(msg = "Hence, function "+func.__name__+" CACHED.")
                        cached = True
                    else:
                        raise ValueError(f'File extension {extension_desired_fn} not supported.')

                except: # Run the function and save the object
                    logging.info(msg = "Executing "+func.__name__+" ..." )
                    obj_var = func(*args, **kwargs) # run the function and obtain the resulting object
                    try:
                        os.makedirs(path_fn, exist_ok=True) # Ensure that the host folder exists
                        
                        if 'pkl' in extension_desired_fn:
                            write_pickle(object = obj_var, path=path_fn, filename=filename_fn+'.pkl')
                        elif 'cvs' in extension_desired_fn:
                            obj_var.to_csv(path_or_buf=os.path.join(path_fn,filename_fn+".csv"))
                        elif 'json' in extension_desired_fn:
                            with open(os.path.join(path_fn, filename_fn+".json"), "w") as outfile:
                                json.dump(obj_var, outfile)
                        elif 'npy' in extension_desired_fn:
                            np.save(os.path.join(path_fn, filename_fn+".npy"), obj_var)
                        elif 'npz' in extension_desired_fn:
                            if isinstance(obj_var, dict):
                                # If fn output is a dictionary, save each key-value pair
                                np.savez_compressed(os.path.join(path_fn, filename_fn+".npz"), **obj_var)
                            elif isinstance(obj_var, tuple):
                                logging.error(f'The decorated function cannot return several arrays it must return a dictionary with key-array pairs. \n For safety the arrays were saved as a tuple in a.pkl file')
                                raise ValueError('')
                            elif isinstance(obj_var, np.ndarray):
                                # If output is only one array, save it with the default key as a dictionary
                                np.savez_compressed(os.path.join(path_fn, filename_fn+".npz"), obj_var)    
                            else:
                                raise ValueError(f'The decorated function must return a dictionary with key-array pairs.')                            
                        else:
                            raise ValueError('No developed option is valid for input combination of python object and desired extension.')
                    except: # Default saving is in ".pkl" format
                        logging.info(msg = f"Selected save option {extension_desired_fn} FAILED, \n Defaulting to pickle saving as: {os.path.join(path_fn,filename_fn+'.pkl')}")
                        write_pickle(object = obj_var, path=path_fn, filename=filename_fn+'.pkl')

                    logging.info(msg = "Function "+func.__name__+" EXECUTION COMPLETE & RESULT FILE SAVED.")
                    
                return obj_var, cached
            
            else:
                raise ValueError(f'Either "path", "filename" arguments were not passed to the decorator or instance of the function {func.__name__}.')

        return wrapper
    return decorator

def file_based_figure_saving(filename:str=None, path:str=None, format:str='png', dpi:int=300):
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            
            # Call the original function to create the figure
            fig = plot_func(*args, **kwargs)
            
            # Get path and filename from the decorated function's arguments
            func_path = kwargs.get('path', path)
            func_filename = os.path.splitext(kwargs.get('filename', filename))[0]
            func_format = kwargs.get('format', format).split(".")[-1] 
            
            # Raise an exception if either is not passed in the plotting function or decorator
            if func_filename == None:
                raise ValueError('No filename passed either in the plotting function or its decorator.')
            if func_path == None:
                raise ValueError('No path passed either in the plotting function or its decorator.') 
        
            if not os.path.exists(os.path.join(func_path,f"{func_filename}.{func_format}")):
                os.makedirs(func_path, exist_ok=True)
                # Save the figure to the specified path
                fig.savefig(os.path.join(func_path, f"{func_filename}.{func_format}"), format=func_format, dpi=dpi, bbox_inches='tight')
                logging.info(f"Figure SAVED to {func_path}/{func_filename}.{func_format}")
            else:
                logging.info(f"Figure already exists at {func_path} as {func_filename}.{func_format}. Figure was not regenerated neither saved.")
                
        return wrapper
    return decorator

def file_based_fig_ax_objects(filename:str=None, path:str=None):
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            # Get path and filename from the decorated function's arguments
            func_path = kwargs.get('path', path)
            func_filename = os.path.splitext(kwargs.get('filename', filename))[0]
            # Try to retrieve the plot objects if the exists in the path
            try:
                # Load the figure and axes
                with open(os.path.join(func_path, func_filename+'.pkl'), 'rb') as f:
                    fig, ax = pickle.load(f)
                logging.info(f"Figure and axis FOUND and CACHED.")
                return fig, ax
            
            # If retrieving plot objects fail (do not exist or other reason),
            # call the original function to create the figure
            except:
                fig, ax = plot_func(*args, **kwargs)
                with open(os.path.join(func_path, func_filename+'.pkl'), 'wb') as f:
                    pickle.dump((fig, ax), f)
                logging.info(f"Figure and axis object SAVED as {func_path}/{func_filename}.pkl")
                return fig, ax
        return wrapper
    return decorator


#######################
######## Utils ########
#######################

def gen_plot_from_fn(fn, **kwargs):
    result = fn(**kwargs)
    try: # in case the function returns an axis
        return result.get_figure()
    except: 
        if isinstance(result, Figure):
            return result
        else:
            raise TypeError('The inputted function does not retrieve neither a matplotlib axis nor a figure object.')