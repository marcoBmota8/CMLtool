import os
import logging

import pandas as pd
import numpy as np
import json
import pickle

from CML_tool.Utils import write_pickle, read_pickle

# Configure the logging module
logging.basicConfig(level=logging.INFO)

def file_based_cacheing(path:str=None, filename:str=None,  extension_desired:str='.pkl'):
    """
    File based cacheing. 
    
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
    if any, and .append pkl at the end.
    It then saves the data/python object.

    If reading fails, we assume the specified file does not exist and
    the function will be run and the result saved.

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
                        
            if (path_fn is not None) and (filename_fn is not None):
        
                try: # Try to retrieve the object
                    if 'pkl' in extension_desired_fn:
                        obj_var = read_pickle(path=path_fn, filename=filename_fn+".pkl")
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                    elif 'cvs' in extension_desired_fn:
                        obj_var = pd.read_csv(filepath = os.path.join(path_fn,filename_fn+".csv"))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                    elif 'json' in extension_desired_fn:
                        with open(os.path.join(path_fn, filename_fn+".json"),'r') as openfile:
                            obj_var = json.load(openfile)
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                    elif 'npy' in extension_desired_fn:
                        obj_var = np.load(os.path.join(path_fn, filename_fn+".npy"))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")                        
                    else:
                        raise ValueError('File not found or file caching failed.')

                except: # Run the function and save the object
                    logging.info(msg = "Executing "+func.__name__+" ..." )
                    obj_var = func(*args, **kwargs) # run the function and obtain the resulting object
                    try:
                        if 'pkl' in extension_desired_fn:
                            os.makedirs(path_fn, exist_ok=True) # Ensure that the host folder exists
                            write_pickle(object = obj_var, path=path_fn, filename=filename_fn+'.pkl')
                        elif 'cvs' in extension_desired_fn:
                            os.makedirs(path_fn, exist_ok=True) # Ensure that the host folder exists
                            obj_var.to_csv(path_or_buf=os.path.join(path_fn,filename_fn+".csv"))
                        elif 'json' in extension_desired_fn:
                            os.makedirs(path_fn, exist_ok=True) # Ensure that the host folder exists
                            with open(os.path.join(path_fn, filename_fn+".json"), "w") as outfile:
                                json.dump(obj_var, outfile)
                        elif 'npy' in extension_desired_fn:
                            os.makedirs(path_fn, exist_ok=True) # Ensure that the host folder exists
                            np.save(os.path.join(path_fn, filename_fn+".npy"), obj_var)
                        else:
                            raise ValueError('No developed option is valid for input combination of python object and desired extension.')
                    except: # Default saving is in ".pkl" format
                        logging.info(msg = f"Defaulting to pickle saving as: {os.path.join(path_fn,filename_fn+'.pkl')}")
                        write_pickle(object = obj_var, path=path_fn, filename=filename_fn+'.pkl')

                    logging.info(msg = "Function "+func.__name__+" EXECUTION COMPLETE & RESULT FILE SAVED.")

                return obj_var
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