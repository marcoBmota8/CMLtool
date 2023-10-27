import os
import logging

import pandas as pd
import json
import matplotlib.pyplot as plt

from CML_tool.Utils import write_pickle, read_pickle

# Configure the logging module
logging.basicConfig(level=logging.INFO)

def file_based_cacheing(path: str, file_name,  extension_desired = '.pkl'):
    """
    File based cacheing. 
        1. It attempts to read the file and return the object.
        2. If fails:
            1. It runs the function it decorates.
            2. Saves the result to the path+file_name
            3. Returns the result object.
    
    Current implementation allows to save and return:
      -python objects as pickle files.
      -pandas dataframes as .csv files.
    
    If the user-specified saving fails, the decorator default behavior is
    to override the user-specified file_name by removing the specified extension 
    if any, and .append pkl at the end.
    It then saves the data/python object.

    If reading fails, we assume the specified file doe snot exists and
    the function will be run.

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if (path is not None) and (file_name is not None):
        
                try:
                    if 'pkl' in extension_desired:
                        obj_var = read_pickle(path=path, filename=file_name)
                        logging.info(msg = "Function "+func.__name__+" CACHED.")

                    elif 'cvs' in extension_desired:
                        obj_var = pd.read_csv(filepath = os.path.join(path,file_name))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")
                    
                    elif 'json' in extension_desired:
                        with open(os.path.join(path, file_name),'r') as openfile:
                            obj_var = json.load(openfile)
                        logging.info(msg = "Function "+func.__name__+" CACHED.")

                    else:
                        raise ValueError('File not found or file caching failed.')

                except:
                    logging.info(msg = "Executing "+func.__name__+" ..." )
                    obj_var = func(*args, **kwargs)
                    # User-specified saving
                    try:
                        if 'pkl' in extension_desired:
                            os.makedirs(path, exist_ok=True) # Ensure that the host folder exists
                            write_pickle(object = obj_var, path=path, filename=file_name)

                        elif 'cvs' in extension_desired:
                            os.makedirs(path, exist_ok=True) # Ensure that the host folder exists
                            obj_var.to_csv(path_or_buf=os.path.join(path,file_name))

                        elif 'json' in extension_desired:
                            os.makedirs(path, exist_ok=True) # Ensure that the host folder exists
                            with open(os.path.join(path, file_name), "w") as outfile:
                                json.dump(obj_var, outfile)

                        else:
                            raise ValueError('No developed option is valid for input combination of python object and desired extension.')
                    # Default saving
                    except:
                        logging.info(msg = "Defaulting to pickle saving...")
                        file_name_except = os.path.splitext(file_name)[0]
                        write_pickle(object = obj_var, path=path, filename=file_name_except+'.pkl')

                    logging.info(msg = "Function "+func.__name__+" EXECUTION COMPLETE & RESULT FILE SAVED.")

                return obj_var
            else:
                pass

        return wrapper
    return decorator

def file_based_figure_saving(filename:str=None, path:str=None, format:str='png', dpi:int=300):
    def decorator(plot_func):
        def wrapper(*args, **kwargs):
            
            # Call the original function to create the figure
            fig, ax = plot_func(*args, **kwargs)
            
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
                fig.savefig(os.path.join(func_path, f"{func_filename}.{func_format}"), format=func_format, dpi=dpi)
                logging.info(f"Figure SAVED to {func_path}/{func_filename}.{func_format}")
            else:
                logging.info(f"Figure already exists at {func_path} as {func_filename}.{func_format}. Figure was not regenerated neither saved.")
                
        return wrapper
    return decorator

