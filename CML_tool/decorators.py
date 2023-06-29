import os
import logging
import pandas as pd
from CML_tool.Utils import write_pickle, read_pickle

# Configure the logging module
logging.basicConfig(level=logging.INFO)

def file_based_cacheing(path, filename, extension_desired = '.pkl'):
    """
    File based cacheing. 
        1. It attempts to read the file and return the object.
        2. If fails:
            1. It runs the function it decorates.
            2. Saves the result to the path+filename
            3. Returns the result object.
    
    Current implementation allows to save and return:
      -python objects as pickle files.
      -pandas dataframes as .csv files.
    
    If the user-specified saving fails, the decorator default behavior is
    to override the user-specified filename by removing the specified extension 
    if any, and .append pkl at the end.
    It then saves the data/python object.

    If reading fails, we assume the specified file doe snot exists and
    the function will be run.

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if (path is not None) and (filename is not None):
        
                try:
                    if 'pkl' in extension_desired:
                        bj_var = read_pickle(path=path, filename=filename)
                        logging.info(msg = "Function "+func.__name__+" CACHED.")

                    elif 'cvs' in extension_desired:
                        bj_var = pd.read_csv(filepath = os.path.join(path,filename))
                        logging.info(msg = "Function "+func.__name__+" CACHED.")

                    else:
                        raise ValueError('File not found or file caching failed.')

                except:
                    logging.info(msg = "Executing "+func.__name__+" ..." )
                    obj_var = func(*args, **kwargs)
                    try:
                        if 'pkl' in extension_desired:
                            write_pickle(object = obj_var, path=path, filename=filename)
                        elif 'cvs' in extension_desired:
                            obj_var.to_csv(path_or_buf=os.path.join(path,filename))
                        else:
                            raise ValueError('No developed option is valid for input combination of python object and desired extension.')
                    except:
                        logging.info(msg = "Defaulting to pickle saving...")
                        filename = os.path.splitext(filename)[0]
                        write_pickle(object = obj_var, path=path, filename=filename+'.pkl')

                    logging.info(msg = "Function "+func.__name__+" EXECUTION COMPLETE & RESULT FILE SAVED.")

                return obj_var
            else:
                pass

        return wrapper
    return decorator
