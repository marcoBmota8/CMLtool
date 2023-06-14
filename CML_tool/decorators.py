import os
import logging
from CML_tool.Utils import write_pickle, read_pickle

# Configure the logging module
logging.basicConfig(level=logging.INFO)

def file_based_cacheing(path, filename):
    """
    File based cacheing. 
        It attempts to read the file and if fails:
            1. It runs the function it decorates.
            2. Saves the result to the path+filename
            3. Returns the result object.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if (path is not None) and (filename is not None):
                try:
                    obj_var = read_pickle(path=path, filename=filename)
                    logging.info(msg = "Function "+func.__name__+" CACHED.")
                except:
                    logging.info(msg = "Executing "+func.__name__+" ..." )
                    obj_var = func(*args, **kwargs)
                    write_pickle(object = obj_var, path=path, filename=filename)
                    logging.info(msg = "Function "+func.__name__+" EXECUTION COMPLETE & RESULT FILE SAVED.")

                return obj_var
            else:
                pass

        return wrapper
    return decorator
