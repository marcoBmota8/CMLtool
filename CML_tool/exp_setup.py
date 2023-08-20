import os
import json
import itertools
from CML_tool.Utils import flatten_list

def results_folder_tree(root_dir, metadata,create_model_subfolder=True, results_name=None):
    '''
    Create the folder tree to store results for a given experiment root directory and metadata.
    First, this function attempts to create a '.../Results' (or '.../Results/result_name' if any string is passed under 'result_name')
    folder in the main path of the script where it is ran if surch directory does not already exist.
    Second, if indicated through 'create_model_subfolder' it attempts to create a subfolder under Results for the model architecture 
    employed if this subfolder does not exits.
    Third, it takes the path to the input data and uses the dataset_name in the metadata to create a subsubfolder /Results/model_name/dataset_name 
    to store the results.
    Args:
        -root_dir: Experiment root directory
        -metadata: Experiment metadata dictionary
        -results_name: string to be passed in case a subfolder is needed between 'Results' and model_subfolder
        -create_model_subfolder: boolean indicating if model subfolder level is desired
    Out:
        -experiment directory
    '''

    dataset_name = metadata["dataset_name"]

    if isinstance(dataset_name, list) and len(dataset_name)==1:
        dataset_name = dataset_name[0]
    elif isinstance(dataset_name, str):
        pass
    elif dataset_name==None:
        pass
    else:
        raise ValueError("The format of the dataset_name metadata is WRONG. It is neither a list if one element neither a single string.")
    
    # Generate directory for results 
    if results_name !=None:
        results_dir = os.path.join(root_dir, 'Results', results_name)
    else:
        results_dir = os.path.join(root_dir, 'Results')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=False)

    if create_model_subfolder:
        # Generate directory for model architecture
        model_dir = os.path.join(results_dir, metadata['model_name'])

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=False)
    else:
        model_dir = results_dir

    #Generate directory the experiment
    if dataset_name !=None:
        exp_dir = os.path.join(model_dir, dataset_name)
    else:
        exp_dir = model_dir


    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=False)
    
    return exp_dir


def save_metadata(metadata, exp_dir):
    '''
    Save metadata to a file in the experiment directory.

    It first checks whether there exist a previous run metadata.json.
    In such case it saves metadata_i.json, where i is the latest version of the metadata.
    This strategy allows to preserve a history of succesful runs where cacheing may have been
    used to bypass existing large computations.
    '''
    if os.path.exists(os.path.join(exp_dir,"metadata.json")):
        meta_exists = True
        cont = 0
        while meta_exists:
            cont +=1
            if not os.path.exists(os.path.join(exp_dir,f"metadata_{cont}.json")):
                meta_exists=False
                metadata_file = os.path.join(exp_dir, f"metadata_{cont}.json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)
    else:    
        metadata_file = os.path.join(exp_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    return

def get_exps_dicts(battery_exps_dict):
    '''
    Args:
        -battery_exps_dict: Dictionary with the same keys as the metadata one wants to iterate over and
            generate experiments for.
    Out:
        -List of dictionaries with such key:value pairs.
    '''
    # Get the keys and values from the dictionary
    keys = list(battery_exps_dict.keys())
    values = list(battery_exps_dict.values())

    combinations = []

    if len(keys)<len(flatten_list(values)):
        # Iterate through all combinations of values
        for combination in itertools.product(*values):
            # Create a new dictionary for the combination
            combinations.append(dict(zip(keys, combination)))
        return combinations
    else:
        return [battery_exps_dict]
    

 
        

