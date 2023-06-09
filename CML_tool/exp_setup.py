import os
import json
import itertools

def results_folder_tree(root_dir, metadata, results_name = ''):
    '''
    Create the folder tree to store results for a given experiment root directory and metadata.
    First, this function creates a Results folder in the main path as the script where it is ran if it does not already exist.
    Second, it attempts to create a subfolder under Results for the model architecture employed if this does not exits.
    Third, it takes the path to the input data and uses the folder name to create a subsubfolder under .../Results/model_name to store the results.
    Args:
        -root_dir: Experiment root directory
        -metadata: Experiment metadata dictionary
    Out:
        -experiment directory
    '''

    dataset_name = metadata["dataset_name"]
    
    # Generate directory for results 
    results_dir = os.path.join(root_dir, 'Results', results_name)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=False)

    # Generate directory for model architecture
    model_dir = os.path.join(results_dir, metadata['model_name'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)

    #Generate directory the experiment
    exp_dir = os.path.join(model_dir, dataset_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=False)
    
    return exp_dir


def save_metadata(metadata, exp_dir):
    '''
    Save metadata to a file in the experiment directory.
    '''
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
    # Iterate through all combinations of values
    for combination in itertools.product(*values):
        # Create a new dictionary for the combination
        combinations.append(dict(zip(keys, combination)))
    return combinations
        

