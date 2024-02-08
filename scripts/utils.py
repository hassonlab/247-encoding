import json
import os
import pickle
from datetime import datetime


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return function_wrapper


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    return datum

def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


def write_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join(dictionary["full_output_dir"], "config.json")
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
