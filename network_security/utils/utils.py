"""Utility functions for YAML file handling, object persistence, and numpy array serialization."""

import sys
from pyaml import yaml
import numpy as np
import pickle
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML file from disk and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file.
    """
    try:
        logging.info(f"Reading YAML file from {file_path}")
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        logging.info(f"Successfully read YAML file from {file_path}")
        return data
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, data: dict) -> None:
    """Write the given dictionary to a YAML file at the provided path.

    Args:
        file_path (str): Destination YAML file path.
        data (dict): Dictionary to serialize to YAML.

    Returns:
        None
    """
    try:
        logging.info(f"Writing data to YAML file at {file_path}")
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
        logging.info(f"Successfully wrote data to YAML file at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(
    file_path: str, array: np.ndarray, target: np.ndarray
) -> None:
    """Save feature and target arrays together in a single .npz file.

    Args:
        file_path (str): Destination file path for the compressed NumPy archive.
        array (np.ndarray): Feature array to save.
        target (np.ndarray): Target array to save.

    Returns:
        None
    """
    try:
        logging.info(f"Saving transformed array and target to {file_path}")
        np.savez_compressed(file_path, array=array, target=target)
        logging.info(f"Successfully saved transformed data to {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """Serialize and persist a Python object using pickle.

    Args:
        file_path (str): Destination file path for the serialized object.
        obj (object): Python object to serialize.

    Returns:
        None
    """
    try:
        logging.info(f"Saving object to {file_path}")
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
