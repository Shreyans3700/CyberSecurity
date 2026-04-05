import sys
from pyaml import yaml
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException


def read_yaml_file(file_path: str) -> dict:
    try:
        logging.info(f"Reading YAML file from {file_path}")
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        logging.info(f"Successfully read YAML file from {file_path}")
        return data
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, data: dict) -> None:
    try:
        logging.info(f"Writing data to YAML file at {file_path}")
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
        logging.info(f"Successfully wrote data to YAML file at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
