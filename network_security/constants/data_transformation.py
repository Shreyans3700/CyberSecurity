from dataclasses import dataclass
import numpy as np


class DataTransformationConstants:
    """
    Class for data transformation constants.
    """

    TARGET_COLUMN: str = "Result"
    DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
    DATA_TRANSFORMATION_MODEL_DIR_NAME: str = "data_transformation_models"
    DATA_TRANSFORMATION_CONFIGS: dict = {
        "n_neighbors": 5,
        "weights": "uniform",
        "missing_values": np.nan,
    }


@dataclass
class DataTransformationArtifact:
    """
    Class for data transformation artifact.
    """

    transformed_train_file_path: str
    transformed_test_file_path: str
    data_transformation_model_dir_path: str
