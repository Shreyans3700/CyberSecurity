import os
import sys
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from network_security.utils.utils import save_numpy_array_data, save_object
from network_security.constants.data_validation import DataValidationArtifact
from network_security.constants.data_transformation import (
    DataTransformationConstants,
    DataTransformationArtifact,
)
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging


class DataTransformation:
    """Transforms validated datasets into feature arrays and saves preprocessing artifacts."""

    def __init__(self, data_validation: DataValidationArtifact):
        """Initialize transformation paths, target column, and preprocessing configuration."""
        self.data_validation_dir: DataValidationArtifact = data_validation
        self.data_transformation_dir = os.path.join(
            os.getcwd(), DataTransformationConstants.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_data_dir = os.path.join(
            os.getcwd(),
            DataTransformationConstants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )
        self.data_transformation_model_dir = os.path.join(
            os.getcwd(), DataTransformationConstants.DATA_TRANSFORMATION_MODEL_DIR_NAME
        )

        self.target_column = DataTransformationConstants.TARGET_COLUMN
        self.transformation_config = (
            DataTransformationConstants.DATA_TRANSFORMATION_CONFIGS
        )

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Read a CSV file from disk and return a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def get_data_transformer_object(self) -> KNNImputer:
        """Create and return a KNNImputer pipeline object based on configured settings.

        Returns:
            KNNImputer: Fitted transformer pipeline component.
        """
        try:
            imputer = KNNImputer(
                n_neighbors=self.transformation_config["n_neighbors"],
                weights=self.transformation_config["weights"],
                missing_values=self.transformation_config["missing_values"],
                strategy=self.transformation_config["imputer_strategy"],
            )
            return Pipeline(steps=[("imputer", imputer)])
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Run the data transformation workflow and return transformation artifacts.

        Returns:
            DataTransformationArtifact: Artifact containing transformed dataset paths and preprocessing model path.
        """
        logging.info("Initiating data transformation")
        try:
            train_df = DataTransformation.read_data(
                self.data_validation_dir.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_dir.valid_test_file_path
            )

            ## training dataframe
            X_train = train_df.drop(columns=[self.target_column], axis=1)
            y_train = train_df[self.target_column]

            ## testing dataframe
            X_test = test_df.drop(columns=[self.target_column], axis=1)
            y_test = test_df[self.target_column]

            ## ipreprocessor
            preprocessor = self.get_data_transformer_object()

            preprocessor_obj = preprocessor.fit(X_train)
            X_train_transformed = preprocessor_obj.transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            ## save transformed data
            os.makedirs(self.transformed_data_dir, exist_ok=True)
            transformed_train_file_path = os.path.join(
                self.transformed_data_dir, "transformed_train_data.npz"
            )
            transformed_test_file_path = os.path.join(
                self.transformed_data_dir, "transformed_test_data.npz"
            )
            save_numpy_array_data(
                transformed_train_file_path, array=X_train_transformed, target=y_train
            )
            save_numpy_array_data(
                transformed_test_file_path, array=X_test_transformed, target=y_test
            )
            logging.info("Saved transformed data")
            ## save preprocessor object
            os.makedirs(self.data_transformation_model_dir, exist_ok=True)
            preprocessor_file_path = os.path.join(
                self.data_transformation_model_dir, "preprocessor.pkl"
            )
            save_object(
                file_path=preprocessor_file_path,
                obj=preprocessor_obj,
            )
            logging.info("Saved preprocessor object")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                data_transformation_model_dir_path=preprocessor_file_path,
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
