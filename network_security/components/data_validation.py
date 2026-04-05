import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from network_security.utils.utils import read_yaml_file, write_yaml_file
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.constants.training_pipeline import IngestionArtifact
from network_security.constants.data_validation import DataValidationConstants
from network_security.constants.data_validation import DataValidationArtifact


class DataValidation:
    def __init__(self):
        self.data_validation_dir = os.path.join(
            os.getcwd(), DataValidationConstants.DATA_VALIDATION_DIR_NAME
        )
        self.data_validation_valid_dir = os.path.join(
            self.data_validation_dir, DataValidationConstants.DATA_VALIDATION_VALID_DIR
        )
        self.data_validation_invalid_dir = os.path.join(
            self.data_validation_dir,
            DataValidationConstants.DATA_VALIDATION_INVALID_DIR,
        )
        self.data_validation_report_file = os.path.join(
            self.data_validation_dir,
            DataValidationConstants.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )
        self.validation_schema_file_path = os.path.join(
            os.getcwd(), DataValidationConstants.DATA_VALIDATION_SCHEMA
        )
        self.validation_artifact = DataValidationArtifact()

    def read_source_data(self, file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading source data from {file_path}")
            data = pd.read_csv(file_path, index_col=False)
            logging.info(f"Successfully read data from {file_path}")
            return data
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, data: pd.DataFrame) -> bool:
        try:
            logging.info("Validating number of columns in the data")
            schema_config = read_yaml_file(self.validation_schema_file_path)
            # logging.info(f"Schema configuration loaded: {schema_config}")
            no_of_columns = len(schema_config["columns"])
            logging.info(f"Expected number of columns: {no_of_columns}")
            if data.shape[1] == no_of_columns:
                logging.info("Number of columns is as expected")
                return True
            else:
                logging.info("Number of columns is not as expected")
                return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_data_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        try:
            logging.info("Detecting data drift between base and current dataframes")
            status = True
            report = {}
            for column in base_df.columns:
                base_data = base_df[column]
                current_data = current_df[column]
                sample_dist_drift = ks_2samp(base_data, current_data)
                if sample_dist_drift.pvalue > threshold:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column: {
                            "p_value": float(sample_dist_drift.pvalue),
                            "drift_status": is_found,
                        }
                    }
                )

            drift_report_dir = os.path.dirname(self.data_validation_report_file)
            os.makedirs(drift_report_dir, exist_ok=True)
            write_yaml_file(file_path=self.data_validation_report_file, data=report)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(
        self, artifact: IngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Initiating Data Validation")
            train_file_path = artifact.train_file_path
            test_file_path = artifact.test_file_path

            train_data = self.read_source_data(train_file_path)
            test_data = self.read_source_data(test_file_path)

            ## validate number of columns
            training_data_status = self.validate_number_of_columns(train_data)

            if training_data_status:
                logging.info("Training data has the expected number of columns")
            else:
                logging.info(
                    "Training data does not have the expected number of columns"
                )
                os.makedirs(self.data_validation_invalid_dir, exist_ok=True)
                train_data.to_csv(
                    os.path.join(self.data_validation_invalid_dir, "train.csv"),
                    index=False,
                )
                self.validation_artifact.invalid_train_file_path = os.path.join(
                    self.data_validation_invalid_dir, "train.csv"
                )

            testing_data_status = self.validate_number_of_columns(test_data)
            if testing_data_status:
                logging.info("Testing data has the expected number of columns")
            else:
                logging.info(
                    "Testing data does not have the expected number of columns"
                )
                os.makedirs(self.data_validation_invalid_dir, exist_ok=True)
                test_data.to_csv(
                    os.path.join(self.data_validation_invalid_dir, "test.csv"),
                    index=False,
                )
                self.validation_artifact.invalid_test_file_path = os.path.join(
                    self.data_validation_invalid_dir, "test.csv"
                )

            ## detect if numerical columns in train and test data have the same distribution
            ## need to implement this

            ## let's check data drift
            status = self.detect_data_drift(base_df=train_data, current_df=test_data)

            if status:
                logging.info("No data drift detected between training and testing data")
                os.makedirs(self.data_validation_valid_dir, exist_ok=True)
                if training_data_status:
                    train_data.to_csv(
                        os.path.join(self.data_validation_valid_dir, "train.csv"),
                        index=False,
                    )
                    self.validation_artifact.valid_train_file_path = os.path.join(
                        self.data_validation_valid_dir, "train.csv"
                    )
                if testing_data_status:
                    test_data.to_csv(
                        os.path.join(self.data_validation_valid_dir, "test.csv"),
                        index=False,
                    )
                    self.validation_artifact.valid_test_file_path = os.path.join(
                        self.data_validation_valid_dir, "test.csv"
                    )
                self.validation_artifact.data_drift_report_file_path = (
                    self.data_validation_report_file
                )
            else:
                logging.info("Data drift detected between training and testing data")

            return {
                "valid_train_file_path": self.validation_artifact.valid_train_file_path,
                "valid_test_file_path": self.validation_artifact.valid_test_file_path,
                "invalid_train_file_path": self.validation_artifact.invalid_train_file_path,
                "invalid_test_file_path": self.validation_artifact.invalid_test_file_path,
                "data_drift_report_file_path": self.validation_artifact.data_drift_report_file_path,
            }

        except Exception as e:
            raise NetworkSecurityException(e, sys)
