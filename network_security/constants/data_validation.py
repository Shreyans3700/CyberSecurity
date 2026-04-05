from dataclasses import dataclass


@dataclass
class DataValidationConstants:
    """
    Constants related to data validation in network security.
    """

    DATA_VALIDATION_DIR_NAME: str = "data_validation"
    DATA_VALIDATION_VALID_DIR: str = "validated"
    DATA_VALIDATION_INVALID_DIR: str = "invalid"
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"
    DATA_VALIDATION_SCHEMA: str = "data_schema/schema.yaml"


@dataclass
class DataValidationArtifact:
    """
    Artifact class to hold information about the data validation process.
    """

    validation_status: bool = False
    drift_report_file_path: str = None
    valid_train_file_path: str = None
    valid_test_file_path: str = None
    invalid_train_file_path: str = None
    invalid_test_file_path: str = None
