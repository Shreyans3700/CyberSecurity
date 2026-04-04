from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


## Training Pipeline Configurations
@dataclass
class TrainingPipelineConfig:
    PIPELINE_NAME: str = "Network_Security_Pipeline"
    ARTIFACT_DIR: str = "artifact"
    TARGET_COLUMN: str = "Result"
    RAW_FILE_NAME: str = "raw.csv"
    TRAIN_FILE_NAME: str = "train.csv"
    TEST_FILE_NAME: str = "test.csv"
    DATA_TRAIN_TEST_SPLIT_RATIO = 0.2

    ## MONGODB CONFIG
    DATABASE_NAME = "NetworkSecurity"
    COLLECTION_NAME = "PhishingData"
    DATABASE_URL = os.getenv("MONGODB_URL_KEY")


## Ingestion Artifact - this is the output of data ingestion component which will be used as input for data transformation component
@dataclass
class IngestionArtifact:
    train_file_path: str
    test_file_path: str
