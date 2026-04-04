import os
import sys
import pymongo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from network_security.constants.training_pipeline import TrainingPipelineConfig
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.constants.training_pipeline import IngestionArtifact


class DataIngestionPipeline:
    """
    This class is responsible for ingesting data from MongoDB, normalizing it, and saving it as CSV files for training and testing. It also handles the creation of necessary directories and manages the file paths for the raw, train, and test datasets.
    """

    def __init__(self):
        self.data_dir = os.path.join(os.getcwd(), TrainingPipelineConfig.ARTIFACT_DIR)
        self.raw_file_path = os.path.join(
            self.data_dir, TrainingPipelineConfig.RAW_FILE_NAME
        )
        self.train_file_path = os.path.join(
            self.data_dir, TrainingPipelineConfig.TRAIN_FILE_NAME
        )
        self.test_file_path = os.path.join(
            self.data_dir, TrainingPipelineConfig.TEST_FILE_NAME
        )
        self.mongo_client = pymongo.MongoClient(TrainingPipelineConfig.DATABASE_URL)
        self.database_name = TrainingPipelineConfig.DATABASE_NAME
        self.collection_name = TrainingPipelineConfig.COLLECTION_NAME
        self.train_test_split_ratio = TrainingPipelineConfig.DATA_TRAIN_TEST_SPLIT_RATIO
        self.artifact_dir = IngestionArtifact

    def ingest_collection_from_mongodb(self):
        try:
            db = self.mongo_client[self.database_name]
            collection = db[self.collection_name]
            data = list(collection.find())
            logging.info("Data fetched successfully from MongoDB")
            return data
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def normalize_data(self, df):
        try:
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)
                logging.info("Dropped '_id' column from the dataframe")

            df = df.replace({"na": np.nan})
            logging.info("Data normalization completed successfully")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def save_data_to_csv(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logging.info(f"Data saved successfully to {file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def ingest_data(self):
        try:
            ## create the directory if it does not exist
            os.makedirs(self.data_dir, exist_ok=True)
            logging.info("Directory created successfully for data ingestion")

            ## fetch the data from mongo db and save as csv file
            data = self.ingest_collection_from_mongodb()

            ## convert the data to pandas dataframe
            df = pd.DataFrame(data)

            df = self.normalize_data(df)

            ## save the data to raw file path
            self.save_data_to_csv(df, self.raw_file_path)
            logging.info("Data saved successfully to raw file path")

            train_df, test_df = train_test_split(
                df, test_size=self.train_test_split_ratio, random_state=42
            )
            logging.info("Data split successfully into train and test sets")

            ## save the train and test data to respective file paths
            self.save_data_to_csv(train_df, self.train_file_path)
            self.save_data_to_csv(test_df, self.test_file_path)
            logging.info(
                "Train and test data saved successfully to respective file paths"
            )

            self.artifact_dir.train_file_path = self.train_file_path
            self.artifact_dir.test_file_path = self.test_file_path

            return self.artifact_dir

        except Exception as e:
            raise NetworkSecurityException(e, sys)
