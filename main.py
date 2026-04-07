from network_security.components.data_ingestion import DataIngestionPipeline
from network_security.components.data_validation import DataValidation
import sys
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.components.data_transformation import DataTransformation

if __name__=="__main__":
    try:
        logging.info("Starting the network security data processing pipeline.")
        
        # Step 1: Data Ingestion
        data_ingestion = DataIngestionPipeline()
        data_artifact = data_ingestion.ingest_data()
        training_data_path = data_artifact.train_file_path
        test_data_path = data_artifact.test_file_path
        
        # Step 2: Data Validation
        data_validation = DataValidation()
        validation_artifact = data_validation.initiate_data_validation(data_artifact)
        
        # Step 3: Data Transformation
        data_transformation = DataTransformation(data_validation=validation_artifact)
        transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(transformation_artifact.transformed_train_file_path)
        logging.info(transformation_artifact.transformed_test_file_path)
        logging.info(transformation_artifact.data_transformation_model_dir_path)
        
        logging.info("Network security data processing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error occurred in main.py: {e}")
        raise NetworkSecurityException(e, sys)