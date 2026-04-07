from network_security.components.data_ingestion import DataIngestionPipeline
from network_security.components.data_validation import DataValidation
import sys
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer

if __name__ == "__main__":
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

        ## Step 4: Model Training
        model_trainer = ModelTrainer(transformation_artifact)
        model_artifact, models = model_trainer.initiate_model_trainer()

        logging.info(model_artifact.model_file_path)

        logging.info(
            "Network security data processing pipeline completed successfully."
        )
    except Exception as e:
        logging.error(f"Error occurred in main.py: {e}")
        raise NetworkSecurityException(e, sys)
