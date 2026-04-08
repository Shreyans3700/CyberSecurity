import sys
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.training_pipeline import (
    TrainingPipelineConfig,
    IngestionArtifact,
)
from network_security.constants.data_validation import (
    DataValidationArtifact,
)
from network_security.constants.data_transformation import (
    DataTransformationArtifact,
)
from network_security.constants.model_trainer import (
    ModelTrainerArtifact,
)
from network_security.components.data_ingestion import DataIngestionPipeline
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> IngestionArtifact:
        try:
            logging.info("Starting the data ingestion config.")
            data_ingestion = DataIngestionPipeline()
            ingestion_artifact = data_ingestion.ingest_data()
            logging.info("Data ingestion is complete")
            return ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: IngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            validation_object = DataValidation()
            validation_artifact = validation_object.initiate_data_validation(
                data_ingestion_artifact
            )
            logging.info("Data Validation completed")

            return validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation.")
            data_transformation_object = DataTransformation(
                data_validation=data_validation_artifact
            )
            transformation_artifact = (
                data_transformation_object.initiate_data_transformation()
            )
            logging.info("Data transformation is also complete")
            return transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_training(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("initiating model training")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact
            )
            model_train_artifact = model_trainer.initiate_model_trainer()
            logging.info("model training complete")
            return model_train_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self) -> ModelTrainerArtifact:
        try:
            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(
                data_ingestion_artifact=ingestion_artifact
            )
            transformation_artifact = self.start_data_transformation(
                data_validation_artifact=validation_artifact
            )
            model_training_artifact = self.start_model_training(
                data_transformation_artifact=transformation_artifact
            )

            return model_training_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
