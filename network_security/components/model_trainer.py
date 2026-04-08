import os
import sys
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException
from network_security.constants.data_transformation import DataTransformationArtifact
from network_security.constants.model_trainer import (
    ModelTrainerArtifact,
    ModelTrainerConstants,
)
from network_security.utils.ml_utils import NetworkModel, evaluate_classification_models
from network_security.utils.utils import save_object, load_object, load_numpy_array_data
from network_security.utils.classification_metric import get_classification_metrics


class ModelTrainer:
    """
    Class for training machine learning models on transformed data.
    """

    def __init__(self, data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize model trainer with paths to transformed data and model directory.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Artifact containing paths to transformed data and model directory.
        """
        try:
            self.model_trainer_dir = os.path.join(
                os.getcwd(), ModelTrainerConstants.MODEL_TRAINER_DIR_NAME
            )
            self.model_trained_file_path = os.path.join(
                self.model_trainer_dir,
                ModelTrainerConstants.MODEL_TRAINER_MODEL_FILE_NAME,
            )
            self.expected_accuracy = ModelTrainerConstants.MODEL_TRAINER_EXPECTED_SCORE
            self.overfitting_underfitting_threshold = (
                ModelTrainerConstants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
            )
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def track_mlflow(self, best_model, classification_metric):
        try:
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score

                mlflow.log_metric("f1_score ", f1_score)
                mlflow.log_metric("precision ", precision_score)
                mlflow.log_metric("recall ", recall_score)
                mlflow.sklearn.log_model(best_model, "model")
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train a machine learning model on the provided training data and evaluate it on the test data.

        Args:
            X_train (np.ndarray): Training feature array.
            y_train (np.ndarray): Training target array.
            X_test (np.ndarray): Testing feature array.
            y_test (np.ndarray): Testing target array.
        Returns:
            NetworkModel: Trained machine learning model wrapped in a NetworkModel class.
        """
        models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(),
            "Random Forest": RandomForestClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
        }

        model_report: dict = evaluate_classification_models(
            models, X_train, y_train, X_test, y_test
        )

        best_model_score = 0.0
        best_model_name = None
        best_model = None

        for model_name, metrics in model_report.items():
            test_f1_score = metrics["test_metrics"].f1_score
            logging.info(f"{model_name} - Test F1 Score: {test_f1_score}")
            if test_f1_score > best_model_score:
                best_model_score = test_f1_score
                best_model_name = model_name
                best_model = metrics["trained_model"]

        if best_model_score < self.expected_accuracy:
            raise NetworkSecurityException(
                f"No model achieved the expected accuracy of {self.expected_accuracy}. Best model: {best_model_name} with F1 Score: {best_model_score}",
                sys,
            )
        logging.info(
            f"Best model selected: {best_model_name} with F1 Score: {best_model_score}"
        )

        y_train_pred = best_model.predict(X_train)

        classification_train_metrics = get_classification_metrics(y_train, y_train_pred)

        self.track_mlflow(best_model, classification_train_metrics)

        y_test_pred = best_model.predict(X_test)

        classification_test_metrics = get_classification_metrics(y_test, y_test_pred)

        self.track_mlflow(best_model, classification_test_metrics)

        preprocessor = load_object(
            f"{self.data_transformation_artifact.data_transformation_model_dir_path}/preprocessor.pkl"
        )

        network_model = NetworkModel(model=best_model, preprocessor=preprocessor)

        logging.info(
            f"Classification metrics for the best model on training data: {classification_train_metrics}"
        )

        save_object("final_models/model.pkl", best_model)
        save_object("final_models/preprocessor.pkl", preprocessor)

        model_trainer_artifact = ModelTrainerArtifact(
            model_file_path=self.model_trained_file_path,
            model_dir_path=self.model_trainer_dir,
            trained_metric_artifact=classification_train_metrics,
            test_metric_artifact=classification_test_metrics,
        )

        return model_trainer_artifact, network_model

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Train a machine learning model on the transformed data and evaluate its performance.

        Returns:
            ModelTrainerArtifact: Artifact containing the path to the trained model and evaluation metrics.
        """
        try:
            logging.info(
                "Loading transformed training and testing data. and starting model training"
            )
            train_data = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_data = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_data
            X_test, y_test = test_data

            model_artifact, model_network = self.train_model(
                X_train, y_train, X_test, y_test
            )

            return model_artifact, model_network

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
