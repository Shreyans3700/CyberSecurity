import sys
from network_security.exception.exception import NetworkSecurityException
from network_security.utils.classification_metric import get_classification_metrics
from network_security.logging.logger import logging


class NetworkModel:
    def __init__(self, model, preprocessor):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys)


def evaluate_classification_models(models, X_train, y_train, X_test, y_test):
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_metrics = get_classification_metrics(y_train, y_train_pred)
            test_metrics = get_classification_metrics(y_test, y_test_pred)
            model_report[model_name] = {
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "trained_model": model,
            }
            logging.info(f"Completed training and evaluation for model: {model_name}")
        return model_report
    except Exception as e:
        raise NetworkSecurityException(e, sys)
