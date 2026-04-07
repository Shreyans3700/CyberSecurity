from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from network_security.constants.model_trainer import ClassificationMetricArtifact
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging


def get_classification_metrics(y_true, y_pred) -> ClassificationMetricArtifact:
    """Calculate and return classification metrics (F1 score, precision, recall).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    Returns:
        ClassificationMetricArtifact: Object containing calculated metrics.
    """
    try:
        logging.info("Calculating classification metrics.")
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        logging.info(
            f"Calculated metrics - F1 Score: {f1}, Precision: {precision}, Recall: {recall}"
        )
        return ClassificationMetricArtifact(
            f1_score=f1, precision_score=precision, recall_score=recall
        )
    except Exception as e:
        logging.error(f"Error calculating classification metrics: {e}")
        raise NetworkSecurityException(e, sys)
