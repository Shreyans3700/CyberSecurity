from dataclasses import dataclass


class ModelTrainerConstants:
    """
    Class for model trainer constants.
    """

    MODEL_TRAINER_DIR_NAME: str = "model_trainer"
    MODEL_TRAINER_MODEL_FILE_NAME: str = "model.pkl"
    MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
    MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05


@dataclass
class ClassificationMetricArtifact:
    """
    Class for classification metric artifact.
    """

    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Class for model trainer artifact.
    """

    model_dir_path: str
    model_file_path: str
    trained_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
