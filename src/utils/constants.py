import os


class PATHS:
    # The root path for the project.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # The path where the models are saved.
    SAVED_MODELS_PATH = PROJECT_ROOT + "/saved_models"

    # Three saved models for distillation.
    TEACHER_PATH = SAVED_MODELS_PATH + "/teacher"
    SMALL_PATH = SAVED_MODELS_PATH + "/small"
    STUDENT_PATH = SAVED_MODELS_PATH + "/student"
    AVG_PATH = SAVED_MODELS_PATH + "/avg"


class MODEL_PARAMS:
    # Batch size for the models.
    BATCH_SIZE = 32

    # Save the model after training.
    IS_SAVE = False

    # Cross validation
    CV = 3
