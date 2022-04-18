import os

ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TRAINING_DATA_DIRECTORY = f"{ROOT_DIRECTORY}/training_data"
LOGS_DIRECTORY = f"{TRAINING_DATA_DIRECTORY}/logs"
MODELS_DIRECTORY = f"{TRAINING_DATA_DIRECTORY}/models"
TRAINED_MODELS_DIRECTORY = f"{ROOT_DIRECTORY}/trained_models"
