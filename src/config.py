"""Shared project configuration."""

PROJECT_NAME = "Arxiv Classification"
DATASET_NAME = "arxiv_train_multiclass"
DATASET_PROJECT = PROJECT_NAME
DATASET_VERSION = "1.0"

QUEUE_NAME = "students"

BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256

TOPICS = [
    "art",
    "computer vision",
    "food",
    "games",
    "medicine",
    "microbiome",
    "physics",
    "transformers",
]
