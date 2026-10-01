"""Data Entry Repository - stores Title/Description/Skills entries for keyword analysis."""
from app.infra.mongodb.connection import get_database

COLLECTION_NAME = "data_entries"
PROPOSAL_TRAINING_COLLECTION_NAME = "proposal_training_entries"


def get_data_entries_collection():
    db = get_database()
    return db[COLLECTION_NAME]


def get_proposal_training_entries_collection():
    db = get_database()
    return db[PROPOSAL_TRAINING_COLLECTION_NAME]
