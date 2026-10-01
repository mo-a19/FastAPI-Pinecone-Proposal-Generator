"""MongoDB infrastructure layer - connection management and repositories."""

from app.infra.mongodb.connection import get_database, close_database
from app.infra.mongodb.base_repository import BaseRepository

__all__ = [
    "get_database",
    "close_database", 
    "BaseRepository",
]
