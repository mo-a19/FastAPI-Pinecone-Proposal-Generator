"""
MongoDB Connection Management

Centralized connection handling for MongoDB.
Extracted from db.py to support repository pattern.
"""
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import certifi

from app.config import settings

logger = logging.getLogger(__name__)

# Global connection instance
_client: Optional[MongoClient] = None
_database: Optional[Database] = None


def connect(
    connection_string: str = None,
    db_name: str = None
) -> Database:
    """
    Establish MongoDB connection with SSL configuration.
    
    Args:
        connection_string: MongoDB URI (defaults to settings)
        db_name: Database name (defaults to settings)
        
    Returns:
        MongoDB Database instance
        
    Raises:
        ConnectionFailure: If connection fails
    """
    global _client, _database
    
    if _database is not None:
        return _database
    
    conn_str = connection_string or settings.MONGODB_URI
    database_name = db_name or settings.MONGODB_DB_NAME
    
    try:
        _client = MongoClient(
            conn_str,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=15000,
            socketTimeoutMS=15000,
            retryWrites=True,
            maxPoolSize=50,
            minPoolSize=10,
            tls=True,
            tlsCAFile=certifi.where()
        )
        
        # Test connection
        _client.admin.command('ping')
        _database = _client[database_name]
        
        logger.info(f"Connected to MongoDB: {database_name}")
        return _database
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise


def get_database() -> Database:
    """
    Get the database instance, connecting if necessary.
    
    Returns:
        MongoDB Database instance
    """
    global _database
    if _database is None:
        return connect()
    return _database


def get_client() -> MongoClient:
    """
    Get the MongoClient instance.
    
    Returns:
        MongoClient instance
    """
    global _client
    if _client is None:
        connect()
    return _client


def close_database():
    """Close the database connection."""
    global _client, _database
    if _client:
        _client.close()
        _client = None
        _database = None
        logger.info("Disconnected from MongoDB")


def get_collection(collection_name: str):
    """
    Get a collection from the database.
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        MongoDB Collection
    """
    db = get_database()
    return db[collection_name]
