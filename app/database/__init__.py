"""MongoDB client factory and shared database utilities."""

from app.database.mongo import MongoDatabase, create_mongo_database

__all__ = ["MongoDatabase", "create_mongo_database"]
