# app/core/database.py
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.models.user_profile import UserProfile
from app.core.config import settings

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    database = None

db = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        # Get MongoDB URL from settings
        mongodb_url = getattr(settings, 'MONGODB_URL', 'mongodb://localhost:27017')
        db_name = getattr(settings, 'DATABASE_NAME', 'face_recognition')
        
        logger.info(f"Connecting to MongoDB at: {mongodb_url}")
        
        # Create Motor client
        db.client = AsyncIOMotorClient(mongodb_url)
        db.database = db.client[db_name]
        
        # Test the connection
        await db.client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        # Initialize Beanie with the UserProfile model
        await init_beanie(
            database=db.database,
            document_models=[UserProfile]  # Add other models here as needed
        )
        logger.info("Beanie initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")

# Alternative: Direct MongoDB operations without Beanie
from pymongo import MongoClient
from datetime import datetime
import asyncio

class DirectMongoUserProfileDB:
    """Direct MongoDB operations as fallback"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        
    async def connect(self):
        """Initialize direct MongoDB connection"""
        try:
            mongodb_url = getattr(settings, 'MONGODB_URL', 'mongodb://localhost:27017')
            db_name = getattr(settings, 'DATABASE_NAME', 'face_recognition')
            
            # Use AsyncIOMotorClient for async operations
            from motor.motor_asyncio import AsyncIOMotorClient
            self.client = AsyncIOMotorClient(mongodb_url)
            self.db = self.client[db_name]
            self.collection = self.db.user_profiles
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Direct MongoDB connection successful")
            return True
        except Exception as e:
            logger.error(f"Direct MongoDB connection failed: {e}")
            return False
    
    async def save_user_profile(
        self,
        user_id: str,
        embedding: list,
        confidence: float = None,
        quality_score: float = None,
        bbox: list = None,
        image_data: bytes = None,
        image_path: str = None
    ) -> bool:
        """Save user profile using direct MongoDB operations"""
        try:
            if not self.collection:
                await self.connect()
            
            now = datetime.utcnow()
            
            # Prepare document
            doc = {
                "user_id": user_id,
                "embedding": embedding,
                "confidence": confidence,
                "quality_score": quality_score,
                "bbox": bbox,
                "image_data": image_data,
                "image_path": image_path,
                "updated_at": now
            }
            
            # Remove None values
            doc = {k: v for k, v in doc.items() if v is not None}
            
            # Upsert operation
            result = await self.collection.update_one(
                {"user_id": user_id},
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": now}
                },
                upsert=True
            )
            
            logger.info(f"Profile saved for {user_id} (upserted: {result.upserted_id is not None})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save profile for {user_id}: {e}")
            return False
    
    async def get_user_profile(self, user_id: str) -> dict:
        """Get user profile using direct MongoDB operations"""
        try:
            if not self.collection:
                await self.connect()
            
            doc = await self.collection.find_one({"user_id": user_id})
            return doc
            
        except Exception as e:
            logger.error(f"Failed to get profile for {user_id}: {e}")
            return None

# Create global instance
direct_db = DirectMongoUserProfileDB()