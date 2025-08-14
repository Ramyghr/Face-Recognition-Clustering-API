# app/db/user_profile_operations.py
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import re
from app.core.database import direct_db

logger = logging.getLogger(__name__)

class UserProfileDB:
    @staticmethod
    async def save_user_profile(
        user_id: str,
        embedding: List[float],
        confidence: Optional[float] = None,
        quality_score: Optional[float] = None,
        bbox: Optional[List[float]] = None,
        image_data: Optional[bytes] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """Upsert a user profile with face embedding data"""
        try:
            # First validate the user_id format
            if not re.match(r"^[\w\-\.@]{1,64}$", user_id):
                raise ValueError(f"Invalid user_id format: {user_id}")
            
            # Validate embedding is not empty
            if not embedding or len(embedding) < 128:
                raise ValueError("Embedding must have at least 128 dimensions")
            
            logger.info(f"Attempting to save profile for user: {user_id}")
            
            # Try Beanie first, fallback to direct MongoDB
            beanie_success = False
            try:
                from app.models.user_profile import UserProfile
                
                # Check if Beanie is initialized by trying to use it
                existing_doc = await UserProfile.find_one({"user_id": user_id})
                
                now = datetime.utcnow()
                
                if existing_doc:
                    # Update existing document
                    update_data = {
                        "embedding": embedding,
                        "updated_at": now,
                        "confidence": confidence,
                        "quality_score": quality_score,
                        "bbox": bbox
                    }
                    # Remove None values
                    update_data = {k: v for k, v in update_data.items() if v is not None}
                    
                    await existing_doc.update({"$set": update_data})
                    logger.info(f"Updated existing profile for {user_id} using Beanie")
                else:
                    # Create new document
                    new_doc = UserProfile(
                        user_id=user_id,
                        embedding=embedding,
                        confidence=confidence,
                        quality_score=quality_score,
                        bbox=bbox,
                        image_data=image_data,
                        image_path=image_path,
                        created_at=now,
                        updated_at=now,
                    )
                    await new_doc.insert()
                    logger.info(f"Created new profile for {user_id} using Beanie")
                
                beanie_success = True
                return True
                
            except Exception as beanie_error:
                logger.warning(f"Beanie operation failed for {user_id}: {beanie_error}")
                beanie_success = False
            
            # If Beanie failed, use direct MongoDB operations
            if not beanie_success:
                logger.info(f"Using direct MongoDB for {user_id}")
                
                # Ensure direct_db is connected
                if not hasattr(direct_db, 'collection') or direct_db.collection is None:
                    logger.info("Initializing direct_db connection")
                    await direct_db.connect()
                
                # Check if collection is properly initialized
                if direct_db.collection is None:
                    logger.error("Direct MongoDB collection is None after connection attempt")
                    return False
                
                now = datetime.utcnow()
                
                # Prepare document data
                doc_data = {
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
                doc_data = {k: v for k, v in doc_data.items() if v is not None}
                
                # Use upsert operation (update if exists, insert if not)
                try:
                    result = await direct_db.collection.update_one(
                        {"user_id": user_id},
                        {
                            "$set": doc_data,
                            "$setOnInsert": {"created_at": now}
                        },
                        upsert=True
                    )
                    
                    if result.upserted_id:
                        logger.info(f"Inserted new profile for {user_id} using direct MongoDB")
                    else:
                        logger.info(f"Updated existing profile for {user_id} using direct MongoDB")
                    
                    return True
                    
                except Exception as mongo_error:
                    logger.error(f"Direct MongoDB operation failed for {user_id}: {mongo_error}")
                    return False
            
        except Exception as e:
            logger.error(f"Failed to save profile for {user_id}: {str(e)}", exc_info=True)
            return False

    @staticmethod
    async def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user profile by user ID"""
        try:
            logger.debug(f"Fetching profile for user: {user_id}")
            
            # Try Beanie first
            try:
                from app.models.user_profile import UserProfile
                doc = await UserProfile.find_one({"user_id": user_id})
                if doc:
                    logger.debug(f"Found profile for {user_id} using Beanie")
                    return doc
            except Exception as beanie_error:
                logger.warning(f"Beanie query failed for {user_id}: {beanie_error}")
            
            # Fallback to direct MongoDB
            try:
                # Ensure direct_db is connected
                if not hasattr(direct_db, 'collection') or direct_db.collection is None:
                    await direct_db.connect()
                
                # Check if collection is properly initialized
                if direct_db.collection is None:
                    logger.error("Direct MongoDB collection is None")
                    return None
                
                doc = await direct_db.collection.find_one({"user_id": user_id})
                if doc:
                    logger.debug(f"Found profile for {user_id} using direct MongoDB")
                    return doc
                else:
                    logger.debug(f"No profile found for {user_id}")
                    return None
                    
            except Exception as mongo_error:
                logger.error(f"Direct MongoDB query failed for {user_id}: {mongo_error}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching profile for {user_id}: {e}")
            return None
    
    @staticmethod
    async def delete_user_profile(user_id: str) -> bool:
        """Delete user profile by user ID"""
        try:
            logger.info(f"Deleting profile for user: {user_id}")
            
            # Try Beanie first
            try:
                from app.models.user_profile import UserProfile
                doc = await UserProfile.find_one({"user_id": user_id})
                if doc:
                    await doc.delete()
                    logger.info(f"Deleted profile for {user_id} using Beanie")
                    return True
                else:
                    logger.warning(f"Profile not found for deletion via Beanie: {user_id}")
            except Exception as beanie_error:
                logger.warning(f"Beanie delete failed for {user_id}: {beanie_error}")
            
            # Fallback to direct MongoDB
            try:
                # Ensure direct_db is connected
                if not hasattr(direct_db, 'collection') or direct_db.collection is None:
                    await direct_db.connect()
                
                # Check if collection is properly initialized
                if direct_db.collection is None:
                    logger.error("Direct MongoDB collection is None")
                    return False
                
                result = await direct_db.collection.delete_one({"user_id": user_id})
                success = result.deleted_count > 0
                if success:
                    logger.info(f"Deleted profile for {user_id} using direct MongoDB")
                else:
                    logger.warning(f"Profile not found for deletion via MongoDB: {user_id}")
                return success
                
            except Exception as mongo_error:
                logger.error(f"Direct MongoDB delete failed for {user_id}: {mongo_error}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting profile for {user_id}: {e}")
            return False
    
    @staticmethod
    async def profile_exists(user_id: str) -> bool:
        """Check if user profile exists"""
        try:
            doc = await UserProfileDB.get_user_profile(user_id)
            exists = doc is not None
            logger.debug(f"Profile exists check for {user_id}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking profile existence for {user_id}: {e}")
            return False
    
    @staticmethod
    async def list_all_profiles(limit: int = 100) -> List[Dict[str, Any]]:
        """List all user profiles (for debugging)"""
        try:
            profiles = []
            
            # Try Beanie first
            try:
                from app.models.user_profile import UserProfile
                docs = await UserProfile.find_all().limit(limit).to_list()
                if docs:
                    profiles = [doc.dict() for doc in docs]
                    logger.info(f"Found {len(profiles)} profiles using Beanie")
                    return profiles
            except Exception as beanie_error:
                logger.warning(f"Beanie list failed: {beanie_error}")
            
            # Fallback to direct MongoDB
            try:
                # Ensure direct_db is connected
                if not hasattr(direct_db, 'collection') or direct_db.collection is None:
                    await direct_db.connect()
                
                if direct_db.collection is None:
                    logger.error("Direct MongoDB collection is None")
                    return []
                
                cursor = direct_db.collection.find({}).limit(limit)
                profiles = await cursor.to_list(length=limit)
                
                # Convert ObjectId to string for JSON serialization
                for profile in profiles:
                    if '_id' in profile:
                        profile['_id'] = str(profile['_id'])
                
                logger.info(f"Found {len(profiles)} profiles using direct MongoDB")
                return profiles
                
            except Exception as mongo_error:
                logger.error(f"Direct MongoDB list failed: {mongo_error}")
                return []
                
        except Exception as e:
            logger.error(f"Error listing profiles: {e}")
            return []