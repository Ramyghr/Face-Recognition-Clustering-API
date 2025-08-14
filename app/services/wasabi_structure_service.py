# app/services/wasabi_structure_service.py
import uuid
import hashlib
import time
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UserIDManager:
    """Utility class for managing user IDs in Wasabi structure"""
    
    @staticmethod
    def generate_uuid_user_id() -> str:
        """Generate a UUID-based user ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_timestamp_user_id() -> str:
        """Generate a timestamp-based user ID"""
        timestamp = int(time.time() * 1000)  # milliseconds
        return f"user_{timestamp}"
    
    @staticmethod
    def generate_hash_user_id(email: Optional[str] = None, name: Optional[str] = None) -> str:
        """Generate a hash-based user ID from email or name"""
        if email:
            return f"user_{hashlib.md5(email.encode()).hexdigest()[:12]}"
        elif name:
            return f"user_{hashlib.md5(name.encode()).hexdigest()[:12]}"
        else:
            return UserIDManager.generate_uuid_user_id()
    
    @staticmethod
    def generate_sequential_user_id(current_max_id: int = 0) -> str:
        """Generate a sequential user ID"""
        return f"user_{current_max_id + 1:06d}"  # e.g., user_000001
    
    @staticmethod
    def is_valid_user_id(user_id: str) -> bool:
        """Validate user ID format"""
        if not user_id or len(user_id.strip()) == 0:
            return False
        
        # Check for valid characters (alphanumeric, hyphens, underscores)
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', user_id))

class WasabiStructureManager:
    """Helper class for managing Wasabi bucket structure"""
    
    # Configuration
    MAIN_BUCKET = "face-recognition-app"
    ANONYMOUS_PREFIX = "anonymous-images"
    USER_PROFILES_PREFIX = "user-profiles"
    TEMP_PREFIX = "temp-uploads"
    CLUSTERING_RESULTS_PREFIX = "clustering-results"
    
    @staticmethod
    def get_anonymous_images_path(batch_name: str = "default") -> str:
        """Get path for anonymous images clustering"""
        # Clean batch name
        clean_batch = WasabiStructureManager._clean_path_component(batch_name)
        return f"{WasabiStructureManager.ANONYMOUS_PREFIX}/{clean_batch}"
    
    @staticmethod
    def get_user_profile_path(user_id: str) -> str:
        """Get path for user profile image"""
        clean_user_id = WasabiStructureManager._clean_path_component(user_id)
        return f"{WasabiStructureManager.USER_PROFILES_PREFIX}/{clean_user_id}"
    
    @staticmethod
    def get_user_profile_image_key(user_id: str, filename: str = "profile.jpg") -> str:
        """Get full S3 key for user profile image"""
        clean_filename = WasabiStructureManager._clean_filename(filename)
        return f"{WasabiStructureManager.get_user_profile_path(user_id)}/{clean_filename}"
    
    @staticmethod
    def get_temp_upload_path(session_id: str) -> str:
        """Get path for temporary uploads"""
        clean_session = WasabiStructureManager._clean_path_component(session_id)
        return f"{WasabiStructureManager.TEMP_PREFIX}/{clean_session}"
    
    @staticmethod
    def get_clustering_results_path(bucket_name: str, sub_bucket: str) -> str:
        """Get path for clustering results storage"""
        clean_bucket = WasabiStructureManager._clean_path_component(bucket_name)
        clean_sub = WasabiStructureManager._clean_path_component(sub_bucket)
        return f"{WasabiStructureManager.CLUSTERING_RESULTS_PREFIX}/{clean_bucket}/{clean_sub}"
    
    @staticmethod
    def parse_image_path(image_path: str) -> Dict[str, Optional[str]]:
        """Parse an image path to extract bucket structure information"""
        try:
            parts = image_path.strip('/').split('/')
            
            result = {
                "bucket": None,
                "main_category": None,
                "sub_category": None,
                "filename": None,
                "user_id": None,
                "batch_name": None,
                "is_anonymous": False,
                "is_user_profile": False,
                "is_temp": False
            }
            
            if len(parts) >= 2:
                result["main_category"] = parts[0]
                
                if parts[0] == WasabiStructureManager.ANONYMOUS_PREFIX:
                    result["is_anonymous"] = True
                    result["batch_name"] = parts[1] if len(parts) > 1 else None
                    result["filename"] = parts[-1] if len(parts) > 2 else parts[1]
                
                elif parts[0] == WasabiStructureManager.USER_PROFILES_PREFIX:
                    result["is_user_profile"] = True
                    result["user_id"] = parts[1] if len(parts) > 1 else None
                    result["filename"] = parts[-1] if len(parts) > 2 else "profile.jpg"
                
                elif parts[0] == WasabiStructureManager.TEMP_PREFIX:
                    result["is_temp"] = True
                    result["sub_category"] = parts[1] if len(parts) > 1 else None
                    result["filename"] = parts[-1] if len(parts) > 2 else parts[1]
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing image path {image_path}: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def generate_batch_name(prefix: str = "batch") -> str:
        """Generate a unique batch name for anonymous images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    @staticmethod
    def _clean_path_component(component: str) -> str:
        """Clean a path component to be safe for S3"""
        if not component:
            return "default"
        
        # Replace unsafe characters
        import re
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', component)
        
        # Remove multiple underscores
        cleaned = re.sub(r'_{2,}', '_', cleaned)
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        # Ensure it's not empty
        if not cleaned:
            cleaned = "default"
        
        return cleaned.lower()
    
    @staticmethod
    def _clean_filename(filename: str) -> str:
        """Clean filename to be safe for S3"""
        if not filename:
            return "file.jpg"
        
        # Keep original extension
        import os
        name, ext = os.path.splitext(filename)
        
        # Clean the name part
        clean_name = WasabiStructureManager._clean_path_component(name)
        
        # Ensure we have an extension
        if not ext:
            ext = ".jpg"
        
        return f"{clean_name}{ext}"

class WasabiBucketOrganizer:
    """Service for organizing and managing Wasabi bucket contents"""
    
    def __init__(self, structure_manager: WasabiStructureManager):
        self.structure = structure_manager
    
    async def organize_anonymous_batch(
        self, 
        file_paths: List[str], 
        batch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Organize a batch of anonymous images"""
        try:
            if not batch_name:
                batch_name = self.structure.generate_batch_name("anonymous")
            
            organized_paths = []
            batch_path = self.structure.get_anonymous_images_path(batch_name)
            
            for i, file_path in enumerate(file_paths):
                # Generate organized path
                filename = f"image_{i+1:04d}.jpg"  # e.g., image_0001.jpg
                organized_path = f"{batch_path}/{filename}"
                organized_paths.append(organized_path)
            
            return {
                "batch_name": batch_name,
                "batch_path": batch_path,
                "organized_paths": organized_paths,
                "total_files": len(file_paths),
                "bucket": self.structure.MAIN_BUCKET
            }
            
        except Exception as e:
            logger.error(f"Error organizing anonymous batch: {e}")
            raise
    
    async def organize_user_profiles(
        self, 
        user_files: Dict[str, str]  # user_id -> file_path
    ) -> Dict[str, Dict[str, str]]:
        """Organize user profile images"""
        try:
            organized_users = {}
            
            for user_id, file_path in user_files.items():
                if not UserIDManager.is_valid_user_id(user_id):
                    logger.warning(f"Invalid user ID: {user_id}")
                    continue
                
                profile_key = self.structure.get_user_profile_image_key(user_id)
                profile_path = self.structure.get_user_profile_path(user_id)
                
                organized_users[user_id] = {
                    "original_path": file_path,
                    "profile_key": profile_key,
                    "profile_path": profile_path,
                    "bucket": self.structure.MAIN_BUCKET
                }
            
            return organized_users
            
        except Exception as e:
            logger.error(f"Error organizing user profiles: {e}")
            raise
    
    async def get_bucket_statistics(self) -> Dict[str, Any]:
        """Get statistics about bucket organization"""
        # This would typically query your Wasabi bucket
        # For now, return structure information
        return {
            "bucket_structure": {
                "main_bucket": self.structure.MAIN_BUCKET,
                "categories": {
                    "anonymous_images": self.structure.ANONYMOUS_PREFIX,
                    "user_profiles": self.structure.USER_PROFILES_PREFIX,
                    "temp_uploads": self.structure.TEMP_PREFIX,
                    "clustering_results": self.structure.CLUSTERING_RESULTS_PREFIX
                }
            },
            "organization_rules": {
                "anonymous_batches": "Organized by batch name with sequential numbering",
                "user_profiles": "One folder per user with profile.jpg",
                "temp_uploads": "Session-based temporary storage",
                "clustering_results": "Results stored by bucket/sub-bucket structure"
            }
        }

# Factory function to create instances
def create_wasabi_organizer() -> WasabiBucketOrganizer:
    """Create a configured Wasabi bucket organizer"""
    structure_manager = WasabiStructureManager()
    return WasabiBucketOrganizer(structure_manager)