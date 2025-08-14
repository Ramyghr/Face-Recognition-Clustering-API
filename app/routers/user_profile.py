# app/routers/user_profile.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Query
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import re
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# FIXED: Import the corrected service
from app.services.face_detection_service import face_detection_service
from app.schemas.detection import DetectionResponse
from app.core.config import settings

router = APIRouter(tags=["User Face Detection"])
logger = logging.getLogger(__name__)

def validate_image_content(content: bytes) -> None:
    """Validate image content and basic integrity"""
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded"
        )
    
    try:
        Image.open(BytesIO(content)).verify()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

def validate_user_id(user_id: str) -> None:
    """Prevent NoSQL injection and validate user ID format"""
    if not re.match(r"^[\w\-\.@]{1,64}$", user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format. Use alphanumeric characters with .-_@ only"
        )

@router.post("/detect-face", response_model=DetectionResponse, summary="Detect face and create user profile")
async def detect_and_create_profile(
    user_id: str = Form(..., description="Unique user identifier (alphanumeric with .-_@)"),
    file: UploadFile = File(..., description="Image file with a clear front-facing face"),
    # Optional quality parameters for testing
    min_face_size: int = Form(30, description="Minimum face size in pixels"),
    min_confidence: float = Form(0.4, description="Minimum detection confidence (0.0-1.0)"),
    blur_threshold: float = Form(50, description="Minimum blur threshold (higher = less blurry)")
):
    """
    Processes an image to detect faces with high accuracy, extracts facial embeddings, 
    and stores them in the database. Requires exactly one high-quality face in the image.
    
    Key validations:
    - Strict user ID format validation
    - Image content verification
    - Exactly one face detection
    - Face quality checks (size, clarity, confidence)
    
    Returns detailed detection results with quality metrics.
    """
    try:
        # Validate inputs
        validate_user_id(user_id)
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be an image"
            )
        
        content = await file.read()
        validate_image_content(content)
        
        # Log image details
        logger.info(f"Processing image: {file.filename}, size: {len(content)} bytes, type: {file.content_type}")
        
        # Process image and detect faces with custom quality parameters
        logger.info(f"Processing face detection for user: {user_id} with params: min_size={min_face_size}, min_conf={min_confidence}, blur_thresh={blur_threshold}")
        
        result = await face_detection_service.detect_and_save_user_profile(
            user_id=user_id, 
            image_data=content,
            original_filename=file.filename,
            min_face_size=min_face_size,
            min_confidence=min_confidence,
            blur_threshold=blur_threshold
        )
        
        logger.info(f"Detection result: {result}")
        
        # Handle detection results with better error reporting
        if result.get('has_face') and result.get('single_face'):
            return DetectionResponse(
                user_id=user_id,
                has_face=True,
                message="Face detected and profile saved successfully",
                face_quality=result.get('quality_score'),
                confidence=result.get('confidence')
            )
        else:
            # More specific error handling with detailed info
            reason = result.get('reason', 'Face detection failed')
            db_error = result.get('db_error')
            
            # Include quality metrics in the response for debugging
            response_data = {
                "user_id": user_id,
                "has_face": result.get('has_face', False),
                "message": reason,
                "reason": reason
            }
            
            # Add debugging information
            if 'quality_metrics' in result:
                response_data['debug_info'] = {
                    'quality_metrics': result['quality_metrics'],
                    'detailed_checks': result.get('detailed_checks', [])
                }
            
            if db_error:
                logger.error(f"Database error for user {user_id}: {db_error}")
                response_data['message'] = f"Database error: {db_error}"
                response_data['reason'] = f"Database error: {db_error}"
            
            # Return as JSON response to include debug info
            return JSONResponse(
                status_code=200,
                content=response_data
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Face detection failed for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during face processing: {str(e)}"
        )

@router.put("/update-face/{user_id}", response_model=DetectionResponse, summary="Update user profile with new face image")
async def update_user_face(
    user_id: str,
    file: UploadFile = File(..., description="New image file to update profile"),
    # Optional quality parameters for testing
    min_face_size: int = Form(30, description="Minimum face size in pixels"),
    min_confidence: float = Form(0.4, description="Minimum detection confidence (0.0-1.0)"),
    blur_threshold: float = Form(50, description="Minimum blur threshold (higher = less blurry)")
):
    """Update existing user profile with a new face image using enhanced validation"""
    try:
        validate_user_id(user_id)
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be an image"
            )
        
        content = await file.read()
        validate_image_content(content)
        
        logger.info(f"Updating face profile for user: {user_id}")
        result = await face_detection_service.detect_and_save_user_profile(
            user_id=user_id,
            image_data=content,
            original_filename=file.filename,
            update_existing=True,
            min_face_size=min_face_size,
            min_confidence=min_confidence,
            blur_threshold=blur_threshold
        )
        
        if result.get('has_face') and result.get('single_face'):
            return DetectionResponse(
                user_id=user_id,
                has_face=True,
                message="Profile updated successfully",
                face_quality=result.get('quality_score'),
                confidence=result.get('confidence')
            )
        else:
            reason = result.get('reason', 'Face detection failed')
            db_error = result.get('db_error')
            
            response_data = {
                "user_id": user_id,
                "has_face": result.get('has_face', False),
                "message": reason,
                "reason": reason
            }
            
            if 'quality_metrics' in result:
                response_data['debug_info'] = {
                    'quality_metrics': result['quality_metrics'],
                    'detailed_checks': result.get('detailed_checks', [])
                }
            
            if db_error:
                logger.error(f"Database error for user {user_id}: {db_error}")
                response_data['message'] = f"Database error: {db_error}"
                response_data['reason'] = f"Database error: {db_error}"
            
            return JSONResponse(
                status_code=200,
                content=response_data
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Profile update failed for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during profile update: {str(e)}"
        )

@router.get("/debug-quality-settings", summary="Get current quality check settings")
async def get_quality_settings():
    """Get the current default quality check settings"""
    return {
        "default_min_face_size": 30,
        "default_min_confidence": 0.4,
        "default_blur_threshold": 50,
        "description": {
            "min_face_size": "Minimum face size in pixels (width or height)",
            "min_confidence": "Minimum detection confidence score (0.0 to 1.0)",
            "blur_threshold": "Minimum blur score (higher values = less blurry)"
        },
        "suggestions": {
            "very_lenient": {"min_face_size": 20, "min_confidence": 0.3, "blur_threshold": 30},
            "lenient": {"min_face_size": 30, "min_confidence": 0.4, "blur_threshold": 50},
            "normal": {"min_face_size": 40, "min_confidence": 0.5, "blur_threshold": 80},
            "strict": {"min_face_size": 60, "min_confidence": 0.7, "blur_threshold": 120}
        }
    }

@router.get("/debug-profiles", summary="List all stored profiles (debugging)")
async def debug_list_profiles(limit: int = Query(10, description="Maximum number of profiles to return")):
    """Debug endpoint to list all stored user profiles"""
    try:
        from app.db.user_profile_operations import UserProfileDB
        profiles = await UserProfileDB.list_all_profiles(limit=limit)
        
        # Remove sensitive data for debugging
        safe_profiles = []
        for profile in profiles:
            safe_profile = {
                "user_id": profile.get("user_id"),
                "embedding_size": len(profile.get("embedding", [])),
                "confidence": profile.get("confidence"),
                "quality_score": profile.get("quality_score"),
                "created_at": profile.get("created_at"),
                "updated_at": profile.get("updated_at"),
                "has_bbox": bool(profile.get("bbox"))
            }
            safe_profiles.append(safe_profile)
        
        return {
            "total_profiles": len(safe_profiles),
            "profiles": safe_profiles
        }
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return {"error": str(e), "total_profiles": 0, "profiles": []}

@router.get("/debug-profile/{user_id}", summary="Get specific profile details (debugging)")
async def debug_get_profile(user_id: str):
    """Debug endpoint to get specific user profile details"""
    try:
        from app.db.user_profile_operations import UserProfileDB
        
        validate_user_id(user_id)
        profile = await UserProfileDB.get_user_profile(user_id)
        
        if not profile:
            return {"user_id": user_id, "exists": False, "profile": None}
        
        # Return safe profile data
        safe_profile = {
            "user_id": profile.get("user_id"),
            "embedding_size": len(profile.get("embedding", [])),
            "confidence": profile.get("confidence"),
            "quality_score": profile.get("quality_score"),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
            "has_bbox": bool(profile.get("bbox")),
            "bbox": profile.get("bbox") if profile.get("bbox") else None
        }
        
        return {"user_id": user_id, "exists": True, "profile": safe_profile}
        
    except Exception as e:
        logger.error(f"Error getting profile for {user_id}: {e}")
        return {"user_id": user_id, "exists": False, "error": str(e)}

@router.delete("/debug-profile/{user_id}", summary="Delete specific profile (debugging)")
async def debug_delete_profile(user_id: str):
    """Debug endpoint to delete specific user profile"""
    try:
        from app.db.user_profile_operations import UserProfileDB
        
        validate_user_id(user_id)
        success = await UserProfileDB.delete_user_profile(user_id)
        
        return {
            "user_id": user_id,
            "deleted": success,
            "message": f"Profile for {user_id} {'deleted successfully' if success else 'not found or deletion failed'}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting profile for {user_id}: {e}")
        return {"user_id": user_id, "deleted": False, "error": str(e)}