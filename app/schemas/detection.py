# app/schemas/detection.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class DetectionResponse(BaseModel):
    """Response model for face detection endpoints"""
    user_id: str
    has_face: bool
    message: str
    face_quality: Optional[float] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

class UserProfileResponse(BaseModel):
    """Response model for user profile information"""
    user_id: str
    has_embedding: bool
    embedding_size: int
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

class UserIdentificationResponse(BaseModel):
    """Response for user identification in clusters"""
    user_id: str
    cluster_id: str
    bucket_path: str
    is_user_in_cluster: bool
    best_similarity: float
    matching_images: List[dict]
    total_cluster_images: int

class MatchingImage(BaseModel):
    """Individual matching image information"""
    image_path: str
    similarity: float
    face_id: str

class ClusterMatch(BaseModel):
    """Cluster matching information"""
    cluster_id: str
    cluster_size: int
    best_similarity: float
    matches_count: int
    matching_images: List[MatchingImage]

class UserSearchResponse(BaseModel):
    """Response for searching user across all clusters"""
    user_found: bool
    user_id: str
    bucket_path: str
    total_clusters_searched: int
    matching_clusters: List[ClusterMatch]
    total_matches: int
    reason: Optional[str] = None