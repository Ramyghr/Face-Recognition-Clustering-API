# app/schemas/clustering.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class FileInput(BaseModel):
    """Input file for clustering"""
    file_path: str = Field(..., description="Path to the file in storage")
    file_name: Optional[str] = Field(None, description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file metadata")

class Payload(BaseModel):
    """Enhanced payload for clustering operations"""
    files: List[FileInput] = Field(..., min_items=1, description="List of files to process")
    bucket_name: Optional[str] = Field("face-recognition-app", description="Wasabi bucket name")
    sub_bucket: Optional[str] = Field("anonymous-images", description="Sub bucket/folder name")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.1, le=1.0, description="Similarity threshold for user matching")
    noise_similarity_threshold: Optional[float] = Field(0.5, ge=0.1, le=1.0, description="Similarity threshold for noise processing")
    client_ids: Optional[List[str]] = Field(default_factory=list, description="Optional client IDs to consider")
    
    @validator('files')
    def validate_files(cls, v):
        if not v:
            raise ValueError("At least one file must be provided")
        return v
    
    @validator('similarity_threshold', 'noise_similarity_threshold')
    def validate_thresholds(cls, v):
        if not 0.1 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.1 and 1.0")
        return v

class AlbumInfo(BaseModel):
    """Information about an identified album"""
    client_id: Optional[str] = Field(None, description="Client/User ID if identified")
    image_paths: List[str] = Field(..., description="List of image paths in this album")
    anonymous: bool = Field(True, description="Whether this album is anonymous")
    similarity_score: Optional[float] = Field(None, description="Best similarity score if matched")
    cluster_size: int = Field(..., description="Number of images in the cluster")

class AnonymousCluster(BaseModel):
    """Information about an anonymous cluster"""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    image_paths: List[str] = Field(..., description="List of image paths in this cluster")
    anonymous: bool = Field(True, description="Always true for anonymous clusters")
    max_similarity: Optional[float] = Field(None, description="Maximum similarity found but below threshold")

class NoiseInfo(BaseModel):
    """Information about noise images"""
    image_paths: List[str] = Field(default_factory=list, description="List of noise image paths")
    anonymous: bool = Field(True, description="Noise is always anonymous")
    matched_noise: Optional[List[Dict[str, Any]]] = Field(None, description="Noise images that matched users")

class ProcessingStats(BaseModel):
    """Statistics about the processing"""
    total_images: int = Field(..., description="Total number of images processed")
    total_clusters: int = Field(..., description="Total number of clusters found")
    matched_clusters: int = Field(..., description="Number of clusters matched to users")
    anonymous_clusters: int = Field(..., description="Number of anonymous clusters")
    noise_images: int = Field(..., description="Number of noise images")
    users_available: int = Field(..., description="Number of users available for matching")

class ClusteringResponse(BaseModel):
    """Response from clustering operations"""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(..., description="Clustering results data")

class EnhancedClusteringResult(BaseModel):
    """Enhanced clustering result with user identification"""
    albums: List[AlbumInfo] = Field(default_factory=list, description="Identified user albums")
    anonymous_clusters: List[AnonymousCluster] = Field(default_factory=list, description="Anonymous clusters")
    noise: NoiseInfo = Field(default_factory=NoiseInfo, description="Noise information")
    processing_stats: ProcessingStats = Field(..., description="Processing statistics")

class UserIdentificationRequest(BaseModel):
    """Request for user identification in clusters"""
    user_ids: Optional[List[str]] = Field(None, description="Specific user IDs to search for (all if None)")
    similarity_threshold: float = Field(0.6, ge=0.1, le=1.0, description="Similarity threshold")
    include_details: bool = Field(True, description="Include detailed matching information")

class ClusterMatch(BaseModel):
    """Information about a user match in a cluster"""
    user_id: str = Field(..., description="Matched user ID")
    best_similarity: float = Field(..., description="Best similarity score")
    matching_faces: List[Dict[str, Any]] = Field(..., description="Details of matching faces")
    match_count: int = Field(..., description="Number of matching faces")

class IdentifiedCluster(BaseModel):
    """Cluster with identified users"""
    cluster_id: str = Field(..., description="Cluster identifier")
    cluster_size: int = Field(..., description="Number of images in cluster")
    image_paths: List[str] = Field(..., description="Image paths in cluster")
    identified_users: Dict[str, ClusterMatch] = Field(..., description="Users identified in this cluster")

class UserIdentificationResponse(BaseModel):
    """Response from user identification"""
    bucket_path: str = Field(..., description="Bucket path processed")
    clustering_id: str = Field(..., description="Clustering ID")
    total_clusters: int = Field(..., description="Total number of clusters")
    clusters_with_identifications: int = Field(..., description="Clusters with user identifications")
    users_searched: List[str] = Field(..., description="User IDs that were searched")
    similarity_threshold: float = Field(..., description="Similarity threshold used")
    identified_clusters: List[IdentifiedCluster] = Field(..., description="Clusters with identifications")

# Legacy support - keep old schemas for backward compatibility
class LegacyPayload(BaseModel):
    """Legacy payload format for backward compatibility"""
    files: List[str] = Field(..., description="List of file paths")
    client_ids: Optional[List[str]] = Field(default_factory=list, description="Client IDs")

class Clusters(BaseModel):
    """Response model for clustering results"""
    clusters: List[AlbumInfo] = Field(default_factory=list, description="List of identified clusters")
    anonymous_clusters: List[AnonymousCluster] = Field(default_factory=list, description="Anonymous clusters")
    noise: NoiseInfo = Field(default_factory=NoiseInfo, description="Noise information")
    stats: ProcessingStats = Field(..., description="Processing statistics")

class ClientResult(BaseModel):
    """Legacy client result format"""
    client_id: Optional[str] = None
    image_paths: List[str] = []
    anonymous: bool = True
    
class FileItem(BaseModel):
    """Represents a file item with its key and optional metadata"""
    fileKey: str = Field(..., description="The key/path of the file in storage")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional file metadata")