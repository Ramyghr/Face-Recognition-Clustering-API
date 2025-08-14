# app/models/user_profile.py
from datetime import datetime
from typing import List, Optional,Dict,Any
from beanie import Document, Indexed
from bson import ObjectId
from pymongo import IndexModel
from pydantic import Field, BaseModel, ConfigDict, validator
import re


class UserProfile(Document):
    """
    Represents a user's facial profile with embedding and metadata
    """

    # MongoDB document ID
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")

    # Unique user identifier (email, username, etc.)
    user_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique user identifier"
    )

    # Facial embedding vector
    embedding: List[float] = Field(..., min_items=128, max_items=1024)

    # Optional fields
    image_data: Optional[bytes] = None
    image_path: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0)
    bbox: Optional[List[float]] = Field(None, min_items=4, max_items=4)  # [x, y, w, h]

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()},
        exclude_none=True  # Exclude None values from JSON
    )

    class Settings:
        name = "user_profiles"
        use_revision = True  # ✅ must be a boolean, not a method
        indexes = [
            IndexModel([("user_id", 1)], name="user_id_unique_idx", unique=True),
            IndexModel([("created_at", -1)], name="created_at_desc_idx"),
            IndexModel([("updated_at", -1)], name="updated_at_desc_idx"),
        ]

    @validator("user_id")
    def validate_user_id(cls, v):
        if not re.match(r"^[\w\-\.@]{1,64}$", v):
            raise ValueError("User ID must be 1-64 chars: alphanumeric with .-_@")
        return v

    @validator("embedding")
    def validate_embedding(cls, v):
        if len(v) < 128:
            raise ValueError("Embedding must have at least 128 dimensions")
        return v

    def update_timestamp(self):
        """Update the 'updated_at' timestamp"""
        self.updated_at = datetime.utcnow()

    class BBox(BaseModel):
        """Bounding box model for validation"""
        x: float = Field(..., ge=0.0, le=1.0)
        y: float = Field(..., ge=0.0, le=1.0)
        width: float = Field(..., ge=0.0, le=1.0)
        height: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def validate_bbox(cls, bbox: List[float]):
        """Validate bounding box coordinates"""
        if bbox and len(bbox) == 4:
            try:
                cls.BBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])
                return True
            except Exception:
                pass
        return False
class AssignedCluster(BaseModel):
    clustering_id: str
    cluster_id: str
    bucket_path: str
    similarity: float
    strategy: str
    assigned_at: datetime
    
assigned_clusters: Optional[List[AssignedCluster]] = None
current_cluster: Optional[Dict[str, Any]] = None  # per-bucket override, optional