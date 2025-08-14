# models/cluster_assignments.py
from datetime import datetime
from typing import Optional, List
from beanie import Document
from pydantic import Field, ConfigDict
from bson import ObjectId
from pymongo import IndexModel


class ClusterAssignment(Document):
    """
    Represents a mapping from an anonymous cluster to a known user profile.
    One row per (clustering_id, cluster_id) assignment.
    """
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    
    # ids / keys
    clustering_id: ObjectId
    cluster_id: str
    bucket_path: str  # e.g. "uwas-classification-recette/usersAvatars"
    
    # target user
    user_id: str  # from user_profiles.user_id
    similarity: float
    strategy: str = "centroid"  # or "vote"
    
    # additional metadata
    face_count: int = 0
    image_paths: List[str] = Field(default_factory=list)
    face_ids: List[str] = Field(default_factory=list)
    
    # timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(arbitrary_types_allowed=True, json_encoders={ObjectId: str})
    
    class Settings:
        name = "cluster_assignments"
        indexes = [
            IndexModel([("clustering_id", 1), ("cluster_id", 1)], name="clustering_cluster_unique", unique=True),
            IndexModel([("user_id", 1)], name="user_id_idx"),
            IndexModel([("bucket_path", 1)], name="bucket_path_idx"),
            IndexModel([("created_at", -1)], name="created_at_desc"),
        ]
    
    def touch(self):
        self.updated_at = datetime.utcnow()