# app/models/face_clustering_models.py

from beanie import Document
from pydantic import Field, ConfigDict, BaseModel,validator,field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pymongo import IndexModel
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)


from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            try:
                return ObjectId(v)
            except Exception:
                raise ValueError("Invalid ObjectId")
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string", example="507f1f77bcf86cd799439011")

class PersonClusterInfo(BaseModel):
    """Information about a person-based cluster"""
    person_id: str  # Unique identifier for this person
    owner_face_id: str  # Face ID of the representative/strongest face
    owner_embedding: List[float]  # Primary embedding for this person
    image_paths: List[str]  # All images this person appears in
    face_ids: List[str]  # All face IDs for this person across images
    confidence_scores: List[float]  # Confidence for each detection
    quality_scores: List[float]  # Quality scores for each face
    size: int  # Total number of images this person appears in
    avg_confidence: float  # Average confidence across all detections
    best_quality_score: float  # Highest quality score among all faces

class ImageOverlapStats(BaseModel):
    """Statistics about how images overlap across person clusters"""
    single_person_images: int = 0
    multi_person_images: int = 0
    max_persons_per_image: int = 0
    avg_persons_per_image: float = 0.0  # Changed from int to float

class PersonBasedClusteringResult(Document):
    """Results from person-based clustering where each person gets their own cluster"""
    bucket_path: str
    person_clusters: List[PersonClusterInfo] = []
    unassigned_faces: List[str] = []
    unassigned_face_ids: List[str] = []
    total_images: int = 0
    total_faces: int = 0
    total_persons: int = 0
    image_overlap_stats: ImageOverlapStats = Field(default_factory=ImageOverlapStats)
    processing_stats: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "person_based_clustering_results"
        indexes = [
            IndexModel([("bucket_path", 1), ("timestamp", -1)], name="bucket_timestamp_idx"),
            IndexModel([("total_persons", -1)], name="persons_count_idx")
        ]

class ClusterInfo(BaseModel):
    """Legacy cluster info - kept for backward compatibility"""
    cluster_id: str
    image_paths: List[str]
    face_ids: List[str]
    size: int
    centroid: Optional[List[float]] = None

class ClusterResultResponse(BaseModel):
    """Updated response model for person-based clustering"""
    bucket: str
    person_clusters: List[Dict[str, Any]]  # Person-based clusters
    unassigned: List[str]  # Unassigned images
    timestamp: datetime
    stats: Dict[str, Any]  # Processing statistics dictionary

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()},
    )

class FaceEmbeddingBase(Document):
    """Enhanced face embedding with person assignment"""
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    face_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    clustering_id: Optional[PyObjectId] = None
    image_path: str
    embedding: List[float] = Field(default_factory=list)
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Person-based clustering fields
    person_id: Optional[str] = None  # Which person this face belongs to
    is_owner_face: bool = False  # True if this is the representative face for the person
    cluster_confidence: Optional[float] = None  # Confidence of assignment to person

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

    class Settings:
        name = "face_embeddings_enhanced"
        indexes = [
            IndexModel([("image_path", 1)]),
            IndexModel([("person_id", 1)]),
            IndexModel([("timestamp", -1)]),
            IndexModel([("face_id", 1)], unique=True),
            IndexModel([("clustering_id", 1)]),
            IndexModel([("is_owner_face", 1)])
        ]

# Keep the original ClusteringResult for backward compatibility
class ClusteringResult(Document):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    bucket_path: str
    clusters: List[ClusterInfo]
    noise: List[str]
    noise_face_ids: List[str]
    total_images: int
    total_faces: int
    num_clusters: int
    processing_stats: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )
    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_ts(cls, v):
        if isinstance(v, datetime): return v
        if isinstance(v, str):
            try: return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except: return datetime.utcnow()
        return datetime.utcnow()
    @validator('id', pre=True, always=True)
    def validate_id(cls, v):
        if v is None:
            return ObjectId()
        if isinstance(v, str):
            try:
                return ObjectId(v)
            except:
                return ObjectId()
        return v
    class Settings:
        name = "clustering_results"
        indexes = [
            IndexModel(
                [("bucket_path", 1), ("timestamp", -1)],
                name="bucket_path_1_timestamp_-1"
            )
        ]