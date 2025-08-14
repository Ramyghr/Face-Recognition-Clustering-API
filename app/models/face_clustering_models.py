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

class ClusterInfo(BaseModel):
    cluster_id: str
    image_paths: List[str]
    face_ids: List[str]
    size: int
    centroid: Optional[List[float]] = None

class ClusterResultResponse(BaseModel):
    bucket: str
    clusters: List[List[str]]  # List of image paths in each cluster
    noise: List[str]  # List of image paths marked as noise
    timestamp: datetime
    stats: Dict[str, Any]  # Processing statistics dictionary

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()},
    )
class FaceEmbeddingBase(Document):
    """Fixed version of FaceEmbedding that prevents _id conflicts"""
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")  # Explicit ObjectId
    face_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    clustering_id: Optional[PyObjectId] = None
    image_path: str
    embedding: List[float] = Field(default_factory=list)
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    quality_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cluster_id: str = Field(default="unassigned")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )

    class Settings:
        name = "face_embeddings_fixed"
        indexes = [
            IndexModel([("image_path", 1)]),
            IndexModel([("cluster_id", 1)]),
            IndexModel([("timestamp", -1)]),
            IndexModel([("face_id", 1)], unique=True),
            IndexModel([("clustering_id", 1)])
        ]
    

class ClusteringResult(Document):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")  # Explicit ObjectId
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
