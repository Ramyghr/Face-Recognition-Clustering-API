from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import os
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # ======================
    # Application Configuration
    # ======================
    CORS_ORIGINS: str
    @field_validator("CORS_ORIGINS")
    def parse_cors_origins(cls, v):
        return [origin.strip() for origin in v.split(",")] if "," in v else v

    PROJECT_NAME: str = "Clustering API"
    DEBUG: bool = False
    ENV: str = "dev"
    # LOCAL_STORAGE_PATH: str = "./local-data"
    # CLUSTER_INPUT_PATH: str = "./local-data/input"
    # CLUSTER_OUTPUT_PATH: str = "./local-data/clustered"

    # ======================
    # Model Paths
    # ======================
    YOLO_MODEL_PATH: str = "./app/ai_models/yolov11n-face.onnx"
    EMBED_MODEL_PATH: str = "./app/ai_models/Facenet.onnx"
    EMBED_MODEL_NAME: str

    # ======================
    # Image Processing
    # ======================
    MAX_IMAGE_PIXELS: int = 25_000_000
    MAX_IMAGE_SIZE_BYTES: int = 10_000_000
    MAX_IMAGE_DIMENSION: int = 640
    JPEG_QUALITY: int = 75
    FACE_SIZE_TARGET: tuple = (160, 160)

    # ======================
    # Face Detection
    # ======================
    MIN_FACE_SIZE: int = 30
    MIN_FACE_AREA_RATIO: float = 0.01
    MAX_ASPECT_RATIO: float = 1.4
    MIN_ASPECT_RATIO: float = 0.7
    MIN_CONFIDENCE: float = 0.50
    BLUR_THRESHOLD: int = 90
    CENTER_MARGIN: float = 0.1
    MAX_FACES_PER_IMAGE: int = 5

    # ======================
    # Clustering Parameters
    # ======================
    CLUSTERING_DISTANCE_THRESHOLD: float = 0.20
    SIMILARITY_THRESHOLD: float = 0.72
    MERGE_THRESHOLD: float = 0.40
    MIN_CLUSTER_SIZE: int = 3
    CLUSTER_EPS: float = 0.75
    CLUSTER_PROB_THRESHOLD: float = 0.45
    OUTLIER_STD_MULTIPLIER: float = 1.8
    MIN_QUALITY_SCORE: float = 0.25

    # ======================
    # Performance Settings
    # ======================
    EMBEDDING_BATCH_SIZE: int = 12
    DEFAULT_BATCH_SIZE: int = 15
    BATCH_SIZE: int = 15
    MAX_CONCURRENT_DOWNLOADS: int = 6          # Reduced from 10
    DEFAULT_MAX_CONCURRENT: int = 6
    MAX_CONCURRENT_PROCESSING: int = 6
    MEMORY_THRESHOLD: float = 70.0

    # ======================
    # Timeout Settings
    # ======================
    DOWNLOAD_TIMEOUT_SECONDS: int = 50         # Increased from 15
    BATCH_TIMEOUT_SECONDS: int = 120
    PIPELINE_TIMEOUT_SECONDS: int = 2400
    PROCESSING_TIMEOUT: int = 240
    CLUSTERING_TIMEOUT_SECONDS: int = 600
    MAX_RETRY_ATTEMPTS: int = 3                # New: automatic retries
    TCP_KEEPALIVE_INTERVAL: int = 60
    CONNECTION_POOL_SIZE: int = 6
    # ======================
    # Quality Control
    # ======================
    ENABLE_FACE_AUGMENTATION: bool = True
    ENABLE_QUALITY_FILTERING: bool = True
    ENABLE_FACE_ALIGNMENT: bool = True
    CONFIDENCE_WEIGHT: float = 0.5
    SIZE_WEIGHT: float = 0.25
    BLUR_WEIGHT: float = 0.25
    # ======================
    # Qdrant
    # ======================
    QDRANT_URL: Optional[str] = None  # e.g. "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HOST: Optional[str] = None  # Alternative to URL
    QDRANT_PORT: Optional[int] = None  # Alternative to URL
    QDRANT_STORAGE_PATH: str = "./qdrant_data"  # For local persistent storage
    # ======================
    # Retry Configuration
    # ======================
    RETRY_BACKOFF_BASE: float = 2.0
    MAX_RETRY_DELAY: int = 60
    MIN_RETRY_DELAY: int = 5
    # ======================
    #Assignment Similarity 
    # ======================
    ASSIGNMENT_SIMILARITY_THRESHOLD: float = 0.65
    # ======================
    # Noise Handling
    # ======================
    NOISE_RECLUSTER_THRESHOLD: float = 0.60
    NOISE_MIN_CLUSTER_SIZE: int = 2
    # ======================
    # Network Optimization
    # ======================
    S3_MAX_ATTEMPTS: int = 5
    S3_TIMEOUT: int = 120
    S3_CONNECT_TIMEOUT: int = 30
    S3_READ_TIMEOUT: int = 60
    S3_RETRY_MODE: str = "adaptive"
    # ======================
    # Database Configuration
    # ======================
    MONGO_URI: str
    MONGO_DB: str
    MONGO_DB_NAME: str = "uwas-recette"
    MONGO_AUTH_SOURCE: str = "admin"
    MONGO_CONNECT_TIMEOUT_MS: int = 2000
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "uwas-recette"
    # ======================
    # Database Optimization
    # ======================
    MONGO_RETRY_WRITES: bool = True             # Handle duplicate errors
    MONGO_SOCKET_TIMEOUT_MS: int = 30000        # Increased timeout
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 5000
    # ======================
    # Qdrant Congif
    # ======================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    # ======================
    # Storage Configuration
    # ======================
    WASABI_ENDPOINT: str
    WASABI_ACCESS_KEY: str
    WASABI_SECRET_KEY: str
    WASABI_BUCKET_NAME: str
    WASABI_REGION: str

    model_config = SettingsConfigDict(
        env_file=f".env.{os.environ.get('ENV', 'dev')}",
        env_file_encoding='utf-8'
    )

settings = Settings()
