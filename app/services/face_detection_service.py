# app/services/face_detection_service.py
import re
import logging
from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError, ImageOps
from typing import Optional, Dict, List, Any
import asyncio
from contextlib import asynccontextmanager
from app.models.user_profile import UserProfile
from app.core.config import settings
from app.services import yolo_detector
from app.db.face_clustering_operations import FaceClusteringDB
from app.services.ai import inference_pipeline, init_face_model
from app.services.S3_functions import storage_service
import os

# FIXED: Import the corrected UserProfileDB
from app.db.user_profile_operations import UserProfileDB

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()  # Enable HEIC/HEIF if library is present
except ImportError:
    pass # HEIC/HEIF support is optional

# Configure PIL security
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = getattr(settings, 'MAX_IMAGE_PIXELS', 89478485)

logger = logging.getLogger(__name__)

class FaceDetectionService:
    """
    Enhanced face detection service with proper async support and database integration
    """
    
    def __init__(self):
        self.model_initialized = False
        self.yolo_initialized = False
        
    async def _ensure_models_initialized(self):
        """Ensure AI models are initialized"""
        try:
            # Initialize face embedding model (async)
            if not self.model_initialized:
                await init_face_model(model_path=settings.EMBED_MODEL_PATH)
                self.model_initialized = True

            # Initialize YOLO model (SYNC – do NOT await)
            if not self.yolo_initialized:
                try:
                    yolo_detector.init_yolo_model()  # uses snapshot -> settings -> HF
                    self.yolo_initialized = True
                except Exception as e:
                    logger.error(f"YOLO init failed: {e}")
                    raise RuntimeError("Couldn't initialize face detector")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize AI models: {e}")
    
    def _calculate_blur(self, face: np.ndarray) -> float:
        """Calculate blur using Laplacian variance"""
        try:
            if face is None or face.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY) if len(face.shape) == 3 else face
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.debug(f"Calculated blur score: {blur_score}")
            return blur_score
        except Exception as e:
            logger.warning(f"Blur calculation failed: {e}")
            return 0.0
    
    def _filter_quality(
        self, 
        face_data: Dict, 
        min_size: int = 30,  # Reduced from 40 to be more lenient
        min_confidence: float = 0.5,  # Reduced from 0.7 to be more lenient
        blur_threshold: float = 80  # Reduced from 120 to be more lenient
    ) -> Dict[str, Any]:
        """
        Filter low-quality faces with detailed logging
        Returns dict with pass/fail status and detailed metrics
        """
        try:
            face = face_data.get("face")
            if face is None or face.size == 0:
                return {
                    "passed": False,
                    "reason": "No face data or empty face",
                    "metrics": {}
                }
            
            h, w = face.shape[:2]
            confidence = face_data.get("confidence", 0)
            blur_score = self._calculate_blur(face)
            
            metrics = {
                "face_size": (h, w),
                "confidence": confidence,
                "blur_score": blur_score,
                "min_size_required": min_size,
                "min_confidence_required": min_confidence,
                "blur_threshold": blur_threshold
            }
            
            # Detailed checks with logging
            checks = []
            
            # Size check
            if h < min_size or w < min_size:
                checks.append(f"Size too small: {h}x{w} < {min_size}")
            else:
                checks.append(f"Size OK: {h}x{w} >= {min_size}")
            
            # Confidence check
            if confidence < min_confidence:
                checks.append(f"Confidence too low: {confidence:.3f} < {min_confidence}")
            else:
                checks.append(f"Confidence OK: {confidence:.3f} >= {min_confidence}")
            
            # Blur check
            if blur_score < blur_threshold:
                checks.append(f"Too blurry: {blur_score:.2f} < {blur_threshold}")
            else:
                checks.append(f"Blur OK: {blur_score:.2f} >= {blur_threshold}")
            
            # Overall result
            passed = (h >= min_size and w >= min_size and 
                     confidence >= min_confidence and 
                     blur_score >= blur_threshold)
            
            logger.info(f"Quality check results: {'; '.join(checks)}")
            
            return {
                "passed": passed,
                "reason": "Quality check failed: " + "; ".join([c for c in checks if "too" in c.lower() or "low" in c.lower()]) if not passed else "Quality check passed",
                "metrics": metrics,
                "detailed_checks": checks
            }
                
        except Exception as e:
            logger.warning(f"Quality filtering failed: {e}")
            return {
                "passed": False,
                "reason": f"Quality check error: {str(e)}",
                "metrics": {}
            }
    
    def _sanitize_image(self, image_data: bytes, max_dim: int = 2048) -> Optional[np.ndarray]:
        try:
            max_size = getattr(settings, 'MAX_IMAGE_SIZE_BYTES', 20 * 1024 * 1024)
            if len(image_data) > max_size:
                raise ValueError(f"Image too large ({len(image_data)/1024/1024:.2f}MB > {max_size/1024/1024:.2f}MB)")
            
            with Image.open(BytesIO(image_data)) as img:
                logger.info(f"Original image: {img.size}, mode: {img.mode}")
                
                # Auto-orient based on EXIF
                img = ImageOps.exif_transpose(img)
                
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert("RGB")
                
                # Resize if too large
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to: {new_size}")
                
                # Convert to numpy array (BGR for OpenCV)
                bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                logger.info(f"Final BGR image shape: {bgr_img.shape}")
                return bgr_img
                
        except UnidentifiedImageError:
            raise ValueError("Corrupt or unsupported image format")
        except Exception as e:
            raise RuntimeError(f"Image processing failed: {e}")
        
    def _validate_user_id(self, user_id: str):
        """Prevent NoSQL injection through user IDs"""
        if not re.match(r"^[\w\-\.@]{1,64}$", user_id):
            logger.warning(f"Invalid user ID attempt: {user_id}")
            raise ValueError("User ID contains invalid characters")
    
    async def detect_and_save_user_profile(
        self, 
        user_id: str, 
        image_data: bytes, 
        original_filename: Optional[str] = None,
        update_existing: bool = False,
        # Quality parameters - made more lenient for testing
        min_face_size: int = 30,
        min_confidence: float = 0.4,
        blur_threshold: float = 50
    ) -> Dict[str, Any]:
        """
        Detect face in image and save/update user profile
        Returns detailed result with face detection status
        """
        try:
            # Validate user ID
            self._validate_user_id(user_id)
            logger.info(f"Processing face detection for user: {user_id}")

            # Ensure models are ready
            await self._ensure_models_initialized()

            # Process image
            bgr_img = self._sanitize_image(image_data)
            if bgr_img is None:
                return {
                    "has_face": False,
                    "single_face": False,
                    "reason": "Image processing failed"
                }

            # Detect faces (use detect_faces -> (faces, single))
            faces, single = yolo_detector.detect_faces(bgr_img)

            logger.info(f"Detected {len(faces)} faces for user {user_id}")

            # Validate count (exactly one)
            if len(faces) == 0:
                return {"has_face": False, "single_face": False, "reason": "No faces detected in image"}
            if not single:
                return {
                    "has_face": True,
                    "single_face": False,
                    "reason": f"Multiple faces detected ({len(faces)}). Please use image with single face."
                }

            face_data = faces[0]
            logger.info(f"Face data keys: {list(face_data.keys())}")

            # Enhanced quality filtering with detailed feedback
            quality_result = self._filter_quality(
                face_data, 
                min_size=min_face_size,
                min_confidence=min_confidence,
                blur_threshold=blur_threshold
            )
            
            logger.info(f"Quality check result: {quality_result}")
            
            if not quality_result["passed"]:
                return {
                    "has_face": True,
                    "single_face": False,
                    "reason": quality_result["reason"],
                    "quality_metrics": quality_result["metrics"],
                    "detailed_checks": quality_result.get("detailed_checks", [])
                }

            # Generate embedding
            face_img = face_data["face"]
            logger.info(f"Face image shape for embedding: {face_img.shape}")
            
            try:
                embedding = inference_pipeline(
                    image=face_img,
                    model_path=settings.EMBED_MODEL_PATH,
                    model_name="FACENET",
                    size=(160, 160),
                    use_augmentation=False
                )
                if embedding is None:
                    return {
                        "has_face": True,
                        "single_face": False,
                        "reason": "Failed to generate face embedding"
                    }
                embedding_list = [float(x) for x in embedding]
                logger.info(f"Generated embedding with {len(embedding_list)} dimensions")
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return {
                    "has_face": True,
                    "single_face": False,
                    "reason": "Face embedding generation failed"
                }

            # Quality metrics
            bbox = face_data.get("bbox", [])
            confidence = face_data.get("confidence", 0.0)
            quality_score = None
            if bbox and len(bbox) >= 4:
                area = bbox[2] * bbox[3]
                quality_score = confidence * area

            logger.info(f"Attempting to save profile: user_id={user_id}, embedding_size={len(embedding_list)}, confidence={confidence}")

            # Save to DB using corrected UserProfileDB
            try:
                success = await UserProfileDB.save_user_profile(
                    user_id=user_id,
                    embedding=embedding_list,
                    confidence=confidence,
                    quality_score=quality_score,
                    bbox=bbox,
                    image_data=None,  # Optional: store image data if needed
                    image_path=None   # Optional: store image path if needed
                )
                
                logger.info(f"Database save result: {success}")
                
                if not success:
                    logger.error(f"Failed to save profile for {user_id}")
                    return {
                        "has_face": True,
                        "single_face": False,
                        "reason": "Failed to save user profile to database",
                        "db_error": True
                    }

                # Optional verification (with better error handling)
                try:
                    saved_profile = await UserProfileDB.get_user_profile(user_id)
                    if saved_profile:
                        logger.info(f"Profile verification successful for {user_id}")
                    else:
                        logger.warning(f"Profile verification failed for {user_id}, but save operation reported success")
                except Exception as verify_error:
                    logger.warning(f"Profile verification error for {user_id}: {verify_error} (but save was successful)")

                logger.info(f"Successfully saved profile for user {user_id}")
                return {
                    "has_face": True,
                    "single_face": True,
                    "user_id": user_id,
                    "embedding_size": len(embedding_list),
                    "confidence": confidence,
                    "quality_score": quality_score,
                    "face_bbox": bbox,
                    "saved_to_db": True,
                    "quality_metrics": quality_result["metrics"]
                }

            except Exception as e:
                logger.error(f"Database save failed for user {user_id}: {e}", exc_info=True)
                return {
                    "has_face": True, 
                    "single_face": False, 
                    "reason": f"Database save failed: {str(e)}",
                    "db_error": str(e)
                }

        except Exception as e:
            logger.error(f"Face detection failed for user {user_id}: {e}", exc_info=True)
            return {
                "has_face": False, 
                "single_face": False, 
                "reason": f"Processing error: {str(e)}"
            }

    async def identify_user_in_cluster(
        self,
        bucket_name: str,
        sub_bucket: str,
        cluster_id: str,
        user_embedding: List[float],
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Check if a user's face appears in a specific cluster using cosine similarity
        """
        try:
            # Get all faces in the cluster
            cluster_faces = await FaceClusteringDB.get_cluster_faces(
                bucket_name, sub_bucket, cluster_id
            )
            
            if not cluster_faces:
                return {
                    "is_match": False,
                    "reason": "Cluster not found or empty",
                    "total_cluster_images": 0
                }
            
            # Convert user embedding to numpy array
            user_emb = np.array(user_embedding, dtype=np.float32)
            user_norm = np.linalg.norm(user_emb)
            if user_norm > 0:
                user_emb = user_emb / user_norm  # Normalize
            else:
                return {
                    "is_match": False,
                    "reason": "Invalid user embedding",
                    "total_cluster_images": len(cluster_faces)
                }
            
            best_similarity = 0.0
            matching_images = []
            
            # Compare with each face in cluster
            for face_doc in cluster_faces:
                try:
                    if not face_doc.embedding:
                        continue
                    
                    # Convert cluster face embedding to numpy array
                    cluster_emb = np.array(face_doc.embedding, dtype=np.float32)
                    cluster_norm = np.linalg.norm(cluster_emb)
                    if cluster_norm > 0:
                        cluster_emb = cluster_emb / cluster_norm  # Normalize
                    else:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(user_emb, cluster_emb)
                    
                    if similarity > similarity_threshold:
                        matching_images.append({
                            "image_path": face_doc.image_path,
                            "similarity": float(similarity),
                            "face_id": face_doc.face_id,
                            "confidence": face_doc.confidence,
                            "quality_score": face_doc.quality_score
                        })
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                
                except Exception as e:
                    logger.warning(f"Error comparing with face {face_doc.face_id}: {e}")
                    continue
            
            # Sort matches by similarity (highest first)
            matching_images.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "is_match": len(matching_images) > 0,
                "best_similarity": float(best_similarity),
                "matching_images": matching_images,
                "total_cluster_images": len(cluster_faces),
                "matches_count": len(matching_images),
                "cluster_id": cluster_id
            }
        
        except Exception as e:
            logger.error(f"Error identifying user in cluster: {e}")
            return {
                "is_match": False,
                "reason": f"Identification failed: {str(e)}",
                "total_cluster_images": 0
            }
    
    async def find_user_across_all_clusters(
        self,
        bucket_name: str,
        sub_bucket: str,
        user_id: str,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search for a user across all clusters in a bucket
        """
        try:
            # Get user profile
            user_profile = await UserProfileDB.get_user_profile(user_id)
            if not user_profile or not user_profile.embedding:
                return {
                    "user_found": False,
                    "reason": "User profile not found or no embedding"
                }
            
            # Get latest clustering result
            clustering_result = await FaceClusteringDB.get_latest_clustering_result(
                bucket_name, sub_bucket
            )
            
            if not clustering_result:
                return {
                    "user_found": False,
                    "reason": "No clustering results found for this bucket"
                }
            
            user_clusters = []
            
            # Check each cluster
            for cluster_info in clustering_result.clusters:
                cluster_result = await self.identify_user_in_cluster(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    cluster_id=cluster_info.cluster_id,
                    user_embedding=user_profile.embedding,
                    similarity_threshold=similarity_threshold
                )
                
                if cluster_result["is_match"]:
                    user_clusters.append({
                        "cluster_id": cluster_info.cluster_id,
                        "cluster_size": cluster_info.size,
                        "best_similarity": cluster_result["best_similarity"],
                        "matches_count": cluster_result["matches_count"],
                        "matching_images": cluster_result["matching_images"]
                    })
            
            # Sort clusters by best similarity
            user_clusters.sort(key=lambda x: x["best_similarity"], reverse=True)
            
            return {
                "user_found": len(user_clusters) > 0,
                "user_id": user_id,
                "bucket_path": f"{bucket_name}/{sub_bucket}",
                "total_clusters_searched": len(clustering_result.clusters),
                "matching_clusters": user_clusters,
                "total_matches": sum(c["matches_count"] for c in user_clusters)
            }
        
        except Exception as e:
            logger.error(f"Error finding user across clusters: {e}")
            return {
                "user_found": False,
                "reason": f"Search failed: {str(e)}"
            }
    
    @asynccontextmanager
    async def _connection_manager(self):
        """Context manager for handling connections"""
        try:
            yield
        finally:
            # Clean up any connections
            try:
                import gc
                gc.collect()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

# Create singleton instance
face_detection_service = FaceDetectionService()