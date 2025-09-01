#db/person_clustering_operations.py
from beanie import Document, init_beanie 
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Set, Tuple
from pymongo import IndexModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.services.qdrant_service import qdrant_service
from app.models.face_clustering_models import FaceEmbeddingBase as FaceEmbedding, PersonClusterInfo, PersonBasedClusteringResult
from app.core.config import settings
from app.models.user_profile import UserProfile
import numpy as np
from collections import defaultdict
from app.models.face_clustering_models import FaceEmbeddingBase as FaceEmbedding, ClusterInfo, ClusteringResult

logger = logging.getLogger(__name__)

# Global database connection
_database_instance = None
# Cache for dynamic models
_model_cache = {}

def _convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for MongoDB/Pydantic compatibility
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    else:
        return obj

def _coerce_timestamp_in_raw(raw: dict) -> dict:
    """Ensure raw['timestamp'] is a datetime (fixes legacy 'timestamp' string)."""
    ts = raw.get("timestamp")
    if not isinstance(ts, datetime):
        raw["timestamp"] = datetime.utcnow()
    return raw

def get_person_face_embedding_model(bucket_name: str, sub_bucket: str):
    """Create dynamic FaceEmbedding model for person-based clustering"""
    collection_name = f"person_faces_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
    
    cache_key = f"person_face_embedding_{collection_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    class DynamicPersonFaceEmbedding(FaceEmbedding):
        class Settings:
            name = collection_name
            indexes = [
                IndexModel([("image_path", 1)], name=f"{collection_name}_image_path_idx"),
                IndexModel([("person_id", 1)], name=f"{collection_name}_person_id_idx"),
                IndexModel([("timestamp", -1)], name=f"{collection_name}_timestamp_idx"),
                IndexModel([("face_id", 1)], name=f"{collection_name}_face_id_idx", unique=True),
                IndexModel([("clustering_id", 1)], name=f"{collection_name}_clustering_id_idx"),
                IndexModel([("is_owner_face", 1)], name=f"{collection_name}_is_owner_idx"),
                IndexModel([("person_id", 1), ("is_owner_face", -1)], name=f"{collection_name}_person_owner_idx")
            ]
    
    _model_cache[cache_key] = DynamicPersonFaceEmbedding
    return DynamicPersonFaceEmbedding

def get_person_clustering_result_model(bucket_name: str, sub_bucket: str):
    """Create dynamic PersonBasedClusteringResult model"""
    collection_name = f"person_clustering_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
    
    cache_key = f"person_clustering_result_{collection_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    class DynamicPersonClusteringResult(PersonBasedClusteringResult):
        class Settings:
            name = collection_name
            indexes = [
                IndexModel([("bucket_path", 1), ("timestamp", -1)], name=f"{collection_name}_bucket_timestamp_idx"),
                IndexModel([("total_persons", -1)], name=f"{collection_name}_persons_count_idx")
            ]
    
    _model_cache[cache_key] = DynamicPersonClusteringResult
    return DynamicPersonClusteringResult

async def initialize_dynamic_model(model_class):
    """Initialize a dynamic model by creating its collection if needed and setting up direct collection access"""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        collection_name = model_class.Settings.name

        # Create collection if it doesn't exist
        existing_collections = await db.list_collection_names()
        if collection_name not in existing_collections:
            await db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")

        # Set up direct collection access for the model
        collection = db[collection_name]
        
        # Store collection reference directly on model class
        model_class._motor_collection = collection
        model_class._collection = collection
        
        # Create indexes manually
        indexes = getattr(model_class.Settings, "indexes", None)
        if indexes:
            to_create = []
            
            # Get existing indexes to avoid duplicates
            try:
                existing_indexes = await collection.list_indexes().to_list(length=None)
                existing_index_names = {idx.get('name', '') for idx in existing_indexes}
            except Exception:
                existing_index_names = set()
            
            for idx in indexes:
                try:
                    if isinstance(idx, IndexModel):
                        if idx.document.get('name', '') not in existing_index_names:
                            to_create.append(idx)
                    elif isinstance(idx, (list, tuple)):
                        index_model = IndexModel(idx)
                        if index_model.document.get('name', '') not in existing_index_names:
                            to_create.append(index_model)
                    elif isinstance(idx, dict) and "keys" in idx:
                        opts = {k: v for k, v in idx.items() if k != "keys"}
                        index_model = IndexModel(idx["keys"], **opts)
                        if index_model.document.get('name', '') not in existing_index_names:
                            to_create.append(index_model)
                except Exception as idx_error:
                    logger.warning(f"Index preparation error: {idx_error}")
                    continue

            if to_create:
                try:
                    await collection.create_indexes(to_create)
                    logger.info(f"Created {len(to_create)} indexes for {collection_name}")
                except Exception as idx_error:
                    logger.warning(f"Index creation warning for {collection_name}: {idx_error}")

        # Verify collection access
        try:
            await collection.find_one({})
            logger.info(f"Successfully initialized dynamic model: {collection_name}")
            return True
        except Exception as verify_error:
            logger.error(f"Model verification failed for {collection_name}: {verify_error}")
            return False
            
    except Exception as e:
        logger.error(f"Dynamic model initialization error: {e}")
        return False

class PersonBasedClusteringDB:
    """Database operations for person-based face clustering"""
    
    @staticmethod
    async def save_person_face_metadata(
        bucket_name: str, 
        sub_bucket: str,
        image_path: str,
        face_id: str,
        person_id: Optional[str] = None,
        is_owner_face: bool = False,
        bbox: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        quality_score: Optional[float] = None,
        cluster_confidence: Optional[float] = None,
        clustering_id: Optional[ObjectId] = None
    ) -> Optional[str]:
        """Save face metadata with person assignment using direct MongoDB operations"""
        try:
            PersonFaceModel = get_person_face_embedding_model(bucket_name, sub_bucket)
            
            # Initialize the model first
            ok = await initialize_dynamic_model(PersonFaceModel)
            if not ok:
                raise RuntimeError(f"Failed to initialize PersonFaceModel for {bucket_name}/{sub_bucket}")
            
            # Create document data directly
            doc_data = {
                "_id": ObjectId(),
                "face_id": face_id,
                "image_path": image_path,
                "person_id": person_id,
                "is_owner_face": is_owner_face,
                "bbox": bbox,
                "confidence": confidence,
                "quality_score": quality_score,
                "cluster_confidence": cluster_confidence,
                "clustering_id": clustering_id,
                "embedding": [],  # Empty, stored in Qdrant
                "timestamp": datetime.utcnow()
            }
            
            # Insert directly using motor collection
            await PersonFaceModel._motor_collection.insert_one(doc_data)
            logger.info(f"Saved person face metadata: {face_id}, person: {person_id}, owner: {is_owner_face}")
            return face_id
            
        except Exception as e:
            logger.error(f"Error saving person face metadata: {str(e)}", exc_info=True)
            return None

    @staticmethod
    async def save_person_clustering_result(
        bucket_name: str,
        sub_bucket: str,
        person_clusters: List[PersonClusterInfo],
        unassigned_faces: List[str],
        unassigned_face_ids: List[str],
        total_images: int,
        total_faces: int,
        processing_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[ObjectId]:
        """Save person-based clustering results using direct MongoDB operations"""
        try:
            # Calculate overlap statistics
            image_appearances = defaultdict(int)
            for cluster in person_clusters:
                for img_path in cluster.image_paths:
                    image_appearances[img_path] += 1
            
            # Convert numpy types to native Python types
            overlap_stats = {
                "single_person_images": sum(1 for count in image_appearances.values() if count == 1),
                "multi_person_images": sum(1 for count in image_appearances.values() if count > 1),
                "max_persons_per_image": max(image_appearances.values()) if image_appearances else 0,
                "avg_persons_per_image": float(np.mean(list(image_appearances.values()))) if image_appearances else 0.0
            }
            
            # Normalize processing stats
            processing_stats = processing_stats or {}
            ts = processing_stats.get("timestamp")
            if ts is None:
                processing_stats["timestamp"] = datetime.utcnow()
            elif isinstance(ts, str):
                try:
                    processing_stats["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    processing_stats["timestamp"] = datetime.utcnow()

            PersonClusteringModel = get_person_clustering_result_model(bucket_name, sub_bucket)
            
            # Initialize the dynamic model
            ok = await initialize_dynamic_model(PersonClusteringModel)
            if not ok:
                raise RuntimeError(f"Failed to initialize collection for {bucket_name}/{sub_bucket}")
            
            # Convert PersonClusterInfo objects to dict format
            person_clusters_dict = []
            for cluster in person_clusters:
                cluster_dict = {
                    "person_id": cluster.person_id,
                    "owner_face_id": cluster.owner_face_id,
                    "owner_embedding": cluster.owner_embedding,
                    "image_paths": cluster.image_paths,
                    "face_ids": cluster.face_ids,
                    "confidence_scores": [float(x) for x in cluster.confidence_scores] if cluster.confidence_scores else [],
                    "quality_scores": [float(x) for x in cluster.quality_scores] if cluster.quality_scores else [],
                    "size": cluster.size,
                    "avg_confidence": float(cluster.avg_confidence),
                    "best_quality_score": float(cluster.best_quality_score)
                }
                person_clusters_dict.append(cluster_dict)

            # Create document data directly
            doc_id = ObjectId()
            doc_data = {
                "_id": doc_id,
                "bucket_path": f"{bucket_name}/{sub_bucket}",
                "person_clusters": person_clusters_dict,
                "unassigned_faces": unassigned_faces,
                "unassigned_face_ids": unassigned_face_ids,
                "total_images": total_images,
                "total_faces": total_faces,
                "total_persons": len(person_clusters),
                "image_overlap_stats": overlap_stats,
                "processing_stats": _convert_numpy_types(processing_stats),
                "timestamp": datetime.utcnow(),
            }

            # Insert directly using motor collection
            await PersonClusteringModel._motor_collection.insert_one(doc_data)
            logger.info("Saved person clustering result: %s", str(doc_id))
            return doc_id

        except Exception as e:
            logger.error("Error saving person clustering result: %s", e, exc_info=True)
            return None

    @staticmethod
    async def save_complete_person_clustering_pipeline(
        bucket_name: str,
        sub_bucket: str,
        person_clusters: Dict[str, Dict],  # person_id -> cluster data
        unassigned_faces: List[str],
        face_id_mapping: Dict[str, str],  # image_path -> face_id
        face_metadata: Dict[str, Dict],   # face_id -> metadata
        embeddings_data: Dict[str, List[float]],  # face_id -> embedding
        processing_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[ObjectId]:
        """Complete pipeline for person-based clustering with proper face_id mapping"""
        try:
            logger.info("Starting person-based clustering pipeline for %s/%s", bucket_name, sub_bucket)

            # Create reverse mapping for lookups: face_id -> image_path
            face_id_to_image_path = {face_id: image_path for image_path, face_id in face_id_mapping.items()}

            # 1) Save face metadata for all faces
            all_face_ids = set()
            for person_id, cluster_data in person_clusters.items():
                for face_id in cluster_data["face_ids"]:
                    all_face_ids.add(face_id)
                    
                    # Check if this is the owner face
                    is_owner = face_id == cluster_data["owner_face_id"]
                    
                    # Get image path using reverse lookup
                    image_path = face_id_to_image_path.get(face_id)
                    
                    if not image_path:
                        logger.warning(f"No image path found for face_id: {face_id}")
                        continue
                    
                    metadata = face_metadata.get(face_id, {})
                    
                    saved_id = await PersonBasedClusteringDB.save_person_face_metadata(
                        bucket_name=bucket_name,
                        sub_bucket=sub_bucket,
                        image_path=image_path,
                        face_id=face_id,
                        person_id=person_id,
                        is_owner_face=is_owner,
                        bbox=metadata.get("bbox"),
                        confidence=metadata.get("confidence"),
                        quality_score=metadata.get("quality_score"),
                        cluster_confidence=metadata.get("cluster_confidence", metadata.get("confidence")),
                        clustering_id=None,  # Will be set after clustering result is saved
                    )
                    
                    if not saved_id:
                        logger.warning(f"Failed to save metadata for face_id: {face_id}")

            # Add unassigned faces
            for image_path in unassigned_faces:
                face_id = face_id_mapping.get(image_path)
                if face_id:
                    all_face_ids.add(face_id)
                    metadata = face_metadata.get(face_id, {})
                    
                    saved_id = await PersonBasedClusteringDB.save_person_face_metadata(
                        bucket_name=bucket_name,
                        sub_bucket=sub_bucket,
                        image_path=image_path,
                        face_id=face_id,
                        person_id=None,  # Unassigned
                        is_owner_face=False,
                        bbox=metadata.get("bbox"),
                        confidence=metadata.get("confidence"),
                        quality_score=metadata.get("quality_score"),
                        clustering_id=None,
                    )
                    
                    if not saved_id:
                        logger.warning(f"Failed to save unassigned face metadata for: {image_path}")

            # 2) Prepare PersonClusterInfo objects with numpy type conversion
            person_cluster_infos = []
            for person_id, cluster_data in person_clusters.items():
                # Convert numpy types to native Python types
                confidence_scores = cluster_data.get("confidence_scores", [])
                quality_scores = cluster_data.get("quality_scores", [])
                
                # Ensure all values are native Python types
                avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
                best_quality = float(max(quality_scores)) if quality_scores else 0.0
                
                cluster_info = PersonClusterInfo(
                    person_id=person_id,
                    owner_face_id=cluster_data["owner_face_id"],
                    owner_embedding=cluster_data.get("owner_embedding", []),
                    image_paths=cluster_data["image_paths"],
                    face_ids=cluster_data["face_ids"],
                    confidence_scores=[float(x) for x in confidence_scores] if confidence_scores else [],
                    quality_scores=[float(x) for x in quality_scores] if quality_scores else [],
                    size=len(cluster_data["image_paths"]),
                    avg_confidence=avg_confidence,
                    best_quality_score=best_quality
                )
                person_cluster_infos.append(cluster_info)

            # 3) Calculate totals
            total_images = len(set(face_id_mapping.keys()))  # Unique images
            total_faces = len(all_face_ids)
            unassigned_face_ids = [face_id_mapping.get(path) for path in unassigned_faces if face_id_mapping.get(path)]

            # 4) Save clustering result
            clustering_result_id = await PersonBasedClusteringDB.save_person_clustering_result(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                person_clusters=person_cluster_infos,
                unassigned_faces=unassigned_faces,
                unassigned_face_ids=unassigned_face_ids,
                total_images=total_images,
                total_faces=total_faces,
                processing_stats=processing_stats,
            )

            if not clustering_result_id:
                raise RuntimeError("Failed to save person clustering result")

            # 5) Update face metadata with clustering_id
            await PersonBasedClusteringDB.update_faces_with_clustering_id(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                face_ids=list(all_face_ids),
                clustering_id=clustering_result_id,
            )

            # 6) Save embeddings to Qdrant with person assignments
            qdrant_success = await PersonBasedClusteringDB.save_person_embeddings_to_qdrant(
                clustering_id=str(clustering_result_id),
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                person_clusters=person_clusters,
                unassigned_faces=unassigned_faces,
                face_id_mapping=face_id_mapping,
                face_metadata=face_metadata,
                embeddings_data=embeddings_data,
            )

            if not qdrant_success:
                logger.warning("Qdrant storage failed, but MongoDB metadata saved")

            logger.info("Person clustering pipeline completed: %s", str(clustering_result_id))
            return clustering_result_id

        except Exception as e:
            logger.error("Error in person clustering pipeline: %s", e, exc_info=True)
            return None

    @staticmethod
    async def save_person_embeddings_to_qdrant(
        clustering_id: str,
        bucket_name: str,
        sub_bucket: str,
        person_clusters: Dict[str, Dict],
        unassigned_faces: List[str],
        face_id_mapping: Dict[str, str],
        face_metadata: Dict[str, Dict],
        embeddings_data: Dict[str, List[float]],
    ) -> bool:
        """Save person-based embeddings to Qdrant with enhanced metadata"""
        try:
            logger.info(f"[QDRANT] Saving person-based embeddings for clustering {clustering_id}")
            
            # Create reverse mapping for lookups
            face_id_to_image_path = {face_id: image_path for image_path, face_id in face_id_mapping.items()}
            
            embeddings = []
            payloads = []
            
            # Process person clusters
            for person_id, cluster_data in person_clusters.items():
                for i, face_id in enumerate(cluster_data["face_ids"]):
                    if face_id not in embeddings_data:
                        logger.warning(f"No embedding found for face_id: {face_id}")
                        continue
                    
                    # Find image path for this face
                    image_path = face_id_to_image_path.get(face_id)
                    
                    if not image_path:
                        logger.warning(f"No image path found for face_id: {face_id}")
                        continue
                    
                    embedding = embeddings_data[face_id]
                    metadata = face_metadata.get(face_id, {})
                    is_owner = face_id == cluster_data["owner_face_id"]
                    
                    embeddings.append(embedding)
                    
                    payload = {
                        "face_id": face_id,
                        "image_path": f"{bucket_name}/{image_path}",
                        "person_id": person_id,
                        "is_owner_face": str(is_owner),
                        "clustering_id": clustering_id,
                        "bucket_name": bucket_name,
                        "sub_bucket": sub_bucket,
                        "bbox": str(metadata.get("bbox", [])),
                        "confidence": str(metadata.get("confidence", 0.0)),
                        "quality_score": str(metadata.get("quality_score", 0.0)),
                        "cluster_confidence": str(metadata.get("cluster_confidence", metadata.get("confidence", 0.0)))
                    }
                    payloads.append(payload)
            
            # Process unassigned faces
            for image_path in unassigned_faces:
                face_id = face_id_mapping.get(image_path)
                if not face_id or face_id not in embeddings_data:
                    continue
                
                embedding = embeddings_data[face_id]
                metadata = face_metadata.get(face_id, {})
                
                embeddings.append(embedding)
                
                payload = {
                    "face_id": face_id,
                    "image_path": f"{bucket_name}/{image_path}",
                    "person_id": "unassigned",
                    "is_owner_face": "false",
                    "clustering_id": clustering_id,
                    "bucket_name": bucket_name,
                    "sub_bucket": sub_bucket,
                    "bbox": str(metadata.get("bbox", [])),
                    "confidence": str(metadata.get("confidence", 0.0)),
                    "quality_score": str(metadata.get("quality_score", 0.0)),
                    "cluster_confidence": "0.0"
                }
                payloads.append(payload)
            
            # Batch insert to Qdrant
            if embeddings:
                success = qdrant_service.add_face_embeddings_with_payloads(
                    embeddings=embeddings,
                    payloads=payloads
                )
                
                if success:
                    logger.info(f"[QDRANT] Successfully saved {len(embeddings)} person-based embeddings")
                    return True
                else:
                    logger.error(f"[QDRANT] Failed to save embeddings")
                    return False
            else:
                logger.warning(f"[QDRANT] No valid embeddings to save")
                return False
                
        except Exception as e:
            logger.error(f"[QDRANT] Error saving person embeddings: {e}")
            return False

    @staticmethod
    async def update_faces_with_clustering_id(
        bucket_name: str,
        sub_bucket: str,
        face_ids: List[str],
        clustering_id: ObjectId
    ) -> bool:
        """Update face metadata with clustering_id reference using direct MongoDB operations"""
        try:
            PersonFaceModel = get_person_face_embedding_model(bucket_name, sub_bucket)
            
            # Use direct MongoDB update operation
            result = await PersonFaceModel._motor_collection.update_many(
                {"face_id": {"$in": face_ids}},
                {"$set": {"clustering_id": clustering_id}}
            )
            
            updated_count = result.modified_count
            logger.info(f"Updated {updated_count}/{len(face_ids)} face metadata with clustering_id {clustering_id}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating faces with clustering_id: {e}")
            return False

    @staticmethod
    async def get_latest_person_clustering_result(bucket_name: str, sub_bucket: str):
        """Get the most recent person-based clustering result"""
        try:
            path = f"{bucket_name}/{sub_bucket}"
            PersonClusteringModel = get_person_clustering_result_model(bucket_name, sub_bucket)
            await initialize_dynamic_model(PersonClusteringModel)

            # Use direct MongoDB query
            raw = await PersonClusteringModel._motor_collection.find_one(
                {"bucket_path": path},
                sort=[("timestamp", -1)]
            )
            
            if not raw:
                return None

            # Coerce timestamp if needed
            ts = raw.get("timestamp")
            if not isinstance(ts, datetime):
                raw["timestamp"] = datetime.utcnow()

            # Create model instance manually
            doc = PersonBasedClusteringResult.model_validate(raw)
            return doc
            
        except Exception as e:
            logger.error("Error getting person clustering result: %s", e, exc_info=True)
            return None

    @staticmethod
    async def get_person_cluster_details(bucket_name: str, sub_bucket: str, person_id: str):
        """Get detailed information about a specific person's cluster"""
        try:
            PersonFaceModel = get_person_face_embedding_model(bucket_name, sub_bucket)
            await initialize_dynamic_model(PersonFaceModel)
            
            # Use direct MongoDB query
            person_faces = await PersonFaceModel._motor_collection.find(
                {"person_id": person_id},
                {
                    "face_id": 1,
                    "image_path": 1,
                    "bbox": 1,
                    "confidence": 1,
                    "quality_score": 1,
                    "is_owner_face": 1,
                    "cluster_confidence": 1,
                    "timestamp": 1
                }
            ).to_list(length=None)
            
            if not person_faces:
                return None
            
            # Get owner face
            owner_face = next((f for f in person_faces if f.get("is_owner_face")), person_faces[0])
            
            return {
                "person_id": person_id,
                "total_appearances": len(person_faces),
                "owner_face": {
                    "face_id": owner_face["face_id"],
                    "image_path": owner_face["image_path"],
                    "confidence": owner_face.get("confidence"),
                    "quality_score": owner_face.get("quality_score")
                },
                "all_appearances": [
                    {
                        "face_id": face["face_id"],
                        "image_path": face["image_path"],
                        "confidence": face.get("confidence"),
                        "quality_score": face.get("quality_score"),
                        "is_owner": face.get("is_owner_face", False),
                        "cluster_confidence": face.get("cluster_confidence"),
                        "bbox": face.get("bbox")
                    } for face in person_faces
                ],
                "image_paths": [face["image_path"] for face in person_faces],
                "avg_confidence": float(np.mean([face.get("confidence", 0) for face in person_faces])),
                "best_quality": float(max([face.get("quality_score", 0) for face in person_faces]))
            }
            
        except Exception as e:
            logger.error(f"Error getting person cluster details: {e}")
            return None

    @staticmethod
    async def get_clustering_statistics(bucket_name: str, sub_bucket: str):
        """Get comprehensive statistics about person-based clustering"""
        try:
            clustering_result = await PersonBasedClusteringDB.get_latest_person_clustering_result(bucket_name, sub_bucket)
            if not clustering_result:
                return None

            PersonFaceModel = get_person_face_embedding_model(bucket_name, sub_bucket)
            
            # Get face counts per person
            person_stats = []
            for cluster in clustering_result.person_clusters:
                person_stats.append({
                    "person_id": cluster["person_id"] if isinstance(cluster, dict) else cluster.person_id,
                    "total_appearances": cluster["size"] if isinstance(cluster, dict) else cluster.size,
                    "avg_confidence": float(cluster["avg_confidence"]) if isinstance(cluster, dict) else float(cluster.avg_confidence),
                    "best_quality": float(cluster["best_quality_score"]) if isinstance(cluster, dict) else float(cluster.best_quality_score),
                    "owner_face_id": cluster["owner_face_id"] if isinstance(cluster, dict) else cluster.owner_face_id
                })
            
            # Sort by appearances (most appearances first)
            person_stats.sort(key=lambda x: x["total_appearances"], reverse=True)
            
            # Convert all potentially problematic types
            stats = {
                "clustering_id": str(clustering_result.id),
                "bucket_path": clustering_result.bucket_path,
                "timestamp": clustering_result.timestamp,
                "total_persons": int(clustering_result.total_persons),
                "total_images": int(clustering_result.total_images),
                "total_faces": int(clustering_result.total_faces),
                "unassigned_faces": len(clustering_result.unassigned_faces),
                "image_overlap_stats": _convert_numpy_types(clustering_result.image_overlap_stats),
                "person_statistics": person_stats[:10],  # Top 10 persons
                "processing_stats": _convert_numpy_types(clustering_result.processing_stats)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting clustering statistics: {e}")
            return None

# Keep database initialization functions compatible
async def get_database():
    """Get or create database instance"""
    global _database_instance
    if _database_instance is None:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        _database_instance = client[settings.DATABASE_NAME]
        
        # Initialize base models with the database
        await init_beanie(
            database=_database_instance,
            document_models=[FaceEmbedding, PersonBasedClusteringResult, UserProfile]
        )
    return _database_instance

async def init_person_clustering_db():
    """Initialize the person-based clustering database collections"""
    try:
        await get_database()
        logger.info("Person clustering database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing person clustering database: {e}")
        raise e
    
def get_face_embedding_model(bucket_name: str, sub_bucket: str):
    """Create dynamic FaceEmbedding model for specific bucket"""
    collection_name = f"face_embeddings_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
    
    cache_key = f"face_embedding_{collection_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    class DynamicFaceEmbedding(FaceEmbedding):
        class Settings:
            name = collection_name
            indexes = [
                IndexModel([("image_path", 1)], name=f"{collection_name}_image_path_idx"),
                IndexModel([("cluster_id", 1)], name=f"{collection_name}_cluster_id_idx"),
                IndexModel([("timestamp", -1)], name=f"{collection_name}_timestamp_idx"),
                IndexModel([("face_id", 1)], name=f"{collection_name}_face_id_idx", unique=True),
                IndexModel([("clustering_id", 1)], name=f"{collection_name}_clustering_id_idx")
            ]
    
    _model_cache[cache_key] = DynamicFaceEmbedding
    return DynamicFaceEmbedding

def get_clustering_result_model(bucket_name: str, sub_bucket: str):
    """Create a dynamic ClusteringResult model for specific bucket"""
    collection_name = f"clustering_results_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
    
    cache_key = f"clustering_result_{collection_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    class DynamicClusteringResult(ClusteringResult):
        class Settings:
            name = collection_name
            indexes = [
                IndexModel([("bucket_path", 1), ("timestamp", -1)], name=f"{collection_name}_bucket_timestamp_idx"),
            ]
    
    _model_cache[cache_key] = DynamicClusteringResult
    return DynamicClusteringResult

class FaceClusteringDB:
    """Utility class for managing face clustering database operations"""
    
    @staticmethod
    async def save_face_metadata(
        bucket_name: str, 
        sub_bucket: str,
        image_path: str,
        bbox: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        quality_score: Optional[float] = None,
        clustering_id: Optional[ObjectId] = None
    ) -> Optional[str]:
        """Save face metadata to MongoDB (without embedding)"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            if not await initialize_dynamic_model(FaceEmbeddingModel):
                raise Exception("Failed to initialize dynamic model")
            
            face_id = str(uuid.uuid4())
            
            # Create document data directly
            doc_data = {
                "_id": ObjectId(),
                "face_id": face_id,
                "image_path": image_path,
                "bbox": bbox,
                "confidence": confidence,
                "quality_score": quality_score,
                "clustering_id": clustering_id,
                "embedding": [],  # Empty embedding for metadata-only storage
                "timestamp": datetime.utcnow()
            }
                
            # Insert directly using motor collection
            await FaceEmbeddingModel._motor_collection.insert_one(doc_data)
            
            logger.info(f"Successfully saved face metadata for {image_path} with face_id {face_id}")
            return face_id
            
        except Exception as e:
            logger.error(f"Error saving face metadata for {image_path}: {str(e)}", exc_info=True)
            return None

    @staticmethod
    async def save_clustering_result_as_metadata(
        bucket_name: str,
        sub_bucket: str,
        clusters: List[List[str]],
        noise: List[str],
        face_id_mapping: Dict[str, str],
        processing_stats: Optional[Dict[str, Any]] = None,
    ):
        """Save clustering result as metadata only"""
        try:
            # normalize stats & timestamps
            processing_stats = processing_stats or {}
            ts = processing_stats.get("timestamp")
            if ts is None:
                processing_stats["timestamp"] = datetime.utcnow()
            else:
                if isinstance(ts, str):
                    try:
                        processing_stats["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        processing_stats["timestamp"] = datetime.utcnow()

            # build ClusterInfo objects
            cluster_infos = []
            for i, image_paths in enumerate(clusters):
                cluster_id = f"cluster_{i}"
                face_ids = []
                for p in image_paths:
                    fid = face_id_mapping.get(p)
                    if fid:
                        face_ids.append(fid)
                cluster_infos.append({
                    "cluster_id": cluster_id,
                    "image_paths": image_paths,
                    "face_ids": face_ids,
                    "size": len(image_paths),
                })

            # noise face_ids
            noise_face_ids = []
            for p in noise:
                fid = face_id_mapping.get(p)
                if fid:
                    noise_face_ids.append(fid)

            # compute totals
            total_images = sum(len(c) for c in clusters) + len(noise)
            total_faces = len(face_id_mapping)
            num_clusters = len(clusters)

            ClusteringResultModel = get_clustering_result_model(bucket_name, sub_bucket)

            # ensure collection exists & indexed
            ok = await initialize_dynamic_model(ClusteringResultModel)
            if not ok:
                raise RuntimeError("Failed to initialize clustering result model")

            # Create document data directly
            doc_id = ObjectId()
            doc_data = {
                "_id": doc_id,
                "bucket_path": f"{bucket_name}/{sub_bucket}",
                "clusters": cluster_infos,
                "noise": noise,
                "noise_face_ids": noise_face_ids,
                "total_images": total_images,
                "total_faces": total_faces,
                "num_clusters": num_clusters,
                "processing_stats": _convert_numpy_types(processing_stats),
                "timestamp": datetime.utcnow(),
            }
            
            # Insert directly using motor collection
            await ClusteringResultModel._motor_collection.insert_one(doc_data)
            logger.info("Saved clustering metadata: %s", str(doc_id))
            return doc_id
        except Exception as e:
            logger.error("Error saving clustering metadata: %s", e, exc_info=True)
            return None

    @staticmethod
    async def cleanup_problem_documents(bucket_name: str, sub_bucket: str):
        """Clean up documents with invalid _id values"""
        try:
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.DATABASE_NAME]
            
            collection_name = f"face_embeddings_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
            
            # Delete documents with string _id values
            await db[collection_name].delete_many({
                "_id": {"$type": "string"}
            })
            
            # Delete documents with specific problematic _id values
            await db[collection_name].delete_many({
                "_id": {"$in": ["_id", "id"]}
            })
            
            logger.info(f"Cleaned problem documents from {collection_name}")
        except Exception as e:
            logger.error(f"Error cleaning problem documents: {e}")

    @staticmethod
    async def save_complete_clustering_pipeline_fixed(
        bucket_name: str,
        sub_bucket: str,
        embeddings_data: Dict[str, List[float]],  # face_id -> embedding
        face_metadata: Dict[str, Dict],           # face_id -> metadata
        clusters: List[List[str]],
        noise: List[str],
        face_id_mapping: Dict[str, str],          # image_path -> face_id
        processing_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[ObjectId]:
        """Complete pipeline with separated storage: MongoDB (metadata) + Qdrant (vectors)"""
        try:
            logger.info("Starting complete clustering pipeline for %s/%s", bucket_name, sub_bucket)

            # 1) save face metadata docs
            saved_face_ids, failed = [], 0
            for image_path, face_id in face_id_mapping.items():
                md = face_metadata.get(face_id, {})
                saved_id = await FaceClusteringDB.save_face_metadata(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    image_path=image_path,
                    bbox=md.get("bbox"),
                    confidence=md.get("confidence"),
                    quality_score=md.get("quality_score"),
                    clustering_id=None,
                )
                if saved_id:
                    saved_face_ids.append(saved_id)
                else:
                    failed += 1
                    logger.warning("Failed to save metadata for %s", image_path)

            if not saved_face_ids:
                raise RuntimeError("No face metadata could be saved")

            # 2) normalize processing_stats timestamp
            processing_stats = processing_stats or {}
            ts = processing_stats.get("timestamp")
            if ts is None:
                processing_stats["timestamp"] = datetime.utcnow()
            else:
                if isinstance(ts, str):
                    try:
                        processing_stats["timestamp"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        processing_stats["timestamp"] = datetime.utcnow()

            # 3) save clustering result doc
            clustering_result_id = await FaceClusteringDB.save_clustering_result_as_metadata(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                clusters=clusters,
                noise=noise,
                face_id_mapping=face_id_mapping,
                processing_stats=processing_stats,
            )
            if not clustering_result_id:
                raise RuntimeError("Failed to save clustering metadata")

            # 4) backfill clustering_id in face metadata
            ok = await FaceClusteringDB.update_face_metadata_with_clustering_id(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                face_ids=list(face_id_mapping.values()),
                clustering_id=clustering_result_id,
            )
            if not ok:
                logger.warning("Some face metadata did not get clustering_id")

            # 5) push vectors to Qdrant
            q_ok = await FaceClusteringDB.save_embeddings_to_qdrant_with_metadata(
                clustering_id=str(clustering_result_id),
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                embeddings_data=embeddings_data,
                face_metadata=face_metadata,
                clusters=clusters,
                noise=noise,
                face_id_mapping=face_id_mapping,
            )
            if not q_ok:
                logger.warning("Vectors not saved to Qdrant, but Mongo metadata is saved")

            logger.info("Complete pipeline saved with ID: %s", str(clustering_result_id))
            return clustering_result_id

        except Exception as e:
            logger.error("Error in complete clustering pipeline: %s", e, exc_info=True)
            return None
        
    @staticmethod
    async def update_face_metadata_with_clustering_id(
        bucket_name: str,
        sub_bucket: str,
        face_ids: List[str],
        clustering_id: ObjectId
    ) -> bool:
        """Update face metadata with clustering_id reference"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            # Use direct MongoDB update operation
            result = await FaceEmbeddingModel._motor_collection.update_many(
                {"face_id": {"$in": face_ids}},
                {"$set": {"clustering_id": clustering_id}}
            )
            
            updated_count = result.modified_count
            logger.info(f"Updated {updated_count}/{len(face_ids)} face metadata entries with clustering_id {clustering_id}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating face metadata with clustering_id: {e}")
            return False

    @staticmethod
    async def save_embeddings_to_qdrant_with_metadata(
        clustering_id: str,
        bucket_name: str,
        sub_bucket: str,
        embeddings_data: Dict[str, List[float]],
        face_metadata: Dict[str, Dict],
        clusters: List[List[str]],
        noise: List[str],
        face_id_mapping: Dict[str, str]
    ) -> bool:
        """Save embeddings to Qdrant with proper clustering metadata"""
        try:
            logger.info(f"[QDRANT] Starting to save embeddings for clustering {clustering_id}")
            
            # Prepare data for batch insertion
            embeddings = []
            payloads = []
            
            # Process clusters
            for cluster_idx, cluster_images in enumerate(clusters):
                cluster_id = f"cluster_{cluster_idx}"
                
                for image_path in cluster_images:
                    face_id = face_id_mapping.get(image_path)
                    if not face_id or face_id not in embeddings_data:
                        logger.warning(f"No face_id or embedding found for image_path: {image_path}")
                        continue
                    
                    embedding = embeddings_data[face_id]
                    metadata = face_metadata.get(face_id, {})
                    
                    embeddings.append(embedding)
                    
                    # Create payload with all metadata
                    payload = {
                        "face_id": face_id,
                        "image_path": f"{bucket_name}/{image_path}",
                        "cluster_id": cluster_id,
                        "clustering_id": clustering_id,
                        "bucket_name": bucket_name,
                        "sub_bucket": sub_bucket,
                        "bbox": str(metadata.get("bbox", [])),
                        "confidence": str(metadata.get("confidence", 0.0)),
                        "quality_score": str(metadata.get("quality_score", 0.0))
                    }
                    payloads.append(payload)
            
            # Process noise
            for image_path in noise:
                face_id = face_id_mapping.get(image_path)
                if not face_id or face_id not in embeddings_data:
                    logger.warning(f"No face_id or embedding found for noise image_path: {image_path}")
                    continue
                
                embedding = embeddings_data[face_id]
                metadata = face_metadata.get(face_id, {})
                
                embeddings.append(embedding)
                
                payload = {
                    "face_id": face_id,
                    "image_path": f"{bucket_name}/{image_path}",
                    "cluster_id": "noise",
                    "clustering_id": clustering_id,
                    "bucket_name": bucket_name,
                    "sub_bucket": sub_bucket,
                    "bbox": str(metadata.get("bbox", [])),
                    "confidence": str(metadata.get("confidence", 0.0)),
                    "quality_score": str(metadata.get("quality_score", 0.0))
                }
                payloads.append(payload)
            
            # Batch insert into Qdrant
            if embeddings:
                logger.info(f"[QDRANT] Attempting to save {len(embeddings)} embeddings")
                success = qdrant_service.add_face_embeddings_with_payloads(
                    embeddings=embeddings,
                    payloads=payloads
                )
                
                if success:
                    logger.info(f"[QDRANT] Successfully saved {len(embeddings)} embeddings")
                    return True
                else:
                    logger.error(f"[QDRANT] Failed to save embeddings")
                    return False
            else:
                logger.warning(f"[QDRANT] No valid embeddings to save")
                return False
                
        except Exception as e:
            logger.error(f"[QDRANT] Error saving embeddings with metadata: {e}")
            return False

    @staticmethod
    async def get_latest_clustering_result(bucket_name: str, sub_bucket: str):
        """Get the most recent clustering result for a bucket/sub_bucket"""
        try:
            path = f"{bucket_name}/{sub_bucket}"
            ClusteringResultModel = get_clustering_result_model(bucket_name, sub_bucket)
            await initialize_dynamic_model(ClusteringResultModel)

            # Use direct MongoDB query
            raw = await ClusteringResultModel._motor_collection.find_one(
                {"bucket_path": path},
                sort=[("timestamp", -1)]
            )
            
            if not raw:
                return None

            # coerce timestamp to a valid datetime if it's wrong
            ts = raw.get("timestamp")
            if not isinstance(ts, datetime):
                raw["timestamp"] = datetime.utcnow()

            # run through model validation to keep shape consistent
            doc = ClusteringResult.model_validate(raw)
            return doc
        except Exception as e:
            logger.error("Error getting clustering result: %s", e, exc_info=True)
            return None
    
    @staticmethod
    async def get_cluster_faces(bucket_name: str, sub_bucket: str, cluster_id: str):
        """Get all faces in a specific cluster with only necessary fields"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            await initialize_dynamic_model(FaceEmbeddingModel)
            
            faces = await FaceEmbeddingModel._motor_collection.find(
                {"cluster_id": cluster_id},
                {
                    "face_id": 1,
                    "image_path": 1,
                    "bbox": 1,
                    "confidence": 1,
                    "quality_score": 1,
                    "clustering_id": 1
                }
            ).to_list(length=None)
            
            return faces
        except Exception as e:
            logger.error(f"Error getting cluster faces: {e}")
            return []
    
    @staticmethod
    async def delete_bucket_data(bucket_name: str, sub_bucket: str):
        """Delete all data for a specific bucket"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            ClusteringResultModel = get_clustering_result_model(bucket_name, sub_bucket)
            
            await initialize_dynamic_model(FaceEmbeddingModel)
            await initialize_dynamic_model(ClusteringResultModel)
            
            # Delete using direct MongoDB operations
            face_result = await FaceEmbeddingModel._motor_collection.delete_many({})
            cluster_result = await ClusteringResultModel._motor_collection.delete_many({})
            
            # Also delete from Qdrant
            qdrant_service.delete_by_bucket_path(f"{bucket_name}/{sub_bucket}")
            
            return {
                "deleted_face_embeddings": face_result.deleted_count if face_result else 0,
                "deleted_clustering_results": cluster_result.deleted_count if cluster_result else 0
            }
        except Exception as e:
            logger.error(f"Error deleting bucket data: {e}")
            return {"deleted_face_embeddings": 0, "deleted_clustering_results": 0}

async def init_face_clustering_db():
    """Initialize the face clustering database collections"""
    try:
        await get_database()
        logger.info("Face clustering database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing face clustering database: {e}")
        raise e