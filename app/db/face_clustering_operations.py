#db/face_clustering_operations.py
from beanie import Document, init_beanie 
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from pymongo import IndexModel
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from app.services.qdrant_service import qdrant_service
from app.models.face_clustering_models import FaceEmbeddingBase as FaceEmbedding, ClusterInfo, ClusteringResult
from app.core.config import settings
from app.models.user_profile import UserProfile
logger = logging.getLogger(__name__)
# Global database connection
_database_instance = None
# Cache for dynamic models
_model_cache = {}
def _coerce_timestamp_in_raw(raw: dict) -> dict:
    """Ensure raw['timestamp'] is a datetime (fixes legacy 'timestamp' string)."""
    ts = raw.get("timestamp")
    if not isinstance(ts, datetime):
        # If it's a string like "timestamp" or missing, force a sane datetime
        raw["timestamp"] = datetime.utcnow()
    return raw

async def get_clustering_by_id(clustering_id: str):
    """
    Load a clustering doc by _id from the base collection first.
    If not found, try dynamic collection fallback.
    """
    # 1) Try base collection (ClusteringResult.Settings.name == "clustering_results")
    try:
        oid = ObjectId(clustering_id)
    except Exception:
        raise ValueError(f"Invalid clustering_id: {clustering_id}")

    doc = await ClusteringResult.get(oid)
    if doc:
        return doc

    # 1b) Fallback via raw (handles malformed timestamp)
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    raw = await db[ClusteringResult.Settings.name].find_one({"_id": oid})
    if raw:
        raw = _coerce_timestamp_in_raw(raw)
        return ClusteringResult.model_validate(raw)

    # 2) Dynamic collection fallback (if your pipeline saved in per-bucket coll)
    # We don't know the bucket/sub_bucket here, so we scan a handful of dynamic collections
    # If you keep many, you can skip this block to avoid scanning.
    for name in await db.list_collection_names():
        if not name.startswith("clustering_results_"):
            continue
        raw = await db[name].find_one({"_id": oid})
        if raw:
            raw = _coerce_timestamp_in_raw(raw)
            try:
                # Validate against the base model; it shares the same schema
                return ClusteringResult.model_validate(raw)
            except Exception:
                # If dynamic schema diverges, just return raw or raise
                return ClusteringResult.model_validate(raw)

    return None

async def get_latest_clustering_for_bucket(bucket_name: str, sub_bucket: str):
    """
    Return the latest clustering for f"{bucket}/{sub_bucket}".
    Prefer base collection; fall back to dynamic collection if needed.
    """
    bucket_path = f"{bucket_name}/{sub_bucket}"

    # 1) Base collection, proper datetime timestamps
    doc = await ClusteringResult.find(
        {"bucket_path": bucket_path, "timestamp": {"$type": 9}}
    ).sort([("timestamp", -1)]).limit(1).first_or_none()
    if doc:
        return doc

    # 1b) Base collection, raw fallback if timestamp is a bad string
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    raw = await db[ClusteringResult.Settings.name].find_one(
        {"bucket_path": bucket_path}, sort=[("_id", -1)]
    )
    if raw:
        raw = _coerce_timestamp_in_raw(raw)
        return ClusteringResult.model_validate(raw)

    # 2) Dynamic per-bucket collection fallback
    dyn_name = f"clustering_results_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
    if dyn_name in await db.list_collection_names():
        raw = await db[dyn_name].find_one({}, sort=[("_id", -1)])
        if raw:
            raw = _coerce_timestamp_in_raw(raw)
            return ClusteringResult.model_validate(raw)

    return None
async def get_database():
    """Get or create database instance"""
    global _database_instance
    if _database_instance is None:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        _database_instance = client[settings.DATABASE_NAME]
        
        # Initialize base models with the database
        await init_beanie(
            database=_database_instance,
            document_models=[FaceEmbedding, ClusteringResult, UserProfile]
        )
    return _database_instance

async def initialize_dynamic_model(model_class):
    """Initialize a dynamic model by creating its collection if needed - FIXED"""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        collection_name = model_class.Settings.name

        # Create collection if it doesn't exist
        if collection_name not in await db.list_collection_names():
            await db.create_collection(collection_name)

        # Create indexes (handle proper IndexModel usage)
        indexes = getattr(model_class.Settings, "indexes", None)
        if indexes:
            collection = db[collection_name]
            to_create = []
            for idx in indexes:
                if isinstance(idx, IndexModel):
                    to_create.append(idx)
                elif isinstance(idx, (list, tuple)):
                    # allow [('field', 1)] style
                    to_create.append(IndexModel(idx))
                elif isinstance(idx, dict) and "keys" in idx:
                    # legacy dict format: {"keys": [("field", 1)], "unique": True, ...}
                    opts = {k: v for k, v in idx.items() if k != "keys"}
                    to_create.append(IndexModel(idx["keys"], **opts))
                else:
                    logger.warning(f"Skipping unexpected index spec: {idx!r}")

            if to_create:
                await collection.create_indexes(to_create)  # <-- the correct API for IndexModel

        return True
    except Exception as e:
        logger.error(f"Dynamic model initialization error: {e}")
        return False
async def cleanup_problem_collections():
    """Clean up collections that have the '_id' problem"""
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        collections = await db.list_collection_names()
        
        for collection_name in collections:
            if "face_embeddings" in collection_name:
                # Check if collection has documents with _id: "_id"
                problem_count = await db[collection_name].count_documents({"_id": "_id"})
                
                if problem_count > 0:
                    logger.warning(f"Dropping problem collection: {collection_name}")
                    await db[collection_name].drop()
        
        logger.info("Problem collection cleanup completed")
        return True
    except Exception as e:
        logger.error(f"Error during problem collection cleanup: {e}")
        return False
def get_face_embedding_model(bucket_name: str, sub_bucket: str):
    """Create dynamic FaceEmbedding model for specific bucket - FIXED"""
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
    """Create a dynamic ClusteringResult model for specific bucket - FIXED"""
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

class DatabaseSynchronizer:
    """Utility to keep Qdrant and MongoDB in sync"""
    
    @staticmethod
    async def verify_consistency(bucket_name: str, sub_bucket: str) -> Dict[str, Any]:
        """Verify that both databases have the same data"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            # Get all MongoDB face IDs
            mongo_face_ids: Set[str] = set()
            async for face in FaceEmbeddingModel.find_all():
                mongo_face_ids.add(face.face_id)
            
            # Get all Qdrant IDs
            qdrant_ids = set()
            try:
                points = qdrant_service.get_all_face_ids()
                qdrant_ids = set(points)
            except Exception as e:
                logger.error(f"Error getting Qdrant IDs: {e}")
            
            # Find discrepancies
            missing_in_qdrant = mongo_face_ids - qdrant_ids
            missing_in_mongo = qdrant_ids - mongo_face_ids
            
            return {
                "mongo_count": len(mongo_face_ids),
                "qdrant_count": len(qdrant_ids),
                "missing_in_qdrant": list(missing_in_qdrant),
                "missing_in_mongo": list(missing_in_mongo),
                "consistent": not (missing_in_qdrant or missing_in_mongo)
            }
            
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def repair_inconsistencies(bucket_name: str, sub_bucket: str) -> Dict[str, Any]:
        """Attempt to repair inconsistencies between databases"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            report = await DatabaseSynchronizer.verify_consistency(bucket_name, sub_bucket)
            
            if report.get("consistent", False):
                return {"status": "already_consistent", "report": report}
                
            # Repair faces missing in Qdrant
            if report["missing_in_qdrant"]:
                missing_faces = await FaceEmbeddingModel.find(
                    {"face_id": {"$in": list(report["missing_in_qdrant"])}}
                ).to_list()
                
                for face in missing_faces:
                    await face.delete()
                    logger.warning(f"Deleted MongoDB document for face_id {face.face_id} - missing in Qdrant")
                    
            # Repair faces missing in MongoDB
            if report["missing_in_mongo"]:
                qdrant_faces = qdrant_service.get_embeddings_by_face_ids(
                    list(report["missing_in_mongo"])
                )
                
                for face_data in qdrant_faces:
                    payload = face_data["payload"]
                    
                    # Create MongoDB entry
                    clustering_id = payload.get("clustering_id")
                    if clustering_id and ObjectId.is_valid(clustering_id):
                        clustering_id = ObjectId(clustering_id)
                    
                    face_doc = FaceEmbeddingModel(
                        face_id=payload.get("face_id", str(uuid.uuid4())),
                        image_path=payload.get("image_path", ""),
                        bbox=eval(payload["bbox"]) if payload.get("bbox") else None,
                        confidence=float(payload["confidence"]) if payload.get("confidence") else None,
                        quality_score=float(payload["quality_score"]) if payload.get("quality_score") else None,
                        clustering_id=clustering_id,
                        cluster_id=payload.get("cluster_id", "unassigned")
                    )
                    
                    await face_doc.insert()
                    logger.info(f"Recovered MongoDB entry for face_id {face_doc.face_id}")
            
            return {
                "status": "repair_attempted",
                "original_report": report,
                "new_status": await DatabaseSynchronizer.verify_consistency(bucket_name, sub_bucket)
            }
            
        except Exception as e:
            logger.error(f"Repair failed: {e}")
            return {"status": "failed", "error": str(e)}

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
        """Save face metadata to MongoDB (without embedding) - FIXED"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            if not await initialize_dynamic_model(FaceEmbeddingModel):
                raise Exception("Failed to initialize dynamic model")
            
            # Create document with explicit ObjectId
            face_doc = FaceEmbeddingModel(
                id=ObjectId(),
                face_id=str(uuid.uuid4()),
                image_path=image_path,
                bbox=bbox,
                confidence=confidence,
                quality_score=quality_score,
                clustering_id=clustering_id,
                embedding=[]  # Empty embedding for metadata-only storage
            )
                
            # Use insert() method
            await face_doc.insert()
            
            logger.info(f"Successfully saved face metadata for {image_path} with face_id {face_doc.face_id}")
            return face_doc.face_id
            
        except Exception as e:
            logger.error(f"Error saving face metadata for {image_path}: {str(e)}", exc_info=True)
            return None
    async def save_clustering_result_as_metadata(
        bucket_name: str,
        sub_bucket: str,
        clusters: List[List[str]],
        noise: List[str],
        face_id_mapping: Dict[str, str],
        processing_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Save clustering result as metadata only.
        - Ensures top-level `timestamp` is a datetime (not the literal string "timestamp")
        - Ensures `processing_stats["timestamp"]` is a proper datetime (or ISO string)
        - Calculates total_images correctly from all cluster image_paths + noise
        """
        try:
            # normalize stats & timestamps
            processing_stats = processing_stats or {}
            ts = processing_stats.get("timestamp")
            if ts is None:
                processing_stats["timestamp"] = datetime.utcnow()
            else:
                # tolerate strings; coerce to datetime where possible
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
                cluster_infos.append(
                    ClusterInfo(
                        cluster_id=cluster_id,
                        image_paths=image_paths,
                        face_ids=face_ids,
                        size=len(image_paths),
                    )
                )

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

            # write document (top-level timestamp must be datetime)
            doc = ClusteringResultModel(
                id=ObjectId(),
                bucket_path=f"{bucket_name}/{sub_bucket}",
                clusters=cluster_infos,
                noise=noise,
                noise_face_ids=noise_face_ids,
                total_images=total_images,
                total_faces=total_faces,
                num_clusters=num_clusters,
                processing_stats=processing_stats,
                timestamp=datetime.utcnow(),
            )
            await doc.insert()
            logger.info("Saved clustering metadata: %s", str(doc.id))
            return doc.id
        except Exception as e:
            logger.error("Error saving clustering metadata: %s", e, exc_info=True)
            return None
    @staticmethod
    async def cleanup_problem_documents(bucket_name: str, sub_bucket: str):
        """Clean up documents with invalid _id values"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            from app.core.config import settings
            
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
        """
        Complete pipeline with separated storage: MongoDB (metadata) + Qdrant (vectors).
        Ensures timestamps are always valid datetimes and never the literal "timestamp".
        """
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
        """Update face metadata with clustering_id reference - FIXED"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            updated_count = 0
            for face_id in face_ids:
                try:
                    face_doc = await FaceEmbeddingModel.find_one({"face_id": face_id})
                    if face_doc:
                        face_doc.clustering_id = clustering_id
                        await face_doc.save()
                        updated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to update face_id {face_id}: {e}")
                    continue
                    
            logger.info(f"Updated {updated_count}/{len(face_ids)} face metadata entries with clustering_id {clustering_id}")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating face metadata with clustering_id: {e}")
            return False

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
        """Save embeddings to Qdrant with proper clustering metadata - FIXED"""
        try:
            logger.info(f"[QDRANT] Starting to save embeddings for clustering {clustering_id}")
            
            # Prepare data for batch insertion
            embeddings = []
            face_ids = []
            image_paths = []
            cluster_ids = []
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
                    face_ids.append(face_id)
                    image_paths.append(f"{bucket_name}/{image_path}")
                    cluster_ids.append(cluster_id)
                    
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
                face_ids.append(face_id)
                image_paths.append(f"{bucket_name}/{image_path}")
                cluster_ids.append("noise")
                
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
    async def update_face_embeddings_with_cluster_assignments(
        bucket_name: str,
        sub_bucket: str,
        cluster_assignments: Dict[str, str],  # face_id -> cluster_id
        clustering_id: ObjectId
    ) -> bool:
        """Update face embeddings with cluster assignments and clustering_id"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            updated_count = 0
            for face_id, cluster_id in cluster_assignments.items():
                # Update the document
                result = await FaceEmbeddingModel.find_one({"face_id": face_id})
                if result:
                    result.cluster_id = cluster_id
                    result.clustering_id = clustering_id
                    await result.save()
                    updated_count += 1
                    
            logger.info(f"Updated {updated_count} face embeddings with cluster assignments")
            return updated_count > 0
            
        except Exception as e:
            logger.error(f"Error updating face embeddings with cluster assignments: {e}")
            return False
    @staticmethod
    async def update_face_embeddings_with_clustering_id(
        bucket_name: str,
        sub_bucket: str,
        face_ids: List[str],
        clustering_id: ObjectId
    ) -> bool:
        """Update face embeddings with clustering_id reference"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            # Update all face embeddings with the clustering_id
            result = await FaceEmbeddingModel.find({"face_id": {"$in": face_ids}}).update_many(
                {"$set": {"clustering_id": clustering_id}}
            )
            
            # Also update Qdrant with clustering_id
            await qdrant_service.update_clustering_ids(face_ids, str(clustering_id))
            
            logger.info(f"Updated {result.modified_count} face embeddings with clustering_id {clustering_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating face embeddings with clustering_id: {e}")
            return False

    @staticmethod
    async def update_cluster_assignments_with_qdrant(
        bucket_name: str,
        sub_bucket: str,
        cluster_assignments: Dict[str, str],
        clustering_id: ObjectId
    ) -> bool:
        """Update cluster assignments for faces in both MongoDB and Qdrant"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            
            # Get all faces that need updating
            faces_to_update = []
            for image_path, cluster_id in cluster_assignments.items():
                face = await FaceEmbeddingModel.find_one({"image_path": image_path})
                if face:
                    faces_to_update.append({
                        "face_id": face.face_id,
                        "cluster_id": cluster_id,
                        "image_path": image_path
                    })
            
            # Update MongoDB
            for face_update in faces_to_update:
                await FaceEmbeddingModel.find_one({"face_id": face_update["face_id"]}).update(
                    {"$set": {"cluster_id": face_update["cluster_id"]}}
                )
            
            # Update Qdrant
            face_ids = [f["face_id"] for f in faces_to_update]
            cluster_ids = [f["cluster_id"] for f in faces_to_update]
            
            success = await qdrant_service.update_cluster_assignments(
                face_ids, cluster_ids, str(clustering_id)
            )
            
            if success:
                logger.info(f"Updated {len(cluster_assignments)} cluster assignments in both databases")
                return True
            else:
                logger.error("Failed to update Qdrant cluster assignments")
                return False
                
        except Exception as e:
            logger.error(f"Error updating cluster assignments: {e}")
            return False

    @staticmethod
    async def save_complete_clustering_pipeline(
        bucket_name: str,
        sub_bucket: str,
        face_data: List[Dict[str, Any]],
        clusters: List[List[str]],
        noise: List[str],
        processing_stats: Optional[Dict] = None
    ) -> Optional[ObjectId]:
        try:
            # Create face_id mapping first
            face_id_mapping = {}
            
            # Save all face embeddings without clustering_id first
            for face_info in face_data:
                face_id = await FaceClusteringDB.save_face_embedding(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    image_path=face_info["image_path"],
                    embedding=face_info["embedding"],
                    bbox=face_info.get("bbox"),
                    confidence=face_info.get("confidence"),
                    quality_score=face_info.get("quality_score"),
                    clustering_id=None
                )
                if face_id:
                    face_id_mapping[face_info["image_path"]] = face_id
            
            # Save clustering result
            clustering_result_id = await FaceClusteringDB.save_clustering_result(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                clusters=clusters,
                noise=noise,
                face_id_mapping=face_id_mapping,
                processing_stats=processing_stats
            )
            
            if not clustering_result_id:
                raise Exception("Failed to save clustering result")
            
            # Update all faces with clustering_id
            await FaceClusteringDB.update_face_embeddings_with_clustering_id(
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                face_ids=list(face_id_mapping.values()),
                clustering_id=clustering_result_id
            )
            
            # Create cluster assignments
            cluster_assignments = {}
            for i, cluster_images in enumerate(clusters):
                cluster_id = f"cluster_{i}"
                for img_path in cluster_images:
                    if img_path in face_id_mapping:
                        cluster_assignments[face_id_mapping[img_path]] = cluster_id
            
            for img_path in noise:
                if img_path in face_id_mapping:
                    cluster_assignments[face_id_mapping[img_path]] = "noise"
            
            # Update cluster assignments
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            for face_id, cluster_id in cluster_assignments.items():
                # Update MongoDB
                await FaceEmbeddingModel.find_one({"face_id": face_id}).update(
                    {"$set": {"cluster_id": cluster_id}}
                )
                
                # Update Qdrant
                qdrant_service.update_cluster_assignments(
                    face_ids=[face_id],
                    cluster_ids=[cluster_id],
                    clustering_id=str(clustering_result_id)
                )
            
            return clustering_result_id
        except Exception as e:
            logger.error(f"Error in clustering pipeline: {e}")
            return None

    @staticmethod
    async def verify_cluster_data(bucket_name: str, sub_bucket: str, cluster_id: str):
        """Verify that cluster data exists in both collections"""
        try:
            ClusteringResultModel = get_clustering_result_model(bucket_name, sub_bucket)
            result = await ClusteringResultModel.find_one(
                {"bucket_path": f"{bucket_name}/{sub_bucket}"},
                sort=[("timestamp", -1)]
            )
            
            if not result:
                return {"status": "error", "message": "No clustering results found"}
            
            target_cluster = None
            for cluster in result.clusters:
                if cluster.cluster_id == cluster_id:
                    target_cluster = cluster
                    break
                    
            if not target_cluster:
                return {"status": "error", "message": f"Cluster {cluster_id} not found in results"}
            
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            missing_faces = []
            
            for face_id in target_cluster.face_ids:
                face = await FaceEmbeddingModel.find_one({"face_id": face_id})
                if not face:
                    missing_faces.append(face_id)
            
            return {
                "status": "success",
                "cluster_exists": True,
                "clustering_id": str(result.id),
                "total_faces": len(target_cluster.face_ids),
                "missing_faces": missing_faces,
                "missing_count": len(missing_faces)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @staticmethod
    async def get_latest_clustering_result(bucket_name: str, sub_bucket: str):
        """
        Get the most recent clustering result for a bucket/sub_bucket.
        - First, only consider docs whose `timestamp` is a MongoDB date (type 9)
        - If none found, fallback to raw Motor, coerce bad `timestamp` and parse through the model
        """
        try:
            path = f"{bucket_name}/{sub_bucket}"
            ClusteringResultModel = get_clustering_result_model(bucket_name, sub_bucket)

            # 1) Prefer documents where timestamp is a real BSON date
            doc = await ClusteringResultModel.find(
                {"bucket_path": path, "timestamp": {"$type": 9}}
            ).sort([("timestamp", -1)]).limit(1).first_or_none()
            if doc:
                return doc

            # 2) Fallback: raw driver sort by _id if some legacy docs have bad strings
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.DATABASE_NAME]
            coll = db[ClusteringResultModel.Settings.name]
            raw = await coll.find_one({"bucket_path": path}, sort=[("_id", -1)])
            if not raw:
                return None

            # coerce timestamp to a valid datetime if it's wrong
            ts = raw.get("timestamp")
            if not isinstance(ts, datetime):
                raw["timestamp"] = datetime.utcnow()

            # run through Beanie model validation to keep shape consistent
            doc = ClusteringResultModel.model_validate(raw)
            return doc
        except Exception as e:
            logger.error("Error getting clustering result: %s", e, exc_info=True)
            return None
    
    @staticmethod
    async def get_cluster_faces(bucket_name: str, sub_bucket: str, cluster_id: str):
        """Get all faces in a specific cluster with only necessary fields"""
        try:
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            faces = []
            async for face_doc in FaceEmbeddingModel.find(
                {"cluster_id": cluster_id},
                projection={
                    "face_id": 1,
                    "image_path": 1,
                    "bbox": 1,
                    "confidence": 1,
                    "quality_score": 1,
                    "clustering_id": 1
                }
            ):
                faces.append(face_doc)
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
            
            face_result = await FaceEmbeddingModel.delete_all()
            cluster_result = await ClusteringResultModel.delete_all()
            
            # Also delete from Qdrant
            qdrant_service.delete_by_bucket_path(f"{bucket_name}/{sub_bucket}")
            
            return {
                "deleted_face_embeddings": face_result.deleted_count if face_result else 0,
                "deleted_clustering_results": cluster_result.deleted_count if cluster_result else 0
            }
        except Exception as e:
            logger.error(f"Error deleting bucket data: {e}")
            return {"deleted_face_embeddings": 0, "deleted_clustering_results": 0}


class ConnectionManager:
    """Manages database and external service connections"""
    
    @staticmethod
    async def close_connections():
        """Close any open connections"""
        try:
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

class DatabaseCleanup:
    """Utility for cleaning up conflicting indexes and documents"""
    
    @staticmethod
    async def drop_conflicting_indexes():
        """Drop conflicting indexes if they exist"""
        try:
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.DATABASE_NAME]
            
            collections = await db.list_collection_names()
            
            for collection_name in collections:
                if "clustering_results" in collection_name:
                    collection = db[collection_name]
                    
                    indexes = await collection.list_indexes().to_list(length=None)
                    
                    for index in indexes:
                        if index.get("name") == "clustering_id_1":
                            await collection.drop_index("clustering_id_1")
                            logger.info(f"Dropped conflicting index 'clustering_id_1' from {collection_name}")
                            
            logger.info("Database cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    @staticmethod
    async def clean_duplicate_documents(bucket_name: str, sub_bucket: str):
        """Clean up any duplicate documents that might exist"""
        try:
            client = AsyncIOMotorClient(settings.MONGODB_URL)
            db = client[settings.DATABASE_NAME]
            
            collection_name = f"clustering_results_{bucket_name}_{sub_bucket}".replace("/", "_").replace("-", "_").lower()
            collection = db[collection_name]
            
            await collection.delete_many({
                "_id": {"$in": ["_id", "id", "cluster", "clustering"]}
            })
            
            count = await collection.count_documents({})
            if count < 100:
                await collection.drop()
                logger.info(f"Dropped entire collection {collection_name}")
            else:
                logger.info(f"Cleaned problematic documents from {collection_name}")
                
        except Exception as e:
            logger.error(f"Error cleaning duplicate documents: {e}")
            try:
                await collection.drop()
                logger.info(f"Force-dropped collection {collection_name} due to cleanup error")
            except Exception as drop_error:
                logger.error(f"Failed to drop collection: {drop_error}")


    
async def init_face_clustering_db():
    """Initialize the face clustering database collections - FIXED"""
    try:
        # Initialize database connection
        await get_database()
        logger.info("Face clustering database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing face clustering database: {e}")
        raise e