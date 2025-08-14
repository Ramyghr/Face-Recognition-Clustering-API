# app/routers/bucket_clustering.py
from fastapi import APIRouter, HTTPException, Query
from app.services.S3_functions import storage_service
#from app.services.optimized_clustering.optimized_clustering_with_db import optimized_pipeline_with_db
from app.services.optimized_clustering.optimized_clustering_with_db import optimized_pipeline_with_db
from app.core.config import settings
from app.models.face_clustering_models import ClusterResultResponse
from app.models.face_clustering import FaceClusteringDB
from app.services import yolo_detector, ai
from app.services.qdrant_service import qdrant_service  # Changed to Qdrant
from botocore.exceptions import ClientError
from typing import List, Optional
import logging
import datetime
import traceback
import asyncio
import time
from app.schemas.clustering import FileItem
from datetime import datetime
from beanie import Document
from bson import ObjectId
from app.db.face_clustering_operations import init_face_clustering_db, get_face_embedding_model


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Optimized Bucket Clustering with Databases"])


# app/routers/bucket_clustering.py
@router.post("/cluster-bucket-with-db/{bucket_name}/{sub_bucket}", response_model=ClusterResultResponse)
async def cluster_bucket_with_database_fixed(
    bucket_name: str, 
    sub_bucket: str,
    max_images: Optional[int] = Query(None),
    batch_size: int = Query(30),
    skip_quality_filter: bool = Query(False),
    max_concurrent: int = Query(12)
):
    """Clusters faces in a given S3 bucket path with FIXED database storage - metadata in MongoDB, embeddings in Qdrant."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Initialize database with fixed initialization
    #await init_face_clustering_db()

    try:
        # Validate and adjust parameters
        batch_size = min(batch_size, 50)
        max_concurrent = min(max_concurrent, 20)

        prefix = f"{bucket_name}/{sub_bucket}/"
        logger.info(f"Starting FIXED clustering for path: {prefix}")

        # List and filter image keys
        try:
            raw_keys = await storage_service.list_avatar_keys(prefix=prefix)
            image_keys = [
                key for key in raw_keys
                if any(key.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"])
                and not key.endswith('/')
                and key != prefix
            ]
            
            if not image_keys:
                raise HTTPException(404, f"No valid images found in {prefix}")

            if max_images and max_images > 0:
                image_keys = image_keys[:max_images]

        except asyncio.TimeoutError:
            raise HTTPException(408, "Timeout while listing bucket")
        except Exception as e:
            logger.error(f"Error listing bucket contents: {e}")
            raise HTTPException(500, "Failed to list bucket contents")
 
        # Prepare file list
        files = [FileItem(fileKey=key, fileName=key.split("/")[-1]) for key in image_keys]
        original_count = len(image_keys)

        # Initialize models
        try:
            # Load models sequentially to avoid memory issues
            yolo_detector.load_yolo_from_hf()
            await ai.init_face_model(settings.EMBED_MODEL_PATH)
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise HTTPException(500, f"Model initialization failed: {str(e)}")

        # Process pipeline with FIXED DB integration
        try:
            result = await optimized_pipeline_with_db.process_optimized_pipeline_with_db_fixed(
                files=files,
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                batch_size=batch_size,
                skip_quality_filter=skip_quality_filter,
                max_concurrent=max_concurrent
            )
            
            # Ensure we have a valid result
            if not result or "clusters" not in result:
                raise HTTPException(500, "FIXED clustering returned invalid result")
            logger.info("FIXED clustering completed, verifying database consistency...")

        except asyncio.TimeoutError:
            raise HTTPException(408, "FIXED clustering timed out")
        except Exception as e:
            logger.error(f"FIXED clustering pipeline error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(500, f"FIXED clustering failed: {str(e)}")

        # Prepare statistics
        total_time = time.time() - start_time
        stats = {
            "total_processing_time": round(total_time, 2),
            "original_file_count": original_count,
            "processed_file_count": len(image_keys),
            "batch_size_used": batch_size,
            "max_concurrent_used": max_concurrent,
            "avg_processing_time_per_image": round(total_time / len(image_keys), 3) if image_keys else 0,
            "files_per_second": round(len(image_keys) / total_time, 2) if total_time > 0 else 0,
            "database_integrated": True,
            "qdrant_integrated": True,
            "storage_architecture": "metadata_in_mongodb_embeddings_in_qdrant",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update result stats
        result.setdefault("stats", {}).update(stats)

        # Verify storage with improved methods
        try:
            # Verify MongoDB metadata was created
            FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
            face_count = await FaceEmbeddingModel.count()
            logger.info(f"Created {face_count} face metadata entries in MongoDB")
            
            # Verify Qdrant embeddings
            clustering_id = result["stats"].get("clustering_id")
            if clustering_id:
                qdrant_stats = qdrant_service.get_clustering_stats(clustering_id)
                qdrant_count = qdrant_stats.get("total_embeddings", 0)
                logger.info(f"Qdrant contains {qdrant_count} face embeddings for clustering {clustering_id}")
                
                # Add verification stats
                result["stats"]["mongodb_metadata_count"] = face_count
                result["stats"]["qdrant_embeddings_count"] = qdrant_count
                result["stats"]["storage_consistency"] = face_count == qdrant_count
            
        except Exception as verify_error:
            logger.warning(f"Storage verification warning: {verify_error}")
            result["stats"]["storage_verification"] = "failed"
        
        return ClusterResultResponse(
            bucket=f"{bucket_name}/{sub_bucket}",
            clusters=result["clusters"],
            noise=result["noise"],
            timestamp=datetime.utcnow(),
            stats=result.get("stats", {})
        )

    except HTTPException as http_ex:
        logger.error(f"HTTP Exception during FIXED clustering: {str(http_ex.detail)}")
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error during FIXED clustering: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Unexpected FIXED clustering error: {str(e)}")

# Add this new endpoint for verifying the fixed storage architecture
@router.get("/verify-fixed-storage/{bucket_name}/{sub_bucket}")
async def verify_fixed_storage_architecture(bucket_name: str, sub_bucket: str):
    """Verify that the fixed storage architecture is working properly"""
    try:
        # Get latest clustering result
        clustering_result = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        if not clustering_result:
            raise HTTPException(404, "No clustering results found")
        
        clustering_id = str(clustering_result.id)
        
        # Verify MongoDB metadata
        FaceEmbeddingModel = get_face_embedding_model(bucket_name, sub_bucket)
        mongodb_faces = []
        async for face in FaceEmbeddingModel.find({"clustering_id": ObjectId(clustering_id)}):
            mongodb_faces.append({
                "face_id": face.face_id,
                "image_path": face.image_path,
                "has_embedding": len(face.embedding) > 0,  # Should be False
                "cluster_id": face.cluster_id
            })
        
        # Verify Qdrant embeddings
        qdrant_embeddings = qdrant_service.get_embeddings_by_clustering_id(clustering_id)
        qdrant_stats = qdrant_service.get_clustering_stats(clustering_id)
        
        # Cross-reference face IDs
        mongodb_face_ids = {face["face_id"] for face in mongodb_faces}
        qdrant_face_ids = {emb["payload"]["face_id"] for emb in qdrant_embeddings}
        
        consistency_check = {
            "matching_face_ids": len(mongodb_face_ids & qdrant_face_ids),
            "mongodb_only": len(mongodb_face_ids - qdrant_face_ids),
            "qdrant_only": len(qdrant_face_ids - mongodb_face_ids),
            "total_consistency": mongodb_face_ids == qdrant_face_ids
        }
        
        return {
            "clustering_id": clustering_id,
            "bucket_path": f"{bucket_name}/{sub_bucket}",
            "storage_architecture": "FIXED: metadata in MongoDB, embeddings in Qdrant",
            "mongodb_stats": {
                "total_metadata_entries": len(mongodb_faces),
                "faces_with_embeddings": sum(1 for f in mongodb_faces if f["has_embedding"]),
                "faces_without_embeddings": sum(1 for f in mongodb_faces if not f["has_embedding"])
            },
            "qdrant_stats": qdrant_stats,
            "consistency_check": consistency_check,
            "architecture_working": (
                consistency_check["total_consistency"] and 
                all(not f["has_embedding"] for f in mongodb_faces) and
                len(qdrant_embeddings) > 0
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fixed storage verification failed: {e}")
        raise HTTPException(500, f"Verification failed: {str(e)}")
    
@router.delete("/cleanup-qdrant/{bucket_name}/{sub_bucket}")
async def cleanup_qdrant_data(bucket_name: str, sub_bucket: str):
    """Clean up Qdrant data for a specific bucket (for testing/maintenance)"""
    try:
        # Get latest clustering result
        clustering_result = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        if not clustering_result:
            raise HTTPException(404, "No clustering results found")
        
        clustering_id = str(clustering_result.id)
        
        # Delete Qdrant data for this clustering
        operation_id = qdrant_service.delete_clustering_data(clustering_id)
        
        return {
            "clustering_id": clustering_id,
            "bucket_path": f"{bucket_name}/{sub_bucket}",
            "operation_id": operation_id,
            "cleanup_success": operation_id is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Qdrant cleanup failed: {e}")
        raise HTTPException(500, f"Cleanup failed: {str(e)}")

@router.get("/cluster-status-db/{bucket_name}/{sub_bucket}")
async def get_clustering_status_from_db(bucket_name: str, sub_bucket: str):
    """Get the latest clustering status from database including Qdrant stats"""
    try:
        cluster_result = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        
        if not cluster_result:
            raise HTTPException(404, f"No clustering results found for {bucket_name}/{sub_bucket}")
        
        clustering_id = str(cluster_result.id)
        
        # Get Qdrant statistics
        qdrant_stats = qdrant_service.get_clustering_stats(clustering_id)
        
        return {
            "bucket_path": cluster_result.bucket_path,
            "clustering_id": clustering_id,
            "timestamp": cluster_result.timestamp,
            "num_clusters": cluster_result.num_clusters,
            "total_images": cluster_result.total_images,
            "total_faces": cluster_result.total_faces,
            "noise_count": len(cluster_result.noise),
            "cluster_details": [
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "face_count": len(cluster.face_ids)
                }
                for cluster in cluster_result.clusters
            ],
            "processing_stats": cluster_result.processing_stats,
            "qdrant_stats": {
                "total_embeddings": qdrant_stats.get("total_embeddings", 0),
                "clusters_with_embeddings": qdrant_stats.get("clusters_with_embeddings", 0),
                "noise_embeddings": qdrant_stats.get("noise_embeddings", 0)
            }
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error fetching clustering status from DB: {e}")
        raise HTTPException(500, "Failed to fetch clustering status from database")

@router.get("/cluster-details-db/{bucket_name}/{sub_bucket}/{cluster_id}")
async def get_cluster_details_from_db(bucket_name: str, sub_bucket: str, cluster_id: str):
    """Get detailed information about a specific cluster from database and Qdrant"""
    try:
        cluster_result = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        
        if not cluster_result:
            raise HTTPException(404, f"No clustering results found for {bucket_name}/{sub_bucket}")
        
        clustering_id = str(cluster_result.id)
        
        target_cluster = None
        for cluster in cluster_result.clusters:
            if cluster.cluster_id == cluster_id:
                target_cluster = cluster
                break
                
        if not target_cluster:
            raise HTTPException(404, f"Cluster {cluster_id} not found in {bucket_name}/{sub_bucket}")
        
        # Get embeddings from Qdrant for this cluster
        qdrant_embeddings = qdrant_service.get_embeddings_by_cluster_and_clustering_id(
            clustering_id=clustering_id,
            cluster_id=cluster_id
        )
        
        # Get faces from MongoDB
        cluster_faces = await FaceClusteringDB.get_cluster_faces(bucket_name, sub_bucket, cluster_id)
        
        # Format response
        face_details = []
        for face in cluster_faces:
            # Find corresponding Qdrant embedding
            qdrant_face = next(
                (qf for qf in qdrant_embeddings if qf["payload"]["face_id"] == face.face_id),
                None
            )
            
            face_details.append({
                "face_id": face.face_id,
                "image_path": face.image_path,
                "confidence": face.confidence,
                "quality_score": face.quality_score,
                "bbox": face.bbox,
                "embedding_available": qdrant_face is not None,
                "embedding_size": len(qdrant_face["vector"]) if qdrant_face else 0,
                "timestamp": face.timestamp
            })
        
        return {
            "clustering_id": clustering_id,
            "cluster_id": cluster_id,
            "bucket_path": f"{bucket_name}/{sub_bucket}",
            "total_faces": len(face_details),
            "cluster_size": target_cluster.size,
            "faces": face_details,
            "image_paths": target_cluster.image_paths,
            "face_ids": target_cluster.face_ids,
            "embeddings_in_qdrant": len(qdrant_embeddings)
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error getting cluster details from DB: {e}")
        raise HTTPException(500, "Failed to get cluster details from database")