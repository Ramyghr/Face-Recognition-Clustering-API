# app/routers/bucket_clustering.py
from fastapi import APIRouter, HTTPException, Query, Path
from app.services.S3_functions import storage_service
from app.services.person_based_clustering_pipeline import person_clustering_pipeline
from app.core.config import settings
from app.models.face_clustering_models import ClusterResultResponse
from app.db.face_clustering_operations import PersonBasedClusteringDB
from app.services import yolo_detector, ai
from app.services.qdrant_service import qdrant_service
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Person-Based Bucket Clustering"])

@router.post("/cluster-bucket-persons/{bucket_name}/{sub_bucket}", response_model=ClusterResultResponse)
async def cluster_bucket_by_persons(
    bucket_name: str, 
    sub_bucket: str,
    max_images: Optional[int] = Query(None),
    batch_size: int = Query(30),
    skip_quality_filter: bool = Query(False),
    max_concurrent: int = Query(12)
):
    """Person-based clustering where each person gets their own cluster with all their appearances"""
    logger = logging.getLogger(__name__)
    start_time = time.time()

    try:
        # Validate and adjust parameters
        batch_size = min(batch_size, 50)
        max_concurrent = min(max_concurrent, 20)

        prefix = f"{bucket_name}/{sub_bucket}/"
        logger.info(f"Starting person-based clustering for path: {prefix}")

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
            yolo_detector.load_yolo_from_hf()
            await ai.init_face_model(settings.EMBED_MODEL_PATH)
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise HTTPException(500, f"Model initialization failed: {str(e)}")

        # Process person-based clustering pipeline
        try:
            result = await person_clustering_pipeline.process_person_based_clustering_pipeline(
                files=files,
                bucket_name=bucket_name,
                sub_bucket=sub_bucket,
                batch_size=batch_size,
                skip_quality_filter=skip_quality_filter,
                max_concurrent=max_concurrent
            )
            
            if not result or "person_clusters" not in result:
                raise HTTPException(500, "Person clustering returned invalid result")
                
            logger.info("Person clustering completed successfully")

        except asyncio.TimeoutError:
            raise HTTPException(408, "Person clustering timed out")
        except Exception as e:
            logger.error(f"Person clustering pipeline error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(500, f"Person clustering failed: {str(e)}")

        # Prepare comprehensive statistics
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
            "storage_architecture": "person_based_metadata_in_mongodb_embeddings_in_qdrant",
            "clustering_method": "person_based_with_image_overlaps",
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update result stats
        result.setdefault("stats", {}).update(stats)

        # Convert person_clusters to the expected format for response
        formatted_person_clusters = []
        for cluster in result.get("person_clusters", []):
            formatted_person_clusters.append({
                "person_id": cluster["person_id"],
                "image_paths": cluster["image_paths"],
                "total_appearances": cluster["total_appearances"],
                "unique_images": cluster["unique_images"],
                "owner_face_id": cluster["owner_face_id"],
                "avg_confidence": cluster["avg_confidence"],
                "best_quality": cluster["best_quality"]
            })

        return ClusterResultResponse(
            bucket=f"{bucket_name}/{sub_bucket}",
            person_clusters=formatted_person_clusters,
            unassigned=result.get("unassigned", []),
            timestamp=datetime.utcnow(),
            stats=result.get("stats", {})
        )

    except HTTPException as http_ex:
        logger.error(f"HTTP Exception during person clustering: {str(http_ex.detail)}")
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error during person clustering: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Unexpected person clustering error: {str(e)}")

@router.get("/person-cluster-status/{bucket_name}/{sub_bucket}")
async def get_person_clustering_status(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name"),
    clustering_id: str = Query(None, description="Specific clustering ID to get status for"),
    collection_name: str = Query(None, description="MongoDB collection name (auto-constructed if not provided)"),
    show_details: bool = Query(False, description="Include detailed cluster information")
):
    """Get the person-based clustering status from database"""
    try:
        # Construct collection name dynamically if not provided
        if collection_name is None:
            formatted_bucket = bucket_name.replace('-', '_')
            collection_name = f"person_clustering_{formatted_bucket}_{sub_bucket}"
        
        bucket_path = f"{bucket_name}/{sub_bucket}"
        
        # Direct MongoDB access to get clustering data
        from app.services.cluster_linker import _get_direct_database
        client = None
        
        try:
            client, database = await _get_direct_database()
            collection = database[collection_name]
            
            if clustering_id:
                # Get specific clustering document
                if not ObjectId.is_valid(clustering_id):
                    raise HTTPException(status_code=400, detail="Invalid clustering_id format")
                
                doc = await collection.find_one({"_id": ObjectId(clustering_id)})
                if not doc:
                    raise HTTPException(status_code=404, detail=f"Clustering {clustering_id} not found in {collection_name}")
                
                clustering_docs = [doc]
            else:
                # Get all clustering documents for this bucket_path, sorted by timestamp (newest first)
                cursor = collection.find({"bucket_path": bucket_path}).sort("timestamp", -1)
                clustering_docs = await cursor.to_list(length=None)
                
                if not clustering_docs:
                    raise HTTPException(status_code=404, detail=f"No person clustering results found for {bucket_path} in collection {collection_name}")
        
        finally:
            if client:
                client.close()
        
        # Process the clustering documents
        if clustering_id or len(clustering_docs) == 1:
            # Return single clustering status
            doc = clustering_docs[0]
            
            person_clusters = doc.get("person_clusters", [])
            
            # Extract top persons by cluster size
            top_persons = []
            if isinstance(person_clusters, list):
                # Sort by cluster size (number of faces)
                sorted_clusters = sorted(
                    person_clusters, 
                    key=lambda x: x.get("size", 0) if isinstance(x, dict) else 0, 
                    reverse=True
                )
                
                for i, cluster in enumerate(sorted_clusters[:5]):  # Top 5
                    if isinstance(cluster, dict):
                        top_persons.append({
                            "person_rank": i + 1,
                            "person_id": cluster.get("person_id", f"person_{i}"),
                            "cluster_id": cluster.get("cluster_id", f"cluster_{i}"),
                            "face_count": cluster.get("size", 0),
                            "avg_confidence": cluster.get("avg_confidence", 0.0),
                            "owner_face_id": cluster.get("owner_face_id", ""),
                            "sample_images": cluster.get("image_paths", [])[:3] if show_details else []
                        })
            
            # Get assignment information if available
            assignment_info = {"total_assigned": 0, "total_users": 0}
            try:
                from app.db.cluster_assignment_operations import ClusterAssignmentDB
                assignments = await ClusterAssignmentDB.get_assignments_by_clustering(ObjectId(str(doc["_id"])))
                assignment_info = {
                    "total_assigned": len(assignments),
                    "total_users": len(set(a.user_id for a in assignments)),
                    "avg_similarity": sum(a.similarity for a in assignments) / len(assignments) if assignments else 0
                }
            except Exception as e:
                logger.warning(f"Could not fetch assignment info: {e}")
            
            return {
                "bucket_path": bucket_path,
                "collection_name": collection_name,
                "clustering_id": str(doc["_id"]),
                "timestamp": doc.get("timestamp"),
                "clustering_type": "person_based_with_overlaps",
                "total_persons": doc.get("total_persons", 0),
                "total_images": doc.get("total_images", 0),
                "total_faces": doc.get("total_faces", 0),
                "unassigned_faces": len(doc.get("unassigned_faces", [])),
                "unassigned_face_ids": len(doc.get("unassigned_face_ids", [])),
                "person_clusters_count": len(person_clusters),
                "image_overlap_stats": doc.get("image_overlap_stats", {}),
                "processing_stats": doc.get("processing_stats", {}),
                "assignment_info": assignment_info,
                "top_persons": top_persons,
                "detailed_clusters": person_clusters if show_details else None
            }
        
        else:
            # Return multiple clustering summaries
            summaries = []
            for doc in clustering_docs:
                person_clusters = doc.get("person_clusters", [])
                
                # Get assignment count
                assignment_count = 0
                try:
                    from app.db.cluster_assignment_operations import ClusterAssignmentDB
                    assignments = await ClusterAssignmentDB.get_assignments_by_clustering(ObjectId(str(doc["_id"])))
                    assignment_count = len(assignments)
                except:
                    pass
                
                summaries.append({
                    "clustering_id": str(doc["_id"]),
                    "timestamp": doc.get("timestamp"),
                    "total_persons": doc.get("total_persons", 0),
                    "total_images": doc.get("total_images", 0),
                    "total_faces": doc.get("total_faces", 0),
                    "person_clusters_count": len(person_clusters),
                    "unassigned_faces": len(doc.get("unassigned_faces", [])),
                    "assignments_count": assignment_count,
                    "status_url": f"/person-cluster-status/{bucket_name}/{sub_bucket}?clustering_id={str(doc['_id'])}"
                })
            
            return {
                "bucket_path": bucket_path,
                "collection_name": collection_name,
                "total_clustering_documents": len(summaries),
                "latest_clustering": summaries[0] if summaries else None,
                "all_clusterings": summaries,
                "usage": {
                    "get_specific": f"/person-cluster-status/{bucket_name}/{sub_bucket}?clustering_id=<clustering_id>",
                    "show_details": f"/person-cluster-status/{bucket_name}/{sub_bucket}?clustering_id=<clustering_id>&show_details=true"
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching person clustering status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch person clustering status: {str(e)}")


@router.get("/person-details/{bucket_name}/{sub_bucket}/{person_id}")
async def get_person_cluster_details(bucket_name: str, sub_bucket: str, person_id: str):
    """Get detailed information about a specific person's appearances"""
    try:
        person_details = await PersonBasedClusteringDB.get_person_cluster_details(bucket_name, sub_bucket, person_id)
        
        if not person_details:
            raise HTTPException(404, f"Person {person_id} not found in {bucket_name}/{sub_bucket}")
        
        # Get Qdrant embeddings for this person
        clustering_result = await PersonBasedClusteringDB.get_latest_person_clustering_result(bucket_name, sub_bucket)
        if clustering_result:
            clustering_id = str(clustering_result.id)
            qdrant_embeddings = qdrant_service.get_embeddings_by_person_and_clustering_id(
                clustering_id=clustering_id,
                person_id=person_id
            )
            
            person_details["embeddings_in_qdrant"] = len(qdrant_embeddings) if qdrant_embeddings else 0
        else:
            person_details["embeddings_in_qdrant"] = 0
        
        return person_details
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Error getting person details: {e}")
        raise HTTPException(500, "Failed to get person details")


@router.delete("/cleanup-person-data/{bucket_name}/{sub_bucket}")
async def cleanup_person_clustering_data(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name"),
    clustering_id: str = Query(..., description="Required: Specific clustering ID to clean up"),
    collection_name: str = Query(None, description="MongoDB collection name (auto-constructed if not provided)"),
    cleanup_assignments: bool = Query(True, description="Also cleanup cluster assignments"),
    cleanup_qdrant: bool = Query(True, description="Also cleanup Qdrant data"),
    dry_run: bool = Query(False, description="Preview cleanup without actually deleting")
):
    """Clean up specific person-based clustering data (for testing/maintenance)"""
    try:
        # Validate clustering_id
        if not ObjectId.is_valid(clustering_id):
            raise HTTPException(status_code=400, detail="Invalid clustering_id format")
        
        # Construct collection name dynamically if not provided
        if collection_name is None:
            formatted_bucket = bucket_name.replace('-', '_')
            collection_name = f"person_clustering_{formatted_bucket}_{sub_bucket}"
        
        bucket_path = f"{bucket_name}/{sub_bucket}"
        
        # First, verify the clustering document exists
        from app.services.cluster_linker import _get_direct_database
        client = None
        
        try:
            client, database = await _get_direct_database()
            collection = database[collection_name]
            
            doc = await collection.find_one({"_id": ObjectId(clustering_id)})
            if not doc:
                raise HTTPException(status_code=404, detail=f"Clustering {clustering_id} not found in {collection_name}")
            
            cleanup_results = {
                "clustering_id": clustering_id,
                "bucket_path": bucket_path,
                "collection_name": collection_name,
                "dry_run": dry_run,
                "cleanup_steps": {},
                "errors": []
            }
            
            # Step 1: Cleanup cluster assignments if requested
            if cleanup_assignments:
                try:
                    from app.db.cluster_assignment_operations import ClusterAssignmentDB
                    
                    # Get current assignment count
                    assignments = await ClusterAssignmentDB.get_assignments_by_clustering(ObjectId(clustering_id))
                    assignment_count = len(assignments)
                    
                    if not dry_run and assignment_count > 0:
                        deleted_count = await ClusterAssignmentDB.delete_assignments_by_clustering(ObjectId(clustering_id))
                        cleanup_results["cleanup_steps"]["assignments"] = {
                            "found": assignment_count,
                            "deleted": deleted_count,
                            "success": deleted_count > 0
                        }
                    else:
                        cleanup_results["cleanup_steps"]["assignments"] = {
                            "found": assignment_count,
                            "deleted": 0 if dry_run else 0,
                            "success": True,
                            "note": "dry_run mode" if dry_run else "no assignments to delete"
                        }
                        
                except Exception as e:
                    cleanup_results["errors"].append(f"Assignment cleanup failed: {str(e)}")
                    cleanup_results["cleanup_steps"]["assignments"] = {"error": str(e)}
            
            # Step 2: Cleanup Qdrant data if requested
            if cleanup_qdrant:
                try:
                    if not dry_run:
                        operation_id = qdrant_service.delete_clustering_data(clustering_id)
                        cleanup_results["cleanup_steps"]["qdrant"] = {
                            "operation_id": operation_id,
                            "success": operation_id is not None
                        }
                    else:
                        cleanup_results["cleanup_steps"]["qdrant"] = {
                            "operation_id": None,
                            "success": True,
                            "note": "dry_run mode - would delete Qdrant data"
                        }
                except Exception as e:
                    cleanup_results["errors"].append(f"Qdrant cleanup failed: {str(e)}")
                    cleanup_results["cleanup_steps"]["qdrant"] = {"error": str(e)}
            
            # Step 3: Get clustering document info for reference
            person_clusters_count = len(doc.get("person_clusters", []))
            cleanup_results["document_info"] = {
                "total_persons": doc.get("total_persons", 0),
                "total_faces": doc.get("total_faces", 0),
                "total_images": doc.get("total_images", 0),
                "person_clusters_count": person_clusters_count,
                "timestamp": doc.get("timestamp")
            }
            
            # Step 4: Optionally delete the clustering document itself
            # (Not implemented by default - add parameter if needed)
            
            cleanup_results["summary"] = {
                "total_steps": len(cleanup_results["cleanup_steps"]),
                "successful_steps": len([step for step in cleanup_results["cleanup_steps"].values() if step.get("success", False)]),
                "failed_steps": len(cleanup_results["errors"]),
                "overall_success": len(cleanup_results["errors"]) == 0
            }
            
            return cleanup_results
            
        finally:
            if client:
                client.close()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Person clustering cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")



# Keep the old endpoint for backward compatibility but mark as deprecated
@router.post("/cluster-bucket-with-db/{bucket_name}/{sub_bucket}", response_model=ClusterResultResponse, deprecated=True)
async def cluster_bucket_with_database_legacy(
    bucket_name: str, 
    sub_bucket: str,
    max_images: Optional[int] = Query(None),
    batch_size: int = Query(30),
    skip_quality_filter: bool = Query(False),
    max_concurrent: int = Query(12)
):
    """DEPRECATED: Legacy clustering method. Use /cluster-bucket-persons/ instead for person-based clustering."""
    # Redirect to the new person-based clustering
    return await cluster_bucket_by_persons(
        bucket_name, sub_bucket, max_images, batch_size, skip_quality_filter, max_concurrent
    )