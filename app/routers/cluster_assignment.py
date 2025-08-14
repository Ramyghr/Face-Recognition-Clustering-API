# routers/cluster_assignment.py
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional, Dict, Any
import logging

from app.services.cluster_linker import assign_clusters_to_users  # NOTE: updated import
from app.db.cluster_assignment_operations import ClusterAssignmentDB
from app.models.cluster_assignments import ClusterAssignment
from bson import ObjectId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assignments", tags=["Cluster→User Assignment"])


@router.post("/{bucket_name}/{sub_bucket}")
async def assign_clusters(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name"),
    clustering_id: Optional[str] = Query(None, description="Explicit clustering _id to use"),
    threshold: float = Query(0.65, ge=0.0, le=1.0, description="Cosine similarity threshold"),
    strategy: str = Query("centroid", pattern="^(centroid|vote)$", description="Assignment strategy"),
    dry_run: bool = Query(False, description="Preview without saving"),
    overwrite: bool = Query(False, description="Overwrite existing assignments"),
) -> Dict[str, Any]:
    try:
        result = await assign_clusters_to_users(
            bucket_name=bucket_name,
            sub_bucket=sub_bucket,
            threshold=threshold,
            strategy=strategy,
            dry_run=dry_run,
            overwrite=overwrite,
            clustering_id=clustering_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Assignment failed")
        raise HTTPException(status_code=500, detail=f"Assignment error: {str(e)}")

@router.get("/{bucket_name}/{sub_bucket}")
async def get_assignments(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name")
) -> Dict[str, Any]:
    """Get existing assignments for a bucket/sub_bucket"""
    try:
        # This would need to be implemented in your FaceClusteringDB
        from app.db.face_clustering_operations import FaceClusteringDB
        
        clustering = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        if not clustering:
            raise HTTPException(status_code=404, detail="No clustering found")
        
        assignments = await ClusterAssignmentDB.get_assignments_by_clustering(clustering.id)
        
        return {
            "clustering_id": str(clustering.id),
            "bucket_path": clustering.bucket_path,
            "total_assignments": len(assignments),
            "assignments": [
                {
                    "cluster_id": a.cluster_id,
                    "user_id": a.user_id,
                    "similarity": a.similarity,
                    "face_count": a.face_count,
                    "created_at": a.created_at
                }
                for a in assignments
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get assignments")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_assignments(user_id: str) -> Dict[str, Any]:
    """Get all cluster assignments for a specific user"""
    try:
        summary = await ClusterAssignmentDB.get_user_cluster_summary(user_id)
        return summary
    except Exception as e:
        logger.exception(f"Failed to get assignments for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clustering/{clustering_id}")
async def delete_clustering_assignments(clustering_id: str) -> Dict[str, Any]:
    """Delete all assignments for a specific clustering"""
    try:
        deleted_count = await ClusterAssignmentDB.delete_assignments_by_clustering(
            ObjectId(clustering_id)
        )
        return {"deleted_count": deleted_count}
    except Exception as e:
        logger.exception(f"Failed to delete assignments for clustering {clustering_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/{bucket_name}/{sub_bucket}")
async def debug_clustering(
    bucket_name: str = Path(..., description="Bucket name"), 
    sub_bucket: str = Path(..., description="Sub-bucket name")
) -> Dict[str, Any]:
    """Debug endpoint to check clustering availability"""
    try:
        from app.models.face_clustering_models import ClusteringResult as FaceClustering

        bucket_path = f"{bucket_name}/{sub_bucket}"
        
        # Try direct lookup
        clustering = await FaceClustering.find_one(
            FaceClustering.bucket_path == bucket_path
        ).sort("-timestamp")
        
        if clustering:
            return {
                "found": True,
                "clustering_id": str(clustering.id),
                "bucket_path": clustering.bucket_path,
                "num_clusters": len(clustering.clusters) if clustering.clusters else 0,
                "created_at": clustering.timestamp if hasattr(clustering, 'timestamp') else str(clustering.id.generation_time),
                "total_faces": getattr(clustering, 'total_faces', 0),
                "clusters_info": [
                    {
                        "cluster_id": c.cluster_id,
                        "size": c.size
                    } for c in (clustering.clusters or [])[:5]  # Show first 5 clusters
                ]
            }
        else:
            # Get all clusterings to help debug
            all_clusterings = await FaceClustering.find_all().sort("-timestamp").limit(10).to_list()
            
            return {
                "found": False,
                "searched_for": bucket_path,
                "available_clusterings": [
                    {
                        "id": str(c.id),
                        "bucket_path": c.bucket_path,
                        "created_at": c.timestamp if hasattr(c, 'timestamp') else str(c.id.generation_time),
                        "num_clusters": len(c.clusters) if c.clusters else 0
                    }
                    for c in all_clusterings
                ]
            }
    except Exception as e:
        logger.exception("Debug failed")
        return {
            "error": str(e),
            "searched_for": f"{bucket_name}/{sub_bucket}",
            "suggestion": "Check if the FaceClustering model import path is correct"
        }