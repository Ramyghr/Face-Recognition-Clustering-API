# routers/cluster_assignment.py
from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional, Dict, Any
import logging

from app.services.cluster_linker import assign_clusters_to_users, assign_all_clusterings_to_users
from app.db.cluster_assignment_operations import ClusterAssignmentDB
from app.models.cluster_assignments import ClusterAssignment
from bson import ObjectId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assignments", tags=["Cluster→User Assignment"])


@router.post("/{bucket_name}/{sub_bucket}")
async def assign_clusters(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name"),
    clustering_id: str = Query(..., description="Required clustering _id to use from specific collection"),
    collection_name: str = Query(
        None,  # Changed from hardcoded default to None
        description="MongoDB collection name to read clustering from. If not provided, will be constructed as person_clustering_{bucket_name}_{sub_bucket}"
    ),
    threshold: float = Query(0.65, ge=0.0, le=1.0, description="Cosine similarity threshold"),
    strategy: str = Query("centroid", pattern="^(centroid|vote)$", description="Assignment strategy"),
    dry_run: bool = Query(False, description="Preview without saving"),
    overwrite: bool = Query(False, description="Overwrite existing assignments"),
) -> Dict[str, Any]:
    """
    Assign clusters to users from a MongoDB collection.
    
    The collection name is dynamically constructed based on bucket_name and sub_bucket:
    - bucket_name: uwas-classification-recette, sub_bucket: user -> person_clustering_uwas_classification_recette_user
    - bucket_name: uwas-classification-recette, sub_bucket: test -> person_clustering_uwas_classification_recette_test
    
    You can override this behavior by explicitly providing the collection_name parameter.
    """
    
    try:
        result = await assign_clusters_to_users(
            bucket_name=bucket_name,
            sub_bucket=sub_bucket,
            threshold=threshold,
            strategy=strategy,
            dry_run=dry_run,
            overwrite=overwrite,
            clustering_id=clustering_id,
            collection_name=collection_name  # Will be None if not provided, triggering dynamic construction
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Assignment failed")
        raise HTTPException(status_code=500, detail=f"Assignment error: {str(e)}")


@router.post("/batch")
async def assign_all_clusters_in_collection(
    bucket_name: str = Query(None, description="Bucket name (required if collection_name not provided)"),
    sub_bucket: str = Query(None, description="Sub-bucket name (required if collection_name not provided)"),
    collection_name: str = Query(
        None,
        description="MongoDB collection name to process. If not provided, constructed from bucket_name/sub_bucket"
    ),
    threshold: float = Query(0.65, ge=0.0, le=1.0, description="Cosine similarity threshold"),
    strategy: str = Query("centroid", pattern="^(centroid|vote)$", description="Assignment strategy"),
    dry_run: bool = Query(False, description="Preview without saving"),
    overwrite: bool = Query(False, description="Overwrite existing assignments"),
) -> Dict[str, Any]:
    """
    Process all clustering documents in the specified collection.
    
    Either provide collection_name directly, OR provide bucket_name + sub_bucket to 
    construct the collection name dynamically.
    """
    
    # Validate parameters
    if collection_name is None and (bucket_name is None or sub_bucket is None):
        raise HTTPException(
            status_code=400, 
            detail="Either provide collection_name, or both bucket_name and sub_bucket"
        )
    
    try:
        result = await assign_all_clusterings_to_users(
            collection_name=collection_name,
            bucket_name=bucket_name,
            sub_bucket=sub_bucket,
            threshold=threshold,
            strategy=strategy,
            dry_run=dry_run,
            overwrite=overwrite
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Batch assignment failed")
        raise HTTPException(status_code=500, detail=f"Batch assignment error: {str(e)}")


@router.get("/{bucket_name}/{sub_bucket}")
async def get_assignments(
    bucket_name: str = Path(..., description="Bucket name"),
    sub_bucket: str = Path(..., description="Sub-bucket name"),
    clustering_id: Optional[str] = Query(None, description="Specific clustering ID to get assignments for")
) -> Dict[str, Any]:
    """Get existing assignments for a bucket/sub_bucket"""
    try:
        if clustering_id:
            # Get assignments for specific clustering ID
            if not ObjectId.is_valid(clustering_id):
                raise HTTPException(status_code=400, detail="Invalid clustering_id format")
            
            assignments = await ClusterAssignmentDB.get_assignments_by_clustering(ObjectId(clustering_id))
            
            return {
                "clustering_id": clustering_id,
                "bucket_path": f"{bucket_name}/{sub_bucket}",
                "total_assignments": len(assignments),
                "assignments": [
                    {
                        "cluster_id": a.cluster_id,
                        "user_id": a.user_id,
                        "similarity": a.similarity,
                        "face_count": a.face_count,
                        "strategy": a.strategy,
                        "created_at": a.created_at
                    }
                    for a in assignments
                ]
            }
        else:
            # This would require additional logic to find clustering IDs for bucket/sub_bucket
            # For now, return instructions to use clustering_id parameter
            return {
                "message": "Please provide clustering_id parameter to get specific assignments",
                "example": f"/assignments/{bucket_name}/{sub_bucket}?clustering_id=<your_clustering_id>"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get assignments")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/user/{user_id}")
async def get_user_assignments(
    user_id: str = Path(..., description="User ID to get assignments for"),
    bucket_name: str = Query(None, description="Filter by bucket name"),
    sub_bucket: str = Query(None, description="Filter by sub-bucket"),
    clustering_id: str = Query(None, description="Filter by specific clustering ID"),
    include_details: bool = Query(False, description="Include detailed cluster information")
) -> Dict[str, Any]:
    """Get all cluster assignments for a specific user with optional filtering"""
    try:
        # Get base user summary
        summary = await ClusterAssignmentDB.get_user_cluster_summary(user_id)
        
        # If no specific filtering requested, return the summary
        if not any([bucket_name, sub_bucket, clustering_id]):
            return {
                "user_id": user_id,
                "summary": summary,
                "total_assignments": summary.get("total_assignments", 0),
                "filtering_options": {
                    "by_bucket": f"/assignments/user/{user_id}?bucket_name=<bucket_name>",
                    "by_sub_bucket": f"/assignments/user/{user_id}?bucket_name=<bucket_name>&sub_bucket=<sub_bucket>",
                    "by_clustering": f"/assignments/user/{user_id}?clustering_id=<clustering_id>",
                    "with_details": f"/assignments/user/{user_id}?include_details=true"
                }
            }
        
        # Apply filtering
        from app.models.cluster_assignments import ClusterAssignment
        
        # Build filter query
        filter_query = {"user_id": user_id}
        
        if clustering_id:
            if not ObjectId.is_valid(clustering_id):
                raise HTTPException(status_code=400, detail="Invalid clustering_id format")
            filter_query["clustering_id"] = ObjectId(clustering_id)
        
        if bucket_name or sub_bucket:
            if bucket_name and sub_bucket:
                filter_query["bucket_path"] = f"{bucket_name}/{sub_bucket}"
            elif bucket_name:
                # Use regex to match bucket_name at the start
                filter_query["bucket_path"] = {"$regex": f"^{bucket_name}/"}
        
        # Get filtered assignments
        assignments = await ClusterAssignment.find(filter_query).to_list()
        
        if not assignments:
            return {
                "user_id": user_id,
                "filters_applied": {
                    "bucket_name": bucket_name,
                    "sub_bucket": sub_bucket,
                    "clustering_id": clustering_id
                },
                "total_assignments": 0,
                "assignments": [],
                "message": "No assignments found matching the specified filters"
            }
        
        # Group by clustering for better organization
        clusterings = {}
        total_faces = 0
        
        for assignment in assignments:
            clustering_id_str = str(assignment.clustering_id)
            
            if clustering_id_str not in clusterings:
                clusterings[clustering_id_str] = {
                    "clustering_id": clustering_id_str,
                    "bucket_path": assignment.bucket_path,
                    "clusters": [],
                    "cluster_count": 0,
                    "total_faces": 0,
                    "avg_similarity": 0.0,
                    "similarities": []
                }
            
            cluster_info = {
                "cluster_id": assignment.cluster_id,
                "similarity": assignment.similarity,
                "face_count": assignment.face_count,
                "strategy": assignment.strategy,
                "created_at": assignment.created_at
            }
            
            # Add detailed info if requested
            if include_details:
                cluster_info.update({
                    "image_paths": assignment.image_paths if hasattr(assignment, 'image_paths') else [],
                    "face_ids": assignment.face_ids if hasattr(assignment, 'face_ids') else []
                })
            
            clusterings[clustering_id_str]["clusters"].append(cluster_info)
            clusterings[clustering_id_str]["cluster_count"] += 1
            clusterings[clustering_id_str]["total_faces"] += assignment.face_count
            clusterings[clustering_id_str]["similarities"].append(assignment.similarity)
            
            total_faces += assignment.face_count
        
        # Calculate average similarities for each clustering
        for clustering_info in clusterings.values():
            sims = clustering_info["similarities"]
            clustering_info["avg_similarity"] = round(sum(sims) / len(sims), 4) if sims else 0.0
            del clustering_info["similarities"]  # Remove from response
        
        return {
            "user_id": user_id,
            "filters_applied": {
                "bucket_name": bucket_name,
                "sub_bucket": sub_bucket,
                "clustering_id": clustering_id,
                "include_details": include_details
            },
            "total_assignments": len(assignments),
            "total_clusterings": len(clusterings),
            "total_faces": total_faces,
            "clusterings": list(clusterings.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get assignments for user {user_id}")
        raise HTTPException(status_code=500, detail=str(e))



@router.delete("/clustering/{clustering_id}")
async def delete_clustering_assignments(clustering_id: str) -> Dict[str, Any]:
    """Delete all assignments for a specific clustering"""
    try:
        if not ObjectId.is_valid(clustering_id):
            raise HTTPException(status_code=400, detail="Invalid clustering_id format")
            
        deleted_count = await ClusterAssignmentDB.delete_assignments_by_clustering(
            ObjectId(clustering_id)
        )
        return {
            "clustering_id": clustering_id,
            "deleted_count": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete assignments for clustering {clustering_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug/collections")
async def debug_available_collections(
    bucket_name: str = Query(None, description="Filter collections by bucket name pattern"),
    sub_bucket: str = Query(None, description="Filter collections by sub-bucket pattern"),
    clustering_id: str = Query(None, description="Filter by specific clustering ID"),
    show_details: bool = Query(False, description="Show detailed collection information")
) -> Dict[str, Any]:
    """Debug endpoint to list available clustering collections with optional filtering"""
    client = None
    try:
        # Create direct connection using the same method as cluster_linker
        from app.services.cluster_linker import _get_direct_database
        client, database = await _get_direct_database()
        
        # Get all collection names
        collections = await database.list_collection_names()
        
        # Filter for clustering-related collections
        clustering_collections = [
            name for name in collections 
            if "clustering" in name.lower() or "person_clustering" in name.lower()
        ]
        
        # Apply bucket/sub_bucket filtering
        filtered_collections = clustering_collections
        if bucket_name:
            formatted_bucket = bucket_name.replace('-', '_')
            filtered_collections = [
                name for name in filtered_collections 
                if formatted_bucket in name
            ]
        
        if sub_bucket:
            filtered_collections = [
                name for name in filtered_collections 
                if name.endswith(f"_{sub_bucket}") or sub_bucket in name
            ]
        
        # Get document counts and details for filtered collections
        collection_info = {}
        total_clustering_docs = 0
        clustering_found = False
        
        for coll_name in filtered_collections:
            try:
                # If clustering_id is provided, check if this collection contains it
                if clustering_id:
                    from bson import ObjectId
                    try:
                        # Try to find the specific clustering document
                        clustering_doc = await database[coll_name].find_one({"_id": ObjectId(clustering_id)})
                        if clustering_doc:
                            clustering_found = True
                            # Only include this collection and document in results
                            collection_data = {
                                "document_count": 1,
                                "clustering_id_found": True,
                                "sample_fields": list(clustering_doc.keys()),
                                "has_person_clusters": "person_clusters" in clustering_doc,
                                "person_clusters_count": len(clustering_doc.get("person_clusters", [])),
                                "bucket_path": clustering_doc.get("bucket_path", "unknown"),
                                "total_persons": clustering_doc.get("total_persons", 0),
                                "total_faces": clustering_doc.get("total_faces", 0),
                                "total_images": clustering_doc.get("total_images", 0),
                                "timestamp": clustering_doc.get("timestamp", "unknown"),
                                "clustering_detail": {
                                    "clustering_id": clustering_id,
                                    "bucket_path": clustering_doc.get("bucket_path", "unknown"),
                                    "timestamp": clustering_doc.get("timestamp", "unknown"),
                                    "total_persons": clustering_doc.get("total_persons", 0),
                                    "assignment_url": f"/assignments/clustering/{clustering_id}"
                                }
                            }
                            collection_info[coll_name] = collection_data
                            total_clustering_docs += 1
                            continue
                    except Exception:
                        # Invalid ObjectId or other error, continue to next collection
                        continue
                
                # Regular processing when clustering_id is not specified or not found
                if not clustering_id or not clustering_found:
                    count = await database[coll_name].count_documents({})
                    total_clustering_docs += count
                    
                    collection_data = {
                        "document_count": count
                    }
                    
                    if show_details and count > 0:
                        # Get one sample document to show structure
                        sample = await database[coll_name].find_one({})
                        if sample:
                            collection_data.update({
                                "sample_fields": list(sample.keys()),
                                "has_person_clusters": "person_clusters" in sample,
                                "person_clusters_count": len(sample.get("person_clusters", [])),
                                "bucket_path": sample.get("bucket_path", "unknown"),
                                "total_persons": sample.get("total_persons", 0),
                                "total_faces": sample.get("total_faces", 0),
                                "total_images": sample.get("total_images", 0),
                                "latest_timestamp": sample.get("timestamp", "unknown")
                            })
                            
                            # Show available clustering IDs
                            cursor = database[coll_name].find({}, {"_id": 1, "bucket_path": 1, "timestamp": 1, "total_persons": 1})
                            docs = await cursor.to_list(length=10)  # Limit to first 10
                            
                            collection_data["available_clusterings"] = [
                                {
                                    "clustering_id": str(doc["_id"]),
                                    "bucket_path": doc.get("bucket_path", "unknown"),
                                    "timestamp": doc.get("timestamp", "unknown"),
                                    "total_persons": doc.get("total_persons", 0),
                                    "assignment_url": f"/assignments/clustering/{str(doc['_id'])}"
                                }
                                for doc in docs
                            ]
                    
                    collection_info[coll_name] = collection_data
                
            except Exception as e:
                collection_info[coll_name] = {"error": str(e)}
        
        # Handle case where clustering_id was specified but not found
        if clustering_id and not clustering_found:
            return {
                "error": f"Clustering ID '{clustering_id}' not found in any clustering collections",
                "filters_applied": {
                    "bucket_name": bucket_name,
                    "sub_bucket": sub_bucket,
                    "clustering_id": clustering_id,
                    "show_details": show_details
                },
                "searched_collections": filtered_collections,
                "suggestion": "Check if the clustering_id is correct or try without filtering parameters"
            }
        
        # Generate suggested collection names based on parameters
        suggested_collections = []
        if bucket_name and sub_bucket:
            formatted_bucket = bucket_name.replace('-', '_')
            suggested_name = f"person_clustering_{formatted_bucket}_{sub_bucket}"
            suggested_collections.append({
                "name": suggested_name,
                "exists": suggested_name in clustering_collections,
                "bucket_name": bucket_name,
                "sub_bucket": sub_bucket
            })
        
        return {
            "filters_applied": {
                "bucket_name": bucket_name,
                "sub_bucket": sub_bucket,
                "clustering_id": clustering_id,
                "show_details": show_details
            },
            "total_collections": len(collections),
            "total_clustering_collections": len(clustering_collections),
            "filtered_clustering_collections": len(filtered_collections),
            "total_clustering_documents": total_clustering_docs,
            "clustering_collections": filtered_collections if not show_details and not clustering_id else None,
            "collection_details": collection_info if show_details or clustering_id else None,
            "suggested_collections": suggested_collections,
            "usage_examples": {
                "filter_by_bucket": "/assignments/debug/collections?bucket_name=uwas-classification-recette",
                "filter_by_bucket_and_sub": "/assignments/debug/collections?bucket_name=uwas-classification-recette&sub_bucket=A",
                "filter_by_clustering_id": "/assignments/debug/collections?clustering_id=<your_clustering_id>",
                "show_details": "/assignments/debug/collections?show_details=true",
                "inspect_specific": "/assignments/debug/collection/{collection_name}"
            }
        }
        
    except Exception as e:
        logger.exception("Debug collections failed")
        return {
            "error": str(e),
            "suggestion": "Check if database connection is properly configured",
            "filters_applied": {
                "bucket_name": bucket_name,
                "sub_bucket": sub_bucket,
                "clustering_id": clustering_id,
                "show_details": show_details
            }
        }
    finally:
        if client:
            client.close()

@router.get("/debug/collection/{collection_name}")
async def debug_collection_content(
    collection_name: str = Path(..., description="Collection name to inspect")
) -> Dict[str, Any]:
    """Debug endpoint to inspect a specific collection"""
    client = None
    try:
        # Create direct connection using the same method as cluster_linker
        from app.services.cluster_linker import _get_direct_database
        client, database = await _get_direct_database()
        
        collection = database[collection_name]
        
        # Get basic stats
        doc_count = await collection.count_documents({})
        
        if doc_count == 0:
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "message": "Collection is empty"
            }
        
        # Get sample documents
        sample_docs = await collection.find({}).limit(3).to_list(length=3)
        
        # Analyze structure of first document
        structure_info = {}
        if sample_docs:
            first_doc = sample_docs[0]
            structure_info = {
                "total_fields": len(first_doc.keys()),
                "fields": list(first_doc.keys()),
                "has_clusters": "clusters" in first_doc,
            }
            
            # Check different cluster storage patterns
            clusters = []
            if "clusters" in first_doc:
                clusters = first_doc["clusters"]
                structure_info["cluster_storage"] = "clusters_field"
            elif isinstance(first_doc, list):
                clusters = first_doc
                structure_info["cluster_storage"] = "root_array"
            else:
                # Check for numbered indices (0, 1, 2, etc.)
                i = 0
                numbered_clusters = []
                while str(i) in first_doc:
                    numbered_clusters.append(first_doc[str(i)])
                    i += 1
                if numbered_clusters:
                    clusters = numbered_clusters
                    structure_info["cluster_storage"] = "numbered_indices"
                    structure_info["numbered_cluster_count"] = len(numbered_clusters)
            
            structure_info["clusters_count"] = len(clusters)
            
            # Check if clusters have required fields
            if clusters and isinstance(clusters[0], dict):
                first_cluster = clusters[0]
                structure_info["cluster_fields"] = list(first_cluster.keys())
                structure_info["has_owner_embedding"] = "owner_embedding" in first_cluster
                structure_info["has_face_positions"] = "face_positions" in first_cluster
                
                # Check owner_embedding structure
                if "owner_embedding" in first_cluster:
                    owner_emb = first_cluster["owner_embedding"]
                    if isinstance(owner_emb, list):
                        structure_info["owner_embedding_length"] = len(owner_emb)
                        structure_info["owner_embedding_type"] = "list"
                    else:
                        structure_info["owner_embedding_type"] = type(owner_emb).__name__
                
                # Check face_positions structure
                if "face_positions" in first_cluster:
                    face_pos = first_cluster["face_positions"]
                    if isinstance(face_pos, list):
                        structure_info["face_positions_count"] = len(face_pos)
                        if face_pos:
                            structure_info["face_position_sample"] = face_pos[0] if len(face_pos) > 0 else None
        
        return {
            "collection_name": collection_name,
            "document_count": doc_count,
            "structure_info": structure_info,
            "sample_document_ids": [str(doc["_id"]) for doc in sample_docs],
            "ready_for_assignment": (
                structure_info.get("has_clusters", False) and 
                structure_info.get("has_owner_embedding", False)
            )
        }
        
    except Exception as e:
        logger.exception(f"Debug collection {collection_name} failed")
        return {
            "collection_name": collection_name,
            "error": str(e)
        }
    finally:
        if client:
            client.close()
@router.get("/debug/test-db-connection")
async def test_db_connection():
    try:
        from app.services.cluster_linker import _get_direct_database
        client, database = await _get_direct_database()
        
        # Test basic operations
        collections = await database.list_collection_names()
        client.close()
        
        return {"status": "success", "collections": collections[:5]}
    except Exception as e:
        return {"status": "error", "message": str(e)}