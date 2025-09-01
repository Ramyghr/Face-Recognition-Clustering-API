import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from beanie import init_beanie
from bson import ObjectId

from app.db.cluster_assignment_operations import ClusterAssignmentDB
from app.models.user_profile import UserProfile
from app.models.cluster_assignments import ClusterAssignment
from app.services.qdrant_service import qdrant_service
from app.core.config import settings

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _centroid(vectors: List[List[float]]) -> np.ndarray:
    """Calculate normalized centroid of vectors"""
    if not vectors:
        return np.zeros(128, dtype=np.float32)
    arr = np.asarray(vectors, dtype=np.float32)
    c = arr.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length"""
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v


async def _get_direct_database():
    """Create a direct MongoDB connection for reading clustering data"""
    try:
        # Use settings to create a direct connection
        mongodb_url = getattr(settings, 'MONGODB_URL', None) or getattr(settings, 'MONGO_URI', 'mongodb://localhost:27017')
        db_name = getattr(settings, 'DATABASE_NAME', None) or getattr(settings, 'MONGO_DB', 'face_recognition')
        
        logger.debug(f"Creating direct MongoDB connection to {mongodb_url}/{db_name}")
        
        # Create a new client for this specific operation
        client = AsyncIOMotorClient(mongodb_url)
        database = client[db_name]
        
        # Test the connection
        await client.admin.command('ping')
        logger.debug("Direct MongoDB connection successful")
        
        return client, database
    except Exception as e:
        logger.error(f"Failed to create direct MongoDB connection: {e}")
        raise ValueError(f"Could not create direct database connection: {e}")


async def _load_clustering_from_specific_collection(
    clustering_id: str,
    collection_name: str = "person_clustering_uwas_classification_recette_test"
) -> Dict[str, Any]:
    """
    Load clustering document directly from the specified collection
    """
    client = None
    try:
        # Create direct connection
        client, database = await _get_direct_database()
        collection = database[collection_name]
        
        # Find the document by ObjectId
        if not ObjectId.is_valid(clustering_id):
            raise ValueError(f"Invalid clustering_id: {clustering_id}")
        
        doc = await collection.find_one({"_id": ObjectId(clustering_id)})
        if not doc:
            raise ValueError(f"Clustering {clustering_id} not found in {collection_name}")
        
        logger.info(f"Successfully loaded clustering {clustering_id} from {collection_name}")
        return doc
        
    except Exception as e:
        logger.error(f"Error loading clustering from {collection_name}: {e}")
        raise
    finally:
        # Clean up the connection
        if client:
            client.close()


async def _load_user_profiles() -> Dict[str, np.ndarray]:
    """
    Load user profiles and return {user_id: normalized_embedding}.
    Skips users with missing/empty embeddings.
    """
    try:
        # Try to use Beanie first (since other APIs work fine)
        users = [u async for u in UserProfile.find_all()]
    except Exception as e:
        logger.warning(f"Beanie UserProfile query failed: {e}, trying direct MongoDB access")
        
        # Fallback to direct MongoDB access
        client = None
        try:
            client, database = await _get_direct_database()
            collection = database.user_profiles  # or whatever your user profiles collection is called
            
            cursor = collection.find({})
            raw_users = await cursor.to_list(length=None)
            
            # Convert to a format similar to UserProfile objects
            users = []
            for raw_user in raw_users:
                # Create a simple object with the necessary attributes
                class SimpleUser:
                    def __init__(self, data):
                        self.user_id = data.get('user_id')
                        self.embedding = data.get('embedding')
                
                users.append(SimpleUser(raw_user))
                
        except Exception as direct_e:
            logger.error(f"Direct MongoDB user profile query also failed: {direct_e}")
            raise ValueError(f"Could not load user profiles via Beanie or direct MongoDB: {e}, {direct_e}")
        finally:
            if client:
                client.close()
    
    if not users:
        raise ValueError("No user profiles found")

    user_vecs: Dict[str, np.ndarray] = {}
    for u in users:
        emb = getattr(u, "embedding", None)
        if isinstance(emb, list) and emb:
            v = np.array(emb, dtype=np.float32)
            n = np.linalg.norm(v)
            user_vecs[u.user_id] = (v / n) if n > 0 else v
        else:
            logger.warning("User %s has no embedding; skipped", u.user_id)

    if not user_vecs:
        raise ValueError("All user profiles lack embeddings")

    return user_vecs


def _extract_clusters_from_document(clustering_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract clusters from a clustering document, handling various storage patterns.
    
    Args:
        clustering_doc: The MongoDB document containing clustering data
        
    Returns:
        List of cluster dictionaries
    """
    clusters = []
    
    # Priority order for cluster extraction:
    
    # 1. Check for 'person_clusters' field (your specific case)
    if "person_clusters" in clustering_doc:
        person_clusters = clustering_doc["person_clusters"]
        if isinstance(person_clusters, list):
            clusters = person_clusters
            logger.info(f"Found {len(clusters)} clusters in 'person_clusters' field")
        elif isinstance(person_clusters, dict):
            # If person_clusters is a dict, check if it has numbered keys
            numbered_clusters = []
            i = 0
            while str(i) in person_clusters:
                numbered_clusters.append(person_clusters[str(i)])
                i += 1
            if numbered_clusters:
                clusters = numbered_clusters
                logger.info(f"Found {len(clusters)} clusters in 'person_clusters' with numbered keys")
    
    # 2. Check for standard 'clusters' field
    elif "clusters" in clustering_doc:
        clusters_field = clustering_doc["clusters"]
        if isinstance(clusters_field, list):
            clusters = clusters_field
            logger.info(f"Found {len(clusters)} clusters in 'clusters' field")
    
    # 3. Check if the entire document is an array
    elif isinstance(clustering_doc, list):
        clusters = clustering_doc
        logger.info(f"Document is array with {len(clusters)} clusters")
    
    # 4. Check for numbered indices at root level (0, 1, 2, etc.)
    else:
        numbered_clusters = []
        i = 0
        while str(i) in clustering_doc:
            numbered_clusters.append(clustering_doc[str(i)])
            i += 1
        
        if numbered_clusters:
            clusters = numbered_clusters
            logger.info(f"Found {len(clusters)} clusters with numbered indices at root level")
        else:
            # 5. Last resort: look for any array-like fields
            for key, value in clustering_doc.items():
                if isinstance(value, list) and key not in ['_id', 'unassigned_faces', 'unassigned_face_ids'] and len(value) > 0:
                    # Check if the first element looks like a cluster
                    if isinstance(value[0], dict) and any(field in value[0] for field in ['owner_embedding', 'cluster_id', 'face_positions']):
                        clusters = value
                        logger.info(f"Found {len(clusters)} clusters in field '{key}'")
                        break
    
    if not clusters:
        # Debug: log available fields
        available_fields = list(clustering_doc.keys()) if isinstance(clustering_doc, dict) else "Document is not a dict"
        logger.error(f"No clusters found in document. Available fields: {available_fields}")
        raise ValueError(f"No clusters found in clustering document. Available fields: {available_fields}")
    
    return clusters


async def assign_clusters_to_users(
    bucket_name: str,
    sub_bucket: str,
    threshold: float = 0.65,
    strategy: str = "centroid",
    dry_run: bool = False,
    overwrite: bool = False,
    clustering_id: str | None = None,
    collection_name: str = None
) -> Dict[str, Any]:
    """
    Assign clusters to users using a dynamically determined MongoDB collection.
    """
    
    # Construct collection name dynamically if not provided
    if collection_name is None:
        formatted_bucket = bucket_name.replace('-', '_')
        collection_name = f"person_clustering_{formatted_bucket}_{sub_bucket}"
    
    logger.info(
        "Starting cluster assignment using collection %s (clustering_id=%s, threshold=%.3f, strategy=%s, dry_run=%s, overwrite=%s)",
        collection_name, clustering_id, threshold, strategy, dry_run, overwrite
    )

    if not clustering_id:
        raise ValueError("clustering_id is required when using specific collection")
    
    clustering_doc = await _load_clustering_from_specific_collection(clustering_id, collection_name)
    clustering_oid = ObjectId(clustering_id)
    bucket_path = f"{bucket_name}/{sub_bucket}"

    try:
        clusters = _extract_clusters_from_document(clustering_doc)
    except ValueError as e:
        if isinstance(clustering_doc, dict):
            logger.error(f"Document fields: {list(clustering_doc.keys())}")
            for key, value in list(clustering_doc.items())[:10]:
                if isinstance(value, list):
                    logger.error(f"Field '{key}' is array with {len(value)} elements")
                    if value and isinstance(value[0], dict):
                        logger.error(f"First element of '{key}' has fields: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    logger.error(f"Field '{key}' is object with fields: {list(value.keys())}")
        raise e

    logger.info(f"Successfully extracted {len(clusters)} clusters from document")

    user_vecs = await _load_user_profiles()
    logger.info(f"Loaded {len(user_vecs)} user profiles with embeddings")

    assignments: Dict[str, Dict[str, Any]] = {}
    unassigned: List[str] = []
    to_insert: List[ClusterAssignment] = []

    for i, cluster in enumerate(clusters):
        cluster_id = cluster.get("cluster_id", f"cluster_{i}")
        owner_embedding = cluster.get("owner_embedding", [])
        # Note: Your data doesn't use face_positions, it has direct image_paths and face_ids arrays
        
        logger.debug(f"Processing cluster {cluster_id}")
        
        # Enhanced debug logging for first few clusters
        if i < 3:
            logger.debug(f"Cluster {i} full structure: {cluster.keys() if isinstance(cluster, dict) else 'Not a dict'}")
            if isinstance(cluster, dict):
                logger.debug(f"  - person_id: {cluster.get('person_id', 'N/A')}")
                logger.debug(f"  - size: {cluster.get('size', 'N/A')}")
                logger.debug(f"  - image_paths length: {len(cluster.get('image_paths', []))}")
                logger.debug(f"  - face_ids length: {len(cluster.get('face_ids', []))}")
                logger.debug(f"  - owner_face_id: {cluster.get('owner_face_id', 'N/A')}")
                logger.debug(f"  - avg_confidence: {cluster.get('avg_confidence', 'N/A')}")

        try:
            if owner_embedding and isinstance(owner_embedding, list) and len(owner_embedding) > 0:
                cvec = np.array(owner_embedding, dtype=np.float32)
                n = np.linalg.norm(cvec)
                cvec = cvec / n if n > 0 else cvec
            else:
                logger.warning(f"Cluster {cluster_id} has no owner_embedding; skipping")
                unassigned.append(cluster_id)
                continue

            best_uid, best_sim = None, -1.0
            for uid, uvec in user_vecs.items():
                sim = float(np.dot(cvec, uvec))
                if sim > best_sim:
                    best_sim, best_uid = sim, uid

            if best_uid is None or best_sim < threshold:
                logger.debug(f"Cluster {cluster_id}: best similarity {best_sim:.4f} below threshold {threshold}")
                unassigned.append(cluster_id)
                continue

            # IMPROVED METADATA EXTRACTION - FIXED FOR YOUR DATA STRUCTURE
            image_paths = []
            face_ids = []
            
            # Your data structure has direct arrays, so extract them directly
            cluster_image_paths = cluster.get("image_paths", [])
            cluster_face_ids = cluster.get("face_ids", [])
            
            # Extract image paths
            if isinstance(cluster_image_paths, list):
                image_paths = [str(path) for path in cluster_image_paths if path is not None]
            
            # Extract face IDs
            if isinstance(cluster_face_ids, list):
                face_ids = [str(face_id) for face_id in cluster_face_ids if face_id is not None]
            
            # Get face_count - try multiple approaches
            face_count = 0
            
            # Method 1: Direct 'size' field (as shown in your data)
            if "size" in cluster and isinstance(cluster["size"], int):
                face_count = cluster["size"]
            
            # Method 2: Other count fields
            elif any(field in cluster for field in ["face_count", "count", "num_faces", "faces_count"]):
                for count_field in ["face_count", "count", "num_faces", "faces_count"]:
                    if count_field in cluster and isinstance(cluster[count_field], int):
                        face_count = cluster[count_field]
                        break
            
            # Method 3: Length of extracted arrays
            else:
                face_count = max(len(image_paths), len(face_ids))
            
            # Ensure consistency between arrays and face_count
            if len(image_paths) != face_count or len(face_ids) != face_count:
                logger.debug(f"Cluster {cluster_id}: Array length mismatch - face_count={face_count}, image_paths={len(image_paths)}, face_ids={len(face_ids)}")
                
                # Use the actual data we have rather than synthetic data
                face_count = max(len(image_paths), len(face_ids))
                
                # Pad shorter array to match longer one
                while len(image_paths) < len(face_ids):
                    image_paths.append("")  # Empty string for missing paths
                while len(face_ids) < len(image_paths):
                    face_ids.append(f"unknown_face_{len(face_ids)}")
            
            # Also extract additional metadata if available
            confidence_scores = cluster.get("confidence_scores", [])
            quality_scores = cluster.get("quality_scores", [])
            avg_confidence = cluster.get("avg_confidence", 0.0)
            owner_face_id = cluster.get("owner_face_id", "")
            person_id = cluster.get("person_id", "")
            
            # Store additional metadata in a way that can be used later
            additional_metadata = {
                "confidence_scores": confidence_scores[:10] if isinstance(confidence_scores, list) else [],  # Limit to first 10
                "avg_confidence": avg_confidence,
                "owner_face_id": owner_face_id,
                "person_id": person_id,
                "quality_scores": quality_scores[:5] if isinstance(quality_scores, list) else []  # Limit to first 5
            }
            
            # Debug logging for first few clusters
            if i < 5:
                logger.info(f"Cluster {cluster_id} metadata extraction results:")
                logger.info(f"  - Cluster person_id: {person_id}")
                logger.info(f"  - Cluster owner_face_id: {owner_face_id}")
                logger.info(f"  - Final face_count: {face_count}")
                logger.info(f"  - Extracted {len(image_paths)} image paths: {image_paths[:3]}{'...' if len(image_paths) > 3 else ''}")
                logger.info(f"  - Extracted {len(face_ids)} face IDs: {face_ids[:3]}{'...' if len(face_ids) > 3 else ''}")
                logger.info(f"  - Average confidence: {avg_confidence:.4f}")

            assignments[cluster_id] = {
                "user_id": best_uid,
                "similarity": round(best_sim, 4),
                "face_count": face_count,
                "image_paths": image_paths,
                "face_ids": face_ids,
                "person_id": person_id,
                "owner_face_id": owner_face_id,
                "avg_confidence": round(avg_confidence, 4) if avg_confidence else 0.0,
            }

            if not dry_run:
                to_insert.append(ClusterAssignment(
                    clustering_id=clustering_oid,
                    cluster_id=cluster_id,
                    bucket_path=bucket_path,
                    user_id=best_uid,
                    similarity=best_sim,
                    strategy=strategy,
                    face_ids=face_ids,
                    image_paths=image_paths,
                    face_count=face_count,
                ))

        except Exception as e:
            logger.error("Error processing cluster %s: %s", cluster_id, e)
            unassigned.append(cluster_id)

    # Save assignments
    saved = 0
    if not dry_run and to_insert:
        if overwrite:
            await ClusterAssignmentDB.delete_assignments_by_clustering(clustering_oid)
            logger.info(f"Deleted existing assignments for clustering {clustering_id}")

        for assignment in to_insert:
            try:
                await assignment.insert()
                saved += 1
            except Exception as e:
                logger.warning(f"Insert failed for cluster {assignment.cluster_id}, trying update: {e}")
                try:
                    await ClusterAssignmentDB.update_assignment(
                        clustering_id=assignment.clustering_id,
                        cluster_id=assignment.cluster_id,
                        user_id=assignment.user_id,
                        similarity=assignment.similarity,
                        bucket_path=assignment.bucket_path,
                        strategy=assignment.strategy,
                        face_ids=assignment.face_ids,
                        image_paths=assignment.image_paths,
                        face_count=assignment.face_count,
                    )
                    saved += 1
                except Exception as update_e:
                    logger.error(f"Failed to insert/update assignment for cluster {assignment.cluster_id}: {update_e}")

    result = {
        "clustering_id": str(clustering_oid),
        "collection_name": collection_name,
        "bucket_path": bucket_path,
        "threshold": threshold,
        "strategy": strategy,
        "dry_run": dry_run,
        "total_clusters": len(clusters),
        "assigned_clusters": len(assignments),
        "unassigned_clusters": len(unassigned),
        "assignments": assignments,
        "unassigned_cluster_ids": unassigned,
        "saved_to_db": saved,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    logger.info(f"Assignment completed: {len(assignments)} assigned, {len(unassigned)} unassigned, {saved} saved to DB")
    return result
# Alternative function if you want to process all documents in the collection
async def assign_all_clusterings_to_users(
    collection_name: str = None,  # Make optional
    bucket_name: str = None,      # Add bucket_name parameter
    sub_bucket: str = None,       # Add sub_bucket parameter
    threshold: float = 0.65,
    strategy: str = "centroid",
    dry_run: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Process all clustering documents in the specified collection
    """
    # Construct collection name dynamically if not provided but bucket params are
    if collection_name is None:
        if bucket_name and sub_bucket:
            formatted_bucket = bucket_name.replace('-', '_')
            collection_name = f"person_clustering_{formatted_bucket}_{sub_bucket}"
        else:
            # Fallback to default
            collection_name = "person_clustering_uwas_classification_recette_test"
    
    # Default bucket parameters if not provided
    if bucket_name is None:
        bucket_name = "uwas-classification-recette"
    if sub_bucket is None:
        sub_bucket = "test"
    
    client = None
    try:
        client, database = await _get_direct_database()
        collection = database[collection_name]
        
        # Get all documents
        cursor = collection.find({})
        documents = await cursor.to_list(length=None)
        
        if not documents:
            raise ValueError(f"No clustering documents found in {collection_name}")
        
        results = []
        total_assigned = 0
        total_unassigned = 0
        
        for doc in documents:
            clustering_id = str(doc["_id"])
            logger.info(f"Processing clustering document {clustering_id}")
            
            try:
                result = await assign_clusters_to_users(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    threshold=threshold,
                    strategy=strategy,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    clustering_id=clustering_id,
                    collection_name=collection_name
                )
                results.append(result)
                total_assigned += result["assigned_clusters"]
                total_unassigned += result["unassigned_clusters"]
                
            except Exception as e:
                logger.error(f"Failed to process clustering {clustering_id}: {e}")
                results.append({
                    "clustering_id": clustering_id,
                    "error": str(e)
                })
        
        return {
            "collection_name": collection_name,
            "total_documents_processed": len(documents),
            "total_assigned_clusters": total_assigned,
            "total_unassigned_clusters": total_unassigned,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to process collection {collection_name}: {e}")
        raise
    finally:
        if client:
            client.close()