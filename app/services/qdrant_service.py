import logging
import os
import time
import uuid
from typing import List, Dict, Any, Optional

from bson import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.collection_name = "face_embeddings"
        self.client = self._initialize_client()
        self._ensure_collection_compatibility()

    # ---------- Client init ----------

    def _initialize_client(self):
        """Initialize Qdrant client with multiple fallback options."""
        for attempt in range(self.max_retries):
            try:
                # Prefer URL if provided (e.g., cloud or reverse proxy)
                if getattr(settings, "QDRANT_URL", None):
                    client = self._create_url_client()
                    if client:
                        return client

                # Try host/port (typical Docker mapped port)
                if getattr(settings, "QDRANT_HOST", None) and getattr(settings, "QDRANT_PORT", None):
                    client = self._create_host_port_client()
                    if client:
                        return client

                # Fallback to embedded/local file storage
                return self._create_local_client()

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error("Failed to initialize Qdrant client after %d attempts", self.max_retries)
                    raise RuntimeError(f"Could not connect to Qdrant: {str(e)}")
                logger.warning("Qdrant connection attempt %d failed: %s", attempt + 1, str(e)[:200])
                time.sleep(self.retry_delay)
    def get_embeddings_by_person_and_clustering_id(self, clustering_id: str, person_id: str):
        """Get all embeddings for a specific person in a clustering"""
        try:
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}},
                    {"key": "person_id", "match": {"value": person_id}}
                ]
            }
            
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                with_payload=True,
                with_vectors=True,
                limit=1000
            )
            
            if response and response[0]:
                return [
                    {
                        "id": point.id,
                        "vector": point.vector,
                        "payload": point.payload
                    }
                    for point in response[0]
                ]
            return []
            
        except Exception as e:
            logger.error(f"Error getting embeddings for person {person_id}: {e}")
            return []

    def get_person_statistics(self, clustering_id: str) -> Dict[str, Any]:
        """Get statistics about persons in a clustering"""
        try:
            # Get all points for this clustering
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}}
                ]
            }
            
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                with_payload=True,
                with_vectors=False,
                limit=10000
            )
            
            if not response or not response[0]:
                return {"total_embeddings": 0, "persons": {}}
            
            person_stats = {}
            total_embeddings = len(response[0])
            
            for point in response[0]:
                payload = point.payload
                person_id = payload.get("person_id", "unknown")
                is_owner = payload.get("is_owner_face", "false").lower() == "true"
                
                if person_id not in person_stats:
                    person_stats[person_id] = {
                        "total_embeddings": 0,
                        "owner_faces": 0,
                        "avg_confidence": 0,
                        "confidences": []
                    }
                
                person_stats[person_id]["total_embeddings"] += 1
                if is_owner:
                    person_stats[person_id]["owner_faces"] += 1
                    
                try:
                    confidence = float(payload.get("confidence", "0"))
                    person_stats[person_id]["confidences"].append(confidence)
                except:
                    pass
            
            # Calculate averages
            for person_id, stats in person_stats.items():
                if stats["confidences"]:
                    stats["avg_confidence"] = sum(stats["confidences"]) / len(stats["confidences"])
                del stats["confidences"]  # Remove raw data
            
            return {
                "total_embeddings": total_embeddings,
                "unique_persons": len(person_stats),
                "unassigned_faces": person_stats.get("unassigned", {}).get("total_embeddings", 0),
                "person_distribution": dict(sorted(
                    person_stats.items(), 
                    key=lambda x: x[1]["total_embeddings"], 
                    reverse=True
                ))
            }
            
        except Exception as e:
            logger.error(f"Error getting person statistics: {e}")
            return {"total_embeddings": 0, "persons": {}}

    def search_similar_faces_across_persons(self, query_embedding: List[float], clustering_id: str, limit: int = 10) -> List[Dict]:
        """Search for similar faces across all persons in a clustering"""
        try:
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}}
                ]
            }
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for hit in search_result:
                results.append({
                    "face_id": hit.payload.get("face_id"),
                    "person_id": hit.payload.get("person_id"),
                    "image_path": hit.payload.get("image_path"),
                    "is_owner_face": hit.payload.get("is_owner_face", "false").lower() == "true",
                    "similarity_score": hit.score,
                    "confidence": float(hit.payload.get("confidence", "0")),
                    "quality_score": float(hit.payload.get("quality_score", "0"))
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []

    def get_owner_faces_for_clustering(self, clustering_id: str) -> List[Dict]:
        """Get all owner faces (representative faces) for persons in a clustering"""
        try:
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}},
                    {"key": "is_owner_face", "match": {"value": "true"}}
                ]
            }
            
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                with_payload=True,
                with_vectors=True,
                limit=1000
            )
            
            if response and response[0]:
                return [
                    {
                        "face_id": point.payload.get("face_id"),
                        "person_id": point.payload.get("person_id"),
                        "image_path": point.payload.get("image_path"),
                        "embedding": point.vector,
                        "confidence": float(point.payload.get("confidence", "0")),
                        "quality_score": float(point.payload.get("quality_score", "0"))
                    }
                    for point in response[0]
                ]
            return []
            
        except Exception as e:
            logger.error(f"Error getting owner faces: {e}")
            return []

    def update_person_assignments(self, face_ids: List[str], new_person_id: str, clustering_id: str) -> bool:
        """Update person assignments for specific faces"""
        try:
            # First, get the points to update
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}},
                    {"key": "face_id", "match": {"any": face_ids}}
                ]
            }
            
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                with_payload=True,
                with_vectors=False,
                limit=len(face_ids)
            )
            
            if not response or not response[0]:
                logger.warning(f"No points found for face_ids: {face_ids}")
                return False
            
            # Update each point
            points_to_update = []
            for point in response[0]:
                updated_payload = point.payload.copy()
                updated_payload["person_id"] = new_person_id
                # Reset owner status when reassigning
                updated_payload["is_owner_face"] = "false"
                
                points_to_update.append({
                    "id": point.id,
                    "payload": updated_payload
                })
            
            if points_to_update:
                # Batch update
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={},  # Will be overridden by individual payloads
                    points=points_to_update
                )
                
                logger.info(f"Updated {len(points_to_update)} faces to person {new_person_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating person assignments: {e}")
            return False

    def delete_person_data(self, clustering_id: str, person_id: str) -> bool:
        """Delete all data for a specific person in a clustering"""
        try:
            filter_conditions = {
                "must": [
                    {"key": "clustering_id", "match": {"value": clustering_id}},
                    {"key": "person_id", "match": {"value": person_id}}
                ]
            }
            
            # Delete points matching the filter
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_conditions
            )
            
            logger.info(f"Deleted person data for {person_id} in clustering {clustering_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting person data: {e}")
            return False

    def get_clustering_overview(self, clustering_id: str) -> Dict[str, Any]:
        """Get comprehensive overview of a person-based clustering"""
        try:
            person_stats = self.get_person_statistics(clustering_id)
            owner_faces = self.get_owner_faces_for_clustering(clustering_id)
            
            # Calculate additional metrics
            total_persons = len([p for p in person_stats.get("person_distribution", {}).keys() if p != "unassigned"])
            total_faces = person_stats.get("total_embeddings", 0)
            unassigned_faces = person_stats.get("unassigned_faces", 0)
            
            # Get top persons by appearance count
            top_persons = []
            for person_id, stats in list(person_stats.get("person_distribution", {}).items())[:10]:
                if person_id != "unassigned":
                    owner_face = next((f for f in owner_faces if f["person_id"] == person_id), None)
                    top_persons.append({
                        "person_id": person_id,
                        "total_appearances": stats["total_embeddings"],
                        "avg_confidence": round(stats["avg_confidence"], 3),
                        "owner_face_id": owner_face["face_id"] if owner_face else None,
                        "owner_image_path": owner_face["image_path"] if owner_face else None
                    })
            
            return {
                "clustering_id": clustering_id,
                "total_persons": total_persons,
                "total_faces": total_faces,
                "unassigned_faces": unassigned_faces,
                "assignment_rate": round((total_faces - unassigned_faces) / total_faces * 100, 1) if total_faces > 0 else 0,
                "avg_faces_per_person": round(total_faces / total_persons, 1) if total_persons > 0 else 0,
                "top_persons": top_persons,
                "owner_faces_count": len(owner_faces)
            }
            
        except Exception as e:
            logger.error(f"Error getting clustering overview: {e}")
            return {
                "clustering_id": clustering_id,
                "error": str(e)
            }
    def _create_url_client(self) -> Optional[QdrantClient]:
        try:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=getattr(settings, "QDRANT_API_KEY", None),
                prefer_grpc=False,
                timeout=10,
            )
            client.get_collections()  # probe
            logger.info("Connected to Qdrant at %s", settings.QDRANT_URL)
            return client
        except Exception as e:
            logger.warning("URL connection failed: %s", e)
            return None

    def _create_host_port_client(self) -> Optional[QdrantClient]:
        try:
            client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                prefer_grpc=False,
                timeout=10,
            )
            client.get_collections()  # probe
            logger.info("Connected to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)
            return client
        except Exception as e:
            logger.warning("Host/Port connection failed: %s", e)
            return None

    def _create_local_client(self) -> QdrantClient:
        storage_path = getattr(settings, "QDRANT_STORAGE_PATH", "./qdrant_data")
        os.makedirs(storage_path, exist_ok=True)
        client = QdrantClient(path=storage_path, prefer_grpc=False)
        logger.info("Using local Qdrant storage at %s", storage_path)
        return client

    # ---------- Collections ----------

    def _ensure_collection_compatibility(self):
        """Ensure collection exists with correct vector size (128, cosine)."""
        try:
            collections = self.client.get_collections()
            exists = any(col.name == self.collection_name for col in collections.collections)
            if not exists:
                logger.info("Creating collection %s", self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
                    timeout=10,
                )
            else:
                logger.info("Using existing collection %s", self.collection_name)

            info = self.client.get_collection(self.collection_name)
            # Handle both single and named vectors
            vectors_cfg = info.config.params.vectors
            if hasattr(vectors_cfg, "size"):
                size = vectors_cfg.size
            elif isinstance(vectors_cfg, dict):
                # default unnamed
                size = vectors_cfg.get("", VectorParams(size=0, distance=Distance.COSINE)).size
            else:
                size = 0

            if size != 128:
                raise ValueError(f"Incompatible vector size in collection: {size}")

        except Exception as e:
            logger.error("Collection verification failed: %s", e)
            raise RuntimeError(f"Could not verify Qdrant collection: {e}")

    # ---------- Health ----------

    def health_check(self) -> Dict[str, Any]:
        try:
            collections = self.client.get_collections()
            target = next((c for c in collections.collections if c.name == self.collection_name), None)
            storage = "remote" if (getattr(settings, "QDRANT_URL", None) or
                                   (getattr(settings, "QDRANT_HOST", None) and getattr(settings, "QDRANT_PORT", None))) else "local"
            return {
                "status": "healthy",
                "collection_count": len(collections.collections),
                "target_collection_exists": target is not None,
                "storage_type": storage,
                "details": {
                    "collection_names": [c.name for c in collections.collections],
                    "target_collection_size": getattr(target, "points_count", 0) if target else 0,
                },
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    def get_embeddings_by_clustering_id(self, clustering_id: str) -> List[Dict]:
        try:
            out = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="clustering_id", match=MatchValue(value=clustering_id))]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            return [self._point_to_dict(p) for p in out[0]]
        except Exception as e:
            logger.error("Error getting embeddings by clustering_id %s: %s", clustering_id, e)
            return []

    def add_face_embeddings_with_payloads(
        self,
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> bool:
        """Batch insert vectors + payloads."""
        try:
            if len(embeddings) != len(payloads):
                logger.error("Embeddings and payloads length mismatch")
                return False

            points = []
            for emb, pay in zip(embeddings, payloads):
                points.append({"id": str(uuid.uuid4()), "vector": emb, "payload": pay})

            batch = 100
            for i in range(0, len(points), batch):
                chunk = points[i:i + batch]
                op = self.client.upsert(collection_name=self.collection_name, points=chunk, wait=True)
                if getattr(op, "status", None) != "completed":
                    logger.error("Qdrant upsert failed for batch %d", (i // batch) + 1)
                    return False

            logger.info("Inserted %d embeddings to Qdrant", len(points))
            return True
        except Exception as e:
            logger.error("Error adding embeddings with payloads: %s", e)
            return False

    def get_clustering_stats(self, clustering_id: str) -> Dict[str, int]:
        """Get statistics for a specific clustering result"""
        try:
            embeddings = self.get_embeddings_by_clustering_id(clustering_id)
            
            cluster_counts = {}
            total_embeddings = len(embeddings)
            noise_embeddings = 0
            
            for embedding in embeddings:
                cluster_id = embedding["payload"].get("cluster_id", "unknown")
                if cluster_id == "noise":
                    noise_embeddings += 1
                else:
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            return {
                "total_embeddings": total_embeddings,
                "clusters_with_embeddings": len(cluster_counts),
                "noise_embeddings": noise_embeddings,
                "cluster_distribution": cluster_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting clustering stats for {clustering_id}: {e}")
            return {}

    def get_embeddings_by_cluster_and_clustering_id(self, clustering_id: str, cluster_id: str) -> List[Dict]:
        try:
            out = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="clustering_id", match=MatchValue(value=clustering_id)),
                        FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id)),
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=True,
            )
            return [self._point_to_dict(p) for p in out[0]]
        except Exception as e:
            logger.error(
                "Error getting embeddings for cluster %s in clustering %s: %s",
                cluster_id, clustering_id, e
            )
            return []

    def get_all_face_ids(self) -> List[str]:
        try:
            face_ids: List[str] = []
            out = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            for p in out[0]:
                payload = p.get("payload") if isinstance(p, dict) else getattr(p, "payload", {}) or {}
                fid = payload.get("face_id")
                if fid:
                    face_ids.append(fid)
            return face_ids
        except Exception as e:
            logger.error("Error getting all face IDs: %s", e)
            return []

    def add_face_embeddings(
        self,
        clustering_id: str,
        bucket_name: str,
        sub_bucket: str,
        clusters: List[List[str]],
        noise: List[str],
        embeddings_data: Dict[str, List[float]],
        face_id_mapping: Dict[str, str],
    ) -> bool:
        """Insert embeddings by clusters + noise with payloads."""
        try:
            points: List[PointStruct] = []

            # clusters
            for idx, imgs in enumerate(clusters):
                cid = f"cluster_{idx}"
                for img_path in imgs:
                    fid = face_id_mapping.get(img_path)
                    emb = embeddings_data.get(fid) if fid else None
                    if not fid or not emb:
                        continue
                    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{fid}_{clustering_id}"))
                    points.append(PointStruct(
                        id=pid,
                        vector=emb,
                        payload={
                            "face_id": fid,
                            "image_path": f"{bucket_name}/{img_path}",
                            "cluster_id": cid,
                            "clustering_id": clustering_id,
                            "bucket_path": f"{bucket_name}/{sub_bucket}",
                            "bucket_name": bucket_name,
                            "sub_bucket": sub_bucket,
                        }
                    ))

            # noise
            for img_path in noise:
                fid = face_id_mapping.get(img_path)
                emb = embeddings_data.get(fid) if fid else None
                if not fid or not emb:
                    continue
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{fid}_{clustering_id}"))
                points.append(PointStruct(
                    id=pid,
                    vector=emb,
                    payload={
                        "face_id": fid,
                        "image_path": f"{bucket_name}/{img_path}",
                        "cluster_id": "noise",
                        "clustering_id": clustering_id,
                        "bucket_path": f"{bucket_name}/{sub_bucket}",
                        "bucket_name": bucket_name,
                        "sub_bucket": sub_bucket,
                    }
                ))

            # upsert in batches
            for i in range(0, len(points), 100):
                batch = points[i:i + 100]
                self.client.upsert(collection_name=self.collection_name, points=batch, wait=True)

            logger.info("Saved %d embeddings to Qdrant for clustering %s", len(points), clustering_id)
            return True
        except Exception as e:
            logger.error("Qdrant insertion error: %s", e)
            return False

    def _fallback_individual_inserts(self, points: List[PointStruct]) -> bool:
        """Fallback method to insert points individually"""
        success_count = 0
        for point in points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to insert individual point {point.id}: {e}")
                continue
        
        logger.info(f"Fallback insertion: {success_count}/{len(points)} points inserted")
        return success_count > 0

    async def update_clustering_ids(self, face_ids: List[str], clustering_id: str) -> bool:
        try:
            for fid in face_ids:
                out = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=[FieldCondition(key="face_id", match=MatchValue(value=fid))]),
                    limit=1,
                    with_payload=True,
                    with_vectors=True,
                )
                if out[0]:
                    p = out[0][0]
                    pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
                    vec = p.get("vector") if isinstance(p, dict) else getattr(p, "vector", None)
                    pay = p.get("payload") if isinstance(p, dict) else getattr(p, "payload", {}) or {}
                    pay["clustering_id"] = clustering_id
                    self.client.upsert(self.collection_name, points=[PointStruct(id=pid, vector=vec, payload=pay)])
            return True
        except Exception as e:
            logger.error("Error updating clustering IDs: %s", e)
            return False


    async def update_cluster_assignments(
        self, face_ids: List[str], cluster_ids: List[str], clustering_id: str
    ) -> bool:
        try:
            if len(face_ids) != len(cluster_ids):
                logger.error("Mismatched face_ids and cluster_ids lengths")
                return False

            for fid, cid in zip(face_ids, cluster_ids):
                out = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(must=[FieldCondition(key="face_id", match=MatchValue(value=fid))]),
                    limit=1,
                    with_payload=True,
                    with_vectors=True,
                )
                if out[0]:
                    p = out[0][0]
                    pid = p.get("id") if isinstance(p, dict) else getattr(p, "id", None)
                    vec = p.get("vector") if isinstance(p, dict) else getattr(p, "vector", None)
                    pay = p.get("payload") if isinstance(p, dict) else getattr(p, "payload", {}) or {}
                    pay["cluster_id"] = cid
                    pay["clustering_id"] = clustering_id
                    self.client.upsert(self.collection_name, points=[PointStruct(id=pid, vector=vec, payload=pay)])
            return True
        except Exception as e:
            logger.error("Error updating cluster assignments: %s", e)
            return False

    def delete_by_face_id(self, face_id: str) -> bool:
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="face_id", match=MatchValue(value=face_id))]),
            )
            return True
        except Exception as e:
            logger.error("Error deleting face_id %s: %s", face_id, e)
            return False

    def get_embeddings_by_cluster_id(self, cluster_id: str) -> List[Dict]:
        """Get embeddings by cluster_id"""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
                ),
                limit=10000
            )
            return [self._point_to_dict(point) for point in result[0]]
        except Exception as e:
            logger.error(f"Error getting embeddings by cluster_id {cluster_id}: {e}")
            return []

    def get_embeddings_by_face_ids(self, face_ids: List[str]) -> List[Dict]:
        try:
            out = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    should=[FieldCondition(key="face_id", match=MatchValue(value=fid)) for fid in face_ids]
                ),
                limit=max(len(face_ids), 100),
                with_payload=True,
                with_vectors=True,
            )
            return [self._point_to_dict(p) for p in out[0]]
        except Exception as e:
            logger.error("Error getting embeddings by face IDs: %s", e)
            return []

    def delete_clustering_data(self, clustering_id: str) -> Optional[str]:
        try:
            op = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="clustering_id", match=MatchValue(value=clustering_id))]),
                wait=True,
            )
            if getattr(op, "status", None) == "completed":
                return str(getattr(op, "operation_id", "completed"))
            return None
        except Exception as e:
            logger.error("Error deleting clustering data %s: %s", clustering_id, e)
            return None

    def delete_by_bucket_path(self, bucket_path: str) -> bool:
        try:
            parts = bucket_path.split("/")
            bucket = parts[0]
            sub = parts[1] if len(parts) > 1 else ""
            op = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="bucket_name", match=MatchValue(value=bucket)),
                        FieldCondition(key="sub_bucket", match=MatchValue(value=sub)),
                    ]
                ),
                wait=True,
            )
            return getattr(op, "status", None) == "completed"
        except Exception as e:
            logger.error("Error deleting by bucket path %s: %s", bucket_path, e)
            return False


    @staticmethod
    def _point_to_dict(point) -> Dict[str, Any]:
        """Normalize Qdrant point (object or dict) into a dict."""
        if isinstance(point, dict):
            return {
                "id": point.get("id"),
                "vector": point.get("vector"),
                "payload": point.get("payload") or {},
            }
        pid = getattr(point, "id", None)
        vec = getattr(point, "vector", None)
        pay = getattr(point, "payload", {}) or {}
        return {"id": pid, "vector": vec, "payload": pay}

    

# Create singleton instance
qdrant_service = QdrantService()