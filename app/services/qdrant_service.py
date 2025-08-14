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