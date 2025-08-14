import logging
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from app.db.face_clustering_operations import FaceClusteringDB
from app.db.cluster_assignment_operations import ClusterAssignmentDB
from app.models.user_profile import UserProfile
from app.models.cluster_assignments import ClusterAssignment
from app.services.qdrant_service import qdrant_service
from app.db.face_clustering_operations import (
    FaceClusteringDB,
    get_clustering_result_model,  # <-- REQUIRED for targeting a specific clustering_id
)
from bson import ObjectId

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _centroid(vectors: List[List[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros(128, dtype=np.float32)
    arr = np.asarray(vectors, dtype=np.float32)
    c = arr.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

async def _load_latest_clustering(bucket_name: str, sub_bucket: str):
    """
    Load the latest clustering document for bucket/sub_bucket via DB helper.
    Avoid importing static models (collections are dynamic per bucket+subBucket).
    """
    doc = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
    if not doc:
        bp = f"{bucket_name}/{sub_bucket}"
        raise ValueError(f"No clustering results found for {bp}")
    return doc


async def _load_user_profiles() -> Dict[str, np.ndarray]:
    """
    Load user profiles and return {user_id: normalized_embedding}.
    Skips users with missing/empty embeddings.
    """
    users = [u async for u in UserProfile.find_all()]
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


async def assign_clusters_to_users(
    bucket_name: str,
    sub_bucket: str,
    threshold: float = 0.65,
    strategy: str = "centroid",
    dry_run: bool = False,
    overwrite: bool = False,
    clustering_id: str | None = None,   # <-- NEW
) -> Dict[str, Any]:
    """
    Assign anonymous clusters to users by cosine similarity of cluster centroids.

    If `clustering_id` is provided, assign using that specific clustering document
    from the dynamic collection for (bucket_name, sub_bucket). Otherwise, use the
    latest clustering for that bucket/sub_bucket.
    """
    logger.info(
        "Starting cluster assignment for %s/%s (threshold=%.3f, strategy=%s, dry_run=%s, overwrite=%s, clustering_id=%s)",
        bucket_name, sub_bucket, threshold, strategy, dry_run, overwrite, clustering_id
    )

    # -------------------------------
    # 1) Load clustering document
    # -------------------------------
    if clustering_id:
        # Resolve from the dynamic collection for this bucket/sub-bucket
        Model = get_clustering_result_model(bucket_name, sub_bucket)
        if not ObjectId.is_valid(clustering_id):
            raise ValueError(f"Invalid clustering_id: {clustering_id}")
        doc = await Model.find_one(Model.id == ObjectId(clustering_id))
        if not doc:
            raise ValueError(
                f"Clustering {clustering_id} not found for {bucket_name}/{sub_bucket}"
            )
        clustering = doc
    else:
        # Fallback to the latest
        clustering = await FaceClusteringDB.get_latest_clustering_result(bucket_name, sub_bucket)
        if not clustering:
            raise ValueError(f"No clustering results found for {bucket_name}/{sub_bucket}")

    clustering_oid = clustering.id
    bucket_path = getattr(clustering, "bucket_path", f"{bucket_name}/{sub_bucket}")

    # -------------------------------
    # 2) Load & normalize user vectors
    # -------------------------------
    users = [u async for u in UserProfile.find_all()]
    if not users:
        raise ValueError("No user profiles found")

    user_vecs: Dict[str, np.ndarray] = {}
    for u in users:
        emb = getattr(u, "embedding", None)
        if isinstance(emb, list) and emb:
            v = np.array(emb, dtype=np.float32)
            n = np.linalg.norm(v)
            user_vecs[u.user_id] = (v / n) if n > 0 else v
    if not user_vecs:
        raise ValueError("All user profiles lack embeddings")

    # -------------------------------
    # 3) For each cluster: fetch vectors, compute centroid, match user
    # -------------------------------
    assignments: Dict[str, Dict[str, Any]] = {}
    unassigned: List[str] = []
    to_insert: List[ClusterAssignment] = []

    for c in (clustering.clusters or []):
        cid = c.cluster_id
        size = getattr(c, "size", 0)

        try:
            pts = qdrant_service.get_embeddings_by_cluster_and_clustering_id(
                clustering_id=str(clustering_oid),
                cluster_id=cid
            )

            # Normalize Qdrant result (can be dict or typed object)
            vectors: List[List[float]] = []
            for p in pts:
                vec = p["vector"] if isinstance(p, dict) else getattr(p, "vector", None)
                if vec:
                    vectors.append(vec)

            if not vectors:
                logger.info("Cluster %s has no vectors in Qdrant; leaving unassigned", cid)
                unassigned.append(cid)
                continue

            # centroid & best user
            arr = np.asarray(vectors, dtype=np.float32)
            cvec = arr.mean(axis=0)
            n = np.linalg.norm(cvec)
            cvec = cvec / n if n > 0 else cvec

            best_uid, best_sim = None, -1.0
            for uid, uvec in user_vecs.items():
                sim = float(np.dot(cvec, uvec))  # both unit vectors
                if sim > best_sim:
                    best_sim, best_uid = sim, uid

            if best_uid is None or best_sim < threshold:
                unassigned.append(cid)
                continue

            assignments[cid] = {
                "user_id": best_uid,
                "similarity": round(best_sim, 4),
                "face_count": size,
                "image_paths": (getattr(c, "image_paths", []) or [])[:10],
                "face_ids": (getattr(c, "face_ids", []) or [])[:10],
            }

            if not dry_run:
                to_insert.append(ClusterAssignment(
                    clustering_id=clustering_oid,
                    cluster_id=cid,
                    bucket_path=bucket_path,
                    user_id=best_uid,
                    similarity=best_sim,
                    strategy=strategy,
                    face_ids=getattr(c, "face_ids", []) or [],
                    image_paths=getattr(c, "image_paths", []) or [],
                    face_count=size,
                ))

        except Exception as e:
            logger.error("Error processing cluster %s: %s", cid, e)
            unassigned.append(cid)

    # -------------------------------
    # 4) Persist (optional)
    # -------------------------------
    saved = 0
    if not dry_run and to_insert:
        if overwrite:
            await ClusterAssignmentDB.delete_assignments_by_clustering(clustering_oid)
        for a in to_insert:
            try:
                await a.insert()
                saved += 1
            except Exception:
                await ClusterAssignmentDB.update_assignment(
                    clustering_id=a.clustering_id,
                    cluster_id=a.cluster_id,
                    user_id=a.user_id,
                    similarity=a.similarity,
                    bucket_path=a.bucket_path,
                    strategy=a.strategy,
                    face_ids=a.face_ids,
                    image_paths=a.image_paths,
                    face_count=a.face_count,
                )
                saved += 1

        # try to annotate clustering with a tiny summary
        try:
            clustering.processing_stats = clustering.processing_stats or {}
            clustering.processing_stats["last_assignment"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "threshold": threshold,
                "assigned": len(assignments),
                "unassigned": len(unassigned),
            }
            await clustering.save()
        except Exception:
            pass

    # -------------------------------
    # 5) Response
    # -------------------------------
    return {
        "clustering_id": str(clustering_oid),
        "bucket_path": bucket_path,
        "threshold": threshold,
        "strategy": strategy,
        "dry_run": dry_run,
        "total_clusters": len(clustering.clusters or []),
        "assigned_clusters": len(assignments),
        "unassigned_clusters": len(unassigned),
        "assignments": assignments,
        "unassigned_cluster_ids": unassigned,
        "saved_to_db": saved,
        "timestamp": datetime.utcnow().isoformat(),
    }
