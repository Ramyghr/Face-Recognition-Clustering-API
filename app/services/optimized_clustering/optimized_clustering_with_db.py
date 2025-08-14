# app/services/optimized_clustering_with_db.py (updated)
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import gc
import psutil
from bson import ObjectId
from app.services.qdrant_service import qdrant_service
from app.schemas.clustering import FileItem
from app.services.S3_functions import storage_service
from app.services import yolo_detector, ai
from app.services.face_detection_service import face_detection_service
from app.core.config import settings
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import uuid
from datetime import datetime
# Import the fixed database models
from app.models.face_clustering import FaceClusteringDB

logger = logging.getLogger(__name__)

class OptimizedClusteringPipelineWithDB:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.memory_threshold = 80.0
        
    async def _save_embeddings_to_qdrant(
        self,
        clustering_id: str,
        bucket_name: str,
        sub_bucket: str,
        clusters: List[List[str]],
        noise: List[str],
        embeddings_data: dict,
        face_id_mapping: dict
    ):
        """Save ALL embeddings to Qdrant with proper cluster assignments"""
        try:
            logger.info(f"[QDRANT] Starting to save embeddings for clustering {clustering_id}")
            
            # Prepare data for batch insertion
            embeddings = []
            face_ids = []
            image_paths = []
            cluster_ids = []
            
            # Process clusters
            for cluster_idx, cluster_images in enumerate(clusters):
                cluster_id = f"cluster_{cluster_idx}"
                
                for image_path in cluster_images:
                    face_id = face_id_mapping.get(image_path)
                    if not face_id:
                        logger.warning(f"No face_id found for image_path: {image_path}")
                        continue
                    
                    embedding = embeddings_data.get(face_id)
                    if not embedding:
                        logger.warning(f"No embedding found for face_id: {face_id}")
                        continue
                    
                    embeddings.append(embedding)
                    face_ids.append(face_id)
                    image_paths.append(f"{bucket_name}/{image_path}")  # Full path
                    cluster_ids.append(cluster_id)
            
            # Process noise
            for image_path in noise:
                face_id = face_id_mapping.get(image_path)
                if not face_id:
                    logger.warning(f"No face_id found for noise image_path: {image_path}")
                    continue
                
                embedding = embeddings_data.get(face_id)
                if not embedding:
                    logger.warning(f"No embedding found for noise face_id: {face_id}")
                    continue
                
                embeddings.append(embedding)
                face_ids.append(face_id)
                image_paths.append(f"{bucket_name}/{image_path}")  # Full path
                cluster_ids.append("noise")
            
            # Batch insert into Qdrant
            if embeddings:
                logger.info(f"[QDRANT] Attempting to save {len(embeddings)} embeddings")
                success = qdrant_service.add_face_embeddings(
                    embeddings=embeddings,
                    face_ids=face_ids,
                    image_paths=image_paths,
                    cluster_ids=cluster_ids,
                    clustering_id=clustering_id
                )
                
                if success:
                    logger.info(f"[QDRANT] Successfully saved {len(embeddings)} embeddings")
                    
                    # Update MongoDB with cluster assignments
                    cluster_assignments = {face_id: cluster_id for face_id, cluster_id in zip(face_ids, cluster_ids)}
                    await FaceClusteringDB.update_face_embeddings_with_cluster_assignments(
                        bucket_name, sub_bucket, cluster_assignments, ObjectId(clustering_id)
                    )
                    return True
                else:
                    logger.error(f"[QDRANT] Failed to save embeddings")
                    return False
            else:
                logger.warning(f"[QDRANT] No valid embeddings to save")
                return False
                
        except Exception as e:
            logger.error(f"[QDRANT] Error saving embeddings: {e}")
            return False

    
    async def _process_single_file_fast_with_db_fixed(
        self, 
        file: FileItem, 
        bucket_name: str,
        sub_bucket: str,
        skip_quality_filter: bool
    ) -> Optional[Tuple[List[np.ndarray], int, Optional[str], Dict]]:
        """
        Ultra-fast single file processing with fixed database storage
        Returns: (embeddings, face_count, face_id, metadata)
        """
        try:
            # Download with timeout
            try:
                image_bytes = await asyncio.wait_for(
                    storage_service.get_file(file.fileKey),
                    timeout=15
                )
            except asyncio.TimeoutError:
                logger.warning(f"[OPTIMIZED-DB] Timeout downloading {file.fileKey}")
                return None
            except Exception as e:
                logger.warning(f"[OPTIMIZED-DB] Download failed for {file.fileKey}: {e}")
                return None
            
            # Quick image validation
            if len(image_bytes) > 20 * 1024 * 1024:
                logger.warning(f"[OPTIMIZED-DB] Image too large: {file.fileKey}")
                return None
            
            # Process image
            try:
                bgr_image = face_detection_service._sanitize_image(image_bytes)
                if bgr_image is None:
                    return None
            except Exception as e:
                logger.warning(f"[OPTIMIZED-DB] Image processing failed for {file.fileKey}: {e}")
                return None
            
            # Detect faces with timeout
            try:
                faces, _ = yolo_detector.detect_faces(bgr_image)
                if not faces:
                    return [], 0, None, {}
            except Exception as e:
                logger.warning(f"[OPTIMIZED-DB] Face detection failed for {file.fileKey}: {e}")
                return None
            
            original_face_count = len(faces)
            
            # Filter faces by quality if needed
            if not skip_quality_filter:
                try:
                    faces = self._filter_face_quality_fast(faces)
                except Exception as e:
                    logger.warning(f"[OPTIMIZED-DB] Quality filtering failed for {file.fileKey}: {e}")
            
            if not faces:
                return [], original_face_count, None, {}
            
            # Generate embeddings - take only the best face for speed
            best_face = faces[0]
            face_img = best_face["face"]
            
            # Fast embedding generation without augmentation
            try:
                embedding = ai.get_base_embedding(face_img, settings.EMBED_MODEL_PATH, "FACENET")
                if embedding is None:
                    return [], original_face_count, None, {}
            except Exception as e:
                logger.warning(f"[OPTIMIZED-DB] Embedding generation failed for {file.fileKey}: {e}")
                return [], original_face_count, None, {}
            
            # Create face_id and metadata (DON'T save to MongoDB yet)
            face_id = str(uuid.uuid4())
            bbox = best_face.get("bbox", None)
            confidence = best_face.get("confidence", None)
            quality_score = None
            
            if bbox and confidence:
                quality_score = confidence * (bbox[2] * bbox[3])
            
            # Prepare metadata
            metadata = {
                "bbox": bbox,
                "confidence": confidence,
                "quality_score": quality_score
            }
            
            logger.debug(f"[OPTIMIZED-DB] Successfully processed {file.fileKey} with face_id {face_id}")
            return [embedding], original_face_count, face_id, metadata
            
        except Exception as e:
            logger.error(f"[OPTIMIZED-DB] Unexpected error processing {file.fileKey}: {e}")
            return None
    async def _process_files_optimized_with_db_fixed(
        self, 
        files: List[FileItem], 
        bucket_name: str,
        sub_bucket: str,
        batch_size: int,
        skip_quality_filter: bool,
        max_concurrent: int
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, int], Dict[str, str], Dict[str, List[float]], Dict[str, Dict]]:
        """
        Ultra-optimized file processing with fixed database storage
        Returns: embeddings, file_keys, face_counts, face_id_mapping, embeddings_data, face_metadata
        """
        embeddings = []
        file_keys = []
        face_counts = {}
        face_id_mapping = {}
        embeddings_data = {}  # face_id -> embedding
        face_metadata = {}    # face_id -> metadata
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Process all files with controlled concurrency
        async def process_with_semaphore(file):
            async with semaphore:
                return await self._process_single_file_fast_with_db_fixed(
                    file, bucket_name, sub_bucket, skip_quality_filter
                )
        
        # Process in chunks to avoid memory issues
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files) - 1) // batch_size + 1
            
            logger.info(f"[OPTIMIZED-DB] Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            # Process batch with timeout and error handling
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[process_with_semaphore(file) for file in batch], return_exceptions=True),
                    timeout=180
                )
            except asyncio.TimeoutError:
                logger.warning(f"[OPTIMIZED-DB] Batch {batch_num} timed out, skipping")
                continue
            
            # Collect results
            for file, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"[OPTIMIZED-DB] Error processing {file.fileKey}: {str(result)[:100]}")
                    continue
                
                if result is None:
                    continue
                    
                file_embeddings, face_count, face_id, metadata = result
                if file_embeddings and len(file_embeddings) > 0 and face_id:
                    # Take best embedding per image
                    best_embedding = file_embeddings[0]
                    embeddings.append(best_embedding)
                    file_keys.append(file.fileKey)
                    face_counts[file.fileKey] = face_count
                    face_id_mapping[file.fileKey] = face_id
                    
                    # Store embedding and metadata separately
                    embeddings_data[face_id] = best_embedding.tolist()
                    face_metadata[face_id] = metadata
            
            # Memory management between batches
            if batch_num % 5 == 0:
                self._check_memory()
                logger.info(f"[OPTIMIZED-DB] Processed {len(embeddings)} faces so far...")
        
        logger.info(f"[OPTIMIZED-DB] Final results: {len(embeddings)} embeddings, {len(embeddings_data)} ready for storage")
        return embeddings, file_keys, face_counts, face_id_mapping, embeddings_data, face_metadata

# app/services/optimized_clustering_with_db.py - FIXED PIPELINE FUNCTION

    async def process_optimized_pipeline_with_db_fixed(
        self,
        files: List[FileItem],
        bucket_name: str,
        sub_bucket: str,
        batch_size: int = 30,
        skip_quality_filter: bool = False,
        max_concurrent: int = 12,
    ) -> Dict:
        """
        Ultra-optimized clustering pipeline with FIXED database storage:
        - Stats contain a real datetime `timestamp`
        - Delegates persistence to FaceClusteringDB.save_complete_clustering_pipeline_fixed
        """
        import gc
        gc.collect()
        start_time = time.time()
        logger.info("[OPTIMIZED-DB] Starting FIXED pipeline with %d files for %s/%s",
                    len(files), bucket_name, sub_bucket)

        self._check_memory()

        default_result = {
            "clusters": [],
            "noise": [f.fileKey for f in files],
            "stats": {
                "total_images": 0,
                "total_faces": 0,
                "num_clusters": 0,
                "noise_count": 0,
                "avg_faces_per_image": 0,
                "database_saves": "enabled",
                "timestamp": datetime.utcnow(),  # <- real datetime
            },
        }

        try:
            # 1) process files
            embeddings, file_keys, face_counts, face_id_mapping, embeddings_data, face_metadata = \
                await self._process_files_optimized_with_db_fixed(
                    files, bucket_name, sub_bucket, batch_size, skip_quality_filter, max_concurrent
                )

            if len(embeddings) < 1:
                logger.warning("[OPTIMIZED-DB] No valid faces found for clustering")
                default_result["stats"]["processing_time_seconds"] = round(time.time() - start_time, 2)
                return default_result

            if len(embeddings) == 1:
                # single cluster edge case
                single = {
                    "clusters": [file_keys],
                    "noise": [],
                    "stats": {
                        "total_images": len(file_keys),
                        "total_faces": 1,
                        "num_clusters": 1,
                        "noise_count": 0,
                        "avg_faces_per_image": 1.0,
                        "database_saves": "enabled",
                        "timestamp": datetime.utcnow(),
                    },
                }
                try:
                    clustering_id = await FaceClusteringDB.save_complete_clustering_pipeline_fixed(
                        bucket_name=bucket_name,
                        sub_bucket=sub_bucket,
                        embeddings_data=embeddings_data,
                        face_metadata=face_metadata,
                        clusters=single["clusters"],
                        noise=single["noise"],
                        face_id_mapping=face_id_mapping,
                        processing_stats=single["stats"],
                    )
                    if clustering_id:
                        single["stats"]["clustering_id"] = str(clustering_id)
                        single["stats"]["database_saved"] = True
                except Exception as e:
                    logger.error("[OPTIMIZED-DB] Single-face persistence failed: %s", e)
                    single["stats"]["database_saved"] = False

                single["stats"]["processing_time_seconds"] = round(time.time() - start_time, 2)
                return single

            logger.info("[OPTIMIZED-DB] Generated %d embeddings from %d images", len(embeddings), len(file_keys))

            # 2) cluster
            labels = await self._fast_clustering_fixed(embeddings)

            # 3) format results
            result = self._format_clustering_results(file_keys, labels, face_counts)
            result["stats"]["timestamp"] = datetime.utcnow()  # <- real datetime

            # 4) persist (fixed saver writes valid datetimes)
            try:
                clustering_id = await FaceClusteringDB.save_complete_clustering_pipeline_fixed(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    embeddings_data=embeddings_data,
                    face_metadata=face_metadata,
                    clusters=result["clusters"],
                    noise=result["noise"],
                    face_id_mapping=face_id_mapping,
                    processing_stats=result["stats"],
                )
                if clustering_id:
                    result["stats"]["clustering_id"] = str(clustering_id)
                    result["stats"]["database_saved"] = True
                else:
                    result["stats"]["database_saved"] = False
            except Exception as e:
                logger.error("[OPTIMIZED-DB] Persistence failed: %s", e, exc_info=True)
                result["stats"]["database_saved"] = False

            result["stats"]["processing_time_seconds"] = round(time.time() - start_time, 2)
            return result

        except Exception as e:
            logger.error("[OPTIMIZED-DB] FIXED pipeline failed: %s", e, exc_info=True)
            default_result["stats"]["pipeline_error"] = str(e)
            default_result["stats"]["processing_time_seconds"] = round(time.time() - start_time, 2)
            return default_result

    
    def _filter_face_quality_fast(self, faces: List[Dict]) -> List[Dict]:
        """
        Ultra-fast quality filtering
        """
        if not faces:
            return faces
        
        # Sort by quality score (confidence * area) and take top faces
        scored_faces = []
        for face_data in faces:
            try:
                bbox = face_data.get("bbox", [0, 0, 0, 0])
                confidence = face_data.get("confidence", 0)
                
                if len(bbox) >= 4:
                    area = bbox[2] * bbox[3]
                    
                    # Quick quality check
                    if (bbox[2] >= getattr(settings, 'MIN_FACE_SIZE', 50) and 
                        bbox[3] >= getattr(settings, 'MIN_FACE_SIZE', 50) and 
                        confidence >= getattr(settings, 'MIN_CONFIDENCE', 0.5)):
                        
                        quality_score = confidence * area
                        scored_faces.append((quality_score, face_data))
            except Exception as e:
                logger.warning(f"[OPTIMIZED-DB] Error in quality filtering: {e}")
                continue
        
        # Sort by quality and return top 3 faces max
        scored_faces.sort(key=lambda x: x[0], reverse=True)
        return [face_data for _, face_data in scored_faces[:3]]
    
    async def _fast_clustering_fixed(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Fixed clustering algorithm that handles edge cases properly
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        try:
            # Normalize embeddings
            embeddings_array = np.array(embeddings)
            embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings_normalized)
            
            # Convert to distance matrix and ensure non-negative values
            distance_matrix = 1 - similarity_matrix
            distance_matrix = np.clip(distance_matrix, 0, 2)
            
            # Fill diagonal with zeros
            np.fill_diagonal(distance_matrix, 0)
            
            # Ensure matrix is symmetric
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            
            # Validate distance matrix
            if np.any(distance_matrix < 0) or np.any(np.isnan(distance_matrix)):
                logger.warning("[OPTIMIZED-DB] Invalid distance matrix, using fallback clustering")
                return self._fallback_clustering(embeddings_normalized)
            
            # Use DBSCAN with precomputed distances
            clustering = DBSCAN(
                eps=getattr(settings, 'CLUSTERING_DISTANCE_THRESHOLD', 0.4),
                min_samples=2,
                metric='precomputed'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Post-process clusters
            cluster_labels = self._merge_small_clusters_fast(
                embeddings_normalized, 
                cluster_labels
            )
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"[OPTIMIZED-DB] Clustering failed: {e}, using fallback")
            return self._fallback_clustering(embeddings_normalized)
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simple fallback clustering using euclidean distance
        """
        try:
            clustering = DBSCAN(
                eps=0.5,
                min_samples=2,
                metric='euclidean'
            )
            return clustering.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"[OPTIMIZED-DB] Fallback clustering failed: {e}")
            # Return each item as its own cluster
            return np.arange(len(embeddings))
    
    def _merge_small_clusters_fast(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Fast cluster merging with simplified logic
        """
        try:
            unique_labels = np.unique(labels)
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels if label != -1}
            
            min_cluster_size = getattr(settings, 'MIN_CLUSTER_SIZE', 3)
            merge_threshold = getattr(settings, 'MERGE_THRESHOLD', 0.3)
            
            # Skip merging if we have too many clusters (performance optimization)
            if len(unique_labels) > 50:
                return labels
            
            new_labels = labels.copy()
            
            for current_label in unique_labels:
                if current_label == -1 or cluster_sizes.get(current_label, 0) >= min_cluster_size:
                    continue
                
                # Find closest larger cluster
                current_mask = labels == current_label
                current_centroid = np.mean(embeddings[current_mask], axis=0)
                
                best_similarity = -1
                best_target = None
                
                for target_label in unique_labels:
                    if (target_label == current_label or 
                        target_label == -1 or 
                        cluster_sizes.get(target_label, 0) < min_cluster_size):
                        continue
                    
                    target_mask = labels == target_label
                    target_centroid = np.mean(embeddings[target_mask], axis=0)
                    
                    # Use cosine similarity
                    similarity = np.dot(current_centroid, target_centroid) / (
                        np.linalg.norm(current_centroid) * np.linalg.norm(target_centroid)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = target_label
                
                # Merge if similarity is high enough
                if best_target is not None and best_similarity > (1 - merge_threshold):
                    new_labels[new_labels == current_label] = best_target
            
            return new_labels
            
        except Exception as e:
            logger.warning(f"[OPTIMIZED-DB] Cluster merging failed: {e}")
            return labels
    
    def _format_clustering_results(
        self, 
        file_keys: List[str], 
        cluster_labels: np.ndarray,
        face_counts: Dict[str, int]
    ) -> Dict:
        """
        Format clustering results with comprehensive statistics
        """
        cluster_groups = defaultdict(list)
        
        # Group files by cluster
        for file_key, label in zip(file_keys, cluster_labels):
            cluster_groups[int(label)].append(file_key)
        
        # Separate clusters and noise
        clusters = []
        noise = []
        
        for label, files in cluster_groups.items():
            if label == -1:
                noise.extend(files)
            else:
                clusters.append(files)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        # Comprehensive statistics
        total_faces = sum(face_counts.values())
        avg_faces_per_image = total_faces / len(file_keys) if file_keys else 0
        
        # Cluster size distribution
        cluster_sizes = [len(cluster) for cluster in clusters]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        logger.info(f"[OPTIMIZED-DB] Results: {len(clusters)} clusters, {len(noise)} noise files")
        logger.info(f"[OPTIMIZED-DB] Total faces: {total_faces}, avg per image: {avg_faces_per_image:.2f}")
        logger.info(f"[OPTIMIZED-DB] Average cluster size: {avg_cluster_size:.2f}")
        
        return {
            "clusters": clusters,
            "noise": noise,
            "stats": {
                "total_images": len(file_keys),
                "total_faces": total_faces,
                "num_clusters": len(clusters),
                "noise_count": len(noise),
                "avg_faces_per_image": round(avg_faces_per_image, 2),
                "avg_cluster_size": round(avg_cluster_size, 2),
                "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "cluster_size_distribution": {
                    "small (2-5)": len([s for s in cluster_sizes if 2 <= s <= 5]),
                    "medium (6-15)": len([s for s in cluster_sizes if 6 <= s <= 15]),
                    "large (16+)": len([s for s in cluster_sizes if s >= 16])
                },
                "database_saves": "enabled"
            }
        }
        
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            "total_images": 0,
            "total_faces": 0,
            "num_clusters": 0,
            "noise_count": 0,
            "avg_faces_per_image": 0,
            "processing_time_seconds": 0,
            "database_saves": "enabled"
        }
    
    def _check_memory(self):
        """Check memory usage and clean up if needed"""
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.memory_threshold:
                logger.warning(f"[OPTIMIZED-DB] High memory usage: {memory_percent:.1f}%")
                gc.collect()
        except Exception as e:
            logger.warning(f"[OPTIMIZED-DB] Memory check failed: {e}")
    
    def _cleanup_memory(self, *objects):
        """Clean up large objects"""
        try:
            for obj in objects:
                if obj is not None:
                    del obj
            gc.collect()
        except Exception as e:
            logger.warning(f"[OPTIMIZED-DB] Memory cleanup failed: {e}")

# Create singleton instance
optimized_pipeline_with_db = OptimizedClusteringPipelineWithDB()
