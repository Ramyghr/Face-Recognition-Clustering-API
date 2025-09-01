# app/services/person_based_clustering_pipeline.py
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import uuid
from datetime import datetime
from app.db.face_clustering_operations import PersonBasedClusteringDB
from app.models.face_clustering_models import PersonClusterInfo

logger = logging.getLogger(__name__)

class PersonBasedClusteringPipeline:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.memory_threshold = 80.0
        
    async def _process_single_file_for_person_clustering(
        self, 
        file: FileItem, 
        bucket_name: str,
        sub_bucket: str,
        skip_quality_filter: bool
    ) -> Optional[Tuple[List[np.ndarray], List[str], Dict[str, Dict]]]:
        """
        Process single file and return ALL faces (not just one per image)
        Returns: (embeddings, face_ids, face_metadata)
        """
        try:
            # Download with timeout
            try:
                image_bytes = await asyncio.wait_for(
                    storage_service.get_file(file.fileKey),
                    timeout=15
                )
            except asyncio.TimeoutError:
                logger.warning(f"[PERSON-CLUSTERING] Timeout downloading {file.fileKey}")
                return None
            except Exception as e:
                logger.warning(f"[PERSON-CLUSTERING] Download failed for {file.fileKey}: {e}")
                return None
            
            # Quick image validation
            if len(image_bytes) > 20 * 1024 * 1024:
                logger.warning(f"[PERSON-CLUSTERING] Image too large: {file.fileKey}")
                return None
            
            # Process image
            try:
                bgr_image = face_detection_service._sanitize_image(image_bytes)
                if bgr_image is None:
                    return None
            except Exception as e:
                logger.warning(f"[PERSON-CLUSTERING] Image processing failed for {file.fileKey}: {e}")
                return None
            
            # Detect faces
            try:
                faces, _ = yolo_detector.detect_faces(bgr_image)
                if not faces:
                    return [], [], {}
            except Exception as e:
                logger.warning(f"[PERSON-CLUSTERING] Face detection failed for {file.fileKey}: {e}")
                return None
            
            # Filter faces by quality if needed
            if not skip_quality_filter:
                try:
                    faces = self._filter_face_quality_fast(faces)
                except Exception as e:
                    logger.warning(f"[PERSON-CLUSTERING] Quality filtering failed for {file.fileKey}: {e}")
            
            if not faces:
                return [], [], {}
            
            # Generate embeddings for ALL faces in this image
            embeddings = []
            face_ids = []
            face_metadata = {}
            
            for face_data in faces:
                try:
                    face_img = face_data["face"]
                    embedding = ai.get_base_embedding(face_img, settings.EMBED_MODEL_PATH, "FACENET")
                    
                    if embedding is not None:
                        face_id = str(uuid.uuid4())
                        
                        bbox = face_data.get("bbox", None)
                        confidence = face_data.get("confidence", None)
                        quality_score = None
                        
                        if bbox and confidence:
                            quality_score = confidence * (bbox[2] * bbox[3])
                        
                        embeddings.append(embedding)
                        face_ids.append(face_id)
                        
                        # Store metadata for each face
                        face_metadata[face_id] = {
                            "image_path": file.fileKey,
                            "bbox": bbox,
                            "confidence": confidence,
                            "quality_score": quality_score
                        }
                        
                except Exception as e:
                    logger.warning(f"[PERSON-CLUSTERING] Failed to process face in {file.fileKey}: {e}")
                    continue
            
            logger.debug(f"[PERSON-CLUSTERING] Processed {file.fileKey}: {len(embeddings)} faces")
            return embeddings, face_ids, face_metadata
            
        except Exception as e:
            logger.error(f"[PERSON-CLUSTERING] Unexpected error processing {file.fileKey}: {e}")
            return None

    async def _process_files_for_person_clustering(
        self, 
        files: List[FileItem], 
        bucket_name: str,
        sub_bucket: str,
        batch_size: int,
        skip_quality_filter: bool,
        max_concurrent: int
    ) -> Tuple[List[np.ndarray], List[str], Dict[str, str], Dict[str, List[float]], Dict[str, Dict]]:
        """
        Process all files and collect ALL faces for person-based clustering
        Returns: all_embeddings, all_face_ids, face_to_image_mapping, embeddings_data, face_metadata
        """
        all_embeddings = []
        all_face_ids = []
        face_to_image_mapping = {}  # face_id -> image_path
        embeddings_data = {}  # face_id -> embedding
        face_metadata = {}  # face_id -> metadata
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file):
            async with semaphore:
                return await self._process_single_file_for_person_clustering(
                    file, bucket_name, sub_bucket, skip_quality_filter
                )
        
        # Process in chunks
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files) - 1) // batch_size + 1
            
            logger.info(f"[PERSON-CLUSTERING] Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[process_with_semaphore(file) for file in batch], return_exceptions=True),
                    timeout=180
                )
            except asyncio.TimeoutError:
                logger.warning(f"[PERSON-CLUSTERING] Batch {batch_num} timed out, skipping")
                continue
            
            # Collect results
            for file, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"[PERSON-CLUSTERING] Error processing {file.fileKey}: {str(result)[:100]}")
                    continue
                
                if result is None:
                    continue
                    
                file_embeddings, file_face_ids, file_face_metadata = result
                
                for embedding, face_id in zip(file_embeddings, file_face_ids):
                    all_embeddings.append(embedding)
                    all_face_ids.append(face_id)
                    face_to_image_mapping[face_id] = file.fileKey
                    embeddings_data[face_id] = embedding.tolist()
                    face_metadata[face_id] = file_face_metadata[face_id]
            
            # Memory management
            if batch_num % 5 == 0:
                self._check_memory()
                logger.info(f"[PERSON-CLUSTERING] Processed {len(all_embeddings)} faces so far...")
        
        logger.info(f"[PERSON-CLUSTERING] Final results: {len(all_embeddings)} total faces from {len(files)} images")
        return all_embeddings, all_face_ids, face_to_image_mapping, embeddings_data, face_metadata

    def _perform_person_clustering(
        self, 
        embeddings: List[np.ndarray], 
        face_ids: List[str],
        face_metadata: Dict[str, Dict]
    ) -> Dict[str, List[str]]:
        """
        FIXED: Perform person-based clustering using proper similarity matching
        Returns: person_id -> [face_ids] mapping
        """
        if not embeddings:
            return {}
        
        try:
            logger.info(f"[PERSON-CLUSTERING] Starting FIXED person clustering with {len(embeddings)} faces")
            
            # Normalize embeddings
            embeddings_array = np.array(embeddings)
            embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
            
            # Calculate full similarity matrix
            similarity_matrix = cosine_similarity(embeddings_normalized)
            
            # Use stricter threshold from settings
            similarity_threshold = getattr(settings, 'PERSON_SIMILARITY_THRESHOLD', 0.75)
            logger.info(f"[PERSON-CLUSTERING] Using similarity threshold: {similarity_threshold}")
            
            # FIXED ALGORITHM: Use graph-based clustering approach
            person_clusters = {}
            face_to_person = {}
            person_counter = 0
            
            # Create adjacency list of similar faces
            similar_faces = defaultdict(set)
            for i in range(len(face_ids)):
                for j in range(i + 1, len(face_ids)):
                    if similarity_matrix[i][j] >= similarity_threshold:
                        similar_faces[i].add(j)
                        similar_faces[j].add(i)
            
            # Use connected components to find person clusters
            visited = set()
            
            for i, face_id in enumerate(face_ids):
                if i in visited:
                    continue
                
                # Start a new person cluster using DFS/BFS
                person_id = f"person_{person_counter}"
                person_counter += 1
                current_cluster = []
                
                # DFS to find all connected faces
                stack = [i]
                while stack:
                    current_idx = stack.pop()
                    if current_idx in visited:
                        continue
                    
                    visited.add(current_idx)
                    current_face_id = face_ids[current_idx]
                    current_cluster.append(current_face_id)
                    face_to_person[current_face_id] = person_id
                    
                    # Add similar faces to stack
                    for similar_idx in similar_faces[current_idx]:
                        if similar_idx not in visited:
                            stack.append(similar_idx)
                
                if current_cluster:
                    person_clusters[person_id] = current_cluster
            
            logger.info(f"[PERSON-CLUSTERING] Created {len(person_clusters)} person clusters")
            
            # Post-processing: Merge clusters with very high similarity between centroids
            merged_clusters = self._merge_similar_person_clusters(
                person_clusters, embeddings_normalized, face_ids, similarity_threshold + 0.1
            )
            
            # Log final statistics
            final_cluster_sizes = [len(faces) for faces in merged_clusters.values()]
            if final_cluster_sizes:
                logger.info(f"[PERSON-CLUSTERING] Final cluster sizes: min={min(final_cluster_sizes)}, "
                        f"max={max(final_cluster_sizes)}, avg={np.mean(final_cluster_sizes):.1f}")
            
            return merged_clusters
            
        except Exception as e:
            logger.error(f"[PERSON-CLUSTERING] Clustering failed: {e}")
            # Fallback: each face is its own person
            return {f"person_{i}": [face_id] for i, face_id in enumerate(face_ids)}

    def _merge_similar_person_clusters(
        self,
        person_clusters: Dict[str, List[str]],
        embeddings_normalized: np.ndarray,
        face_ids: List[str],
        merge_threshold: float = 0.85
    ) -> Dict[str, List[str]]:
        """
        NEW METHOD: Merge person clusters that represent the same person
        """
        if len(person_clusters) < 2:
            return person_clusters
        
        logger.info(f"[PERSON-CLUSTERING] Checking for duplicate persons with threshold {merge_threshold}")
        
        # Calculate centroid for each person cluster
        cluster_centroids = {}
        cluster_ids = list(person_clusters.keys())
        
        for person_id, cluster_face_ids in person_clusters.items():
            # Find indices of faces in this cluster
            face_indices = [face_ids.index(fid) for fid in cluster_face_ids if fid in face_ids]
            
            if face_indices:
                cluster_embeddings = embeddings_normalized[face_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                cluster_centroids[person_id] = centroid
        
        # Calculate similarity between cluster centroids
        centroid_similarities = {}
        for i, person_id1 in enumerate(cluster_ids):
            for j, person_id2 in enumerate(cluster_ids[i+1:], i+1):
                if person_id1 in cluster_centroids and person_id2 in cluster_centroids:
                    similarity = cosine_similarity(
                        [cluster_centroids[person_id1]], 
                        [cluster_centroids[person_id2]]
                    )[0][0]
                    centroid_similarities[(person_id1, person_id2)] = similarity
        
        # Find clusters to merge
        merge_groups = []
        processed = set()
        
        for (person_id1, person_id2), similarity in centroid_similarities.items():
            if similarity >= merge_threshold:
                if person_id1 not in processed and person_id2 not in processed:
                    merge_groups.append([person_id1, person_id2])
                    processed.add(person_id1)
                    processed.add(person_id2)
                    logger.info(f"[PERSON-CLUSTERING] Merging {person_id1} and {person_id2} (similarity: {similarity:.3f})")
        
        # Perform merging
        merged_clusters = {}
        
        # Add non-merged clusters
        for person_id in cluster_ids:
            if person_id not in processed:
                merged_clusters[person_id] = person_clusters[person_id]
        
        # Merge similar clusters
        for merge_group in merge_groups:
            if len(merge_group) < 2:
                continue
            
            # Use first person_id as the target
            target_person_id = merge_group[0]
            merged_face_ids = list(person_clusters[target_person_id])
            
            # Merge other clusters into target
            for person_id in merge_group[1:]:
                merged_face_ids.extend(person_clusters[person_id])
            
            # Remove duplicates
            merged_clusters[target_person_id] = list(set(merged_face_ids))
        
        logger.info(f"[PERSON-CLUSTERING] Merge complete: {len(person_clusters)} -> {len(merged_clusters)} clusters")
        return merged_clusters

    def _enhance_person_clusters(
        self,
        person_clusters: Dict[str, List[str]],
        face_metadata: Dict[str, Dict],
        embeddings_data: Dict[str, List[float]],
        face_to_image_mapping: Dict[str, str]
    ) -> Dict[str, Dict]:
        """
        Enhance person clusters with owner selection and statistics
        Returns: Enhanced cluster data with owners and stats
        """
        enhanced_clusters = {}
        
        for person_id, face_ids in person_clusters.items():
            if not face_ids:
                continue
            
            # Collect data for this person
            confidences = []
            quality_scores = []
            image_paths = []
            
            for face_id in face_ids:
                metadata = face_metadata.get(face_id, {})
                image_path = face_to_image_mapping.get(face_id)
                
                if image_path:
                    image_paths.append(image_path)
                    confidences.append(metadata.get("confidence", 0.0))
                    quality_scores.append(metadata.get("quality_score", 0.0))
            
            if not image_paths:
                continue
            
            # Select owner face (best quality + confidence)
            best_score = -1
            owner_face_id = face_ids[0]
            owner_embedding = embeddings_data.get(face_ids[0], [])
            
            for face_id in face_ids:
                metadata = face_metadata.get(face_id, {})
                confidence = metadata.get("confidence", 0.0)
                quality = metadata.get("quality_score", 0.0)
                
                # Combined score favoring quality and confidence
                combined_score = (quality * 0.6) + (confidence * 0.4)
                
                if combined_score > best_score:
                    best_score = combined_score
                    owner_face_id = face_id
                    owner_embedding = embeddings_data.get(face_id, [])
            
            enhanced_clusters[person_id] = {
                "face_ids": face_ids,
                "owner_face_id": owner_face_id,
                "owner_embedding": owner_embedding,
                "image_paths": list(set(image_paths)),  # Remove duplicates
                "confidence_scores": confidences,
                "quality_scores": quality_scores,
                "total_detections": len(face_ids),
                "unique_images": len(set(image_paths))
            }
        
        logger.info(f"[PERSON-CLUSTERING] Enhanced {len(enhanced_clusters)} person clusters")
        return enhanced_clusters

    async def process_person_based_clustering_pipeline(
        self,
        files: List[FileItem],
        bucket_name: str,
        sub_bucket: str,
        batch_size: int = 30,
        skip_quality_filter: bool = False,
        max_concurrent: int = 12,
    ) -> Dict:
        """
        Complete person-based clustering pipeline
        """
        import gc
        gc.collect()
        start_time = time.time()
        logger.info("[PERSON-CLUSTERING] Starting pipeline with %d files for %s/%s",
                    len(files), bucket_name, sub_bucket)

        self._check_memory()

        try:
            # 1) Process all files to get ALL faces
            all_embeddings, all_face_ids, face_to_image_mapping, embeddings_data, face_metadata = \
                await self._process_files_for_person_clustering(
                    files, bucket_name, sub_bucket, batch_size, skip_quality_filter, max_concurrent
                )

            if len(all_embeddings) < 1:
                logger.warning("[PERSON-CLUSTERING] No valid faces found")
                return self._empty_person_result(files, start_time)

            # 2) Perform person-based clustering
            person_clusters = self._perform_person_clustering(
                all_embeddings, all_face_ids, face_metadata
            )

            if not person_clusters:
                logger.warning("[PERSON-CLUSTERING] No person clusters created")
                return self._empty_person_result(files, start_time)

            # 3) Enhance clusters with owners and statistics
            enhanced_clusters = self._enhance_person_clusters(
                person_clusters, face_metadata, embeddings_data, face_to_image_mapping
            )

            # 4) Prepare results
            result = self._format_person_clustering_results(
                enhanced_clusters, face_to_image_mapping, all_face_ids, start_time
            )

            # 5) Save to databases
            try:
                # Create face_id_mapping for compatibility (image_path -> face_id)
                # For images with multiple faces, we'll use the owner face
                face_id_mapping = {}
                for person_id, cluster_data in enhanced_clusters.items():
                    for image_path in cluster_data["image_paths"]:
                        if image_path not in face_id_mapping:
                            # Find the face_id for this image and person
                            for face_id in cluster_data["face_ids"]:
                                if face_to_image_mapping.get(face_id) == image_path:
                                    face_id_mapping[image_path] = face_id
                                    break

                # Add unassigned images
                for face_id, image_path in face_to_image_mapping.items():
                    if image_path not in face_id_mapping:
                        face_id_mapping[image_path] = face_id

                clustering_id = await PersonBasedClusteringDB.save_complete_person_clustering_pipeline(
                    bucket_name=bucket_name,
                    sub_bucket=sub_bucket,
                    person_clusters=enhanced_clusters,
                    unassigned_faces=result.get("unassigned", []),
                    face_id_mapping=face_id_mapping,
                    face_metadata=face_metadata,
                    embeddings_data=embeddings_data,
                    processing_stats=result["stats"],
                )

                if clustering_id:
                    result["stats"]["clustering_id"] = str(clustering_id)
                    result["stats"]["database_saved"] = True
                else:
                    result["stats"]["database_saved"] = False
                    
            except Exception as e:
                logger.error("[PERSON-CLUSTERING] Persistence failed: %s", e, exc_info=True)
                result["stats"]["database_saved"] = False

            result["stats"]["processing_time_seconds"] = round(time.time() - start_time, 2)
            return result

        except Exception as e:
            logger.error("[PERSON-CLUSTERING] Pipeline failed: %s", e, exc_info=True)
            return self._empty_person_result(files, start_time, error=str(e))

    def _format_person_clustering_results(
        self, 
        enhanced_clusters: Dict[str, Dict],
        face_to_image_mapping: Dict[str, str],
        all_face_ids: List[str],
        start_time: float
    ) -> Dict:
        """Format results for person-based clustering"""
        
        # Create person clusters for response
        person_clusters = []
        all_clustered_images = set()
        
        for person_id, cluster_data in enhanced_clusters.items():
            person_cluster = {
                "person_id": person_id,
                "image_paths": cluster_data["image_paths"],
                "total_appearances": cluster_data["total_detections"],
                "unique_images": cluster_data["unique_images"],
                "owner_face_id": cluster_data["owner_face_id"],
                "avg_confidence": np.mean(cluster_data["confidence_scores"]) if cluster_data["confidence_scores"] else 0.0,
                "best_quality": max(cluster_data["quality_scores"]) if cluster_data["quality_scores"] else 0.0
            }
            person_clusters.append(person_cluster)
            all_clustered_images.update(cluster_data["image_paths"])
        
        # Find unassigned faces/images
        unassigned = []
        for face_id in all_face_ids:
            image_path = face_to_image_mapping.get(face_id)
            if image_path and image_path not in all_clustered_images:
                unassigned.append(image_path)
        
        # Remove duplicates
        unassigned = list(set(unassigned))
        
        # Calculate overlap statistics
        image_person_count = defaultdict(int)
        for cluster_data in enhanced_clusters.values():
            for image_path in cluster_data["image_paths"]:
                image_person_count[image_path] += 1
        
        single_person_images = sum(1 for count in image_person_count.values() if count == 1)
        multi_person_images = sum(1 for count in image_person_count.values() if count > 1)
        max_persons_per_image = max(image_person_count.values()) if image_person_count else 0
        
        # Sort person clusters by number of appearances (descending)
        person_clusters.sort(key=lambda x: x["total_appearances"], reverse=True)
        
        logger.info(f"[PERSON-CLUSTERING] Results: {len(person_clusters)} persons, {len(unassigned)} unassigned")
        logger.info(f"[PERSON-CLUSTERING] Image distribution: {single_person_images} single-person, {multi_person_images} multi-person")
        
        return {
            "person_clusters": person_clusters,
            "unassigned": unassigned,
            "stats": {
                "total_persons": len(person_clusters),
                "total_images": len(set(face_to_image_mapping.values())),
                "total_faces": len(all_face_ids),
                "unassigned_faces": len(unassigned),
                "single_person_images": single_person_images,
                "multi_person_images": multi_person_images,
                "max_persons_per_image": max_persons_per_image,
                "avg_faces_per_person": np.mean([c["total_appearances"] for c in person_clusters]) if person_clusters else 0,
                "database_saves": "enabled",
                "timestamp": datetime.utcnow(),
                "clustering_type": "person_based_with_overlaps"
            }
        }

    def _empty_person_result(self, files: List[FileItem], start_time: float, error: str = None) -> Dict:
        """Return empty result structure for person-based clustering"""
        result = {
            "person_clusters": [],
            "unassigned": [f.fileKey for f in files],
            "stats": {
                "total_persons": 0,
                "total_images": len(files),
                "total_faces": 0,
                "unassigned_faces": len(files),
                "single_person_images": 0,
                "multi_person_images": 0,
                "max_persons_per_image": 0,
                "avg_faces_per_person": 0,
                "database_saves": "enabled",
                "timestamp": datetime.utcnow(),
                "clustering_type": "person_based_with_overlaps",
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        }
        
        if error:
            result["stats"]["pipeline_error"] = error
            
        return result

    def _filter_face_quality_fast(self, faces: List[Dict]) -> List[Dict]:
        """Fast quality filtering for faces"""
        if not faces:
            return faces
        
        scored_faces = []
        for face_data in faces:
            try:
                bbox = face_data.get("bbox", [0, 0, 0, 0])
                confidence = face_data.get("confidence", 0)
                
                if len(bbox) >= 4:
                    area = bbox[2] * bbox[3]
                    
                    if (bbox[2] >= getattr(settings, 'MIN_FACE_SIZE', 50) and 
                        bbox[3] >= getattr(settings, 'MIN_FACE_SIZE', 50) and 
                        confidence >= getattr(settings, 'MIN_CONFIDENCE', 0.5)):
                        
                        quality_score = confidence * area
                        scored_faces.append((quality_score, face_data))
            except Exception as e:
                logger.warning(f"[PERSON-CLUSTERING] Error in quality filtering: {e}")
                continue
        
        # Sort by quality and return top faces
        scored_faces.sort(key=lambda x: x[0], reverse=True)
        return [face_data for _, face_data in scored_faces[:5]]  # Max 5 faces per image

    def _check_memory(self):
        """Check memory usage and clean up if needed"""
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.memory_threshold:
                logger.warning(f"[PERSON-CLUSTERING] High memory usage: {memory_percent:.1f}%")
                gc.collect()
        except Exception as e:
            logger.warning(f"[PERSON-CLUSTERING] Memory check failed: {e}")

# Create singleton instance
person_clustering_pipeline = PersonBasedClusteringPipeline()