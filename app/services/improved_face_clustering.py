# app/services/improved_face_clustering.py
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import gc
import psutil
import cv2

from app.schemas.clustering import FileItem
from app.services.S3_functions import storage_service
from app.services import yolo_detector, ai
from app.db.user_profile_operations import face_detection_service
from app.core.config import settings
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
from scipy import ndimage

logger = logging.getLogger(__name__)

class ImprovedFaceClusteringPipeline:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.memory_threshold = 80.0
        
        # Enhanced clustering parameters
        self.similarity_threshold = 0.88  # Increased from 0.85
        self.merge_threshold = 0.3        # Decreased from 0.35
        self.cluster_prob_threshold = 0.6  # Decreased from 0.65
        self.min_confidence = 0.75        # Increased from 0.7
        self.blur_threshold = 180         # Increased from 150
        self.enable_face_alignment = True
        self.min_cluster_size = 2
        self.max_faces_per_image = 1      # Only use best face per image
        
    async def process_enhanced_pipeline(
        self, 
        files: List[FileItem], 
        batch_size: int = 25,  # Slightly reduced for better quality control
        skip_quality_filter: bool = False,
        max_concurrent: int = 10  # Reduced for more stable processing
    ) -> Dict:
        """
        Enhanced clustering pipeline with improved accuracy and noise handling
        """
        start_time = time.time()
        logger.info(f"[ENHANCED] Starting pipeline with {len(files)} files")
        
        # Memory management
        self._check_memory()
        
        # Step 1: Process files with enhanced quality filtering
        embeddings, file_keys, face_data = await self._process_files_enhanced(
            files, batch_size, skip_quality_filter, max_concurrent
        )
        
        if len(embeddings) < 2:
            logger.warning("[ENHANCED] Not enough valid faces for clustering")
            return {"clusters": [], "noise": file_keys, "stats": self._empty_stats()}
        
        logger.info(f"[ENHANCED] Generated {len(embeddings)} high-quality embeddings")
        
        # Step 2: Enhanced clustering with better separation
        cluster_labels = await self._enhanced_clustering(embeddings, face_data)
        
        # Step 3: Post-process with intelligent noise detection
        cluster_labels = self._post_process_clusters(embeddings, cluster_labels, face_data)
        
        # Step 4: Format results
        result = self._format_enhanced_results(file_keys, cluster_labels, face_data)
        
        total_time = time.time() - start_time
        logger.info(f"[ENHANCED] Pipeline completed in {total_time:.2f} seconds")
        result["stats"]["processing_time_seconds"] = round(total_time, 2)
        
        # Cleanup
        self._cleanup_memory(embeddings)
        
        return result
    
    async def _process_files_enhanced(
        self, 
        files: List[FileItem], 
        batch_size: int,
        skip_quality_filter: bool,
        max_concurrent: int
    ) -> Tuple[List[np.ndarray], List[str], List[Dict]]:
        """
        Enhanced file processing with strict quality control
        """
        embeddings = []
        file_keys = []
        face_data = []  # Store additional face metadata
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(file):
            async with semaphore:
                return await self._process_single_file_enhanced(file, skip_quality_filter)
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(files) - 1) // batch_size + 1
            
            logger.info(f"[ENHANCED] Processing batch {batch_num}/{total_batches}")
            
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[process_with_semaphore(file) for file in batch], 
                                 return_exceptions=True),
                    timeout=150  # Increased timeout for quality processing
                )
            except asyncio.TimeoutError:
                logger.warning(f"[ENHANCED] Batch {batch_num} timed out")
                continue
            
            # Collect results with enhanced validation
            for file, result in zip(batch, batch_results):
                if isinstance(result, Exception) or result is None:
                    continue
                    
                embedding, metadata = result
                if embedding is not None and self._validate_embedding_quality(embedding, metadata):
                    embeddings.append(embedding)
                    file_keys.append(file.fileKey)
                    face_data.append(metadata)
            
            if batch_num % 3 == 0:
                self._check_memory()
        
        return embeddings, file_keys, face_data
    
    async def _process_single_file_enhanced(
        self, 
        file: FileItem, 
        skip_quality_filter: bool
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Enhanced single file processing with strict quality checks
        """
        try:
            # Download with timeout
            try:
                image_bytes = await asyncio.wait_for(
                    storage_service.get_file(file.fileKey),
                    timeout=20
                )
            except asyncio.TimeoutError:
                logger.warning(f"[ENHANCED] Timeout downloading {file.fileKey}")
                return None
            
            # Enhanced image validation
            if len(image_bytes) > 15 * 1024 * 1024:  # 15MB limit
                logger.warning(f"[ENHANCED] Image too large: {file.fileKey}")
                return None
            
            # Process image
            bgr_image = face_detection_service._sanitize_image(image_bytes)
            if bgr_image is None:
                return None
            
            # Detect faces
            faces, _ = yolo_detector.detect_faces(bgr_image)
            if not faces:
                return None
            
            # Enhanced quality filtering
            if not skip_quality_filter:
                faces = self._enhanced_quality_filter(faces, bgr_image)
            
            if not faces:
                return None
            
            # Take only the best face
            best_face = faces[0]
            face_img = best_face["face"]
            
            # Enhanced face preprocessing
            if self.enable_face_alignment:
                face_img = self._align_face(face_img)
            
            # Generate embedding with validation
            embedding = ai.get_base_embedding(face_img, settings.EMBED_MODEL_PATH, "FACENET")
            
            if embedding is not None:
                # Create enhanced metadata
                metadata = {
                    "file_key": file.fileKey,
                    "confidence": best_face.get("confidence", 0),
                    "bbox": best_face.get("bbox", [0, 0, 0, 0]),
                    "blur_score": self._calculate_blur_score(face_img),
                    "face_size": face_img.shape[0] * face_img.shape[1] if len(face_img.shape) > 1 else 0,
                    "quality_score": self._calculate_quality_score(best_face, face_img)
                }
                
                return embedding, metadata
            
            return None
            
        except Exception as e:
            logger.warning(f"[ENHANCED] Error processing {file.fileKey}: {str(e)[:100]}")
            return None
    
    def _enhanced_quality_filter(self, faces: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Enhanced quality filtering with multiple criteria
        """
        if not faces:
            return faces
        
        high_quality_faces = []
        
        for face_data in faces:
            bbox = face_data["bbox"]
            confidence = face_data["confidence"]
            face_img = face_data["face"]
            
            # Basic size and confidence checks
            if (bbox[2] < settings.MIN_FACE_SIZE or 
                bbox[3] < settings.MIN_FACE_SIZE or 
                confidence < self.min_confidence):
                continue
            
            # Blur detection
            blur_score = self._calculate_blur_score(face_img)
            if blur_score < self.blur_threshold:
                logger.debug(f"Face rejected due to blur: {blur_score}")
                continue
            
            # Face area ratio check
            image_area = image.shape[0] * image.shape[1]
            face_area = bbox[2] * bbox[3]
            face_ratio = face_area / image_area
            
            if face_ratio < 0.01:  # Face too small relative to image
                continue
            
            # Aspect ratio check (faces should be roughly square)
            aspect_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                continue
            
            # Calculate composite quality score
            quality_score = (
                confidence * 0.4 +
                (blur_score / 300.0) * 0.3 +
                min(face_ratio * 100, 1.0) * 0.3
            )
            
            face_data["quality_score"] = quality_score
            high_quality_faces.append(face_data)
        
        # Sort by quality and return top faces
        high_quality_faces.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        return high_quality_faces[:self.max_faces_per_image]
    
    def _calculate_blur_score(self, face_img: np.ndarray) -> float:
        """
        Calculate blur score using Laplacian variance
        """
        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0.0
    
    def _calculate_quality_score(self, face_data: Dict, face_img: np.ndarray) -> float:
        """
        Calculate composite quality score
        """
        confidence = face_data.get("confidence", 0)
        bbox = face_data.get("bbox", [0, 0, 0, 0])
        
        # Size score
        face_area = bbox[2] * bbox[3]
        size_score = min(face_area / (100 * 100), 1.0)  # Normalize to 100x100
        
        # Blur score
        blur_score = min(self._calculate_blur_score(face_img) / 300.0, 1.0)
        
        # Composite score
        return confidence * 0.5 + size_score * 0.25 + blur_score * 0.25
    
    def _align_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Simple face alignment using eye detection (placeholder)
        """
        # This is a simplified alignment - in production you'd use landmark detection
        try:
            # Basic histogram equalization for better contrast
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(gray)
                return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            else:
                return cv2.equalizeHist(face_img)
        except:
            return face_img
    
    def _validate_embedding_quality(self, embedding: np.ndarray, metadata: Dict) -> bool:
        """
        Validate embedding quality based on metadata
        """
        if embedding is None or len(embedding) == 0:
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        # Check embedding magnitude (shouldn't be too small)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 0.1:
            return False
        
        # Check quality score threshold
        quality_score = metadata.get("quality_score", 0)
        if quality_score < 0.3:  # Minimum quality threshold
            return False
        
        return True
    
    async def _enhanced_clustering(
        self, 
        embeddings: List[np.ndarray], 
        face_data: List[Dict]
    ) -> np.ndarray:
        """
        Enhanced clustering with better separation control
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        try:
            # Normalize embeddings
            embeddings_array = np.array(embeddings)
            embeddings_normalized = normalize(embeddings_array, norm='l2', axis=1)
            
            # Calculate cosine distance matrix
            distance_matrix = cosine_distances(embeddings_normalized)
            
            # Enhanced hierarchical clustering with stricter parameters
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,  # Convert similarity to distance
                metric='precomputed',
                linkage='average'  # Average linkage for more balanced clusters
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Enhanced cluster post-processing
            cluster_labels = self._enhanced_merge_clusters(
                embeddings_normalized, 
                cluster_labels, 
                face_data
            )
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"[ENHANCED] Clustering failed: {e}")
            return self._fallback_clustering(embeddings_normalized)
    
    def _enhanced_merge_clusters(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray,
        face_data: List[Dict]
    ) -> np.ndarray:
        """
        Enhanced cluster merging with quality-aware decisions
        """
        try:
            unique_labels = np.unique(labels)
            cluster_info = {}
            
            # Calculate cluster information including quality metrics
            for label in unique_labels:
                if label == -1:
                    continue
                    
                mask = labels == label
                cluster_embeddings = embeddings[mask]
                cluster_faces = [face_data[i] for i in np.where(mask)[0]]
                
                cluster_info[label] = {
                    "size": len(cluster_embeddings),
                    "centroid": np.mean(cluster_embeddings, axis=0),
                    "avg_quality": np.mean([f.get("quality_score", 0) for f in cluster_faces]),
                    "avg_confidence": np.mean([f.get("confidence", 0) for f in cluster_faces]),
                    "embeddings": cluster_embeddings
                }
            
            new_labels = labels.copy()
            
            # Merge small clusters with quality considerations
            for current_label in unique_labels:
                if current_label == -1 or current_label not in cluster_info:
                    continue
                    
                current_info = cluster_info[current_label]
                
                # Skip if cluster is large enough
                if current_info["size"] >= self.min_cluster_size:
                    continue
                
                # Find best merge candidate
                best_similarity = -1
                best_target = None
                
                for target_label, target_info in cluster_info.items():
                    if (target_label == current_label or 
                        target_info["size"] < self.min_cluster_size):
                        continue
                    
                    # Calculate similarity between centroids
                    similarity = np.dot(current_info["centroid"], target_info["centroid"])
                    
                    # Quality bonus for high-quality clusters
                    quality_bonus = min(target_info["avg_quality"] * 0.1, 0.05)
                    adjusted_similarity = similarity + quality_bonus
                    
                    if adjusted_similarity > best_similarity:
                        best_similarity = adjusted_similarity
                        best_target = target_label
                
                # Merge with stricter threshold
                if (best_target is not None and 
                    best_similarity > (1 - self.merge_threshold)):
                    new_labels[new_labels == current_label] = best_target
                    logger.debug(f"Merged cluster {current_label} into {best_target} "
                               f"(similarity: {best_similarity:.3f})")
            
            return new_labels
            
        except Exception as e:
            logger.warning(f"[ENHANCED] Cluster merging failed: {e}")
            return labels
    
    def _post_process_clusters(
        self, 
        embeddings: List[np.ndarray], 
        labels: np.ndarray,
        face_data: List[Dict]
    ) -> np.ndarray:
        """
        Post-process clusters with intelligent noise detection
        """
        embeddings_array = np.array(embeddings)
        new_labels = labels.copy()
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
                
            mask = labels == label
            cluster_embeddings = embeddings_array[mask]
            cluster_faces = [face_data[i] for i in np.where(mask)[0]]
            
            if len(cluster_embeddings) < 2:
                continue
            
            # Calculate cluster centroid and cohesion
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_to_centroid = [
                1 - np.dot(emb, centroid) for emb in cluster_embeddings
            ]
            
            # Identify outliers within clusters
            mean_distance = np.mean(distances_to_centroid)
            std_distance = np.std(distances_to_centroid)
            outlier_threshold = mean_distance + 2 * std_distance
            
            indices = np.where(mask)[0]
            for i, (dist, face_info) in enumerate(zip(distances_to_centroid, cluster_faces)):
                # Mark as noise if too far from centroid and low quality
                if (dist > outlier_threshold and 
                    face_info.get("quality_score", 1.0) < self.cluster_prob_threshold):
                    new_labels[indices[i]] = -1
                    logger.debug(f"Moved {face_info['file_key']} to noise "
                               f"(distance: {dist:.3f}, quality: {face_info.get('quality_score', 0):.3f})")
        
        return new_labels
    
    def _fallback_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fallback clustering with conservative parameters
        """
        try:
            # Use DBSCAN as fallback with conservative parameters
            clustering = DBSCAN(
                eps=0.4,  # Conservative epsilon
                min_samples=2,
                metric='cosine'
            )
            return clustering.fit_predict(embeddings)
        except Exception as e:
            logger.error(f"[ENHANCED] Fallback clustering failed: {e}")
            return np.arange(len(embeddings))  # Each item as its own cluster
    
    def _format_enhanced_results(
        self, 
        file_keys: List[str], 
        cluster_labels: np.ndarray,
        face_data: List[Dict]
    ) -> Dict:
        """
        Format results with enhanced statistics
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
        
        # Sort clusters by size
        clusters.sort(key=len, reverse=True)
        
        # Enhanced statistics
        total_faces = len(file_keys)
        valid_faces = total_faces - len(noise)
        noise_ratio = len(noise) / total_faces if total_faces > 0 else 0
        
        # Quality statistics
        avg_quality = np.mean([f.get("quality_score", 0) for f in face_data])
        avg_confidence = np.mean([f.get("confidence", 0) for f in face_data])
        avg_blur_score = np.mean([f.get("blur_score", 0) for f in face_data])
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        
        logger.info(f"[ENHANCED] Results: {len(clusters)} clusters, {len(noise)} noise files")
        logger.info(f"[ENHANCED] Noise ratio: {noise_ratio:.2%}, Avg quality: {avg_quality:.3f}")
        
        return {
            "clusters": clusters,
            "noise": noise,
            "stats": {
                "total_images": total_faces,
                "valid_faces": valid_faces,
                "num_clusters": len(clusters),
                "noise_count": len(noise),
                "noise_ratio": round(noise_ratio, 3),
                "avg_quality_score": round(avg_quality, 3),
                "avg_confidence": round(avg_confidence, 3),
                "avg_blur_score": round(avg_blur_score, 1),
                "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "cluster_size_distribution": {
                    "small (2-3)": len([s for s in cluster_sizes if 2 <= s <= 3]),
                    "medium (4-8)": len([s for s in cluster_sizes if 4 <= s <= 8]),
                    "large (9+)": len([s for s in cluster_sizes if s >= 9])
                },
                "quality_metrics": {
                    "high_quality": len([f for f in face_data if f.get("quality_score", 0) > 0.7]),
                    "medium_quality": len([f for f in face_data if 0.4 <= f.get("quality_score", 0) <= 0.7]),
                    "low_quality": len([f for f in face_data if f.get("quality_score", 0) < 0.4])
                }
            }
        }
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            "total_images": 0,
            "valid_faces": 0,
            "num_clusters": 0,
            "noise_count": 0,
            "noise_ratio": 0,
            "processing_time_seconds": 0
        }
    
    def _check_memory(self):
        """Check memory usage and clean up if needed"""
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.memory_threshold:
            logger.warning(f"[ENHANCED] High memory usage: {memory_percent:.1f}%")
            gc.collect()
    
    def _cleanup_memory(self, *objects):
        """Clean up large objects"""
        for obj in objects:
            del obj
        gc.collect()

# Create singleton instance
enhanced_clustering_pipeline = ImprovedFaceClusteringPipeline()