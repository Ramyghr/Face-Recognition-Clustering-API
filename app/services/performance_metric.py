import time
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Timing metrics
    download_time: float = 0.0
    face_detection_time: float = 0.0
    embedding_time: float = 0.0
    clustering_time: float = 0.0
    total_time: float = 0.0
    
    # Processing metrics
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    timeout_files: int = 0
    
    # Face metrics
    total_faces_detected: int = 0
    valid_faces: int = 0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_samples: list = field(default_factory=list)
    
    # Clustering metrics
    num_clusters: int = 0
    noise_count: int = 0
    largest_cluster_size: int = 0
    
    def finish(self):
        """Mark metrics as finished and calculate totals"""
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        
        if self.memory_samples:
            self.peak_memory_mb = max(self.memory_samples)
            self.avg_memory_mb = sum(self.memory_samples) / len(self.memory_samples)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "timing": {
                "total_time_seconds": round(self.total_time, 2),
                "download_time_seconds": round(self.download_time, 2),
                "face_detection_time_seconds": round(self.face_detection_time, 2),
                "embedding_time_seconds": round(self.embedding_time, 2),
                "clustering_time_seconds": round(self.clustering_time, 2),
            },
            "processing": {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "failed_files": self.failed_files,
                "timeout_files": self.timeout_files,
                "success_rate_percent": round((self.processed_files / self.total_files) * 100, 1) if self.total_files > 0 else 0,
                "files_per_second": round(self.processed_files / self.total_time, 2) if self.total_time > 0 else 0
            },
            "faces": {
                "total_faces_detected": self.total_faces_detected,
                "valid_faces": self.valid_faces,
                "faces_per_image": round(self.total_faces_detected / self.processed_files, 2) if self.processed_files > 0 else 0
            },
            "memory": {
                "peak_memory_mb": round(self.peak_memory_mb, 1),
                "avg_memory_mb": round(self.avg_memory_mb, 1)
            },
            "clustering": {
                "num_clusters": self.num_clusters,
                "noise_count": self.noise_count,
                "largest_cluster_size": self.largest_cluster_size,
                "clustering_efficiency": round((self.num_clusters / self.valid_faces) * 100, 1) if self.valid_faces > 0 else 0
            }
        }

class PerformanceMonitor:
    """Monitor and track performance metrics during clustering"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self._monitoring = False
        self._monitor_task = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._memory_monitor())
    
    async def stop_monitoring(self):
        """Stop monitoring and finalize metrics"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.metrics.finish()
    
    async def _memory_monitor(self):
        """Background task to monitor memory usage"""
        while self._monitoring:
            try:
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                self.metrics.memory_samples.append(memory_mb)
                await asyncio.sleep(5)  # Sample every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
    
    @asynccontextmanager
    async def time_operation(self, operation_name: str):
        """Context manager to time specific operations"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            setattr(self.metrics, f"{operation_name}_time", 
                   getattr(self.metrics, f"{operation_name}_time", 0) + elapsed)
    
    def record_file_processed(self, success: bool = True, timeout: bool = False):
        """Record that a file was processed"""
        if success:
            self.metrics.processed_files += 1
        else:
            self.metrics.failed_files += 1
        
        if timeout:
            self.metrics.timeout_files += 1
    
    def record_faces(self, detected: int, valid: int):
        """Record face detection results"""
        self.metrics.total_faces_detected += detected
        self.metrics.valid_faces += valid
    
    def record_clustering_results(self, clusters: list, noise: list):
        """Record clustering results"""
        self.metrics.num_clusters = len(clusters)
        self.metrics.noise_count = len(noise)
        self.metrics.largest_cluster_size = max(len(cluster) for cluster in clusters) if clusters else 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return self.metrics.to_dict()
    
    def log_performance_summary(self):
        """Log performance summary"""
        summary = self.get_performance_summary()
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total Time: {summary['timing']['total_time_seconds']}s")
        logger.info(f"Processing Rate: {summary['processing']['files_per_second']} files/sec")
        logger.info(f"Success Rate: {summary['processing']['success_rate_percent']}%")
        logger.info(f"Peak Memory: {summary['memory']['peak_memory_mb']} MB")
        logger.info(f"Clusters Found: {summary['clustering']['num_clusters']}")
        logger.info(f"Faces per Image: {summary['faces']['faces_per_image']}")
        logger.info("=" * 30)

