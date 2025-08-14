# services/memory_utils.py
import gc
import psutil
import logging
from typing import Any

logger = logging.getLogger(__name__)

class MemoryManager:
    """Utility class for managing memory during large clustering operations"""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection"""
        gc.collect()
    
    @staticmethod
    def check_memory_threshold(threshold: float = 80.0) -> bool:
        """Check if memory usage exceeds threshold"""
        usage = MemoryManager.get_memory_usage()
        if usage > threshold:
            logger.warning(f"Memory usage high: {usage:.1f}%")
            MemoryManager.force_garbage_collection()
            return True
        return False
    
    @staticmethod
    def clear_large_objects(*objects):
        """Clear large objects and force GC"""
        for obj in objects:
            del obj
        MemoryManager.force_garbage_collection()
