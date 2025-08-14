# services/batch_processor.py
import asyncio
from typing import List, Dict, Callable, Any
import time
import logging
from app.services.memory_utils import MemoryManager

logger = logging.getLogger(__name__)

class AsyncBatchProcessor:
    """Handles async batch processing with memory management"""
    
    def __init__(self, batch_size: int = 20, max_concurrent: int = 8):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.memory_manager = MemoryManager()
    
    async def process_batches(
        self, 
        items: List[Any], 
        processor_func: Callable,
        progress_callback: Callable = None
    ) -> List[Any]:
        """Process items in batches with concurrency control"""
        
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(items), self.batch_size):
            batch = items[batch_idx:batch_idx + self.batch_size]
            
            # Memory check
            self.memory_manager.check_memory_threshold()
            
            # Process batch with semaphore
            async with self.semaphore:
                batch_results = await asyncio.gather(
                    *[processor_func(item) for item in batch],
                    return_exceptions=True
                )
                
                # Filter out exceptions and collect results
                valid_results = [
                    result for result in batch_results 
                    if not isinstance(result, Exception)
                ]
                results.extend(valid_results)
            
            # Progress callback
            if progress_callback:
                progress = (batch_idx + self.batch_size) / len(items)
                await progress_callback(min(progress, 1.0))
            
            logger.info(f"Processed batch {batch_idx//self.batch_size + 1}/{total_batches}")
        
        return results
