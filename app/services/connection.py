# app/services/connection_manager.py
import asyncio
import aiohttp
import logging
from contextlib import asynccontextmanager
from typing import Optional
import gc
import weakref

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages HTTP connections and prevents connection leaks
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_ref = None
        self._connector = None
        
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with proper connection management"""
        if self._session is None or self._session.closed:
            # Create connector with connection limits
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection limit
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=60,  # Keep connections alive for 60 seconds
                enable_cleanup_closed=True  # Enable cleanup of closed connections
            )
            
            # Create session with timeout
            timeout = aiohttp.ClientTimeout(
                total=300,  # 5 minute total timeout
                connect=30,  # 30 second connect timeout
                sock_read=60  # 60 second read timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                connector_owner=True  # Session owns the connector
            )
            
            self._connector = connector
            logger.info("Created new aiohttp session with connection management")
        
        return self._session
    
    async def close_session(self):
        """Properly close the session and all connections"""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
                logger.info("Closed aiohttp session")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
        
        if self._connector:
            try:
                await self._connector.close()
                logger.info("Closed aiohttp connector")
            except Exception as e:
                logger.warning(f"Error closing connector: {e}")
        
        self._session = None
        self._connector = None
    
    @asynccontextmanager
    async def managed_request(self, method: str, url: str, **kwargs):
        """Context manager for making HTTP requests with proper connection management"""
        session = await self.get_session()
        try:
            async with session.request(method, url, **kwargs) as response:
                yield response
        except Exception as e:
            logger.error(f"Request failed for {method} {url}: {e}")
            raise
    
    async def cleanup_connections(self):
        """Force cleanup of all connections"""
        await self.close_session()
        
        # Force garbage collection
        gc.collect()
        
        # Small delay to allow cleanup
        await asyncio.sleep(0.1)

# Global connection manager instance
connection_manager = ConnectionManager()

# Context manager for request lifecycle
@asynccontextmanager
async def managed_http_request(method: str, url: str, **kwargs):
    """
    Context manager for HTTP requests with automatic connection cleanup
    Usage:
        async with managed_http_request('GET', 'https://example.com') as response:
            data = await response.read()
    """
    try:
        async with connection_manager.managed_request(method, url, **kwargs) as response:
            yield response
    finally:
        # Optional: Force cleanup after each request (for debugging connection issues)
        pass

# Decorator for functions that make HTTP requests
def with_connection_cleanup(func):
    """Decorator to ensure connection cleanup after function execution"""
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Force cleanup connections periodically
            if hasattr(func, '_cleanup_counter'):
                func._cleanup_counter += 1
            else:
                func._cleanup_counter = 1
            
            # Cleanup every 50 requests to prevent accumulation
            if func._cleanup_counter % 50 == 0:
                await connection_manager.cleanup_connections()
                logger.info(f"Performed connection cleanup after {func._cleanup_counter} requests")
    
    return wrapper

# Enhanced S3 service with connection management
class S3ConnectionManager:
    """Enhanced S3 service with proper connection management"""
    
    def __init__(self, storage_service):
        self.storage_service = storage_service
        self.connection_manager = ConnectionManager()
    
    @with_connection_cleanup
    async def get_file_managed(self, file_key: str) -> bytes:
        """Get file from S3 with connection management"""
        try:
            # Use the original storage service but with connection cleanup
            result = await self.storage_service.get_file(file_key)
            return result
        except Exception as e:
            logger.error(f"Failed to get file {file_key}: {e}")
            # Force connection cleanup on error
            await self.connection_manager.cleanup_connections()
            raise
    
    @with_connection_cleanup
    async def list_avatar_keys_managed(self, prefix: str = "") -> list:
        """List S3 keys with connection management"""
        try:
            result = await self.storage_service.list_avatar_keys(prefix=prefix)
            return result
        except Exception as e:
            logger.error(f"Failed to list keys with prefix {prefix}: {e}")
            # Force connection cleanup on error
            await self.connection_manager.cleanup_connections()
            raise
    
    async def cleanup_all_connections(self):
        """Cleanup all S3 connections"""
        await self.connection_manager.cleanup_connections()

# Application lifecycle management
class ApplicationLifecycleManager:
    """Manages connections throughout the application lifecycle"""
    
    def __init__(self):
        self.managers = []
    
    def register_manager(self, manager):
        """Register a connection manager"""
        self.managers.append(weakref.ref(manager))
    
    async def startup(self):
        """Application startup tasks"""
        logger.info("Application starting up - initializing connection managers")
    
    async def shutdown(self):
        """Application shutdown tasks - cleanup all connections"""
        logger.info("Application shutting down - cleaning up connections")
        
        # Cleanup all registered managers
        for manager_ref in self.managers:
            manager = manager_ref()
            if manager and hasattr(manager, 'cleanup_connections'):
                try:
                    await manager.cleanup_connections()
                except Exception as e:
                    logger.warning(f"Error during manager cleanup: {e}")
        
        # Global cleanup
        await connection_manager.cleanup_connections()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Connection cleanup completed")

# Global lifecycle manager
app_lifecycle = ApplicationLifecycleManager()

# Utility functions for connection debugging
async def get_connection_stats():
    """Get current connection statistics"""
    stats = {
        "session_active": connection_manager._session is not None and not connection_manager._session.closed,
        "connector_active": connection_manager._connector is not None,
    }
    
    if connection_manager._connector:
        try:
            stats.update({
                "total_connections": len(connection_manager._connector._conns),
                "available_connections": len(connection_manager._connector._available_connections)
            })
        except AttributeError:
            # Connector structure might be different in different aiohttp versions
            stats["connection_details"] = "Not available"
    
    return stats

async def force_connection_cleanup():
    """Force cleanup all connections (for debugging)"""
    logger.info("Forcing connection cleanup...")
    await connection_manager.cleanup_connections()
    gc.collect()
    logger.info("Connection cleanup completed")