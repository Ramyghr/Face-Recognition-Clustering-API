# app/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from mongoengine import connection
from app.routers import cluster_assignment
from app.core import config, mongo
from app.core.config import settings
from app.models.cluster_assignments import ClusterAssignment

# Use the actual model module names you showed earlier
from app.models.face_clustering_models import (
    FaceEmbeddingBase as FaceEmbedding,
    ClusteringResult,
)
from app.models.user_profile import UserProfile

from app.services.connection import app_lifecycle
import app.db.face_clustering_operations as db_ops  # to set _database_instance

# Routers
from app.routers import user_profile, bucket_clustering, s3_debug

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Face Clustering Application...")
    client: AsyncIOMotorClient | None = None
    try:
        # ---- Mongo (Beanie) ----
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        db_ops._database_instance = db  # make available to other modules

        # Single init_beanie at startup
        await init_beanie(
            database=db,
            document_models=[FaceEmbedding, ClusteringResult, UserProfile,ClusterAssignment],
        )
        logger.info("Beanie initialized")

        # ---- MongoEngine (only if you actually need it) ----
        mongo.init_db()
        logger.info("MongoEngine initialized")

        # ---- Other startup bits ----
        await app_lifecycle.startup()

        # stash client for shutdown
        app.state.mongo_client = client
        logger.info("Startup complete")
        yield

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down Face Clustering Application...")
        try:
            await app_lifecycle.shutdown()
        except Exception as e:
            logger.warning(f"app_lifecycle.shutdown error: {e}")

        try:
            if getattr(app.state, "mongo_client", None):
                app.state.mongo_client.close()
                logger.info("Closed Mongo (Beanie) client")
        except Exception as e:
            logger.warning(f"Error closing Mongo client: {e}")

        try:
            connection.disconnect()
            logger.info("Disconnected MongoEngine")
        except Exception as e:
            logger.warning(f"MongoEngine disconnect error: {e}")

# Create app ONCE with lifespan
app = FastAPI(
    title=config.settings.PROJECT_NAME,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health
@app.get("/health")
async def health_check():
    try:
        await UserProfile.find_one()
        me_connected = bool(connection.get_connection())
        return {
            "status": "healthy",
            "beanie_database": "connected",
            "mongoengine_database": "connected" if me_connected else "disconnected",
            "version": "2.0.0",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "version": "2.0.0"}

# Routers (attach to the SAME app)
app.include_router(user_profile.router)
app.include_router(bucket_clustering.router)
app.include_router(s3_debug.router)
app.include_router(cluster_assignment.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
