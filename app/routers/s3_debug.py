from fastapi import APIRouter
from app.services.S3_functions import storage_service

router = APIRouter()

@router.get("/debug/list-avatars")
async def list_avatars():
    keys = await storage_service.list_avatar_keys()
    return {"files": keys}


