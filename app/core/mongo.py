from mongoengine import connect
from app.core.config import settings

def init_db():
    connect(
        db=settings.MONGO_DB,
        host=settings.MONGO_URI,
        alias="default"
    )
