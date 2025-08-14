#this is for local manual testing only Do not use


import os
from pathlib import Path
from app.core.config import settings

# For local storage
def save_local(file_path: str, content: bytes):
    full_path = Path(settings.LOCAL_STORAGE_PATH) / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_bytes(content)

def get_local(file_path: str) -> bytes:
    return (Path(settings.LOCAL_STORAGE_PATH) / file_path).read_bytes()




# Unified interface
def save_file(file_path: str, content: bytes):
        save_local(file_path, content)

def get_file(file_path: str) -> bytes:
    return get_local(file_path)