#S3_functions

import time
from io import BytesIO
import aioboto3 
import cv2
from PIL import Image
from botocore.exceptions import ClientError
from app.core.config import settings
import numpy as np
import logging
import os 
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class StorageService:
    def __init__(self):
        self.session = aioboto3.Session(
            aws_access_key_id=settings.WASABI_ACCESS_KEY,
            aws_secret_access_key=settings.WASABI_SECRET_KEY,
            region_name=settings.WASABI_REGION
        )
        self.bucket_name = settings.WASABI_BUCKET_NAME
        self.endpoint_url = settings.WASABI_ENDPOINT
    
    # S3_functions.py
    async def get_file(self, file_key: str, timeout_sec: float = 10.0) -> bytes:
        try:
            async with self.session.client('s3', endpoint_url=self.endpoint_url) as client:
                try:
                    # Wrap in timeout
                    response = await asyncio.wait_for(
                        client.get_object(Bucket=self.bucket_name, Key=file_key),
                        timeout=timeout_sec
                    )
                    return await response['Body'].read()
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout while downloading {file_key}")
                    raise TimeoutError(f"Timeout: {file_key}")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchKey':
                        logger.error(f"File not found: {file_key}")
                    else:
                        logger.error(f"S3 error retrieving {file_key}: {str(e)}")
                    raise
        except Exception as e:
            logger.critical(f"Storage error: {str(e)}")
            raise 
    
    async def upload_avatar(self, image: np.ndarray, client_id: str) -> str:  # Made async
        """Upload a face crop to Wasabi storage asynchronously"""
        try:
            # Convert image to JPEG in memory (synchronous operation)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)

            # Generate unique filename
            timestamp = int(time.time())
            s3_path = f"usersAvatars/{client_id}.jpg"

            # Async upload
            async with self.session.client(
                's3',
                endpoint_url=self.endpoint_url
            ) as client:  # Async client context manager
                await client.upload_fileobj(
                    buffer,
                    self.bucket_name,
                    s3_path,
                    ExtraArgs={
                        'ContentType': 'image/jpeg',
                        'ACL': 'public-read'
                    }
                )

            return f"{s3_path}"
        
        except Exception as e:
            logger.error(f"Error uploading avatar: {str(e)}")  # Better to use logger
            raise
    
    async def list_avatar_keys(self, prefix: str = "usersAvatars/") -> list:
        # Normalize prefix
        if prefix.startswith(self.bucket_name):
            prefix = prefix.replace(f"{self.bucket_name}/", "")

        logger.warning(f"[S3] Using prefix: {prefix}")  # 👈 LOG THE PREFIX USED

        keys = []
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as client:
            continuation_token = None
            while True:
                params = {
                    "Bucket": self.bucket_name,
                    "Prefix": prefix
                }
                if continuation_token:
                    params["ContinuationToken"] = continuation_token

                response = await client.list_objects_v2(**params)

                if "Contents" in response:
                    found_keys = [obj["Key"] for obj in response["Contents"]]
                    logger.warning(f"[S3] Retrieved keys: {found_keys[:5]}")  # 👈 Show first 5 keys
                    keys.extend(found_keys)

                if not response.get("IsTruncated"):
                    break
                continuation_token = response["NextContinuationToken"]
        return keys

    #Timeout & retry logic 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_file_with_retry(self, file_key: str, timeout_sec: float = 5.0) -> bytes:
        """
        Get file with retry logic and shorter timeout for faster processing
        """
        try:
            async with self.session.client('s3', endpoint_url=self.endpoint_url) as client:
                response = await asyncio.wait_for(
                    client.get_object(Bucket=self.bucket_name, Key=file_key),
                    timeout=timeout_sec
                )
                return await response['Body'].read()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading {file_key}")
            raise
        except Exception as e:
            logger.error(f"Error downloading {file_key}: {e}")
            raise

    async def get_file(self, file_key: str, timeout_sec: float = 5.0) -> bytes:
        """Use the retry version by default"""
        return await self.get_file_with_retry(file_key, timeout_sec)


    async def download_images_from_wasabi(self, target_dir: str):
        async with self.session.client("s3", endpoint_url=self.endpoint_url) as client:
            response = await client.list_objects_v2(Bucket=self.bucket_name)
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = os.path.join(target_dir, os.path.basename(key))
                await client.download_file(self.bucket_name, key, path)
    
# Instantiate service for easy import
storage_service = StorageService()