from google.cloud import storage
from PIL import Image
import io
from datetime import timedelta

from src.core.config import settings
from src.core.logging import logger
from src.core.exceptions import StorageError


class GCSStorageService:
    """Google Cloud Storage service for image storage."""
    
    def __init__(self, bucket_name: str | None = None):
        self.bucket_name = bucket_name or settings.gcs_bucket
        self._client: storage.Client | None = None
        self._bucket: storage.Bucket | None = None
    
    @property
    def client(self) -> storage.Client:
        if self._client is None:
            self._client = storage.Client()
        return self._client
    
    @property
    def bucket(self) -> storage.Bucket:
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket
    
    def upload_image(
        self,
        image: Image.Image,
        path: str,
        format: str = "PNG",
    ) -> str:
        """Upload a PIL Image to GCS."""
        try:
            blob = self.bucket.blob(path)
            
            buffer = io.BytesIO()
            image.save(buffer, format=format)
            buffer.seek(0)
            
            content_type = f"image/{format.lower()}"
            blob.upload_from_file(buffer, content_type=content_type)
            
            logger.info(f"Uploaded image to gs://{self.bucket_name}/{path}")
            return f"gs://{self.bucket_name}/{path}"
            
        except Exception as e:
            raise StorageError(f"Failed to upload image: {e}") from e
    
    def download_image(self, path: str) -> Image.Image:
        """Download an image from GCS."""
        try:
            blob = self.bucket.blob(path)
            buffer = io.BytesIO()
            blob.download_to_file(buffer)
            buffer.seek(0)
            return Image.open(buffer)
            
        except Exception as e:
            raise StorageError(f"Failed to download image: {e}") from e
    
    def get_signed_url(
        self,
        path: str,
        expiration_hours: int = 1,
    ) -> str:
        """Generate a signed URL for temporary access."""
        try:
            blob = self.bucket.blob(path)
            url = blob.generate_signed_url(
                expiration=timedelta(hours=expiration_hours),
                method="GET",
            )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate signed URL: {e}") from e
    
    def delete(self, path: str) -> None:
        """Delete a file from GCS."""
        try:
            blob = self.bucket.blob(path)
            blob.delete()
            logger.info(f"Deleted gs://{self.bucket_name}/{path}")
        except Exception as e:
            raise StorageError(f"Failed to delete: {e}") from e
