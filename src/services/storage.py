import io
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from google.cloud import storage
from PIL import Image

from src.core.config import settings
from src.core.exceptions import StorageError
from src.core.logging import logger


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


class StorageService(Protocol):
    """Protocol for storage services."""

    def upload_image(self, image: Image.Image, path: str, format: str = "PNG") -> str:
        """Upload a PIL Image to storage."""
        ...

    def download_image(self, path: str) -> Image.Image:
        """Download an image from storage."""
        ...

    def get_signed_url(self, path: str, expiration_hours: int = 1) -> str:
        """Generate a signed URL for temporary access."""
        ...

    def delete(self, path: str) -> None:
        """Delete a file from storage."""
        ...


class LocalFileSystemStorage:
    """Local filesystem storage for development."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or "./storage")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using local storage at: {self.base_dir.absolute()}")

    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path."""
        full_path = self.base_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    def upload_image(
        self,
        image: Image.Image,
        path: str,
        format: str = "PNG",
    ) -> str:
        """Save a PIL Image to local filesystem."""
        try:
            full_path = self._get_full_path(path)
            image.save(full_path, format=format)
            logger.info(f"Saved image to {full_path}")
            return str(full_path)

        except Exception as e:
            raise StorageError(f"Failed to save image: {e}") from e

    def download_image(self, path: str) -> Image.Image:
        """Load an image from local filesystem."""
        try:
            full_path = self._get_full_path(path)
            return Image.open(full_path)

        except Exception as e:
            raise StorageError(f"Failed to load image: {e}") from e

    def get_signed_url(
        self,
        path: str,
        expiration_hours: int = 1,
    ) -> str:
        """Return local file path (no signing needed for local files)."""
        full_path = self._get_full_path(path)
        return f"file://{full_path.absolute()}"

    def delete(self, path: str) -> None:
        """Delete a file from local filesystem."""
        try:
            full_path = self._get_full_path(path)
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Deleted {full_path}")
        except Exception as e:
            raise StorageError(f"Failed to delete: {e}") from e


def get_storage_service() -> StorageService:
    """Get the appropriate storage service based on configuration.

    Returns LocalFileSystemStorage if in development mode,
    otherwise returns GCSStorageService for production.
    """
    if not settings.is_production():
        logger.info("Using local filesystem storage (local development)")
        return LocalFileSystemStorage()  # type: ignore
    else:
        logger.info("Using GCS storage (production)")
        return GCSStorageService()  # type: ignore
