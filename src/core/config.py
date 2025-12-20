from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # GCP
    google_cloud_project: str = ""
    gcs_bucket: str = ""

    def validate_gcp_config(self) -> None:
        """Validate that required GCP settings are configured."""
        errors = []

        if not self.google_cloud_project or self.google_cloud_project == "your-project-id":
            errors.append("GOOGLE_CLOUD_PROJECT must be set to your actual GCP project ID")

        if not self.gcs_bucket or self.gcs_bucket == "your-bucket-name":
            errors.append("GCS_BUCKET must be set to your actual GCS bucket name")

        if errors:
            error_msg = "Configuration errors:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and self.google_cloud_project != ""
    
    # Pub/Sub
    pubsub_subscription_upscale: str = "upscale-jobs-sub"
    pubsub_subscription_enhance: str = "enhance-jobs-sub"
    pubsub_subscription_comic: str = "style-comic-jobs-sub"
    pubsub_subscription_aged: str = "style-aged-jobs-sub"
    pubsub_subscription_background_remove: str = "background-remove-jobs-sub"
    
    # Model
    model_cache_dir: str = "/models"
    device: str | None = None  # Auto-detect if None
    torch_dtype: str = "float16"

    # HuggingFace cache environment variables
    @property
    def hf_home(self) -> str:
        """HuggingFace cache directory."""
        return f"{self.model_cache_dir}/huggingface"

    @property
    def transformers_cache(self) -> str:
        """Transformers cache directory."""
        return f"{self.model_cache_dir}/transformers"
    
    # Local dev (optional)
    redis_url: str | None = None
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
