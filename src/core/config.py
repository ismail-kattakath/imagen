from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Configuration is loaded from:
    1. Environment variables (uppercase with underscores)
    2. .env file (if present)

    Example:
        API_HOST=0.0.0.0
        GOOGLE_CLOUD_PROJECT=my-project-id
        GCS_BUCKET=my-bucket
    """

    # API Configuration
    api_host: str = "0.0.0.0"  # API server bind address
    api_port: int = 8000  # API server port
    debug: bool = False  # Enable debug mode (disables GCP validation)

    # Google Cloud Platform Configuration
    google_cloud_project: str = ""  # GCP project ID
    gcs_bucket: str = ""  # Google Cloud Storage bucket name

    def validate_gcp_config(self) -> None:
        """Validate that required GCP settings are configured.

        Checks that GOOGLE_CLOUD_PROJECT and GCS_BUCKET are set to actual
        values (not placeholders).

        Raises:
            ValueError: If any required GCP settings are missing or contain
                       placeholder values.
        """
        errors = []

        if not self.google_cloud_project or self.google_cloud_project == "your-project-id":
            errors.append("GOOGLE_CLOUD_PROJECT must be set to your actual GCP project ID")

        if not self.gcs_bucket or self.gcs_bucket == "your-bucket-name":
            errors.append("GCS_BUCKET must be set to your actual GCS bucket name")

        if errors:
            error_msg = "Configuration errors:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

    def is_production(self) -> bool:
        """Check if running in production mode.

        Returns:
            bool: True if debug=False and GCP project is configured
        """
        return not self.debug and self.google_cloud_project != ""

    # Pub/Sub Subscriptions
    # Each worker type has its own subscription
    pubsub_subscription_upscale: str = "upscale-jobs-sub"
    pubsub_subscription_enhance: str = "enhance-jobs-sub"
    pubsub_subscription_comic: str = "style-comic-jobs-sub"
    pubsub_subscription_aged: str = "style-aged-jobs-sub"
    pubsub_subscription_background_remove: str = "background-remove-jobs-sub"

    # Model Configuration
    model_cache_dir: str = "./models"  # Base directory for cached models
    device: str | None = None  # PyTorch device ('cuda', 'cpu', or None for auto-detect)
    torch_dtype: str = "float16"  # Model precision (float16 for GPU, float32 for CPU)

    # HuggingFace cache environment variables
    @property
    def hf_home(self) -> str:
        """HuggingFace cache directory."""
        return f"{self.model_cache_dir}/huggingface"

    @property
    def transformers_cache(self) -> str:
        """Transformers cache directory."""
        return f"{self.model_cache_dir}/transformers"
    
    # Local Development (Optional)
    # These settings are for local development with alternative services
    redis_url: str | None = None  # Redis URL for local job queue
    minio_endpoint: str | None = None  # MinIO endpoint for local storage
    minio_access_key: str | None = None  # MinIO access key
    minio_secret_key: str | None = None  # MinIO secret key

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses LRU cache to ensure singleton pattern - settings are loaded
    once and reused throughout the application lifecycle.

    Returns:
        Settings: The application settings instance
    """
    return Settings()


settings = get_settings()
