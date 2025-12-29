"""Secret Manager utilities for securely accessing secrets."""

import json
import os
from functools import lru_cache

from google.cloud import secretmanager


@lru_cache(maxsize=128)
def access_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Access a secret from Google Cloud Secret Manager.

    Args:
        project_id: GCP project ID
        secret_id: Secret ID (e.g., "jwt-secret")
        version: Secret version (default: "latest")

    Returns:
        Secret value as string

    Raises:
        Exception: If secret cannot be accessed
    """
    # Check if running locally and use env vars as fallback
    if os.getenv("USE_LOCAL_SECRETS", "false").lower() == "true":
        # Map secret IDs to env var names
        env_var_map = {
            "jwt-secret": "JWT_SECRET",
            "api-keys": "API_KEYS",
            "cors-origins": "CORS_ORIGINS",
        }
        env_var = env_var_map.get(secret_id)
        if env_var and os.getenv(env_var):
            return os.getenv(env_var)

    # Access from Secret Manager
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"

    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def get_secret_or_env(project_id: str, secret_id: str, env_var: str, default: str | None = None) -> str | None:
    """Get secret from Secret Manager or fallback to environment variable.

    Args:
        project_id: GCP project ID
        secret_id: Secret Manager secret ID
        env_var: Environment variable name (fallback)
        default: Default value if neither source is available

    Returns:
        Secret value or default
    """
    # Try environment variable first (for local dev)
    env_value = os.getenv(env_var)
    if env_value:
        return env_value

    # Try Secret Manager (for production)
    if project_id:
        try:
            return access_secret(project_id, secret_id)
        except Exception:
            # If Secret Manager fails, use default
            pass

    return default


def parse_json_secret(secret_value: str | None) -> list | dict | None:
    """Parse a JSON secret value.

    Args:
        secret_value: JSON string or None

    Returns:
        Parsed JSON object or None
    """
    if not secret_value:
        return None

    try:
        return json.loads(secret_value)
    except json.JSONDecodeError:
        return None


def parse_list_secret(secret_value: str | None) -> list[str] | None:
    """Parse a comma-separated list secret.

    Args:
        secret_value: Comma-separated string or None

    Returns:
        List of strings or None
    """
    if not secret_value:
        return None

    return [item.strip() for item in secret_value.split(",") if item.strip()]
