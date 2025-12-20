import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_check():
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
