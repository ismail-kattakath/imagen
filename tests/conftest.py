import pytest
from PIL import Image
import io


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (256, 256), color="red")
    return img


@pytest.fixture
def sample_image_bytes():
    """Create sample image as bytes."""
    img = Image.new("RGB", (256, 256), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()
