from PIL import Image

from src.utils.image import base64_to_image, image_to_base64, resize_for_model


def test_image_to_base64(sample_image):
    """Test image to base64 conversion."""
    b64 = image_to_base64(sample_image)
    assert isinstance(b64, str)
    assert len(b64) > 0


def test_base64_to_image(sample_image):
    """Test base64 to image conversion."""
    b64 = image_to_base64(sample_image)
    restored = base64_to_image(b64)
    assert isinstance(restored, Image.Image)
    assert restored.size == sample_image.size


def test_resize_for_model():
    """Test image resizing for model constraints."""
    img = Image.new("RGB", (1920, 1080), color="blue")
    resized = resize_for_model(img, max_size=1024, divisible_by=8)

    assert max(resized.size) <= 1024
    assert resized.size[0] % 8 == 0
    assert resized.size[1] % 8 == 0
