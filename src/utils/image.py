import base64
import io

from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(data: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(data)
    return Image.open(io.BytesIO(image_data))


def resize_for_model(
    image: Image.Image,
    max_size: int = 1024,
    divisible_by: int = 8,
) -> Image.Image:
    """Resize image to fit model constraints."""
    width, height = image.size

    # Scale down if needed
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        width = int(width * scale)
        height = int(height * scale)

    # Make divisible
    width = (width // divisible_by) * divisible_by
    height = (height // divisible_by) * divisible_by

    return image.resize((width, height), Image.Resampling.LANCZOS)
