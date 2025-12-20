class ImagenError(Exception):
    """Base exception for Imagen platform."""
    pass


class PipelineError(ImagenError):
    """Error during image processing pipeline execution."""
    pass


class StorageError(ImagenError):
    """Error during storage operations."""
    pass


class QueueError(ImagenError):
    """Error during queue operations."""
    pass


class ModelLoadError(PipelineError):
    """Error loading ML model."""
    pass


class ImageProcessingError(PipelineError):
    """Error processing image."""
    pass


class JobNotFoundError(ImagenError):
    """Job not found in the system."""
    pass
