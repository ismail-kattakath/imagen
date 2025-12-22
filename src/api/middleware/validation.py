# =============================================================================
# REQUEST VALIDATION MODULE
# =============================================================================
#
# Validates incoming requests for:
#   - File size limits
#   - File type validation
#   - Image dimension limits
#   - Request body size
#
# =============================================================================

from fastapi import Request, HTTPException, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile
from typing import Optional, List, Set
from PIL import Image
import io

from src.core.config import settings
from src.core.logging import logger


# =============================================================================
# CONFIGURATION
# =============================================================================

# File size limits by tier (in bytes)
FILE_SIZE_LIMITS = {
    "free": 5 * 1024 * 1024,        # 5 MB
    "pro": 25 * 1024 * 1024,        # 25 MB
    "enterprise": 100 * 1024 * 1024, # 100 MB
    "anonymous": 2 * 1024 * 1024,    # 2 MB
}

# Default limit for non-authenticated requests
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Maximum request body size (for JSON payloads)
MAX_BODY_SIZE = 1 * 1024 * 1024  # 1 MB

# Allowed image types
ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}

ALLOWED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif"
}

# Image dimension limits
MAX_IMAGE_DIMENSION = 8192  # 8K
MIN_IMAGE_DIMENSION = 32    # Minimum useful size


# =============================================================================
# SIZE LIMIT MIDDLEWARE
# =============================================================================

class SizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce request size limits.
    
    Checks Content-Length header before reading body.
    """
    
    def __init__(self, app, max_size: int = MAX_BODY_SIZE):
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        # Skip for GET, HEAD, OPTIONS
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return await call_next(request)
        
        # Check Content-Length header
        content_length = request.headers.get("Content-Length")
        if content_length:
            size = int(content_length)
            
            # Get tier-specific limit for file uploads
            if "multipart/form-data" in request.headers.get("Content-Type", ""):
                max_size = self._get_file_size_limit(request)
            else:
                max_size = self.max_size
            
            if size > max_size:
                logger.warning(
                    "Request too large",
                    extra={
                        "content_length": size,
                        "max_size": max_size,
                        "path": request.url.path,
                    }
                )
                from starlette.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request too large. Maximum size: {max_size // (1024*1024)} MB"
                    },
                )
        
        return await call_next(request)
    
    def _get_file_size_limit(self, request: Request) -> int:
        """Get file size limit based on API key tier."""
        api_key = request.headers.get("X-API-Key", "")
        
        if not api_key:
            return FILE_SIZE_LIMITS["anonymous"]
        
        from src.api.middleware.auth import api_key_manager
        key_data = api_key_manager.validate(api_key)
        
        if not key_data:
            return FILE_SIZE_LIMITS["anonymous"]
        
        tier = key_data.get("tier", "free")
        return FILE_SIZE_LIMITS.get(tier, DEFAULT_MAX_FILE_SIZE)


# =============================================================================
# FILE VALIDATION
# =============================================================================

class FileValidator:
    """Validates uploaded files."""
    
    @staticmethod
    async def validate_image(
        file: UploadFile,
        max_size: Optional[int] = None,
        allowed_types: Optional[Set[str]] = None,
        max_dimension: int = MAX_IMAGE_DIMENSION,
        min_dimension: int = MIN_IMAGE_DIMENSION,
    ) -> Image.Image:
        """
        Validate and load an uploaded image.
        
        Args:
            file: Uploaded file
            max_size: Maximum file size in bytes
            allowed_types: Allowed MIME types
            max_dimension: Maximum width/height
            min_dimension: Minimum width/height
        
        Returns:
            PIL Image object
        
        Raises:
            HTTPException: If validation fails
        """
        max_size = max_size or DEFAULT_MAX_FILE_SIZE
        allowed_types = allowed_types or ALLOWED_CONTENT_TYPES
        
        # Check content type
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. "
                       f"Allowed types: {', '.join(allowed_types)}",
            )
        
        # Check file extension
        if file.filename:
            ext = "." + file.filename.lower().split(".")[-1]
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension. "
                           f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                )
        
        # Read file content
        contents = await file.read()
        
        # Check file size
        if len(contents) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {len(contents) // (1024*1024)} MB. "
                       f"Maximum: {max_size // (1024*1024)} MB",
            )
        
        # Validate image
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify it's a valid image
            
            # Re-open after verify (verify() leaves file in unusable state)
            image = Image.open(io.BytesIO(contents))
            
        except Exception as e:
            logger.warning(f"Invalid image file: {e}")
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted image file.",
            )
        
        # Check dimensions
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {width}x{height}. "
                       f"Maximum dimension: {max_dimension}px",
            )
        
        if width < min_dimension or height < min_dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Image too small: {width}x{height}. "
                       f"Minimum dimension: {min_dimension}px",
            )
        
        # Reset file position for potential re-reads
        await file.seek(0)
        
        return image
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """Get image metadata."""
        return {
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "format": image.format,
        }


# Singleton instance
file_validator = FileValidator()


# =============================================================================
# DEPENDENCY FOR VALIDATED IMAGE UPLOAD
# =============================================================================

async def get_validated_image(
    file: UploadFile,
    request: Request,
) -> Image.Image:
    """
    FastAPI dependency to validate and return image.
    
    Usage:
        @router.post("/process")
        async def process(image: Image.Image = Depends(get_validated_image)):
            # image is already validated
            ...
    """
    # Get tier-specific file size limit
    api_key = request.headers.get("X-API-Key", "")
    
    if api_key:
        from src.api.middleware.auth import api_key_manager
        key_data = api_key_manager.validate(api_key)
        tier = key_data.get("tier", "free") if key_data else "anonymous"
    else:
        tier = "anonymous"
    
    max_size = FILE_SIZE_LIMITS.get(tier, DEFAULT_MAX_FILE_SIZE)
    
    return await file_validator.validate_image(file, max_size=max_size)
