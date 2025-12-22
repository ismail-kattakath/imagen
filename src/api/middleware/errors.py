# =============================================================================
# ERROR HANDLING MODULE
# =============================================================================
#
# Provides:
#   - Standardized error responses
#   - Exception handlers for FastAPI
#   - Error logging with context
#
# =============================================================================

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Union
import traceback

from src.api.middleware.logging import ctx_logger, get_request_id


# =============================================================================
# ERROR RESPONSE FORMAT
# =============================================================================

def create_error_response(
    status_code: int,
    message: str,
    error_code: str = None,
    details: dict = None,
) -> JSONResponse:
    """Create standardized error response."""
    body = {
        "error": {
            "code": error_code or f"ERR_{status_code}",
            "message": message,
            "request_id": get_request_id(),
        }
    }
    
    if details:
        body["error"]["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=body,
    )


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

async def http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException],
) -> JSONResponse:
    """Handle HTTP exceptions."""
    ctx_logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=str(exc.detail),
        path=request.url.path,
    )
    
    return create_error_response(
        status_code=exc.status_code,
        message=str(exc.detail),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors."""
    # Format validation errors
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })
    
    ctx_logger.warning(
        "Validation error",
        path=request.url.path,
        errors=errors,
    )
    
    return create_error_response(
        status_code=422,
        message="Request validation failed",
        error_code="VALIDATION_ERROR",
        details={"errors": errors},
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle uncaught exceptions."""
    # Log full traceback
    ctx_logger.error(
        "Unhandled exception",
        path=request.url.path,
        exception_type=type(exc).__name__,
        exception_message=str(exc),
        traceback=traceback.format_exc(),
    )
    
    # Don't expose internal details in production
    from src.core.config import settings
    
    if settings.is_production():
        message = "An internal error occurred. Please try again later."
    else:
        message = f"{type(exc).__name__}: {str(exc)}"
    
    return create_error_response(
        status_code=500,
        message=message,
        error_code="INTERNAL_ERROR",
    )


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ImagenException(Exception):
    """Base exception for Imagen API."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = None,
        details: dict = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details
        super().__init__(message)


class JobNotFoundException(ImagenException):
    """Job not found."""
    
    def __init__(self, job_id: str):
        super().__init__(
            message=f"Job not found: {job_id}",
            status_code=404,
            error_code="JOB_NOT_FOUND",
            details={"job_id": job_id},
        )


class JobProcessingException(ImagenException):
    """Error during job processing."""
    
    def __init__(self, job_id: str, reason: str):
        super().__init__(
            message=f"Job processing failed: {reason}",
            status_code=500,
            error_code="JOB_PROCESSING_ERROR",
            details={"job_id": job_id, "reason": reason},
        )


class QuotaExceededException(ImagenException):
    """Quota exceeded."""
    
    def __init__(self, limit_type: str, limit: int, current: int):
        super().__init__(
            message=f"Quota exceeded: {limit_type}",
            status_code=429,
            error_code="QUOTA_EXCEEDED",
            details={
                "limit_type": limit_type,
                "limit": limit,
                "current": current,
            },
        )


class InvalidImageException(ImagenException):
    """Invalid image file."""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid image: {reason}",
            status_code=400,
            error_code="INVALID_IMAGE",
            details={"reason": reason},
        )


async def imagen_exception_handler(
    request: Request,
    exc: ImagenException,
) -> JSONResponse:
    """Handle Imagen-specific exceptions."""
    ctx_logger.warning(
        "Imagen exception",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )
    
    return create_error_response(
        status_code=exc.status_code,
        message=exc.message,
        error_code=exc.error_code,
        details=exc.details,
    )


# =============================================================================
# REGISTER HANDLERS
# =============================================================================

def register_exception_handlers(app):
    """Register all exception handlers with FastAPI app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ImagenException, imagen_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
