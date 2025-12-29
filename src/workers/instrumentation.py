# =============================================================================
# WORKER OPENTELEMETRY INSTRUMENTATION
# =============================================================================
#
# Provides tracing instrumentation for workers:
#   - Job processing spans
#   - Triton inference spans
#   - Storage operation spans
#   - Error tracking
#
# =============================================================================

import functools
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.core.telemetry import get_tracer

# Get tracer for workers
tracer = get_tracer("imagen.worker")


def traced_operation(operation_name: str, **default_attributes):
    """
    Decorator to create a span for an operation.

    Args:
        operation_name: Name of the operation (will be span name)
        **default_attributes: Default attributes to add to the span

    Example:
        @traced_operation("process_image", image_type="upscale")
        def process_image(self, image):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            with tracer.start_as_current_span(operation_name) as span:
                # Add default attributes
                for key, value in default_attributes.items():
                    span.set_attribute(key, str(value))

                # Add function arguments as attributes if they're simple types
                if kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"arg.{key}", value)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            with tracer.start_as_current_span(operation_name) as span:
                # Add default attributes
                for key, value in default_attributes.items():
                    span.set_attribute(key, str(value))

                # Add function arguments as attributes if they're simple types
                if kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span.set_attribute(f"arg.{key}", value)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


import asyncio


def create_job_span(job_id: str, job_type: str, model_name: str) -> trace.Span:
    """
    Create a new span for job processing.

    Args:
        job_id: Job identifier
        job_type: Type of job (e.g., "upscale", "enhance")
        model_name: Triton model name

    Returns:
        Started span (caller must end it)
    """
    span = tracer.start_span(
        name=f"process_job.{job_type}",
        attributes={
            "job.id": job_id,
            "job.type": job_type,
            "model.name": model_name,
        },
    )
    return span


def create_triton_span(model_name: str, operation: str = "infer") -> trace.Span:
    """
    Create a span for Triton inference.

    Args:
        model_name: Name of the Triton model
        operation: Type of operation (e.g., "infer", "health_check")

    Returns:
        Started span (caller must end it)
    """
    span = tracer.start_span(
        name=f"triton.{operation}",
        attributes={
            "triton.model": model_name,
            "triton.operation": operation,
        },
    )
    return span


def create_storage_span(operation: str, path: str) -> trace.Span:
    """
    Create a span for storage operations.

    Args:
        operation: Type of operation (e.g., "upload", "download")
        path: Storage path

    Returns:
        Started span (caller must end it)
    """
    span = tracer.start_span(
        name=f"storage.{operation}",
        attributes={
            "storage.operation": operation,
            "storage.path": path,
        },
    )
    return span


def add_job_attributes(span: trace.Span, **attributes):
    """
    Add custom attributes to current span.

    Args:
        span: Span to add attributes to
        **attributes: Key-value pairs to add
    """
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
