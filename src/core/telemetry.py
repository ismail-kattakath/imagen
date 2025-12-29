# =============================================================================
# OPENTELEMETRY CONFIGURATION
# =============================================================================
#
# Provides centralized OpenTelemetry setup for:
#   - Distributed tracing across API and workers
#   - Trace context propagation (HTTP, gRPC, Pub/Sub)
#   - Export to Google Cloud Trace (production) or Jaeger (development)
#   - Integration with existing logging and metrics
#
# =============================================================================

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL TRACER
# =============================================================================

_tracer_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None


def get_tracer(name: str = "imagen") -> trace.Tracer:
    """
    Get the configured tracer instance.

    Args:
        name: Tracer name (typically module name)

    Returns:
        OpenTelemetry Tracer
    """
    global _tracer
    if _tracer is None:
        # Return a no-op tracer if not initialized
        return trace.get_tracer(name)
    return _tracer


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================


def setup_telemetry(
    service_name: str,
    environment: str = "development",
    endpoint: Optional[str] = None,
    enable_console_export: bool = False,
    enable_gcp_trace: bool = False,
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service (e.g., "imagen-api", "imagen-worker-upscale")
        environment: Environment name (development, staging, production)
        endpoint: OTLP endpoint (e.g., "http://jaeger:4317" for local Jaeger)
        enable_console_export: Export traces to console (useful for debugging)
        enable_gcp_trace: Export traces to Google Cloud Trace (production)

    Returns:
        Configured TracerProvider
    """
    global _tracer_provider, _tracer

    # Create resource with service information
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            "deployment.environment": environment,
            "service.version": os.getenv("IMAGE_TAG", "dev"),
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add exporters based on configuration
    exporters_added = []

    # 1. Console exporter (for debugging)
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        exporters_added.append("console")

    # 2. OTLP exporter (for Jaeger or other OTLP-compatible backends)
    if endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            exporters_added.append(f"OTLP ({endpoint})")
        except Exception as e:
            logger.warning(f"Failed to initialize OTLP exporter: {e}")

    # 3. Google Cloud Trace exporter (production)
    if enable_gcp_trace:
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

            gcp_exporter = CloudTraceSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(gcp_exporter))
            exporters_added.append("Google Cloud Trace")
        except ImportError:
            logger.warning("Google Cloud Trace exporter not available")
        except Exception as e:
            logger.warning(f"Failed to initialize GCP Trace exporter: {e}")

    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Create tracer
    _tracer = trace.get_tracer(service_name)

    logger.info(
        f"OpenTelemetry initialized for '{service_name}' "
        f"(environment: {environment}, exporters: {', '.join(exporters_added) or 'none'})"
    )

    return _tracer_provider


def setup_auto_instrumentation():
    """
    Enable automatic instrumentation for common libraries.

    Instruments:
    - HTTPX (HTTP client used by storage/queue services)
    - Redis (local development queue)
    - Logging (adds trace context to logs)
    """
    try:
        # Instrument HTTPX
        HTTPXClientInstrumentor().instrument()
        logger.debug("HTTPX instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument HTTPX: {e}")

    try:
        # Instrument Redis
        RedisInstrumentor().instrument()
        logger.debug("Redis instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")

    try:
        # Instrument logging (adds trace context to log records)
        LoggingInstrumentor().instrument(set_logging_format=False)
        logger.debug("Logging instrumentation enabled")
    except Exception as e:
        logger.warning(f"Failed to instrument logging: {e}")


def shutdown_telemetry():
    """Shutdown telemetry and flush all pending spans."""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry shut down")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def add_span_attributes(span: trace.Span, **attributes):
    """
    Add custom attributes to a span.

    Args:
        span: The span to add attributes to
        **attributes: Key-value pairs to add as attributes
    """
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))


def record_exception(span: trace.Span, exception: Exception):
    """
    Record an exception in the current span.

    Args:
        span: The span to record the exception in
        exception: The exception to record
    """
    span.record_exception(exception)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
