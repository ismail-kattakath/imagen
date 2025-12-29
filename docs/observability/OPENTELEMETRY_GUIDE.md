# OpenTelemetry Tracing Guide

Complete guide to distributed tracing in the Imagen platform using OpenTelemetry.

## Overview

The Imagen platform implements comprehensive distributed tracing to provide end-to-end visibility into:
- **API Requests** - HTTP request/response lifecycle
- **Job Processing** - Complete workflow from API → Pub/Sub → Worker → Triton
- **External Dependencies** - GCS, Pub/Sub, Redis, Triton calls
- **Performance** - Latency breakdown for each component
- **Errors** - Exception tracking across service boundaries

---

## Architecture

### Trace Flow

```
API Request (FastAPI)
  ↓
  span: http.server.request
  ↓
Job Creation (Queue Service)
  ↓
  span: pubsub.publish
  └─→ trace context injected as message attributes
       ↓
       Pub/Sub Message
       ↓
Worker Receives Message (Queue Service)
  ↓
  span: pubsub.receive (trace context extracted)
  ↓
Job Processing (Triton Worker)
  ↓
  span: process_job.upscale
    ↓
    span: storage.download
    ↓
    span: triton.infer
    ↓
    span: storage.upload
```

### Components Instrumented

1. **FastAPI (src/api/main.py)**
   - Automatic HTTP span creation
   - Request ID correlation
   - Custom attributes (user, API key presence)

2. **Workers (src/workers/triton_worker.py)**
   - Job processing spans
   - Triton inference spans
   - Storage operation spans

3. **Queue Service (src/services/queue.py)**
   - Trace context propagation via Pub/Sub attributes
   - Context extraction on message receive

4. **Logging (src/api/middleware/logging.py)**
   - Trace ID/Span ID added to all logs
   - Cloud Logging integration

---

## Configuration

### Environment Variables

```bash
# Enable/disable tracing
OTEL_ENABLED=true

# Service name (override per service)
OTEL_SERVICE_NAME=imagen-api

# OTLP endpoint (Jaeger for local dev)
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

# Console exporter (debugging)
OTEL_EXPORTER_CONSOLE=false

# Google Cloud Trace (production)
OTEL_EXPORTER_GCP_TRACE=true

# Sampling strategy
OTEL_TRACES_SAMPLER=parentbased_always_on
```

### Sampling Strategies

- **`always_on`** - Sample 100% of traces (development)
- **`always_off`** - No sampling (disable tracing)
- **`traceidratio`** - Sample X% of traces
- **`parentbased_always_on`** - Always sample if parent is sampled (default)

---

## Local Development

### 1. Start Jaeger

```bash
# Start with docker-compose
docker compose up jaeger

# Or standalone
docker run -d \
  --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -e COLLECTOR_OTLP_ENABLED=true \
  jaegertracing/all-in-one:1.52
```

### 2. Configure API

```bash
# .env
OTEL_ENABLED=true
OTEL_SERVICE_NAME=imagen-api
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### 3. Configure Workers

```bash
# Worker automatically gets service name: imagen-worker-{job_type}
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

### 4. Access Jaeger UI

Open http://localhost:16686 in your browser.

**Search for traces:**
- Service: `imagen-api` or `imagen-worker-upscale`
- Operation: `GET /api/v1/jobs/{id}`, `process_job.upscale`
- Tags: `job.id`, `http.status_code`, `error=true`

---

## Production (Google Cloud Trace)

### 1. Enable Cloud Trace API

```bash
gcloud services enable cloudtrace.googleapis.com
```

### 2. Configure Exporters

```bash
# Cloud Run API
OTEL_ENABLED=true
OTEL_SERVICE_NAME=imagen-api
OTEL_EXPORTER_GCP_TRACE=true
OTEL_EXPORTER_OTLP_ENDPOINT=  # Leave empty for GCP Trace

# Workers (via environment in k8s/workers/*.yaml)
OTEL_ENABLED=true
OTEL_EXPORTER_GCP_TRACE=true
```

### 3. IAM Permissions

Workers and API need Cloud Trace Agent role:

```bash
# For Cloud Run service account
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudtrace.agent"

# For GKE Workload Identity
gcloud iam service-accounts add-iam-policy-binding \
  GKE_SA@PROJECT_ID.iam.gserviceaccount.com \
  --role roles/cloudtrace.agent \
  --member "serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/KSA]"
```

### 4. View Traces

- **Console**: https://console.cloud.google.com/traces
- **Filter by service**: `imagen-api`, `imagen-worker-upscale`
- **Correlate with logs**: Trace ID appears in Cloud Logging

---

## Trace Context Propagation

### Pub/Sub Message Attributes

Trace context is automatically propagated through Pub/Sub messages:

**Publisher (src/services/queue.py)**:
```python
# Inject trace context into message attributes
carrier: dict[str, str] = {}
inject(carrier)  # Adds traceparent, tracestate

future = self.publisher.publish(
    topic_path,
    message_bytes,
    **carrier,  # Propagates context
)
```

**Subscriber (src/services/queue.py)**:
```python
# Extract trace context from message
carrier = dict(message.attributes)
ctx = propagator.extract(carrier)

# Continue trace
with trace.use_span(trace.NonRecordingSpan(ctx)):
    callback(data)
```

This ensures traces flow: **API → Pub/Sub → Worker** seamlessly.

---

## Custom Spans

### Adding Business Logic Spans

```python
from src.core.telemetry import get_tracer

tracer = get_tracer("imagen.my_module")

def process_image(image):
    with tracer.start_as_current_span("image_preprocessing") as span:
        span.set_attribute("image.size", len(image))
        span.set_attribute("image.format", image.format)

        # Your processing logic
        result = expensive_operation(image)

        span.set_attribute("processing.duration_ms", 123.45)
        return result
```

### Recording Errors

```python
try:
    result = risky_operation()
except Exception as e:
    span = trace.get_current_span()
    span.record_exception(e)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
    raise
```

---

## Span Attributes

### Standard Attributes

Automatically added by instrumentation:

- `http.method` - HTTP method (GET, POST)
- `http.url` - Request URL
- `http.status_code` - Response status
- `net.peer.ip` - Client IP address
- `request.id` - Request correlation ID

### Custom Attributes (Imagen-specific)

**API spans**:
- `request.id` - Request ID for log correlation
- `client.ip` - Client IP address
- `auth.api_key_present` - Whether API key was provided

**Job spans**:
- `job.id` - Job identifier
- `job.type` - Job type (upscale, enhance, etc.)
- `job.status` - Job status (processing, completed, failed)
- `job.duration_ms` - Total job duration
- `model.name` - Triton model name
- `inference.duration_ms` - Pure inference time

**Storage spans**:
- `storage.operation` - upload/download
- `storage.path` - GCS path

---

## Log Correlation

Logs automatically include trace context for correlation:

**JSON Log Entry**:
```json
{
  "timestamp": "2025-12-29T18:30:00Z",
  "severity": "INFO",
  "message": "Processing job abc-123",
  "request_id": "req-xyz",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "logging.googleapis.com/trace": "projects/my-project/traces/4bf92f3577b34da6a3ce929d0e0e4736",
  "logging.googleapis.com/spanId": "00f067aa0ba902b7"
}
```

**Benefits**:
- Click trace ID in Cloud Logging → View full trace in Cloud Trace
- Click span ID → Jump to specific span
- All logs for a request grouped by `request_id`

---

## Troubleshooting

### No Traces Appearing

1. **Check OTEL_ENABLED**: Must be `true`
2. **Check endpoint**: `OTEL_EXPORTER_OTLP_ENDPOINT` must be reachable
3. **Check Jaeger logs**: `docker logs jaeger`
4. **Enable console export**: `OTEL_EXPORTER_CONSOLE=true` to see spans in logs

### Traces Not Connecting

1. **Trace context propagation**: Ensure Pub/Sub attributes are not dropped
2. **Check worker initialization**: Workers must call `setup_telemetry()`
3. **Verify span creation**: Worker should create root span for each job

### Performance Impact

- **Overhead**: ~1-2% CPU, negligible latency (<1ms per span)
- **Sampling**: Use `traceidratio` to reduce volume in production
- **Batch export**: Spans are exported asynchronously in batches

---

## Best Practices

### 1. Meaningful Span Names

✅ Good:
- `process_job.upscale`
- `triton.infer`
- `storage.upload`

❌ Bad:
- `function_call`
- `step_1`

### 2. Rich Attributes

Add context that helps debugging:
```python
span.set_attribute("job.id", job_id)
span.set_attribute("model.name", model_name)
span.set_attribute("param.scale", scale_factor)
```

### 3. Error Handling

Always record exceptions:
```python
except Exception as e:
    span.record_exception(e)
    span.set_status(trace.Status(trace.StatusCode.ERROR))
    raise
```

### 4. Sampling

Production sampling example:
```bash
# Sample 10% of traces
OTEL_TRACES_SAMPLER=traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1
```

### 5. Sensitive Data

Never add sensitive data to span attributes:
```python
# ❌ BAD
span.set_attribute("user.password", password)
span.set_attribute("api.key", api_key)

# ✅ GOOD
span.set_attribute("user.id", user_id)
span.set_attribute("auth.method", "api_key")
```

---

## Metrics Integration

Traces complement Prometheus metrics:

**Metrics** - System-wide aggregates:
- Request rate
- Error rate
- P50/P95/P99 latency

**Traces** - Individual request details:
- Exact latency breakdown
- Specific failure cause
- Distributed call graph

Use both for complete observability!

---

## References

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [Google Cloud Trace](https://cloud.google.com/trace/docs)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)

---

**Last Updated**: 2025-12-29
**Status**: Production-ready OpenTelemetry implementation
