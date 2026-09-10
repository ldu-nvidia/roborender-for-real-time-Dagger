"""Authenticated bounded HTTP transport specialization for DAgger v4."""

from .binary_transport_v2 import BatchV2Transport, BoundedBearerBatchV2Transport
from .ndarray_batch_v4 import NDARRAY_BATCH_CONTENT_TYPE_V4


BatchV4Transport = BatchV2Transport


class BoundedBearerBatchV4Transport(BoundedBearerBatchV2Transport):
    """Reuse audited keep-alive/auth while carrying RDB4 bodies."""

    binary_content_type = NDARRAY_BATCH_CONTENT_TYPE_V4
    protocol_label = "v4"
    telemetry_event = "http_batch_request_v4"
