"""Authenticated bounded HTTP transport specialization for DAgger v3."""

from .binary_transport_v2 import BatchV2Transport, BoundedBearerBatchV2Transport
from .ndarray_batch_v3 import NDARRAY_BATCH_CONTENT_TYPE_V3


BatchV3Transport = BatchV2Transport


class BoundedBearerBatchV3Transport(BoundedBearerBatchV2Transport):
    """Reuse the audited keep-alive/auth implementation with v3 wire identity."""

    binary_content_type = NDARRAY_BATCH_CONTENT_TYPE_V3
    protocol_label = "v3"
    telemetry_event = "http_batch_request_v3"
