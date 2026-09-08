"""Network boundaries for the distributed Raven DAgger pipeline.

The package deliberately contains orchestration services only.  Image3F model
execution remains in :mod:`pipeline.dagger.image3f_server`, and Isaac remains
on the RTX actor host.
"""

SERVICE_SCHEMA = "roborender.dagger.services.v1"
