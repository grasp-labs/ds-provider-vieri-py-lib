"""
**File:** ``enums.py``
**Region:** ``ds_provider_vieri_py_lib/enums``

Constants for VIERI provider.

Example:
    >>> ResourceType.VIERI_LINKED_SERVICE
    ds.resource.linked-service.vieri

"""

from enum import StrEnum

# Vieri date format constant
VIERI_DATETIME_FORMAT = "%Y-%m-%d"


class ResourceType(StrEnum):
    """
    Constants for VIERI provider.
    """

    VIERI_LINKED_SERVICE = "ds.resource.linked-service.vieri"
    VIERI_DATASET = "ds.resource.dataset.vieri"


class VieriApiType(StrEnum):
    """
    Enum for supported Vieri API types.

    Attributes:
        IVAR: IVAR API for general read/write operations.
        DATALOADER: Vieri Dataloader API for bulk data loading.
    """

    IVAR = "ivar"
    DATALOADER = "dataloader"
