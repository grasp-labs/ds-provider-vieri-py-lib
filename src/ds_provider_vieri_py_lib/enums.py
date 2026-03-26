"""
**File:** ``enums.py``
**Region:** ``ds_provider_vieri_py_lib/enums``

Constants for VIERI provider.

Example:
    >>> ResourceType.VIERI_LINKED_SERVICE
    ds.resource.linked_service.vieri

"""

from enum import StrEnum


class ResourceType(StrEnum):
    """
    Constants for VIERI provider.
    """

    VIERI_LINKED_SERVICE = "ds.linked_service.vieri"
