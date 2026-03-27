"""
**File:** ``__init__.py``
**Region:** ``ds_provider_vieri_py_lib/linked_service``

Vieri Linked Service

This module implements a linked service for Vieri and provides a session.

Example:
    >>> from ds_provider_vieri_py_lib.linked_service import VieriLinkedService, VieriLinkedServiceSettings
    >>> from uuid import UUID
    >>> settings = VieriLinkedServiceSettings(
    ...     host="https://api.vieri.com",
    ...     subscription_key="your_subscription_key"
    ... )
    >>> linked_service = VieriLinkedService(
    ...     id=UUID("your-uuid-here"),
            name="My Vieri Linked Service",
            version="1.0.0",
    ...     settings=settings
    ... )
    >>> linked_service.test_connection()

"""

from .vieri import VieriLinkedService, VieriLinkedServiceSettings

__all__ = ["VieriLinkedService", "VieriLinkedServiceSettings"]
