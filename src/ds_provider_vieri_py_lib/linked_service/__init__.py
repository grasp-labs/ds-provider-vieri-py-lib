"""
**File:** ``__init__.py``
**Region:** ``ds_provider_simployer_py_lib/linked_service``

Simployer Linked Service

This module implements a linked service for Simployer and provides a session.

Example:
    from ds_provider_simployer_py_lib.linked_service import SimployerLinkedService, SimployerLinkedServiceSettings

    settings = SimployerLinkedServiceSettings(
        host="https://api.simployer.com",
        api_key="your_api_key"
    )
    linked_service = SimployerLinkedService(settings=settings)
    linked_service.test_connection()

"""

from .vieri import VieriLinkedService, VieriLinkedServiceSettings

__all__ = ["VieriLinkedService", "VieriLinkedServiceSettings"]
