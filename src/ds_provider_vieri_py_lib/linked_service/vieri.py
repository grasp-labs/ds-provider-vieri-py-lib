"""
**File:** ``vieri.py``
**Region:** ``ds_provider_vieri_py_lib/linked_service/vieri.py``

Vieri Linked Service

This module implements a linked service for Vieri, allowing users to connect to and interact with
Vieri instance using client credentials.

example usage:
    >>> from ds_provider_vieri_py_lib.linked_service import VieriLinkedService, VieriLinkedServiceSettings
    >>> settings = VieriLinkedServiceSettings(
    ...     host="https://my-vieri-instance.com/api",
    ...     subscription_key="my-subscription-key"
    ... )
    >>> linked_service = VieriLinkedService(settings=settings)
    >>> linked_service.test_connection()

"""

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ds_common_logger_py_lib import Logger
from ds_protocol_http_py_lib import HttpLinkedService, HttpLinkedServiceSettings, enums

from ..enums import ResourceType

logger = Logger.get_logger(__name__, package=True)


@dataclass(kw_only=True)
class VieriLinkedServiceSettings(HttpLinkedServiceSettings):
    """
    Settings required to connect to a Vieri instance, extending HTTP linked service settings.

    Attributes:
        host (str): The base URL for the Vieri API.
        subscription_key (str): The subscription key for authentication.
    """

    host: str
    """ The base URL for the Vieri API (required). """

    subscription_key: str
    """ The subscription key for authentication (required). """

    auth_type: enums.AuthType = enums.AuthType.NO_AUTH
    """ The authentication type for the linked service. For Vieri, this is set to NO_AUTH since we use a
    subscription key in headers. """

    headers: dict[str, str] = field(init=False)
    """ The headers to include in API requests, automatically populated with the subscription key. """

    def __post_init__(self) -> None:
        self.headers = {"Ocp-Apim-Subscription-Key": self.subscription_key, "Content-Type": "application/json"}


VieriLinkedServiceSettingsType = TypeVar("VieriLinkedServiceSettingsType", bound=VieriLinkedServiceSettings)


@dataclass(kw_only=True)
class VieriLinkedService(HttpLinkedService[VieriLinkedServiceSettingsType], Generic[VieriLinkedServiceSettingsType]):
    """
    Linked service for connecting to a Vieri instance using client credentials.

    This class extends the HttpLinkedService and provides functionality specific to Vieri, such as
    handling authentication with a subscription key and making API requests to the Vieri instance.
    """

    settings: VieriLinkedServiceSettingsType
    """ The settings required to connect to the Vieri instance. This includes the base URL and subscription key. """

    @property
    def type(self) -> ResourceType:  # type: ignore[override]
        """
        Get the type of the linked service.

        Returns:
            ResourceType: The resource type for the Vieri linked service.
        """
        return ResourceType.VIERI_LINKED_SERVICE
