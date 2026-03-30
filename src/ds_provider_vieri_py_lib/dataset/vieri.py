"""
**File:** ``vieri.py``
**Region:** ``ds_provider_vieri_py_lib/dataset/vieri.py``

Vieri Dataset

This module implements a dataset for Vieri APIs.

Example:
    >>> from uuid import uuid4
    >>> dataset = VieriDataset(
    ...     id=uuid4(),
    ...     name="employees_dataset",
    ...     version="1.0.0",
    ...     settings=VieriDatasetSettings(
    ...         data_product=VieriDataProducts.EMPLOYEES,
    ...         read=ReadSettings(page_size=100),
    ...     ),
    ...     linked_service=VieriLinkedService(
    ...         id=uuid4(),
    ...         name="vieri_connection",
    ...         version="1.0.0",
    ...         settings=VieriLinkedServiceSettings(
    ...             host="https://api.vieri.com",
    ...             subscription_key="your_subscription_key"
    ...         ),
    ...     ),
    ... )
    >>> linked_service = dataset.linked_service
    >>> linked_service.connect()
    >>> dataset.read()
    >>> data = dataset.output
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import pandas as pd
from ds_common_logger_py_lib import Logger
from ds_resource_plugin_py_lib.common.resource.dataset import DatasetSettings, DatasetStorageFormatType, TabularDataset
from ds_resource_plugin_py_lib.common.resource.dataset.errors import (
    ReadError,
)
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError
from ds_resource_plugin_py_lib.common.serde.deserialize import PandasDeserializer
from ds_resource_plugin_py_lib.common.serde.serialize import PandasSerializer

from ..enums import VIERI_DATETIME_FORMAT, ResourceType
from ..linked_service.vieri import VieriLinkedService

logger = Logger.get_logger(__name__, package=True)


@dataclass(kw_only=True)
class VieriDatasetSettings(DatasetSettings):
    """
    Settings for Vieri dataset, extending the base DatasetSettings.
    Includes Vieri API endpoint and query parameters.
    """

    owner_id: str
    """The owner ID of the dataset in Vieri. This should be a valid identifier corresponding to the dataset owner in Vieri."""

    product_name: str
    """The Vieri product name to connect to. This should match one of the products available in the Vieri API."""

    take: int | None = None
    """Page size for pagination. Optional."""

    skip: int | None = None
    """Offset for pagination. Optional."""

    modified_after: str | None = None
    """ISO8601 string or datetime for incremental loads. Optional."""


VieriDatasetSettingsType = TypeVar(
    "VieriDatasetSettingsType",
    bound=VieriDatasetSettings,
)
VieriLinkedServiceType = TypeVar(
    "VieriLinkedServiceType",
    bound=VieriLinkedService[Any],
)


@dataclass(kw_only=True)
class VieriDataset(
    TabularDataset[VieriLinkedServiceType, VieriDatasetSettingsType, PandasSerializer, PandasDeserializer],
    Generic[VieriLinkedServiceType, VieriDatasetSettingsType],
):
    linked_service: VieriLinkedServiceType
    settings: VieriDatasetSettingsType

    serializer: PandasSerializer | None = field(
        default_factory=lambda: PandasSerializer(format=DatasetStorageFormatType.JSON),
    )
    deserializer: PandasDeserializer | None = field(
        default_factory=lambda: PandasDeserializer(format=DatasetStorageFormatType.JSON),
    )

    @property
    def type(self) -> ResourceType:
        return ResourceType.VIERI_DATASET

    @property
    def supports_checkpoint(self) -> bool:
        """Indicate that this dataset supports checkpointing for incremental loads."""
        return True

    def read(self) -> None:
        """
        Read data from the requested endpoint of the Vieri API, handling pagination and checkpoint for incremental loads.

        Raises:
            ReadError: If reading data fails.
        """
        url = (
            f"{self.linked_service.settings.host}/api/v1/owners/"
            f"{self.settings.owner_id}/products/{self.settings.product_name}/data"
        )
        headers = self.linked_service.settings.headers
        take = self.settings.take or 1000
        skip = self.settings.skip or 0
        params = {}
        modified_after = None
        if self.checkpoint and isinstance(self.checkpoint, dict) and self.checkpoint.get("modified_after"):
            modified_after = self.checkpoint["modified_after"]
            # Try to parse and reformat to ensure correct format
            if modified_after is not None:
                try:
                    modified_after = self.format_vieri_date(self.parse_vieri_date(modified_after))
                except Exception as e:
                    raise ReadError(f"Invalid date format for checkpoint.modified_after: {modified_after}") from e
        elif self.settings.modified_after is not None:
            modified_after = self.settings.modified_after
            # Try to parse and reformat to ensure correct format
            if modified_after is not None:
                try:
                    modified_after = self.format_vieri_date(self.parse_vieri_date(modified_after))
                except Exception as e:
                    raise ReadError(f"Invalid date format for settings.modified_after: {modified_after}") from e
        if modified_after is not None:
            params["modifiedAfter"] = modified_after
        try:
            all_results, latest_modified = self._fetch_all_pages(url, headers, params, take, skip)
            self.output = pd.DataFrame(all_results)
            self._update_checkpoint(latest_modified)
        except Exception as e:
            raise ReadError(f"Failed to fetch data from Vieri API: {e}") from e

    def _fetch_all_pages(
        self,
        url: str,
        headers: dict[str, str],
        params: dict[str, Any],
        take: int,
        skip: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Helper to fetch all paginated results from the Vieri API."""
        all_results: list[dict[str, Any]] = []
        latest_modified = params.get("modifiedAfter")
        while True:
            page_params = params.copy()
            page_params["take"] = take
            page_params["skip"] = skip
            response = self.linked_service.session.get(url, headers=headers, params=page_params)  # type: ignore[attr-defined]
            response.raise_for_status()
            data = response.json()
            results = data["results"] if isinstance(data, dict) and "results" in data else data
            if not results:
                break
            all_results.extend(results)
            for row in results:
                row_modified = row.get("modified", None)
                if row_modified and (latest_modified is None or row_modified > latest_modified):
                    latest_modified = row_modified
            if len(results) < take:
                break
            skip += take
        return all_results, latest_modified

    def _update_checkpoint(self, latest_modified: str | None) -> None:
        """Helper to update the checkpoint with the latest modified value."""
        if latest_modified is not None:
            self.checkpoint = {"modified_after": latest_modified}

    def parse_vieri_date(self, date_str: str) -> datetime:
        """Parse a Vieri date string to a datetime object."""
        return datetime.strptime(date_str, VIERI_DATETIME_FORMAT)

    def format_vieri_date(self, dt: datetime) -> str:
        """Format a datetime object to a Vieri date string."""
        return dt.strftime(VIERI_DATETIME_FORMAT)

    def create(self) -> None:
        """
        Create a new dataset in Vieri.

        Raises:
            CreateError: If creating the dataset fails.
        """
        raise NotImplementedError("Create method not implemented yet for VieriDataset.")

    def update(self) -> None:
        """
        Update the dataset in Vieri.

        Raises:
            UpdateError: If updating the dataset fails.
        """
        raise NotImplementedError("Update method not implemented yet for VieriDataset.")

    def delete(self) -> None:
        """
        Delete the dataset from Vieri.

        Raises:
            DeleteError: If deleting the dataset fails.
        """
        raise NotImplementedError("Delete method not implemented yet for VieriDataset.")

    def close(self) -> None:
        """
        Close any open connections or resources.

        Raises:
            Exception: If closing the dataset fails.
        """
        logger.info("Closing VieriDataset and releasing resources.")

    def rename(self) -> None:
        """Rename is not supported by Vieri provider."""
        raise NotSupportedError("Method (rename) not supported by Vieri provider.")

    def list(self) -> None:
        """List is not supported by Vieri provider."""
        raise NotSupportedError("Method (list) not supported by Vieri provider.")

    def upsert(self) -> None:
        """Upsert is not supported by Vieri provider."""
        raise NotSupportedError("Method (upsert) not supported by Vieri provider.")

    def purge(self) -> None:
        """Purge is not supported by Vieri provider."""
        raise NotSupportedError("Method (purge) not supported by Vieri provider.")
