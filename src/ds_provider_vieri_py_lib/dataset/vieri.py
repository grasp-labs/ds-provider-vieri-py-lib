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

    page_size: int = 20
    """Number of records per page for pagination. Optional, defaults to 20"""

    offset: int = 0
    """Number of records to skip before starting pagination. Optional, defaults to 0"""

    last_modified: str | None = None
    """Vieri date string (YYYY-MM-DD) to filter results modified after this date. Optional filter parameter for API requests."""


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

    # Inherited from base class; type hint silences linter warnings
    output: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

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
        logger.info(
            "Starting read operation for VieriDataset with owner_id=%s, product_name=%s",
            self.settings.owner_id,
            self.settings.product_name,
        )
        all_results: list[dict[str, Any]] = []
        last_offset: int = 0
        final_page_count: int = 0  # Track count of final page for accurate checkpoint

        try:
            url = f"{self.linked_service.settings.host}/{self.settings.owner_id}/api/public/{self.settings.product_name}"

            # Build initial params considering checkpoint state
            params = self._build_request_params()
            last_offset = params.get("Skip", 0)

            # Paginate through all results
            while True:
                response = self.linked_service.connection.get(url, headers=self.linked_service.settings.headers, params=params)
                response.raise_for_status()

                data = response.json()

                # Validate response structure: expect dict with "Results" key containing a list
                if not isinstance(data, dict):
                    raise ReadError(
                        message=f"Invalid Vieri API response: expected dict, got {type(data).__name__}",
                        details={"response_type": type(data).__name__, "response": str(data)[:200]},
                    )

                if "Results" not in data:
                    raise ReadError(
                        message="Invalid Vieri API response: missing 'Results' key in response",
                        details={"response_keys": list(data.keys()), "response": str(data)[:200]},
                    )

                results = data["Results"]

                # Validate that Results is a list
                if not isinstance(results, list):
                    raise ReadError(
                        message=f"Invalid Vieri API response: 'Results' is {type(results).__name__}, expected list",
                        details={"results_type": type(results).__name__, "results": str(results)[:200]},
                    )

                if not results:
                    break

                # Track count of records in this batch
                batch_count = len(results)
                all_results.extend(results)
                last_offset = params.get("Skip", 0)
                final_page_count = batch_count  # Update for checkpoint calculation

                # Check if we got less results than requested (last page)
                if batch_count < params.get("Take", 0):
                    break

                # Increment skip for next page using actual count fetched
                params["Skip"] = params.get("Skip", 0) + batch_count

        except ReadError:
            raise
        except Exception as exc:
            logger.error("Error during read operation: %s", exc)
            # Wrap backend exceptions per contract: never leak raw errors
            raise ReadError(
                message=f"Failed to read data from Vieri API: {exc}",
                details={"owner_id": self.settings.owner_id, "product_name": self.settings.product_name},
            ) from exc
        finally:
            # Always populate output and checkpoint, even with partial results
            self.output = pd.DataFrame(all_results)
            self._build_checkpoint(last_offset, final_page_count)

    def _build_request_params(self) -> dict[str, Any]:
        """
        Build request parameters considering checkpoint state.

        - **Full load** (empty checkpoint): Uses settings for pagination and scope
        - **Incremental load** (populated checkpoint): Uses checkpoint for offset and last_modified

        Settings define the static scope, checkpoint tracks the moving position.
        When both apply (e.g., last_modified), settings provide the lower bound,
        checkpoint narrows further by tracking progress within that scope.

        :return: Dictionary with API request parameters (Skip, Take, ModifiedAfter).
        """
        is_full_load = not self.checkpoint or self.checkpoint == {}

        if is_full_load:
            # Full load: use settings values as scope
            skip = self.settings.offset
            take = self.settings.page_size
            last_modified = self.settings.last_modified
        else:
            # Incremental/resuming load: use checkpoint to continue from last position
            skip = self.checkpoint.get("offset", self.settings.offset)
            take = self.checkpoint.get("page_size", self.settings.page_size)
            # Checkpoint last_modified takes precedence (narrowed from settings scope)
            last_modified = self.checkpoint.get("last_modified", self.settings.last_modified)

        # Build params dict
        params: dict[str, Any] = {"Skip": skip, "Take": take}

        # Add ModifiedAfter filter if specified (from scope or checkpoint)
        # Validate date format using parse_vieri_date
        if last_modified:
            # Validate the date format is correct (raises ValueError if invalid)
            self.parse_vieri_date(last_modified)
            # Use validated date in parameters
            params["ModifiedAfter"] = last_modified

        return params

    def _build_checkpoint(self, last_offset: int, final_page_count: int) -> None:
        """
        Build and set the checkpoint dict with pagination state.

        Saves the last offset to enable resuming from this exact position
        on the next incremental load.

        :param last_offset: The last Skip offset value that was successfully fetched.
        :param final_page_count: The actual number of records in the final batch (for accurate offset).
        """
        if not self.supports_checkpoint:
            return

        # Use getattr to safely access checkpoint from base class and satisfy type checker
        if getattr(self, "checkpoint", None) is None:
            self.checkpoint = {}

        # Get current params to preserve modifiers (like last_modified)
        current_params = self._build_request_params()

        # Update offset for pagination persistence using actual record count from final batch
        # This prevents overshooting on partial/last pages
        self.checkpoint["offset"] = last_offset + final_page_count

        # Preserve last_modified in checkpoint if it exists (for incremental resumption)
        if current_params.get("ModifiedAfter"):
            self.checkpoint["last_modified"] = current_params["ModifiedAfter"]

        logger.debug(
            "Checkpoint set: offset=%s, last_modified=%s",
            self.checkpoint.get("offset"),
            self.checkpoint.get("last_modified"),
        )

    def parse_vieri_date(self, date_str: str) -> datetime:
        """Parse a Vieri date string (YYYY-MM-DD) to a datetime object. Strictly enforces format."""
        try:
            return datetime.strptime(date_str, VIERI_DATETIME_FORMAT)
        except ValueError as e:
            raise ValueError(f"Date must be in '{VIERI_DATETIME_FORMAT}' format, got: {date_str}") from e

    def format_vieri_date(self, dt: datetime) -> str:
        """Format a datetime object to a Vieri date string (YYYY-MM-DD)."""
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
