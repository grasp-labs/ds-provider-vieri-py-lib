"""
**File:** ``vieri.py``
**Region:** ``ds_provider_vieri_py_lib/dataset/vieri.py``

Vieri Dataset

This module implements a dataset for Vieri APIs with support for both read and write operations.

Example (Read):
    >>> from uuid import uuid4
    >>> dataset = VieriDataset(
    ...     id=uuid4(),
    ...     name="accounts_dataset",
    ...     version="1.0.0",
    ...     settings=VieriDatasetSettings(
    ...         owner_id="my_owner",
    ...         product_name="Accounts",
    ...         read=VieriReadSettings(page_size=100),
    ...     ),
    ...     linked_service=VieriLinkedService(
    ...         id=uuid4(),
    ...         name="vieri_connection",
    ...         version="1.0.0",
    ...         settings=VieriLinkedServiceSettings(
    ...             host="https://vieri-api.azure-api.net",
    ...             subscription_key="your_subscription_key"
    ...         ),
    ...     ),
    ... )
    >>> dataset.linked_service.connect()
    >>> dataset.read()
    >>> data = dataset.output
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import pandas as pd
from ds_common_logger_py_lib import Logger
from ds_common_serde_py_lib import Serializable
from ds_resource_plugin_py_lib.common.resource.dataset import DatasetSettings, DatasetStorageFormatType, TabularDataset
from ds_resource_plugin_py_lib.common.resource.dataset.errors import (
    CreateError,
    ReadError,
)
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError
from ds_resource_plugin_py_lib.common.serde.deserialize import PandasDeserializer
from ds_resource_plugin_py_lib.common.serde.serialize import PandasSerializer

from ..enums import VIERI_DATETIME_FORMAT, ResourceType
from ..linked_service.vieri import VieriLinkedService

logger = Logger.get_logger(__name__, package=True)


@dataclass(kw_only=True)
class VieriReadSettings(Serializable):
    """
    Settings specific to the read() operation.

    These settings only apply when reading data from the Vieri API
    and do not affect create() operations.
    """

    page_size: int = 20
    """Number of records per page for pagination. Optional, defaults to 20"""

    offset: int = 0
    """Number of records to skip before starting pagination. Optional, defaults to 0"""

    last_modified: str | None = None
    """Vieri date string (YYYY-MM-DD) to filter results modified after this date. Optional filter parameter for API requests."""


@dataclass(kw_only=True)
class VieriCreateSettings(Serializable):
    """
    Settings specific to the create() operation.

    These settings only apply when writing data to the Vieri API
    and do not affect read() operations.
    """

    write_endpoint: str | None = None
    """Endpoint path for POST operations (e.g., 'vieri-dataloader/LoadAccounts').
    If not set, write operations are not supported."""


@dataclass(kw_only=True)
class VieriDatasetSettings(DatasetSettings):
    """
    Settings for Vieri dataset, extending the base DatasetSettings.
    Separates read and write configuration into method-specific settings.
    """

    owner_id: str
    """The owner ID of the dataset in Vieri. This should be a valid identifier corresponding to the dataset owner in Vieri."""

    product_name: str
    """The Vieri product name to connect to. This should match one of the products available in the Vieri API."""

    read: VieriReadSettings = field(default_factory=VieriReadSettings)
    """Settings for read() operation."""

    create: VieriCreateSettings = field(default_factory=VieriCreateSettings)
    """Settings for create() operation."""


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
            skip = self.settings.read.offset
            take = self.settings.read.page_size
            last_modified = self.settings.read.last_modified
        else:
            # Incremental/resuming load: use checkpoint to continue from last position
            skip = self.checkpoint.get("offset", self.settings.read.offset)
            take = self.checkpoint.get("page_size", self.settings.read.page_size)
            # Checkpoint last_modified takes precedence (narrowed from settings scope)
            last_modified = self.checkpoint.get("last_modified", self.settings.read.last_modified)

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
        Create (insert) new records in Vieri via POST request.

        Uses data from self.input (provided by the caller) and POSTs it to the Vieri write endpoint.
        Populates self.output with affected rows (backend response or copy of self.input).
        Empty input is a no-op and returns immediately.

        Raises:
            CreateError: If the write operation fails (including if the API doesn't support writes).
        """
        # Check if write_endpoint is configured
        if not self.settings.create.write_endpoint:
            raise NotSupportedError("Write operations not supported: write_endpoint is not configured")

        if self.input is None or self.input.empty:
            logger.info("create() called with empty input, returning immediately (no-op)")
            self.output = pd.DataFrame()
            return

        logger.info(
            "Starting create operation for VieriDataset with product_name=%s, records=%d",
            self.settings.product_name,
            len(self.input),
        )

        try:
            # Build write endpoint URL
            url = f"{self.linked_service.settings.host}/{self.settings.create.write_endpoint}"

            # Convert DataFrame to list of dicts for JSON payload
            records = self.input.to_dict(orient="records")
            payload = {"Records": records}

            logger.debug(
                "POSTing %d records to Vieri endpoint: %s",
                len(records),
                url,
            )

            # Send POST request
            response = self.linked_service.connection.post(
                url,
                json=payload,
                headers=self.linked_service.settings.headers,
            )
            response.raise_for_status()

            logger.info(
                "Successfully created %d records in Vieri for product_name=%s",
                len(records),
                self.settings.product_name,
            )

            # The API returns 200 on success; populate output with input records
            self.output = self.input.copy()

        except NotSupportedError:
            raise
        except Exception as exc:
            logger.error("Error during create operation: %s", exc)
            raise CreateError(
                message=f"Failed to create records in Vieri API: {exc}",
                details={
                    "product_name": self.settings.product_name,
                    "records_count": len(self.input),
                },
            ) from exc

    def update(self) -> None:
        """
        Update is not supported by the Vieri provider.

        Raises:
            NotSupportedError: Update operations are not available.
        """
        raise NotSupportedError("Method (update) not supported by Vieri provider.")

    def delete(self) -> None:
        """
        Delete is not supported by the Vieri provider.

        Raises:
            NotSupportedError: Delete operations are not available.
        """
        raise NotSupportedError("Method (delete) not supported by Vieri provider.")

    def close(self) -> None:
        """
        Close any open connections or resources.

        Raises:
            Exception: If closing the dataset fails.
        """
        logger.info("Closing VieriDataset")

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
