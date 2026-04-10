"""Unit tests for VieriDataset with checkpoint support."""

import logging
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock
from uuid import uuid4

import pandas as pd
import pytest
from ds_resource_plugin_py_lib.common.resource.dataset.errors import ReadError
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError

from ds_provider_vieri_py_lib.dataset.vieri import VieriDataset, VieriDatasetSettings
from ds_provider_vieri_py_lib.enums import ResourceType
from ds_provider_vieri_py_lib.linked_service.vieri import VieriLinkedService, VieriLinkedServiceSettings


@pytest.fixture
def mock_linked_service():
    """Mock linked service with mocked connection."""
    settings = VieriLinkedServiceSettings(host="https://api.vieri.com", subscription_key="test_key")
    svc = VieriLinkedService(id=uuid4(), name="test_service", version="1.0.0", settings=settings)
    # Mock the connection property to avoid initialization errors
    type(svc).connection = PropertyMock(return_value=MagicMock())
    return svc


@pytest.fixture
def vieri_settings():
    """Default VieriDatasetSettings for testing."""
    return VieriDatasetSettings(
        owner_id="test_owner",
        product_name="test_product",
        page_size=20,
        offset=0,
        last_modified=None,
    )


@pytest.fixture
def vieri_dataset(mock_linked_service, vieri_settings):
    """Create a VieriDataset instance for testing."""
    return VieriDataset(
        id=uuid4(),
        name="test_vieri_dataset",
        version="1.0.0",
        linked_service=mock_linked_service,
        settings=vieri_settings,
    )


class TestVieriDatasetBasics:
    """Test basic properties and methods."""

    def test_type_property(self, vieri_dataset):
        """Test that type property returns correct ResourceType."""
        assert vieri_dataset.type == ResourceType.VIERI_DATASET

    def test_supports_checkpoint_property(self, vieri_dataset):
        """Test that checkpoint support is enabled."""
        assert vieri_dataset.supports_checkpoint is True

    def test_close_succeeds(self, vieri_dataset, caplog):
        """Test that close() logs and succeeds."""
        caplog.set_level(logging.INFO)
        vieri_dataset.close()
        assert "Closing VieriDataset" in caplog.text

    def test_rename_not_supported(self, vieri_dataset):
        """Test that rename raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.rename()

    def test_list_not_supported(self, vieri_dataset):
        """Test that list raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.list()

    def test_upsert_not_supported(self, vieri_dataset):
        """Test that upsert raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.upsert()

    def test_purge_not_supported(self, vieri_dataset):
        """Test that purge raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.purge()

    def test_create_not_implemented(self, vieri_dataset):
        """Test that create raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            vieri_dataset.create()

    def test_update_not_implemented(self, vieri_dataset):
        """Test that update raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            vieri_dataset.update()

    def test_delete_not_implemented(self, vieri_dataset):
        """Test that delete raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            vieri_dataset.delete()


class TestBuildRequestParams:
    """Test request parameter building with checkpoint logic."""

    def test_build_params_full_load_empty_checkpoint(self, vieri_dataset):
        """Test that full load uses settings when checkpoint is empty."""
        vieri_dataset.checkpoint = {}
        vieri_dataset.settings.page_size = 25
        vieri_dataset.settings.offset = 0
        vieri_dataset.settings.last_modified = "2024-01-01"

        params = vieri_dataset._build_request_params()

        assert params["Skip"] == 0
        assert params["Take"] == 25
        assert params["ModifiedAfter"] == "2024-01-01"

    def test_build_params_full_load_no_checkpoint(self, vieri_dataset):
        """Test that full load uses settings when checkpoint is None."""
        vieri_dataset.checkpoint = None
        vieri_dataset.settings.page_size = 50
        vieri_dataset.settings.offset = 10
        vieri_dataset.settings.last_modified = None

        params = vieri_dataset._build_request_params()

        assert params["Skip"] == 10
        assert params["Take"] == 50
        assert "ModifiedAfter" not in params

    def test_build_params_incremental_load_with_checkpoint(self, vieri_dataset):
        """Test that incremental load uses checkpoint values."""
        vieri_dataset.checkpoint = {
            "offset": 100,
            "page_size": 30,
            "last_modified": "2024-02-01",
        }
        vieri_dataset.settings.page_size = 20
        vieri_dataset.settings.offset = 0
        vieri_dataset.settings.last_modified = "2024-01-01"

        params = vieri_dataset._build_request_params()

        assert params["Skip"] == 100
        assert params["Take"] == 30
        assert params["ModifiedAfter"] == "2024-02-01"


class TestBuildAndSetCheckpoint:
    """Test checkpoint building and setting logic."""

    def test_build_and_set_checkpoint_preserves_modifiers(self, vieri_dataset):
        """Test that checkpoint preserves ModifiedAfter filter."""
        vieri_dataset.settings.page_size = 20
        vieri_dataset.settings.last_modified = "2024-01-01"

        vieri_dataset._build_checkpoint(last_offset=60)

        assert vieri_dataset.checkpoint["offset"] == 80
        assert vieri_dataset.checkpoint["last_modified"] == "2024-01-01"

    def test_build_and_set_checkpoint_no_modifiers(self, vieri_dataset):
        """Test checkpoint without filters."""
        vieri_dataset.settings.page_size = 25
        vieri_dataset.settings.last_modified = None

        vieri_dataset._build_checkpoint(last_offset=50)

        assert vieri_dataset.checkpoint["offset"] == 75


class TestReadOperation:
    """Test the read() method and error handling."""

    def test_read_full_load_success(self, vieri_dataset):
        """Test successful full load read."""
        vieri_dataset.checkpoint = {}
        mock_response = MagicMock()
        mock_response.json.return_value = {"Results": [{"id": 1}, {"id": 2}]}
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        vieri_dataset.read()

        assert isinstance(vieri_dataset.output, pd.DataFrame)
        assert len(vieri_dataset.output) == 2
        assert vieri_dataset.checkpoint["offset"] == 20

    def test_read_sets_output_on_success(self, vieri_dataset):
        """Test that output is set to DataFrame on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Results": [{"id": 1}]}
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        vieri_dataset.read()

        assert isinstance(vieri_dataset.output, pd.DataFrame)
        assert len(vieri_dataset.output) == 1

    def test_read_sets_output_on_error(self, vieri_dataset):
        """Test that output is set to empty DataFrame on error (finally block)."""
        vieri_dataset.linked_service.connection.get = MagicMock(side_effect=Exception("API Error"))

        with pytest.raises(ReadError):
            vieri_dataset.read()

        assert isinstance(vieri_dataset.output, pd.DataFrame)
        assert len(vieri_dataset.output) == 0

    def test_read_wraps_exceptions_in_read_error(self, vieri_dataset):
        """Test that backend exceptions are wrapped in ReadError."""
        vieri_dataset.linked_service.connection.get = MagicMock(side_effect=Exception("Connection failed"))

        with pytest.raises(ReadError) as exc_info:
            vieri_dataset.read()

        assert "Failed to read data from Vieri API" in str(exc_info.value.message)

    def test_read_partial_results_on_error(self, vieri_dataset):
        """Test that partial results are preserved in output on error."""
        mock_response = MagicMock()
        vieri_dataset.settings.page_size = 2
        mock_response.json.side_effect = [
            {"Results": [{"id": 1}, {"id": 2}]},
            Exception("Page 2 fails"),
        ]
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        with pytest.raises(ReadError):
            vieri_dataset.read()

        assert len(vieri_dataset.output) == 2
        assert vieri_dataset.output.iloc[0]["id"] == 1


class TestDateFormatting:
    """Test date parsing and formatting utilities."""

    def test_parse_vieri_date_valid(self, vieri_dataset):
        """Test parsing valid Vieri date format."""
        result = vieri_dataset.parse_vieri_date("2024-03-15")

        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 15

    def test_parse_vieri_date_invalid_raises_error(self, vieri_dataset):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            vieri_dataset.parse_vieri_date("03-15-2024")

    def test_format_vieri_date(self, vieri_dataset):
        """Test formatting datetime to Vieri format."""
        dt = datetime(2024, 3, 15)
        result = vieri_dataset.format_vieri_date(dt)

        assert result == "2024-03-15"
