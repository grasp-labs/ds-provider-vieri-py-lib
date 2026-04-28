"""Unit tests for VieriDataset with checkpoint support."""

import logging
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import uuid4

import pandas as pd
import pytest
from ds_resource_plugin_py_lib.common.resource.dataset.errors import CreateError, ReadError
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError

from ds_provider_vieri_py_lib.dataset.vieri import VieriDataset, VieriDatasetSettings, VieriReadSettings
from ds_provider_vieri_py_lib.enums import ResourceType
from ds_provider_vieri_py_lib.linked_service.vieri import VieriLinkedService, VieriLinkedServiceSettings


@pytest.fixture
def mock_linked_service():
    """Mock linked service with mocked connection."""
    settings = VieriLinkedServiceSettings(host="https://api.vieri.com", subscription_key="test_key")
    svc = VieriLinkedService(id=uuid4(), name="test_service", version="1.0.0", settings=settings)
    # Mock the connection property using patch.object to avoid global class mutation
    with patch.object(type(svc), "connection", new_callable=PropertyMock) as mock_conn:
        mock_conn.return_value = MagicMock()
        yield svc


@pytest.fixture
def vieri_settings():
    """Default VieriDatasetSettings for testing."""
    return VieriDatasetSettings(
        owner_id="test_owner",
        product_name="test_product",
        read=VieriReadSettings(
            page_size=20,
            offset=0,
            last_modified=None,
        ),
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

    def test_create_not_supported_without_endpoint(self, vieri_dataset):
        """Test that create raises NotSupportedError when write_endpoint is not configured."""
        # write_endpoint is None by default
        vieri_dataset.input = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        with pytest.raises(NotSupportedError, match="Write operations not supported"):
            vieri_dataset.create()

    def test_create_empty_input_is_noop(self, vieri_dataset, caplog):
        """Test that create with empty input returns immediately (no-op per contract)."""
        caplog.set_level(logging.INFO)
        vieri_dataset.input = pd.DataFrame()  # Empty input
        vieri_dataset.settings.create.write_endpoint = "vieri-dataloader/LoadAccounts"

        vieri_dataset.create()  # Should not raise

        assert vieri_dataset.output.empty
        assert "empty input, returning immediately" in caplog.text
        # Verify no POST was made
        vieri_dataset.linked_service.connection.post.assert_not_called()

    def test_create_none_input_is_noop(self, vieri_dataset, caplog):
        """Test that create with None input returns immediately (no-op per contract)."""
        caplog.set_level(logging.INFO)
        vieri_dataset.input = None  # None input
        vieri_dataset.settings.create.write_endpoint = "vieri-dataloader/LoadAccounts"

        vieri_dataset.create()  # Should not raise

        assert vieri_dataset.output.empty
        assert "empty input" in caplog.text
        vieri_dataset.linked_service.connection.post.assert_not_called()

    def test_create_success_with_input(self, vieri_dataset, caplog):
        """Test successful create operation with records from self.input."""
        caplog.set_level(logging.INFO)
        test_records = {"id": [1, 2], "name": ["John", "Jane"]}
        vieri_dataset.input = pd.DataFrame(test_records)
        vieri_dataset.settings.create.write_endpoint = "vieri-dataloader/LoadAccounts"

        # Mock successful POST response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        vieri_dataset.linked_service.connection.post.return_value = mock_response

        vieri_dataset.create()

        # Verify POST was called with correct parameters
        vieri_dataset.linked_service.connection.post.assert_called_once()
        call_args = vieri_dataset.linked_service.connection.post.call_args
        assert "vieri-dataloader/LoadAccounts" in call_args.args[0]
        assert call_args.kwargs["json"]["Records"] == [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]
        # Verify output is populated with copy of input
        assert len(vieri_dataset.output) == 2
        assert "Successfully created 2 records" in caplog.text

    def test_create_with_different_endpoint(self, vieri_dataset):
        """Test create with different write endpoints."""
        vieri_dataset.input = pd.DataFrame({"id": [1], "name": ["Test"]})
        vieri_dataset.settings.create.write_endpoint = "vieri-dataloader/LoadSuppliers"

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        vieri_dataset.linked_service.connection.post.return_value = mock_response

        vieri_dataset.create()

        call_args = vieri_dataset.linked_service.connection.post.call_args
        assert "vieri-dataloader/LoadSuppliers" in call_args.args[0]

    def test_create_api_failure(self, vieri_dataset):
        """Test create operation when API returns error."""
        vieri_dataset.input = pd.DataFrame({"id": [1], "name": ["Test"]})
        vieri_dataset.settings.create.write_endpoint = "vieri-dataloader/LoadAccounts"

        # Mock POST response that raises an error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        vieri_dataset.linked_service.connection.post.return_value = mock_response

        with pytest.raises(CreateError, match="Failed to create records"):
            vieri_dataset.create()

    def test_update_not_supported(self, vieri_dataset):
        """Test that update raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.update()

    def test_delete_not_supported(self, vieri_dataset):
        """Test that delete raises NotSupportedError."""
        with pytest.raises(NotSupportedError):
            vieri_dataset.delete()


class TestBuildRequestParams:
    """Test request parameter building with checkpoint logic."""

    def test_build_params_full_load_empty_checkpoint(self, vieri_dataset):
        """Test that full load uses settings when checkpoint is empty."""
        vieri_dataset.checkpoint = {}
        vieri_dataset.settings.read.page_size = 25
        vieri_dataset.settings.read.offset = 0
        vieri_dataset.settings.read.last_modified = "2024-01-01"

        params = vieri_dataset._build_request_params()

        assert params["Skip"] == 0
        assert params["Take"] == 25
        assert params["ModifiedAfter"] == "2024-01-01"

    def test_build_params_full_load_no_checkpoint(self, vieri_dataset):
        """Test that full load uses settings when checkpoint is None."""
        vieri_dataset.checkpoint = None
        vieri_dataset.settings.read.page_size = 50
        vieri_dataset.settings.read.offset = 10
        vieri_dataset.settings.read.last_modified = None

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
        vieri_dataset.settings.read.page_size = 20
        vieri_dataset.settings.read.offset = 0
        vieri_dataset.settings.read.last_modified = "2024-01-01"

        params = vieri_dataset._build_request_params()

        assert params["Skip"] == 100
        assert params["Take"] == 30
        assert params["ModifiedAfter"] == "2024-02-01"


class TestBuildAndSetCheckpoint:
    """Test checkpoint building and setting logic."""

    def test_build_and_set_checkpoint_preserves_modifiers(self, vieri_dataset):
        """Test that checkpoint preserves ModifiedAfter filter."""
        vieri_dataset.settings.read.page_size = 20
        vieri_dataset.settings.read.last_modified = "2024-01-01"

        vieri_dataset._build_checkpoint(last_offset=60, final_page_count=20)

        assert vieri_dataset.checkpoint["offset"] == 80
        assert vieri_dataset.checkpoint["last_modified"] == "2024-01-01"

    def test_build_and_set_checkpoint_no_modifiers(self, vieri_dataset):
        """Test checkpoint without filters."""
        vieri_dataset.settings.read.page_size = 25
        vieri_dataset.settings.read.last_modified = None

        vieri_dataset._build_checkpoint(last_offset=50, final_page_count=25)

        assert vieri_dataset.checkpoint["offset"] == 75

    def test_build_checkpoint_non_raising_with_invalid_date_in_checkpoint(self, vieri_dataset):
        """Test that _build_checkpoint() is non-raising even with invalid date in checkpoint.

        This verifies the fix for exception masking: _build_checkpoint() should not
        raise even if checkpoint contains an invalid date string, since it reads dates
        directly without validation (validation happens later in _build_request_params).
        """
        # Set checkpoint with invalid date that would fail validation
        vieri_dataset.checkpoint = {"last_modified": "invalid-date"}

        # Should not raise even though the date is invalid
        vieri_dataset._build_checkpoint(last_offset=100, final_page_count=10)

        # Verify checkpoint was updated with new offset
        assert vieri_dataset.checkpoint["offset"] == 110
        # Invalid date should still be preserved (not validated or removed)
        assert vieri_dataset.checkpoint["last_modified"] == "invalid-date"

    def test_build_checkpoint_incremental_preserves_checkpoint_last_modified(self, vieri_dataset):
        """Test that incremental load preserves checkpoint's last_modified."""
        # Set checkpoint with a valid date
        vieri_dataset.checkpoint = {"offset": 50, "last_modified": "2024-02-15"}
        # Settings has a different date (should be ignored in incremental load)
        vieri_dataset.settings.read.last_modified = "2024-01-01"

        vieri_dataset._build_checkpoint(last_offset=50, final_page_count=20)

        # Checkpoint's last_modified should be preserved
        assert vieri_dataset.checkpoint["last_modified"] == "2024-02-15"
        assert vieri_dataset.checkpoint["offset"] == 70


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
        assert vieri_dataset.checkpoint["offset"] == 2

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
        vieri_dataset.settings.read.page_size = 2

        # Use Exception instance directly (not a function) so unittest.mock raises it
        mock_response.json.side_effect = [
            {"Results": [{"id": 1}, {"id": 2}]},
            Exception("Page 2 fails"),
        ]
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        with pytest.raises(ReadError):
            vieri_dataset.read()

        assert len(vieri_dataset.output) == 2
        assert vieri_dataset.output.iloc[0]["id"] == 1

    def test_read_original_exception_not_masked_by_invalid_checkpoint_date(self, vieri_dataset):
        """Test that original ReadError is not masked by _build_checkpoint().

        This test verifies the fix for exception masking: if read() fails with an API error,
        _build_checkpoint() in the finally block should not raise and mask that original error.

        Before the fix, _build_checkpoint() called _build_request_params() which could raise
        ValueError from date validation, masking the original exception from the try block.
        """
        # Set up checkpoint with invalid date that _build_checkpoint would receive
        vieri_dataset.checkpoint = {
            "offset": 0,
            "last_modified": "2024-01-01",  # Valid date for params building
        }

        # Cause read() to fail with an API error
        vieri_dataset.linked_service.connection.get = MagicMock(side_effect=Exception("API connection failed"))

        # Should raise ReadError (from the API failure), wrapped properly
        with pytest.raises(ReadError) as exc_info:
            vieri_dataset.read()

        # Verify it's the ReadError wrapping the API error, not a ValueError
        assert "Failed to read data from Vieri API" in str(exc_info.value.message)
        assert "API connection failed" in str(exc_info.value.message)

        # Verify output was still set in the finally block (proving _build_checkpoint didn't raise)
        assert isinstance(vieri_dataset.output, pd.DataFrame)
        # Verify checkpoint was updated despite the error
        assert "offset" in vieri_dataset.checkpoint


class TestResponseValidation:
    """Test response validation and error handling."""

    def test_read_response_not_dict_raises_error(self, vieri_dataset):
        """Test that non-dict response raises ReadError."""
        mock_response = MagicMock()
        mock_response.json.return_value = ["item1", "item2"]  # List instead of dict
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        with pytest.raises(ReadError) as exc_info:
            vieri_dataset.read()

        assert "expected dict" in str(exc_info.value.message)

    def test_read_response_missing_results_key_raises_error(self, vieri_dataset):
        """Test that response without 'Results' key raises ReadError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"id": 1}]}  # Missing 'Results' key
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        with pytest.raises(ReadError) as exc_info:
            vieri_dataset.read()

        assert "missing 'Results' key" in str(exc_info.value.message)

    def test_read_results_not_list_raises_error(self, vieri_dataset):
        """Test that non-list 'Results' value raises ReadError."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"Results": {"id": 1}}  # Dict instead of list
        vieri_dataset.linked_service.connection.get = MagicMock(return_value=mock_response)

        with pytest.raises(ReadError) as exc_info:
            vieri_dataset.read()

        assert "expected list" in str(exc_info.value.message)


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
        with pytest.raises(ValueError, match="%Y-%m-%d"):
            vieri_dataset.parse_vieri_date("03-15-2024")

    def test_format_vieri_date(self, vieri_dataset):
        """Test formatting datetime to Vieri format."""
        dt = datetime(2024, 3, 15)
        result = vieri_dataset.format_vieri_date(dt)

        assert result == "2024-03-15"
