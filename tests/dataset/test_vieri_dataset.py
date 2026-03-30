"""Unit tests for VieriDataset."""

import logging
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from ds_resource_plugin_py_lib.common.resource.dataset.errors import ReadError
from ds_resource_plugin_py_lib.common.resource.errors import NotSupportedError

from ds_provider_vieri_py_lib.dataset.vieri import VieriDataset, VieriDatasetSettings
from ds_provider_vieri_py_lib.enums import ResourceType
from ds_provider_vieri_py_lib.linked_service.vieri import VieriLinkedService, VieriLinkedServiceSettings


@pytest.fixture
def mock_linked_service():
    settings = VieriLinkedServiceSettings(host="https://api.vieri.com", subscription_key="key")
    svc = VieriLinkedService(id="id", name="name", version="1.0.0", settings=settings)
    svc.session = MagicMock()
    return svc


@pytest.fixture
def vieri_settings():
    return VieriDatasetSettings(owner_id="owner", product_name="product")


@pytest.fixture
def vieri_dataset(mock_linked_service, vieri_settings):
    return VieriDataset(
        id=uuid4(), name="test_dataset", version="1.0.0", linked_service=mock_linked_service, settings=vieri_settings
    )


def test_type_property(vieri_dataset):
    assert vieri_dataset.type == ResourceType.VIERI_DATASET


def test_supports_checkpoint_property(vieri_dataset):
    assert vieri_dataset.supports_checkpoint is True


def test_update_checkpoint_sets_checkpoint(vieri_dataset):
    vieri_dataset._update_checkpoint("2024-01-01T00:00:00Z")
    assert vieri_dataset.checkpoint == {"modified_after": "2024-01-01T00:00:00Z"}


def test_update_checkpoint_none_does_not_set(vieri_dataset):
    vieri_dataset.checkpoint = {}
    vieri_dataset._update_checkpoint(None)
    assert vieri_dataset.checkpoint == {}


@patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame")
def test_read_success(mock_df, vieri_dataset):
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 1}], "2024-01-01T00:00:00Z"))
    vieri_dataset.read()
    mock_df.assert_called_once_with([{"a": 1}])
    assert vieri_dataset.output is mock_df.return_value
    assert vieri_dataset.checkpoint == {"modified_after": "2024-01-01T00:00:00Z"}


@patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame")
def test_read_raises_read_error(mock_df, vieri_dataset):
    vieri_dataset._fetch_all_pages = MagicMock(side_effect=Exception("fail"))
    with pytest.raises(ReadError):
        vieri_dataset.read()


def test_fetch_all_pages_pagination(vieri_dataset):
    # Setup mock session.get
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.side_effect = [
        {"results": [{"modified": "2024-01-01T00:00:00Z"}]},
        {"results": []},
    ]
    vieri_dataset.linked_service.session.get = MagicMock(return_value=mock_response)
    results, latest = vieri_dataset._fetch_all_pages(url="url", headers={}, params={}, take=1, skip=0)
    assert results == [{"modified": "2024-01-01T00:00:00Z"}]
    assert latest == "2024-01-01T00:00:00Z"


def test_read_with_checkpoint_modified_after(vieri_dataset):
    vieri_dataset.checkpoint = {"modified_after": "2024-01-01T00:00:00Z"}
    vieri_dataset.settings.modified_after = None
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 1}], "2024-01-01T00:00:00Z"))
    with patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame") as mock_df:
        vieri_dataset.read()
        mock_df.assert_called_once_with([{"a": 1}])
        assert vieri_dataset.output is mock_df.return_value


def test_read_with_settings_modified_after(vieri_dataset):
    vieri_dataset.checkpoint = None
    vieri_dataset.settings.modified_after = "2024-01-02T00:00:00Z"
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 2}], "2024-01-02T00:00:00Z"))
    with patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame") as mock_df:
        vieri_dataset.read()
        mock_df.assert_called_once_with([{"a": 2}])
        assert vieri_dataset.output is mock_df.return_value


def test_read_with_no_modified_after(vieri_dataset):
    vieri_dataset.checkpoint = None
    vieri_dataset.settings.modified_after = None
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 3}], None))
    with patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame") as mock_df:
        vieri_dataset.read()
        mock_df.assert_called_once_with([{"a": 3}])
        assert vieri_dataset.output is mock_df.return_value


def test_fetch_all_pages_modified_logic(vieri_dataset):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.side_effect = [
        {
            "results": [
                {"modified": "2024-01-01T00:00:00Z"},
                {"modified": "2024-01-01T00:00:00Z"},
                {"foo": "bar"},
            ]
        },
        {"results": []},
    ]
    vieri_dataset.linked_service.session.get = MagicMock(return_value=mock_response)
    results, latest = vieri_dataset._fetch_all_pages(
        url="url", headers={}, params={"modifiedAfter": "2024-01-01T00:00:00Z"}, take=3, skip=0
    )
    assert results == [
        {"modified": "2024-01-01T00:00:00Z"},
        {"modified": "2024-01-01T00:00:00Z"},
        {"foo": "bar"},
    ]
    assert latest == "2024-01-01T00:00:00Z"


def test_create_not_implemented(vieri_dataset):
    with pytest.raises(NotImplementedError):
        vieri_dataset.create()


def test_update_not_implemented(vieri_dataset):
    with pytest.raises(NotImplementedError):
        vieri_dataset.update()


def test_delete_not_implemented(vieri_dataset):
    with pytest.raises(NotImplementedError):
        vieri_dataset.delete()


def test_close_logs_info(vieri_dataset, caplog):
    caplog.set_level(logging.INFO)
    vieri_dataset.close()
    assert any("Closing VieriDataset" in m for m in caplog.text.splitlines())


def test_rename_not_supported(vieri_dataset):
    with pytest.raises(NotSupportedError):
        vieri_dataset.rename()


def test_list_not_supported(vieri_dataset):
    with pytest.raises(NotSupportedError):
        vieri_dataset.list()


def test_upsert_not_supported(vieri_dataset):
    with pytest.raises(NotSupportedError):
        vieri_dataset.upsert()


def test_purge_not_supported(vieri_dataset):
    with pytest.raises(NotSupportedError):
        vieri_dataset.purge()


def test_read_with_vieri_date_format_checkpoint(vieri_dataset):
    # Should parse and reformat correctly
    vieri_dataset.checkpoint = {"modified_after": "2024-03-30"}  # matches VIERI_DATETIME_FORMAT
    vieri_dataset.settings.modified_after = None
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 1}], "2024-03-30"))
    with patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame") as mock_df:
        vieri_dataset.read()
        mock_df.assert_called_once_with([{"a": 1}])
        assert vieri_dataset.output is mock_df.return_value


def test_read_with_non_vieri_date_format_checkpoint(vieri_dataset):
    # Should pass through unchanged if not matching VIERI_DATETIME_FORMAT
    vieri_dataset.checkpoint = {"modified_after": "2024-03-30T12:34:56Z"}  # ISO8601
    vieri_dataset.settings.modified_after = None
    vieri_dataset._fetch_all_pages = MagicMock(return_value=([{"a": 2}], "2024-03-30T12:34:56Z"))
    with patch("ds_provider_vieri_py_lib.dataset.vieri.pd.DataFrame") as mock_df:
        vieri_dataset.read()
        mock_df.assert_called_once_with([{"a": 2}])
        assert vieri_dataset.output is mock_df.return_value
