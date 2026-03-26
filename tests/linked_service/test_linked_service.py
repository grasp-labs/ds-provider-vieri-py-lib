import uuid
from unittest.mock import MagicMock, PropertyMock, patch

from ds_protocol_http_py_lib.enums import AuthType

from ds_provider_vieri_py_lib.linked_service.vieri import VieriLinkedService, VieriLinkedServiceSettings


def test_vieri_linked_service_settings_headers():
    settings = VieriLinkedServiceSettings(host="https://vieri.example.com/api", subscription_key="test-key")
    assert settings.headers["Ocp-Apim-Subscription-Key"] == "test-key"
    assert settings.headers["Content-Type"] == "application/json"
    assert settings.auth_type == AuthType.NO_AUTH


def test_vieri_linked_service_type():
    settings = VieriLinkedServiceSettings(host="https://vieri.example.com/api", subscription_key="test-key")
    service = VieriLinkedService(id=uuid.uuid4(), name="vieri-test", version="0.1.0", settings=settings)
    assert service.type() is not None


def test_vieri_linked_service_test_connection():
    settings = VieriLinkedServiceSettings(host="https://vieri.example.com/api", subscription_key="test-key")
    service = VieriLinkedService(id=uuid.uuid4(), name="vieri-test", version="0.1.0", settings=settings)
    with patch.object(type(service), "connection", new_callable=PropertyMock) as mock_conn:
        mock_conn.return_value.get.return_value = MagicMock()
        result, msg = service.test_connection()
        assert result is True
        assert "success" in msg.lower() or "tested" in msg.lower()


def test_vieri_linked_service_test_connection_fail():
    settings = VieriLinkedServiceSettings(host="https://vieri.example.com/api", subscription_key="test-key")
    service = VieriLinkedService(id=uuid.uuid4(), name="vieri-test", version="0.1.0", settings=settings)
    with patch.object(type(service), "connection", new_callable=PropertyMock) as mock_conn:
        mock_conn.return_value.get.side_effect = Exception("fail")
        result, msg = service.test_connection()
        assert result is False
        assert "fail" in msg.lower()
