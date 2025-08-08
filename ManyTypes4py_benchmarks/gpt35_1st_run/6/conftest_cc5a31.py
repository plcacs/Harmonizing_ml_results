from __future__ import annotations
from collections.abc import Awaitable, Callable, Generator
from datetime import datetime
from socket import AddressFamily
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from async_upnp_client.client import UpnpDevice
from async_upnp_client.event_handler import UpnpEventHandler
from async_upnp_client.exceptions import UpnpConnectionError
import pytest
from samsungctl import Remote
from samsungtvws.async_remote import SamsungTVWSAsyncRemote
from samsungtvws.command import SamsungTVCommand
from samsungtvws.encrypted.remote import SamsungTVEncryptedWSAsyncRemote
from samsungtvws.event import ED_INSTALLED_APP_EVENT
from samsungtvws.exceptions import ResponseError
from samsungtvws.remote import ChannelEmitCommand
from homeassistant.components.samsungtv.const import WEBSOCKET_SSL_PORT
from homeassistant.util import dt as dt_util
from .const import SAMPLE_DEVICE_INFO_UE48JU6400, SAMPLE_DEVICE_INFO_WIFI

def mock_setup_entry() -> Generator[Mock, None, None]:
def silent_ssdp_scanner() -> Generator[None, None, None]:
def samsungtv_mock_async_get_local_ip() -> Generator[None, None, None]:
def fake_host_fixture() -> Generator[None, None, None]:
def app_list_delay_fixture() -> Generator[None, None, None]:
def upnp_factory_fixture() -> Generator[Mock, None, None]:
def upnp_device_fixture(upnp_factory: Mock) -> Generator[Mock, None, None]:
def dmr_device_fixture(upnp_device: Mock) -> Generator[Mock, None, None]:
def upnp_notify_server_fixture(upnp_factory: Mock) -> Generator[Mock, None, None]:
def remote_fixture() -> Generator[Mock, None, None]:
def rest_api_fixture() -> Generator[Mock, None, None]:
def rest_api_fixture_non_ssl_only() -> Generator[None, None, None]:
def rest_api_failure_fixture() -> Generator[Mock, None, None]:
def remoteencws_failing_fixture() -> Generator[None, None, None]:
def remotews_fixture() -> Generator[Mock, None, None]:
def remoteencws_fixture() -> Generator[Mock, None, None]:
def mock_now() -> datetime:
def mac_address_fixture() -> Generator[Mock, None, None]:
