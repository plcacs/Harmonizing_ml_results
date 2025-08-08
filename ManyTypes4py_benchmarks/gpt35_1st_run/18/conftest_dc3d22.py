from collections.abc import Generator
import itertools
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch
import warnings
import pytest
import zhaquirks
import zigpy
from zigpy.application import ControllerApplication
import zigpy.backups
import zigpy.config
from zigpy.const import SIG_EP_INPUT, SIG_EP_OUTPUT, SIG_EP_PROFILE, SIG_EP_TYPE
import zigpy.device
import zigpy.group
import zigpy.profiles
import zigpy.quirks
import zigpy.state
import zigpy.types
import zigpy.util
from zigpy.zcl.clusters.general import Basic, Groups
from zigpy.zcl.foundation import Status
import zigpy.zdo.types as zdo_t
from homeassistant.components.zha import const as zha_const
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from .common import patch_cluster as common_patch_cluster
from tests.common import MockConfigEntry
from tests.components.light.conftest import mock_light_profiles
FIXTURE_GRP_ID: int = 4097
FIXTURE_GRP_NAME: str = 'fixture group'
COUNTER_NAMES: List[str] = ['counter_1', 'counter_2', 'counter_3']

@pytest.fixture(scope='package', autouse=True)
def globally_load_quirks() -> None:
    ...

class _FakeApp(ControllerApplication):

    async def add_endpoint(self, descriptor: Any) -> None:
        ...

    async def connect(self) -> None:
        ...

    async def disconnect(self) -> None:
        ...

    async def force_remove(self, dev: Any) -> None:
        ...

    async def load_network_info(self, *, load_devices: bool = False) -> None:
        ...

    async def permit_ncp(self, time_s: int = 60) -> None:
        ...

    async def permit_with_link_key(self, node: Any, link_key: Any, time_s: int = 60) -> None:
        ...

    async def reset_network_info(self) -> None:
        ...

    async def send_packet(self, packet: Any) -> None:
        ...

    async def start_network(self) -> None:
        ...

    async def write_network_info(self, *, network_info: Any, node_info: Any) -> None:
        ...

    async def request(self, device: Any, profile: Any, cluster: Any, src_ep: Any, dst_ep: Any, sequence: Any, data: Any, *, expect_reply: bool = True, use_ieee: bool = False, extended_timeout: bool = False) -> None:
        ...

    async def move_network_to_channel(self, new_channel: int, *, num_broadcasts: int = 5) -> None:
        ...

    def _persist_coordinator_model_strings_in_db(self) -> None:
        ...

def _wrap_mock_instance(obj: Any) -> Any:
    ...

@pytest.fixture
async def zigpy_app_controller() -> ControllerApplication:
    ...

@pytest.fixture(name='config_entry')
async def config_entry_fixture() -> MockConfigEntry:
    ...

@pytest.fixture
def mock_zigpy_connect(zigpy_app_controller: ControllerApplication) -> ControllerApplication:
    ...

@pytest.fixture
def setup_zha(hass: HomeAssistant, config_entry: MockConfigEntry, mock_zigpy_connect: ControllerApplication) -> Callable:
    ...

@pytest.fixture
def cluster_handler() -> Callable:
    ...

@pytest.fixture(autouse=True)
def speed_up_radio_mgr() -> Generator:
    ...

@pytest.fixture
def network_backup() -> zigpy.backups.NetworkBackup:
    ...

@pytest.fixture
def zigpy_device_mock(zigpy_app_controller: ControllerApplication) -> Callable:
    ...
