from datetime import timedelta
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, patch
from aiohasupervisor import SupervisorError
from aiohasupervisor.models import AddonsStats
import pytest
from voluptuous import Invalid
from homeassistant.auth.const import GROUP_ID_ADMIN
from homeassistant.components import frontend, hassio
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.hassio import ADDONS_COORDINATOR, DOMAIN, STORAGE_KEY, get_core_info, get_supervisor_ip, hostname_from_addon_slug, is_hassio as deprecated_is_hassio
from homeassistant.components.hassio.const import REQUEST_REFRESH_DELAY
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, issue_registry as ir
from homeassistant.helpers.hassio import is_hassio
from homeassistant.helpers.service_info.hassio import HassioServiceInfo
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from tests.common import MockConfigEntry, async_fire_time_changed, import_and_test_deprecated_constant
from tests.test_util.aiohttp import AiohttpClientMocker
MOCK_ENVIRON: dict = {'SUPERVISOR': '127.0.0.1', 'SUPERVISOR_TOKEN': 'abcdefgh'}

def extra_os_info() -> dict:
    return {}

def os_info(extra_os_info: dict) -> dict:
    return {'json': {'result': 'ok', 'data': {'version_latest': '1.0.0', 'version': '1.0.0', **extra_os_info}}

@pytest.fixture(autouse=True)
def mock_all(aioclient_mock, os_info, store_info, addon_info, addon_stats, addon_changelog, resolution_info):
    aioclient_mock.post('http://127.0.0.1/homeassistant/options', json={'result': 'ok'})
    aioclient_mock.post('http://127.0.0.1/supervisor/options', json={'result': 'ok'})
    aioclient_mock.get('http://127.0.0.1/info', json={'result': 'ok', 'data': {'supervisor': '222', 'homeassistant': '0.110.0', 'hassos': '1.2.3'}})
    aioclient_mock.get('http://127.0.0.1/host/info', json={'result': 'ok', 'data': {'result': 'ok', 'data': {'chassis': 'vm', 'operating_system': 'Debian GNU/Linux 10 (buster)', 'kernel': '4.19.0-6-amd64'}}})
    aioclient_mock.get('http://127.0.0.1/core/info', json={'result': 'ok', 'data': {'version_latest': '1.0.0', 'version': '1.0.0'}})
    aioclient_mock.get('http://127.0.0.1/os/info', **os_info)
    aioclient_mock.get('http://127.0.0.1/supervisor/info', json={'result': 'ok', 'data': {'version_latest': '1.0.0', 'version': '1.0.0', 'auto_update': True, 'addons': [{'name': 'test', 'slug': 'test', 'state': 'stopped', 'update_available': False, 'version': '1.0.0', 'version_latest': '1.0.0', 'repository': 'core', 'icon': False}, {'name': 'test2', 'slug': 'test2', 'state': 'stopped', 'update_available': False, 'version': '1.0.0', 'version_latest': '1.0.0', 'repository': 'core', 'icon': False}]}})
    aioclient_mock.get('http://127.0.0.1/core/stats', json={'result': 'ok', 'data': {'cpu_percent': 0.99, 'memory_usage': 182611968, 'memory_limit': 3977146368, 'memory_percent': 4.59, 'network_rx': 362570232, 'network_tx': 82374138, 'blk_read': 46010945536, 'blk_write': 15051526144}})
    aioclient_mock.get('http://127.0.0.1/supervisor/stats', json={'result': 'ok', 'data': {'cpu_percent': 0.99, 'memory_usage': 182611968, 'memory_limit': 3977146368, 'memory_percent': 4.59, 'network_rx': 362570232, 'network_tx': 82374138, 'blk_read': 46010945536, 'blk_write': 15051526144}})
    async def mock_addon_stats(addon: str) -> AddonsStats:
        if addon in {'test2', 'test3'}:
            return AddonsStats(cpu_percent=0.8, memory_usage=51941376, memory_limit=3977146368, memory_percent=1.31, network_rx=31338284, network_tx=15692900, blk_read=740077568, blk_write=6004736)
        return AddonsStats(cpu_percent=0.99, memory_usage=182611968, memory_limit=3977146368, memory_percent=4.59, network_rx=362570232, network_tx=82374138, blk_read=46010945536, blk_write=15051526144)
    addon_stats.side_effect = mock_addon_stats
    def mock_addon_info(slug: str) -> Any:
        addon_info.return_value.auto_update = slug == 'test'
        return addon_info.return_value
    addon_info.side_effect = mock_addon_info
    aioclient_mock.get('http://127.0.0.1/ingress/panels', json={'result': 'ok', 'data': {'panels': {}}})
    aioclient_mock.get('http://127.0.0.1/network/info', json={'result': 'ok', 'data': {'host_internet': True, 'supervisor_internet': True}})

async def test_setup_api_ping(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, supervisor_client: Any) -> None:
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, 'hassio', {})
        await hass.async_block_till_done()
    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert get_core_info(hass)['version_latest'] == '1.0.0'
    assert is_hassio(hass)

# Add more annotated test functions here...
