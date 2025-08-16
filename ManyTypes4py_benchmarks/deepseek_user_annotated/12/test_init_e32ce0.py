"""The tests for the hassio component."""

from datetime import timedelta
import logging
import os
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aiohasupervisor import SupervisorError
from aiohasupervisor.models import AddonsStats
import pytest
from voluptuous import Invalid

from homeassistant.auth.const import GROUP_ID_ADMIN
from homeassistant.components import frontend, hassio
from homeassistant.components.binary_sensor import DOMAIN as BINARY_SENSOR_DOMAIN
from homeassistant.components.hassio import (
    ADDONS_COORDINATOR,
    DOMAIN,
    STORAGE_KEY,
    get_core_info,
    get_supervisor_ip,
    hostname_from_addon_slug,
    is_hassio as deprecated_is_hassio,
)
from homeassistant.components.hassio.const import REQUEST_REFRESH_DELAY
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, issue_registry as ir
from homeassistant.helpers.hassio import is_hassio
from homeassistant.helpers.service_info.hassio import HassioServiceInfo
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from tests.common import (
    MockConfigEntry,
    async_fire_time_changed,
    import_and_test_deprecated_constant,
)
from tests.test_util.aiohttp import AiohttpClientMocker

MOCK_ENVIRON: Dict[str, str] = {"SUPERVISOR": "127.0.0.1", "SUPERVISOR_TOKEN": "abcdefgh"}


@pytest.fixture
def extra_os_info() -> Dict[str, Any]:
    """Extra os/info."""
    return {}


@pytest.fixture
def os_info(extra_os_info: Dict[str, Any]) -> Dict[str, Any]:
    """Mock os/info."""
    return {
        "json": {
            "result": "ok",
            "data": {"version_latest": "1.0.0", "version": "1.0.0", **extra_os_info},
        }
    }


@pytest.fixture(autouse=True)
def mock_all(
    aioclient_mock: AiohttpClientMocker,
    os_info: AsyncMock,
    store_info: AsyncMock,
    addon_info: AsyncMock,
    addon_stats: AsyncMock,
    addon_changelog: AsyncMock,
    resolution_info: AsyncMock,
) -> None:
    """Mock all setup requests."""
    aioclient_mock.post("http://127.0.0.1/homeassistant/options", json={"result": "ok"})
    aioclient_mock.post("http://127.0.0.1/supervisor/options", json={"result": "ok"})
    aioclient_mock.get(
        "http://127.0.0.1/info",
        json={
            "result": "ok",
            "data": {
                "supervisor": "222",
                "homeassistant": "0.110.0",
                "hassos": "1.2.3",
            },
        },
    )
    aioclient_mock.get(
        "http://127.0.0.1/host/info",
        json={
            "result": "ok",
            "data": {
                "result": "ok",
                "data": {
                    "chassis": "vm",
                    "operating_system": "Debian GNU/Linux 10 (buster)",
                    "kernel": "4.19.0-6-amd64",
                },
            },
        },
    )
    aioclient_mock.get(
        "http://127.0.0.1/core/info",
        json={"result": "ok", "data": {"version_latest": "1.0.0", "version": "1.0.0"}},
    )
    aioclient_mock.get(
        "http://127.0.0.1/os/info",
        **os_info,
    )
    aioclient_mock.get(
        "http://127.0.0.1/supervisor/info",
        json={
            "result": "ok",
            "data": {
                "version_latest": "1.0.0",
                "version": "1.0.0",
                "auto_update": True,
                "addons": [
                    {
                        "name": "test",
                        "slug": "test",
                        "state": "stopped",
                        "update_available": False,
                        "version": "1.0.0",
                        "version_latest": "1.0.0",
                        "repository": "core",
                        "icon": False,
                    },
                    {
                        "name": "test2",
                        "slug": "test2",
                        "state": "stopped",
                        "update_available": False,
                        "version": "1.0.0",
                        "version_latest": "1.0.0",
                        "repository": "core",
                        "icon": False,
                    },
                ],
            },
        },
    )
    aioclient_mock.get(
        "http://127.0.0.1/core/stats",
        json={
            "result": "ok",
            "data": {
                "cpu_percent": 0.99,
                "memory_usage": 182611968,
                "memory_limit": 3977146368,
                "memory_percent": 4.59,
                "network_rx": 362570232,
                "network_tx": 82374138,
                "blk_read": 46010945536,
                "blk_write": 15051526144,
            },
        },
    )
    aioclient_mock.get(
        "http://127.0.0.1/supervisor/stats",
        json={
            "result": "ok",
            "data": {
                "cpu_percent": 0.99,
                "memory_usage": 182611968,
                "memory_limit": 3977146368,
                "memory_percent": 4.59,
                "network_rx": 362570232,
                "network_tx": 82374138,
                "blk_read": 46010945536,
                "blk_write": 15051526144,
            },
        },
    )

    async def mock_addon_stats(addon: str) -> AddonsStats:
        """Mock addon stats for test and test2."""
        if addon in {"test2", "test3"}:
            return AddonsStats(
                cpu_percent=0.8,
                memory_usage=51941376,
                memory_limit=3977146368,
                memory_percent=1.31,
                network_rx=31338284,
                network_tx=15692900,
                blk_read=740077568,
                blk_write=6004736,
            )
        return AddonsStats(
            cpu_percent=0.99,
            memory_usage=182611968,
            memory_limit=3977146368,
            memory_percent=4.59,
            network_rx=362570232,
            network_tx=82374138,
            blk_read=46010945536,
            blk_write=15051526144,
        )

    addon_stats.side_effect = mock_addon_stats

    def mock_addon_info(slug: str) -> Any:
        addon_info.return_value.auto_update = slug == "test"
        return addon_info.return_value

    addon_info.side_effect = mock_addon_info
    aioclient_mock.get(
        "http://127.0.0.1/ingress/panels", json={"result": "ok", "data": {"panels": {}}}
    )
    aioclient_mock.get(
        "http://127.0.0.1/network/info",
        json={
            "result": "ok",
            "data": {
                "host_internet": True,
                "supervisor_internet": True,
            },
        },
    )


async def test_setup_api_ping(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API ping."""
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {})
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert get_core_info(hass)["version_latest"] == "1.0.0"
    assert is_hassio(hass)


async def test_setup_api_panel(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """Test setup with API ping."""
    assert await async_setup_component(hass, "frontend", {})
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {})
        assert result

    panels = hass.data[frontend.DATA_PANELS]

    assert panels.get("hassio").to_response() == {
        "component_name": "custom",
        "icon": None,
        "title": None,
        "url_path": "hassio",
        "require_admin": True,
        "config_panel_domain": None,
        "config": {
            "_panel_custom": {
                "embed_iframe": True,
                "js_url": "/api/hassio/app/entrypoint.js",
                "name": "hassio-main",
                "trust_external": False,
            }
        },
    }


async def test_setup_api_push_api_data(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push."""
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(
            hass, "hassio", {"http": {"server_port": 9999}, "hassio": {}}
        )
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert not aioclient_mock.mock_calls[0][2]["ssl"]
    assert aioclient_mock.mock_calls[0][2]["port"] == 9999
    assert "watchdog" not in aioclient_mock.mock_calls[0][2]


async def test_setup_api_push_api_data_server_host(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push with active server host."""
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(
            hass,
            "hassio",
            {"http": {"server_port": 9999, "server_host": "127.0.0.1"}, "hassio": {}},
        )
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert not aioclient_mock.mock_calls[0][2]["ssl"]
    assert aioclient_mock.mock_calls[0][2]["port"] == 9999
    assert not aioclient_mock.mock_calls[0][2]["watchdog"]


async def test_setup_api_push_api_data_default(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    hass_storage: Dict[str, Any],
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push default data."""
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {"http": {}, "hassio": {}})
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert not aioclient_mock.mock_calls[0][2]["ssl"]
    assert aioclient_mock.mock_calls[0][2]["port"] == 8123
    refresh_token = aioclient_mock.mock_calls[0][2]["refresh_token"]
    hassio_user = await hass.auth.async_get_user(
        hass_storage[STORAGE_KEY]["data"]["hassio_user"]
    )
    assert hassio_user is not None
    assert hassio_user.system_generated
    assert len(hassio_user.groups) == 1
    assert hassio_user.groups[0].id == GROUP_ID_ADMIN
    assert hassio_user.name == "Supervisor"
    for token in hassio_user.refresh_tokens.values():
        if token.token == refresh_token:
            break
    else:
        pytest.fail("refresh token not found")


async def test_setup_adds_admin_group_to_user(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    hass_storage: Dict[str, Any],
) -> None:
    """Test setup with API push default data."""
    # Create user without admin
    user = await hass.auth.async_create_system_user("Hass.io")
    assert not user.is_admin
    await hass.auth.async_create_refresh_token(user)

    hass_storage[STORAGE_KEY] = {
        "data": {"hassio_user": user.id},
        "key": STORAGE_KEY,
        "version": 1,
    }

    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {"http": {}, "hassio": {}})
        assert result

    assert user.is_admin


async def test_setup_migrate_user_name(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    hass_storage: Dict[str, Any],
) -> None:
    """Test setup with migrating the user name."""
    # Create user with old name
    user = await hass.auth.async_create_system_user("Hass.io")
    await hass.auth.async_create_refresh_token(user)

    hass_storage[STORAGE_KEY] = {
        "data": {"hassio_user": user.id},
        "key": STORAGE_KEY,
        "version": 1,
    }

    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {"http": {}, "hassio": {}})
        assert result

    assert user.name == "Supervisor"


async def test_setup_api_existing_hassio_user(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    hass_storage: Dict[str, Any],
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push default data."""
    user = await hass.auth.async_create_system_user("Hass.io test")
    token = await hass.auth.async_create_refresh_token(user)
    hass_storage[STORAGE_KEY] = {"version": 1, "data": {"hassio_user": user.id}}
    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {"http": {}, "hassio": {}})
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert not aioclient_mock.mock_calls[0][2]["ssl"]
    assert aioclient_mock.mock_calls[0][2]["port"] == 8123
    assert aioclient_mock.mock_calls[0][2]["refresh_token"] == token.token


async def test_setup_core_push_timezone(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push default data."""
    hass.config.time_zone = "testzone"

    with patch.dict(os.environ, MOCK_ENVIRON):
        result = await async_setup_component(hass, "hassio", {"hassio": {}})
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert aioclient_mock.mock_calls[1][2]["timezone"] == "testzone"

    with patch("homeassistant.util.dt.set_default_time_zone"):
        await hass.config.async_update(time_zone="America/New_York")
    await hass.async_block_till_done()
    assert aioclient_mock.mock_calls[-1][2]["timezone"] == "America/New_York"


async def test_setup_hassio_no_additional_data(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    supervisor_client: AsyncMock,
) -> None:
    """Test setup with API push default data."""
    with (
        patch.dict(os.environ, MOCK_ENVIRON),
        patch.dict(os.environ, {"SUPERVISOR_TOKEN": "123456"}),
    ):
        result = await async_setup_component(hass, "hassio", {"hassio": {}})
        await hass.async_block_till_done()

    assert result
    assert aioclient_mock.call_count + len(supervisor_client.mock_calls) == 20
    assert aioclient_mock.mock_calls[-1][3]["Authorization"] == "Bearer 123456"


async def test_fail_setup_without_environ_var(hass: HomeAssistant) -> None:
    """Fail setup if no environ variable set."""
    with patch.dict(os.environ, {}, clear=True):
        result = await async_setup_component(hass, "hassio", {})
        assert not result


async def test_warn_when_cannot_connect(
    hass: