"""The tests for the Cast Media player platform."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import attr
import pychromecast
from pychromecast.const import CAST_TYPE_CHROMECAST, CAST_TYPE_GROUP
import pytest
import yarl

from homeassistant.components import media_player, tts
from homeassistant.components.cast import media_player as cast
from homeassistant.components.cast.const import (
    SIGNAL_HASS_CAST_SHOW_VIEW,
    HomeAssistantControllerData,
)
from homeassistant.components.cast.media_player import ChromecastInfo
from homeassistant.components.media_player import (
    BrowseMedia,
    MediaClass,
    MediaPlayerEntityFeature,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    CAST_APP_ID_HOMEASSISTANT_LOVELACE,
    EVENT_HOMEASSISTANT_STOP,
)
from homeassistant.core import HomeAssistant
from homeassistant.core_config import async_process_ha_core_config
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, network
from homeassistant.helpers.dispatcher import (
    async_dispatcher_connect,
    async_dispatcher_send,
)
from homeassistant.setup import async_setup_component

from tests.common import (
    MockConfigEntry,
    assert_setup_component,
    load_fixture,
    mock_platform,
)
from tests.components.media_player import common
from tests.test_util.aiohttp import AiohttpClientMocker
from tests.typing import WebSocketGenerator

FakeUUID: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e2")
FakeUUID2: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e4")
FakeGroupUUID: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e3")

FAKE_HOST_SERVICE: pychromecast.discovery.HostServiceInfo = pychromecast.discovery.HostServiceInfo("127.0.0.1", 8009)
FAKE_MDNS_SERVICE: pychromecast.discovery.MDNSServiceInfo = pychromecast.discovery.MDNSServiceInfo("the-service")

UNDEFINED: object = object()


def get_fake_chromecast(info: ChromecastInfo) -> MagicMock:
    """Generate a Fake Chromecast object with the specified arguments."""
    mock = MagicMock(uuid=info.uuid)
    mock.app_id = None
    mock.media_controller.status = None
    return mock


def get_fake_chromecast_info(
    *,
    host: str = "192.168.178.42",
    port: int = 8009,
    service: pychromecast.discovery.HostServiceInfo | pychromecast.discovery.MDNSServiceInfo | None = None,
    uuid: UUID | None = FakeUUID,
    cast_type: str | object = UNDEFINED,
    manufacturer: str | object = UNDEFINED,
    model_name: str | object = UNDEFINED,
) -> ChromecastInfo:
    """Generate a Fake ChromecastInfo with the specified arguments."""

    if service is None:
        service = pychromecast.discovery.HostServiceInfo(host, port)
    if cast_type is UNDEFINED:
        cast_type = CAST_TYPE_GROUP if port != 8009 else CAST_TYPE_CHROMECAST
    if manufacturer is UNDEFINED:
        manufacturer = "Nabu Casa"
    if model_name is UNDEFINED:
        model_name = "Chromecast"
    return ChromecastInfo(
        cast_info=pychromecast.models.CastInfo(
            services={service},
            uuid=uuid,
            model_name=model_name,
            friendly_name="Speaker",
            host=host,
            port=port,
            cast_type=cast_type,
            manufacturer=manufacturer,
        )
    )


def get_fake_zconf(host: str = "192.168.178.42", port: int = 8009) -> MagicMock:
    """Generate a Fake Zeroconf object with the specified arguments."""
    parsed_addresses = MagicMock()
    parsed_addresses.return_value = [host]
    service_info = MagicMock(parsed_addresses=parsed_addresses, port=port)
    zconf = MagicMock()
    zconf.get_service_info.return_value = service_info
    return zconf


async def async_setup_cast(
    hass: HomeAssistant, config: dict[str, Any] | None = None
) -> MagicMock:
    """Set up the cast platform."""
    if config is None:
        config = {}
    data = {"ignore_cec": [], "known_hosts": [], "uuid": [], **config}
    with patch(
        "homeassistant.helpers.entity_platform.EntityPlatform._async_schedule_add_entities_for_entry"
    ) as add_entities:
        entry = MockConfigEntry(data=data, domain="cast")
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    return add_entities


async def async_setup_cast_internal_discovery(
    hass: HomeAssistant, config: dict[str, Any] | None = None
) -> tuple[
    Callable[
        [
            pychromecast.discovery.HostServiceInfo
            | pychromecast.discovery.MDNSServiceInfo,
            ChromecastInfo,
        ],
        None,
    ],
    Callable[[str, ChromecastInfo], None],
    MagicMock,
]:
    """Set up the cast platform and the discovery."""
    browser = MagicMock(devices={}, zc={})

    with patch(
        "homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser",
        return_value=browser,
    ) as cast_browser:
        add_entities = await async_setup_cast(hass, config)
        await hass.async_block_till_done(wait_background_tasks=True)
        await hass.async_block_till_done(wait_background_tasks=True)

        assert browser.start_discovery.call_count == 1

        discovery_callback = cast_browser.call_args[0][0].add_cast
        remove_callback = cast_browser.call_args[0][0].remove_cast

    def discover_chromecast(
        service: (
            pychromecast.discovery.HostServiceInfo
            | pychromecast.discovery.MDNSServiceInfo
        ),
        info: ChromecastInfo,
    ) -> None:
        """Discover a chromecast device."""
        browser.devices[info.uuid] = pychromecast.discovery.CastInfo(
            {service},
            info.uuid,
            info.cast_info.model_name,
            info.friendly_name,
            info.cast_info.host,
            info.cast_info.port,
            info.cast_info.cast_type,
            info.cast_info.manufacturer,
        )
        discovery_callback(info.uuid, "")

    def remove_chromecast(service_name: str, info: ChromecastInfo) -> None:
        """Remove a chromecast device."""
        remove_callback(
            info.uuid,
            service_name,
            pychromecast.models.CastInfo(
                set(),
                info.uuid,
                info.cast_info.model_name,
                info.cast_info.friendly_name,
                info.cast_info.host,
                info.cast_info.port,
                info.cast_info.cast_type,
                info.cast_info.manufacturer,
            ),
        )

    return discover_chromecast, remove_chromecast, add_entities


async def async_setup_media_player_cast(hass: HomeAssistant, info: ChromecastInfo) -> tuple[MagicMock, Callable[[str, ChromecastInfo], None]]:
    """Set up a cast config entry."""
    browser = MagicMock(devices={}, zc={})
    chromecast = get_fake_chromecast(info)
    zconf = get_fake_zconf(host=info.cast_info.host, port=info.cast_info.port)

    with (
        patch(
            "homeassistant.components.cast.discovery.pychromecast.get_chromecast_from_cast_info",
            return_value=chromecast,
        ) as get_chromecast,
        patch(
            "homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser",
            return_value=browser,
        ) as cast_browser,
        patch(
            "homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf",
            return_value=zconf,
        ),
    ):
        data = {"ignore_cec": [], "known_hosts": [], "uuid": [str(info.uuid)]}
        entry = MockConfigEntry(data=data, domain="cast")
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
        await hass.async_block_till_done(wait_background_tasks=True)

        discovery_callback = cast_browser.call_args[0][0].add_cast

        browser.devices[info.uuid] = pychromecast.discovery.CastInfo(
            {FAKE_MDNS_SERVICE},
            info.uuid,
            info.cast_info.model_name,
            info.friendly_name,
            info.cast_info.host,
            info.cast_info.port,
            info.cast_info.cast_type,
            info.cast_info.manufacturer,
        )
        discovery_callback(info.uuid, FAKE_MDNS_SERVICE.name)

        await hass.async_block_till_done()
        await hass.async_block_till_done()
        assert get_chromecast.call_count == 1

        def discover_chromecast(service_name: str, info: ChromecastInfo) -> None:
            """Discover a chromecast device."""
            browser.devices[info.uuid] = pychromecast.discovery.CastInfo(
                {FAKE_MDNS_SERVICE},
                info.uuid,
                info.cast_info.model_name,
                info.friendly_name,
                info.cast_info.host,
                info.cast_info.port,
                info.cast_info.cast_type,
                info.cast_info.manufacturer,
            )
            discovery_callback(info.uuid, FAKE_MDNS_SERVICE[1])

        return chromecast, discover_chromecast


def get_status_callbacks(chromecast_mock: MagicMock, mz_mock: MagicMock | None = None) -> tuple[Callable, Callable, Callable] | tuple[Callable, Callable, Callable, Callable]:
    """Get registered status callbacks from the chromecast mock."""
    status_listener = chromecast_mock.register_status_listener.call_args[0][0]
    cast_status_cb = status_listener.new_cast_status

    connection_listener = chromecast_mock.register_connection_listener.call_args[0][0]
    conn_status_cb = connection_listener.new_connection_status

    mc = chromecast_mock.socket_client.media_controller
    media_status_cb = mc.register_status_listener.call_args[0][0].new_media_status

    if not mz_mock:
        return cast_status_cb, conn_status_cb, media_status_cb

    mz_listener = mz_mock.register_listener.call_args[0][1]
    group_media_status_cb = mz_listener.multizone_new_media_status
    return cast_status_cb, conn_status_cb, media_status_cb, group_media_status_cb
