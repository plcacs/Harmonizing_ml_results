from __future__ import annotations
import asyncio
from collections.abc import Callable
import json
from typing import Any, Optional, Tuple, Union
from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import attr
import pychromecast
from pychromecast.const import CAST_TYPE_CHROMECAST, CAST_TYPE_GROUP
import pytest
import yarl

from homeassistant.components import media_player, tts
from homeassistant.components.cast import media_player as cast
from homeassistant.components.cast.const import SIGNAL_HASS_CAST_SHOW_VIEW, HomeAssistantControllerData
from homeassistant.components.cast.media_player import ChromecastInfo
from homeassistant.components.media_player import BrowseMedia, MediaClass, MediaPlayerEntityFeature
from homeassistant.const import ATTR_ENTITY_ID, CAST_APP_ID_HOMEASSISTANT_LOVELACE, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.core_config import async_process_ha_core_config
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, network
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.setup import async_setup_component
from tests.common import MockConfigEntry, assert_setup_component, load_fixture, mock_platform
from tests.components.media_player import common
from tests.test_util.aiohttp import AiohttpClientMocker
from tests.typing import WebSocketGenerator

FakeUUID: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e2")
FakeUUID2: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e4")
FakeGroupUUID: UUID = UUID("57355bce-9364-4aa6-ac1e-eb849dccf9e3")
FAKE_HOST_SERVICE: pychromecast.discovery.HostServiceInfo = pychromecast.discovery.HostServiceInfo("127.0.0.1", 8009)
FAKE_MDNS_SERVICE: pychromecast.discovery.MDNSServiceInfo = pychromecast.discovery.MDNSServiceInfo("the-service")
UNDEFINED = object()


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
    service: Optional[pychromecast.discovery.HostServiceInfo] = None,
    uuid: UUID = FakeUUID,
    cast_type: Any = UNDEFINED,
    manufacturer: Any = UNDEFINED,
    model_name: Any = UNDEFINED,
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


async def async_setup_cast(hass: HomeAssistant, config: Optional[dict[str, Any]] = None) -> MagicMock:
    """Set up the cast platform."""
    if config is None:
        config = {}
    data = {"ignore_cec": [], "known_hosts": [], "uuid": [], **config}
    with patch("homeassistant.helpers.entity_platform.EntityPlatform._async_schedule_add_entities_for_entry") as add_entities:
        entry = MockConfigEntry(data=data, domain="cast")
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return add_entities


async def async_setup_cast_internal_discovery(
    hass: HomeAssistant, config: Optional[dict[str, Any]] = None
) -> Tuple[Callable[[Any, Any], None], Callable[[Any, Any], None], MagicMock]:
    """Set up the cast platform and the discovery."""
    browser = MagicMock(devices={}, zc={})
    with patch("homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser", return_value=browser) as cast_browser:
        add_entities = await async_setup_cast(hass, config)
        await hass.async_block_till_done(wait_background_tasks=True)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert browser.start_discovery.call_count == 1
        discovery_callback = cast_browser.call_args[0][0].add_cast
        remove_callback = cast_browser.call_args[0][0].remove_cast

    def discover_chromecast(service: Any, info: ChromecastInfo) -> None:
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

    def remove_chromecast(service_name: Any, info: ChromecastInfo) -> None:
        """Remove a chromecast device."""
        remove_callback(
            info.uuid,
            service_name,
            pychromecast.models.CastInfo(
                set(), info.uuid, info.cast_info.model_name, info.cast_info.friendly_name, info.cast_info.host, info.cast_info.port, info.cast_info.cast_type, info.cast_info.manufacturer
            ),
        )

    return discover_chromecast, remove_chromecast, add_entities


async def async_setup_media_player_cast(hass: HomeAssistant, info: ChromecastInfo) -> Tuple[MagicMock, Callable[[Any, Any], None]]:
    """Set up a cast config entry."""
    browser = MagicMock(devices={}, zc={})
    chromecast = get_fake_chromecast(info)
    zconf = get_fake_zconf(host=info.cast_info.host, port=info.cast_info.port)
    with patch(
        "homeassistant.components.cast.discovery.pychromecast.get_chromecast_from_cast_info", return_value=chromecast
    ) as get_chromecast, patch(
        "homeassistant.components.cast.discovery.pychromecast.discovery.CastBrowser", return_value=browser
    ) as cast_browser, patch(
        "homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf
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

        def discover_chromecast(service_name: Any, info: ChromecastInfo) -> None:
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


def get_status_callbacks(chromecast_mock: MagicMock, mz_mock: Optional[MagicMock] = None) -> Union[
    Tuple[Callable[[Any], None], Callable[[Any], None], Callable[[Any], None]],
    Tuple[Callable[[Any], None], Callable[[Any], None], Callable[[Any], None], Callable[[Any, Any], None]]
]:
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


async def test_start_discovery_called_once(hass: HomeAssistant, castbrowser_mock: MagicMock) -> None:
    """Test pychromecast.start_discovery called exactly once."""
    await async_setup_cast(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert castbrowser_mock.return_value.start_discovery.call_count == 1
    await async_setup_cast(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert castbrowser_mock.return_value.start_discovery.call_count == 1


async def test_internal_discovery_callback_fill_out_group_fail(
    hass: HomeAssistant, get_multizone_status_mock: MagicMock
) -> None:
    """Test internal discovery automatically filling out information."""
    discover_cast, _, _ = await async_setup_cast_internal_discovery(hass)
    info = get_fake_chromecast_info(host="host1", port=12345, service=FAKE_MDNS_SERVICE)
    zconf = get_fake_zconf(host="host1", port=12345)
    full_info = attr.evolve(
        info,
        cast_info=pychromecast.discovery.CastInfo(
            services=info.cast_info.services,
            uuid=FakeUUID,
            model_name="Chromecast",
            friendly_name="Speaker",
            host=info.cast_info.host,
            port=info.cast_info.port,
            cast_type=info.cast_info.cast_type,
            manufacturer=info.cast_info.manufacturer,
        ),
        is_dynamic_group=False,
    )
    get_multizone_status_mock.assert_not_called()
    get_multizone_status_mock.return_value = None
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf):
        signal = MagicMock()
        async_dispatcher_connect(hass, "cast_discovered", signal)
        discover_cast(FAKE_MDNS_SERVICE, info)
        await hass.async_block_till_done()
        discover = signal.mock_calls[-1][1][0]
        assert discover == full_info
        get_multizone_status_mock.assert_called_once()


async def test_internal_discovery_callback_fill_out_group(
    hass: HomeAssistant, get_multizone_status_mock: MagicMock
) -> None:
    """Test internal discovery automatically filling out information."""
    discover_cast, _, _ = await async_setup_cast_internal_discovery(hass)
    info = get_fake_chromecast_info(host="host1", port=12345, service=FAKE_MDNS_SERVICE)
    zconf = get_fake_zconf(host="host1", port=12345)
    full_info = attr.evolve(
        info,
        cast_info=pychromecast.discovery.CastInfo(
            services=info.cast_info.services,
            uuid=FakeUUID,
            model_name="Chromecast",
            friendly_name="Speaker",
            host=info.cast_info.host,
            port=info.cast_info.port,
            cast_type=info.cast_info.cast_type,
            manufacturer=info.cast_info.manufacturer,
        ),
        is_dynamic_group=False,
    )
    get_multizone_status_mock.assert_not_called()
    get_multizone_status_mock.return_value = None
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf):
        signal = MagicMock()
        async_dispatcher_connect(hass, "cast_discovered", signal)
        discover_cast(FAKE_MDNS_SERVICE, info)
        await hass.async_block_till_done()
        discover = signal.mock_calls[-1][1][0]
        assert discover == full_info
        get_multizone_status_mock.assert_called_once()


async def test_internal_discovery_callback_fill_out_cast_type_manufacturer(
    hass: HomeAssistant, get_cast_type_mock: MagicMock, caplog: Any
) -> None:
    """Test internal discovery automatically filling out information."""
    discover_cast, _, _ = await async_setup_cast_internal_discovery(hass)
    info = get_fake_chromecast_info(host="host1", port=8009, service=FAKE_MDNS_SERVICE, cast_type=None, manufacturer=None)
    info2 = get_fake_chromecast_info(host="host1", port=8009, service=FAKE_MDNS_SERVICE, cast_type=None, manufacturer=None, model_name="Model 101")
    zconf = get_fake_zconf(host="host1", port=8009)
    full_info = attr.evolve(
        info,
        cast_info=pychromecast.discovery.CastInfo(
            services=info.cast_info.services,
            uuid=FakeUUID,
            model_name="Chromecast",
            friendly_name="Speaker",
            host=info.cast_info.host,
            port=info.cast_info.port,
            cast_type="audio",
            manufacturer="TrollTech",
        ),
        is_dynamic_group=None,
    )
    full_info2 = attr.evolve(
        info2,
        cast_info=pychromecast.discovery.CastInfo(
            services=info.cast_info.services,
            uuid=FakeUUID,
            model_name="Model 101",
            friendly_name="Speaker",
            host=info.cast_info.host,
            port=info.cast_info.port,
            cast_type="cast",
            manufacturer="Cyberdyne Systems",
        ),
        is_dynamic_group=None,
    )
    get_cast_type_mock.assert_not_called()
    get_cast_type_mock.return_value = full_info.cast_info
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf):
        signal = MagicMock()
        async_dispatcher_connect(hass, "cast_discovered", signal)
        discover_cast(FAKE_MDNS_SERVICE, info)
        await hass.async_block_till_done()
        get_cast_type_mock.assert_called_once()
        assert get_cast_type_mock.call_count == 1
        discover = signal.mock_calls[2][1][0]
        assert discover == full_info
        assert "Fetched cast details for unknown model 'Chromecast'" in caplog.text
        signal.reset_mock()
        discover_cast(FAKE_MDNS_SERVICE, info)
        await hass.async_block_till_done()
        assert get_cast_type_mock.call_count == 1
        discover = signal.mock_calls[0][1][0]
        assert discover == full_info
        signal.reset_mock()
        get_cast_type_mock.return_value = full_info2.cast_info
        discover_cast(FAKE_MDNS_SERVICE, info2)
        await hass.async_block_till_done()
        assert get_cast_type_mock.call_count == 2
        discover = signal.mock_calls[0][1][0]
        assert discover == full_info2


async def test_stop_discovery_called_on_stop(hass: HomeAssistant, castbrowser_mock: MagicMock) -> None:
    """Test pychromecast.stop_discovery called on shutdown."""
    await async_setup_cast(hass, {})
    await hass.async_block_till_done(wait_background_tasks=True)
    assert castbrowser_mock.return_value.start_discovery.call_count == 1
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done(wait_background_tasks=True)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert castbrowser_mock.return_value.stop_discovery.call_count == 1


async def test_create_cast_device_without_uuid(hass: HomeAssistant) -> None:
    """Test create a cast device with no UUId does not create an entity."""
    info = get_fake_chromecast_info(uuid=None)
    cast_device = cast._async_create_cast_device(hass, info)
    assert cast_device is None


async def test_create_cast_device_with_uuid(hass: HomeAssistant) -> None:
    """Test create cast devices with UUID creates entities."""
    added_casts: set[Any] = hass.data[cast.ADDED_CAST_DEVICES_KEY] = set()
    info = get_fake_chromecast_info()
    cast_device = cast._async_create_cast_device(hass, info)
    assert cast_device is not None
    assert info.uuid in added_casts
    cast_device = cast._async_create_cast_device(hass, info)
    assert cast_device is None


async def test_manual_cast_chromecasts_uuid(hass: HomeAssistant) -> None:
    """Test only wanted casts are added for manual configuration."""
    cast_1 = get_fake_chromecast_info(host="host_1", uuid=FakeUUID)
    cast_2 = get_fake_chromecast_info(host="host_2", uuid=FakeUUID2)
    zconf_1 = get_fake_zconf(host="host_1")
    zconf_2 = get_fake_zconf(host="host_2")
    discover_cast, _, add_dev1 = await async_setup_cast_internal_discovery(hass, config={"uuid": str(FakeUUID)})
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_2):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service2"), cast_2)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 0
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service1"), cast_1)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 1


async def test_auto_cast_chromecasts(hass: HomeAssistant) -> None:
    """Test all discovered casts are added for default configuration."""
    cast_1 = get_fake_chromecast_info(host="some_host")
    cast_2 = get_fake_chromecast_info(host="other_host", uuid=FakeUUID2)
    zconf_1 = get_fake_zconf(host="some_host")
    zconf_2 = get_fake_zconf(host="other_host")
    discover_cast, _, add_dev1 = await async_setup_cast_internal_discovery(hass)
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service2"), cast_2)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 1
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_2):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service1"), cast_1)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 2


async def test_discover_dynamic_group(
    hass: HomeAssistant, entity_registry: Any, get_multizone_status_mock: MagicMock, get_chromecast_mock: MagicMock, caplog: Any
) -> None:
    """Test dynamic group does not create device or entity."""
    cast_1 = get_fake_chromecast_info(host="host_1", port=23456, uuid=FakeUUID)
    cast_2 = get_fake_chromecast_info(host="host_2", port=34567, uuid=FakeUUID2)
    zconf_1 = get_fake_zconf(host="host_1", port=23456)
    zconf_2 = get_fake_zconf(host="host_2", port=34567)
    tmp1 = MagicMock()
    tmp1.uuid = FakeUUID
    tmp2 = MagicMock()
    tmp2.uuid = FakeUUID2
    get_multizone_status_mock.return_value.dynamic_groups = [tmp1, tmp2]
    get_chromecast_mock.assert_not_called()
    discover_cast, remove_cast, add_dev1 = await async_setup_cast_internal_discovery(hass)
    tasks: list[asyncio.Task] = []
    real_create_task = asyncio.create_task

    def create_task(coroutine: Any, name: Any) -> None:
        tasks.append(real_create_task(coroutine))

    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1), patch.object(
        hass, "async_create_background_task", wraps=create_task
    ):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service"), cast_1)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    tasks.clear()
    get_chromecast_mock.assert_called()
    get_chromecast_mock.reset_mock()
    assert add_dev1.call_count == 0
    assert entity_registry.async_get_entity_id("media_player", "cast", cast_1.uuid) is None
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_2), patch.object(
        hass, "async_create_background_task", wraps=create_task
    ):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service"), cast_2)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    tasks.clear()
    get_chromecast_mock.assert_called()
    get_chromecast_mock.reset_mock()
    assert add_dev1.call_count == 0
    assert entity_registry.async_get_entity_id("media_player", "cast", cast_2.uuid) is None
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1), patch.object(
        hass, "async_create_background_task", wraps=create_task
    ):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service"), cast_1)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
    assert len(tasks) == 0
    get_chromecast_mock.assert_not_called()
    assert add_dev1.call_count == 0
    assert entity_registry.async_get_entity_id("media_player", "cast", cast_1.uuid) is None
    assert "Disconnecting from chromecast" not in caplog.text
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1):
        remove_cast(pychromecast.discovery.MDNSServiceInfo("service"), cast_1)
        await hass.async_block_till_done()
        await hass.async_block_till_done()
    assert "Disconnecting from chromecast" in caplog.text


async def test_update_cast_chromecasts(hass: HomeAssistant) -> None:
    """Test discovery of same UUID twice only adds one cast."""
    cast_1 = get_fake_chromecast_info(host="old_host")
    cast_2 = get_fake_chromecast_info(host="new_host")
    zconf_1 = get_fake_zconf(host="old_host")
    zconf_2 = get_fake_zconf(host="new_host")
    discover_cast, _, add_dev1 = await async_setup_cast_internal_discovery(hass)
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_1):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service1"), cast_1)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 1
    with patch("homeassistant.components.cast.discovery.ChromeCastZeroconf.get_zeroconf", return_value=zconf_2):
        discover_cast(pychromecast.discovery.MDNSServiceInfo("service2"), cast_2)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert add_dev1.call_count == 1


async def test_entity_availability(hass: HomeAssistant) -> None:
    """Test handling of connection status."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    state = hass.states.get(entity_id)
    assert state.state == "unavailable"
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "off"
    connection_status = MagicMock()
    connection_status.status = "LOST"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "unavailable"
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "off"
    connection_status = MagicMock()
    connection_status.status = "DISCONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "unavailable"
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "unavailable"


@pytest.mark.parametrize(("port", "entry_type"), [(8009, None), (12345, None)])
async def test_device_registry(
    hass: HomeAssistant, hass_ws_client: Any, device_registry: Any, entity_registry: Any, port: int, entry_type: Any
) -> None:
    """Test device registry integration."""
    assert await async_setup_component(hass, "config", {})
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info(port=port)
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = pychromecast.const.CAST_TYPE_CHROMECAST
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    cast_entry = hass.config_entries.async_entries("cast")[0]
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    entity_entry = entity_registry.async_get(entity_id)
    device_entry = device_registry.async_get(entity_entry.device_id)
    assert entity_entry.device_id == device_entry.id
    assert device_entry.entry_type == entry_type
    chromecast.disconnect.assert_not_called()
    client = await hass_ws_client(hass)
    response = await client.remove_device(device_entry.id, cast_entry.entry_id)
    assert response["success"]
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    chromecast.disconnect.assert_called_once()
    assert entity_registry.async_get(entity_id) is None
    assert device_registry.async_get(entity_entry.device_id) is None


async def test_entity_cast_status(hass: HomeAssistant, entity_registry: Any) -> None:
    """Test handling of cast status."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = pychromecast.const.CAST_TYPE_CHROMECAST
    cast_status_cb, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    assert state.attributes.get("supported_features") == (
        MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )
    cast_status = MagicMock()
    cast_status.volume_level = 0.5
    cast_status.volume_muted = False
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("volume_level") is None
    assert not state.attributes.get("is_volume_muted")
    chromecast.app_id = "1234"
    cast_status = MagicMock()
    cast_status.volume_level = 0.5
    cast_status.volume_muted = False
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("volume_level") == 0.5
    assert not state.attributes.get("is_volume_muted")
    cast_status = MagicMock()
    cast_status.volume_level = 0.2
    cast_status.volume_muted = True
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("volume_level") == 0.2
    assert state.attributes.get("is_volume_muted")
    cast_status = MagicMock()
    cast_status.volume_control_type = "fixed"
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("supported_features") == (
        MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )


@pytest.mark.parametrize(
    ("cast_type", "supported_features", "supported_features_no_media"),
    [
        (
            pychromecast.const.CAST_TYPE_AUDIO,
            MediaPlayerEntityFeature.PAUSE
            | MediaPlayerEntityFeature.PLAY
            | MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.STOP
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
            MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
        ),
        (
            pychromecast.const.CAST_TYPE_CHROMECAST,
            MediaPlayerEntityFeature.PAUSE
            | MediaPlayerEntityFeature.PLAY
            | MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.STOP
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
            MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
        ),
        (
            pychromecast.const.CAST_TYPE_GROUP,
            MediaPlayerEntityFeature.PAUSE
            | MediaPlayerEntityFeature.PLAY
            | MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.STOP
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
            MediaPlayerEntityFeature.PLAY_MEDIA
            | MediaPlayerEntityFeature.TURN_OFF
            | MediaPlayerEntityFeature.TURN_ON
            | MediaPlayerEntityFeature.VOLUME_MUTE
            | MediaPlayerEntityFeature.VOLUME_SET,
        ),
    ],
)
async def test_supported_features(
    hass: HomeAssistant, cast_type: Any, supported_features: int, supported_features_no_media: int
) -> None:
    """Test supported features."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = cast_type
    _, conn_status_cb, media_status_cb = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert state.attributes.get("supported_features") == supported_features_no_media
    media_status = MagicMock(images=None)
    media_status.supports_queue_next = False
    media_status.supports_seek = False
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("supported_features") == supported_features


async def test_entity_browse_media(hass: HomeAssistant, hass_ws_client: Any) -> None:
    """Test we can browse media."""
    await async_setup_component(hass, "media_source", {"media_source": {}})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    client = await hass_ws_client()
    await client.send_json(
        {
            "id": 1,
            "type": "media_player/browse_media",
            "entity_id": "media_player.speaker",
        }
    )
    response = await client.receive_json()
    assert response["success"]
    expected_child_1 = {
        "title": "Epic Sax Guy 10 Hours.mp4",
        "media_class": "video",
        "media_content_type": "video/mp4",
        "media_content_id": "media-source://media_source/local/Epic Sax Guy 10 Hours.mp4",
        "can_play": True,
        "can_expand": False,
        "thumbnail": None,
        "children_media_class": None,
    }
    assert expected_child_1 in response["result"]["children"]
    expected_child_2 = {
        "title": "test.mp3",
        "media_class": "music",
        "media_content_type": "audio/mpeg",
        "media_content_id": "media-source://media_source/local/test.mp3",
        "can_play": True,
        "can_expand": False,
        "thumbnail": None,
        "children_media_class": None,
    }
    assert expected_child_2 in response["result"]["children"]


@pytest.mark.parametrize("cast_type", [pychromecast.const.CAST_TYPE_AUDIO, pychromecast.const.CAST_TYPE_GROUP])
async def test_entity_browse_media_audio_only(hass: HomeAssistant, hass_ws_client: Any, cast_type: Any) -> None:
    """Test we can browse media."""
    await async_setup_component(hass, "media_source", {"media_source": {}})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = cast_type
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    client = await hass_ws_client()
    await client.send_json(
        {
            "id": 1,
            "type": "media_player/browse_media",
            "entity_id": "media_player.speaker",
        }
    )
    response = await client.receive_json()
    assert response["success"]
    expected_child_1 = {
        "title": "Epic Sax Guy 10 Hours.mp4",
        "media_class": "video",
        "media_content_type": "video/mp4",
        "media_content_id": "media-source://media_source/local/Epic Sax Guy 10 Hours.mp4",
        "can_play": True,
        "can_expand": False,
        "thumbnail": None,
        "children_media_class": None,
    }
    assert expected_child_1 not in response["result"]["children"]
    expected_child_2 = {
        "title": "test.mp3",
        "media_class": "music",
        "media_content_type": "audio/mpeg",
        "media_content_id": "media-source://media_source/local/test.mp3",
        "can_play": True,
        "can_expand": False,
        "thumbnail": None,
        "children_media_class": None,
    }
    assert expected_child_2 in response["result"]["children"]


async def test_entity_play_media(hass: HomeAssistant, entity_registry: Any, quick_play_mock: MagicMock) -> None:
    """Test playing media."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "audio",
            media_player.ATTR_MEDIA_CONTENT_ID: "http://example.com/best.mp3",
            media_player.ATTR_MEDIA_EXTRA: {"metadata": {"metadatatype": 3}},
        },
        blocking=True,
    )
    chromecast.media_controller.play_media.assert_not_called()
    quick_play_mock.assert_called_once_with(
        chromecast, "default_media_receiver", {"media_id": "http://example.com/best.mp3", "media_type": "audio", "metadata": {"metadatatype": 3}}
    )


async def test_entity_play_media_cast(hass: HomeAssistant, entity_registry: Any, quick_play_mock: MagicMock) -> None:
    """Test playing media with cast special features."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    await common.async_play_media(hass, "cast", '{"app_id": "abc123"}', entity_id)
    chromecast.start_app.assert_called_once_with("abc123")
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "cast",
            media_player.ATTR_MEDIA_CONTENT_ID: '{"app_name":"youtube"}',
            media_player.ATTR_MEDIA_EXTRA: {"metadata": {"metadatatype": 3}},
        },
        blocking=True,
    )
    quick_play_mock.assert_called_once_with(ANY, "youtube", {"metadata": {"metadatatype": 3}})


async def test_entity_play_media_cast_invalid(hass: HomeAssistant, entity_registry: Any, caplog: Any, quick_play_mock: MagicMock) -> None:
    """Test playing media."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    with pytest.raises(json.decoder.JSONDecodeError):
        await common.async_play_media(hass, "cast", '{"app_id": "abc123"', entity_id)
    assert "Invalid JSON in media_content_id" in caplog.text
    chromecast.start_app.assert_not_called()
    quick_play_mock.assert_not_called()
    await common.async_play_media(hass, "cast", '{"app_id": "abc123", "extra": "data"}', entity_id)
    assert "Extra keys dict_keys(['extra']) were ignored" in caplog.text
    chromecast.start_app.assert_called_once_with("abc123")
    quick_play_mock.assert_not_called()
    quick_play_mock.side_effect = NotImplementedError()
    await common.async_play_media(hass, "cast", '{"app_name": "unknown"}', entity_id)
    quick_play_mock.assert_called_once_with(ANY, "unknown", {})
    assert "App unknown not supported" in caplog.text


async def test_entity_play_media_sign_URL(hass: HomeAssistant, quick_play_mock: MagicMock) -> None:
    """Test playing media."""
    entity_id = "media_player.speaker"
    await async_process_ha_core_config(hass, {"internal_url": "http://example.com:8123"})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    await common.async_play_media(hass, "audio", "/best.mp3", entity_id)
    quick_play_mock.assert_called_once_with(
        chromecast, "default_media_receiver", {"media_id": ANY, "media_type": "audio"}
    )
    assert quick_play_mock.call_args[0][2]["media_id"].startswith("http://example.com:8123/best.mp3?authSig=")


@pytest.mark.parametrize(
    ("url", "fixture", "playlist_item"),
    [
        (
            "https://sverigesradio.se/topsy/direkt/209-hi-mp3.m3u",
            "209-hi-mp3.m3u",
            {"media_id": "https://http-live.sr.se/p4norrbotten-mp3-192", "media_type": "audio", "metadata": {"title": "Sveriges Radio"}},
        ),
        (
            "http://sverigesradio.se/topsy/direkt/164-hi-aac.pls",
            "164-hi-aac.pls",
            {"media_id": "https://http-live.sr.se/p3-aac-192", "media_type": "audio", "metadata": {"title": "Sveriges Radio"}},
        ),
        (
            "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/nonuk/sbr_low/ak/bbc_radio_fourfm.m3u8",
            "bbc_radio_fourfm.m3u8",
            {"media_id": "http://a.files.bbci.co.uk/media/live/manifesto/audio/simulcast/hls/nonuk/sbr_low/ak/bbc_radio_fourfm.m3u8", "media_type": "audio"},
        ),
        (
            "https://sverigesradio.se/209-hi-mp3.m3u",
            "209-hi-mp3_bad_url.m3u",
            {"media_id": "https://sverigesradio.se/209-hi-mp3.m3u", "media_type": "audio"},
        ),
    ],
)
async def test_entity_play_media_playlist(
    hass: HomeAssistant, aioclient_mock: Any, quick_play_mock: MagicMock, url: str, fixture: str, playlist_item: dict[str, Any]
) -> None:
    """Test playing media."""
    entity_id = "media_player.speaker"
    aioclient_mock.get(url, text=load_fixture(fixture, "cast"))
    await async_process_ha_core_config(hass, {"internal_url": "http://example.com:8123"})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    await common.async_play_media(hass, "audio", url, entity_id)
    quick_play_mock.assert_called_once_with(chromecast, "default_media_receiver", playlist_item)


@pytest.mark.parametrize(
    ("cast_type", "default_content_type"),
    [
        (pychromecast.const.CAST_TYPE_AUDIO, "music"),
        (pychromecast.const.CAST_TYPE_GROUP, "music"),
        (pychromecast.const.CAST_TYPE_CHROMECAST, "video"),
    ],
)
async def test_entity_media_content_type(
    hass: HomeAssistant, entity_registry: Any, cast_type: Any, default_content_type: str
) -> None:
    """Test various content types."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = cast_type
    _, conn_status_cb, media_status_cb = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    media_status = MagicMock(images=None)
    media_status.media_is_movie = False
    media_status.media_is_musictrack = False
    media_status.media_is_tvshow = False
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("media_content_type") == default_content_type
    media_status.media_is_tvshow = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("media_content_type") == "tvshow"
    media_status.media_is_tvshow = False
    media_status.media_is_musictrack = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("media_content_type") == "music"
    media_status.media_is_musictrack = True
    media_status.media_is_movie = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("media_content_type") == "movie"


async def test_entity_control(hass: HomeAssistant, entity_registry: Any, quick_play_mock: MagicMock) -> None:
    """Test various device and media controls."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = pychromecast.const.CAST_TYPE_CHROMECAST
    _, conn_status_cb, media_status_cb = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    media_status = MagicMock(images=None)
    media_status.player_state = "PLAYING"
    media_status.supports_queue_next = False
    media_status.supports_seek = False
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "playing"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    assert state.attributes.get("supported_features") == (
        MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.PLAY
        | MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.STOP
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )
    await common.async_turn_on(hass, entity_id)
    quick_play_mock.assert_called_once_with(
        chromecast,
        "default_media_receiver",
        {"media_id": "https://www.home-assistant.io/images/cast/splash.png", "media_type": "image/png"},
    )
    chromecast.quit_app.reset_mock()
    await common.async_turn_off(hass, entity_id)
    chromecast.quit_app.assert_called_once_with()
    await common.async_mute_volume(hass, True, entity_id)
    chromecast.set_volume_muted.assert_called_once_with(True)
    await common.async_set_volume_level(hass, 0.33, entity_id)
    chromecast.set_volume.assert_called_once_with(0.33)
    await common.async_media_play(hass, entity_id)
    chromecast.media_controller.play.assert_called_once_with()
    await common.async_media_pause(hass, entity_id)
    chromecast.media_controller.pause.assert_called_once_with()
    with pytest.raises(HomeAssistantError):
        await common.async_media_previous_track(hass, entity_id)
    chromecast.media_controller.queue_prev.assert_not_called()
    with pytest.raises(HomeAssistantError):
        await common.async_media_next_track(hass, entity_id)
    chromecast.media_controller.queue_next.assert_not_called()
    with pytest.raises(HomeAssistantError):
        await common.async_media_seek(hass, 123, entity_id)
    chromecast.media_controller.seek.assert_not_called()
    media_status = MagicMock(images=None)
    media_status.supports_queue_next = True
    media_status.supports_seek = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.attributes.get("supported_features") == (
        MediaPlayerEntityFeature.PAUSE
        | MediaPlayerEntityFeature.PLAY
        | MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.STOP
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.PREVIOUS_TRACK
        | MediaPlayerEntityFeature.NEXT_TRACK
        | MediaPlayerEntityFeature.SEEK
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )
    await common.async_media_previous_track(hass, entity_id)
    chromecast.media_controller.queue_prev.assert_called_once_with()
    await common.async_media_next_track(hass, entity_id)
    chromecast.media_controller.queue_next.assert_called_once_with()
    await common.async_media_seek(hass, 123, entity_id)
    chromecast.media_controller.seek.assert_called_once_with(123)


@pytest.mark.parametrize(("app_id", "state_no_media"), [(pychromecast.APP_YOUTUBE, "idle"), ("Netflix", "playing")])
async def test_entity_media_states(
    hass: HomeAssistant, entity_registry: Any, app_id: str, state_no_media: str
) -> None:
    """Test various entity media states."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    cast_status_cb, conn_status_cb, media_status_cb = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    chromecast.app_id = app_id
    cast_status = MagicMock()
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == state_no_media
    media_status = MagicMock(images=None)
    media_status.player_state = "BUFFERING"
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "buffering"
    media_status.player_state = "PLAYING"
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    media_status.player_state = None
    media_status.player_is_paused = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "paused"
    media_status.player_is_paused = False
    media_status.player_is_idle = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "idle"
    media_status_cb(None)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == state_no_media
    chromecast.app_id = pychromecast.IDLE_APP_ID
    cast_status = MagicMock()
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "off"
    chromecast.is_idle = False
    cast_status_cb(None)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "unknown"


async def test_entity_media_states_lovelace_app(hass: HomeAssistant, entity_registry: Any) -> None:
    """Test various entity media states when the lovelace app is active."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    cast_status_cb, conn_status_cb, media_status_cb = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    chromecast.app_id = CAST_APP_ID_HOMEASSISTANT_LOVELACE
    cast_status = MagicMock()
    cast_status_cb(cast_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    assert state.attributes.get("supported_features") == (
        MediaPlayerEntityFeature.PLAY_MEDIA
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
    )
    media_status = MagicMock(images=None)
    media_status.player_is_playing = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    media_status.player_is_playing = False
    media_status.player_is_paused = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    media_status.player_is_paused = False
    media_status.player_is_idle = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    chromecast.app_id = pychromecast.IDLE_APP_ID
    media_status.player_is_idle = False
    chromecast.is_idle = True
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "off"
    chromecast.is_idle = False
    media_status_cb(media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "unknown"


async def test_group_media_states(hass: HomeAssistant, entity_registry: Any, mz_mock: MagicMock) -> None:
    """Test media states are read from group if entity has no state."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, media_status_cb, group_media_status_cb = get_status_callbacks(chromecast, mz_mock)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    group_media_status = MagicMock(images=None)
    player_media_status = MagicMock(images=None)
    group_media_status.player_state = "BUFFERING"
    group_media_status_cb(str(FakeGroupUUID), group_media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "buffering"
    group_media_status.player_state = "PLAYING"
    group_media_status_cb(str(FakeGroupUUID), group_media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"
    player_media_status.player_state = None
    player_media_status.player_is_paused = True
    media_status_cb(player_media_status)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "paused"
    player_media_status.player_state = "UNKNOWN"
    media_status_cb(player_media_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state.state == "playing"


async def test_group_media_states_early(hass: HomeAssistant, entity_registry: Any, mz_mock: MagicMock) -> None:
    """Test media states are read from group if entity has no state.

    This tests case asserts group state is polled when the player is created.
    """
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    mz_mock.get_multizone_memberships = MagicMock(return_value=[str(FakeGroupUUID)])
    mz_mock.get_multizone_mediacontroller = MagicMock(return_value=MagicMock(status=MagicMock(images=None, player_state="BUFFERING")))
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _, _ = get_status_callbacks(chromecast, mz_mock)
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "unavailable"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    assert hass.states.get(entity_id).state == "buffering"
    connection_status = MagicMock()
    connection_status.status = "LOST"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    assert hass.states.get(entity_id).state == "unavailable"
    mz_mock.get_multizone_mediacontroller = MagicMock(return_value=MagicMock(status=MagicMock(images=None, player_state="PLAYING")))
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    await hass.async_block_till_done()
    assert hass.states.get(entity_id).state == "playing"


async def test_group_media_control(hass: HomeAssistant, entity_registry: Any, mz_mock: MagicMock, quick_play_mock: MagicMock) -> None:
    """Test media controls are handled by group if entity has no state."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, media_status_cb, group_media_status_cb = get_status_callbacks(chromecast, mz_mock)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    state = hass.states.get(entity_id)
    assert state is not None
    assert state.name == "Speaker"
    assert state.state == "off"
    assert entity_id == entity_registry.async_get_entity_id("media_player", "cast", str(info.uuid))
    group_media_status = MagicMock(images=None)
    player_media_status = MagicMock(images=None)
    group_media_status.player_is_playing = True
    group_media_status_cb(str(FakeGroupUUID), group_media_status)
    await common.async_media_play(hass, entity_id)
    grp_media = mz_mock.get_multizone_mediacontroller(str(FakeGroupUUID))
    assert grp_media.play.called
    assert not chromecast.media_controller.play.called
    player_media_status.player_is_playing = False
    player_media_status.player_is_paused = True
    media_status_cb(player_media_status)
    await common.async_media_pause(hass, entity_id)
    grp_media = mz_mock.get_multizone_mediacontroller(str(FakeGroupUUID))
    assert not grp_media.pause.called
    assert chromecast.media_controller.pause.called
    player_media_status.player_state = "UNKNOWN"
    media_status_cb(player_media_status)
    await common.async_media_stop(hass, entity_id)
    grp_media = mz_mock.get_multizone_mediacontroller(str(FakeGroupUUID))
    assert grp_media.stop.called
    assert not chromecast.media_controller.stop.called
    await common.async_play_media(hass, "music", "http://example.com/best.mp3", entity_id)
    assert not grp_media.play_media.called
    assert not chromecast.media_controller.play_media.called
    quick_play_mock.assert_called_once_with(chromecast, "default_media_receiver", {"media_id": "http://example.com/best.mp3", "media_type": "music"})


async def test_failed_cast_on_idle(hass: HomeAssistant, caplog: Any) -> None:
    """Test no warning when unless player went idle with reason "ERROR"."""
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, _, media_status_cb = get_status_callbacks(chromecast)
    media_status = MagicMock(images=None)
    media_status.player_is_idle = False
    media_status.idle_reason = "ERROR"
    media_status.content_id = "http://example.com:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media" not in caplog.text
    media_status = MagicMock(images=None)
    media_status.player_is_idle = True
    media_status.idle_reason = "Other"
    media_status.content_id = "http://example.com:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media" not in caplog.text
    media_status = MagicMock(images=None)
    media_status.player_is_idle = True
    media_status.idle_reason = "ERROR"
    media_status.content_id = "http://example.com:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media http://example.com:8123/tts.mp3." in caplog.text


async def test_failed_cast_other_url(hass: HomeAssistant, caplog: Any) -> None:
    """Test warning when casting from internal_url fails."""
    await async_setup_component(hass, "homeassistant", {})
    with assert_setup_component(1, tts.DOMAIN):
        assert await async_setup_component(hass, tts.DOMAIN, {tts.DOMAIN: {"platform": "demo"}})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, _, media_status_cb = get_status_callbacks(chromecast)
    media_status = MagicMock(images=None)
    media_status.player_is_idle = True
    media_status.idle_reason = "ERROR"
    media_status.content_id = "http://example.com:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media http://example.com:8123/tts.mp3." in caplog.text


async def test_failed_cast_internal_url(hass: HomeAssistant, caplog: Any) -> None:
    """Test warning when casting from internal_url fails."""
    await async_setup_component(hass, "homeassistant", {})
    await async_process_ha_core_config(hass, {"internal_url": "http://example.local:8123"})
    with assert_setup_component(1, tts.DOMAIN):
        assert await async_setup_component(hass, tts.DOMAIN, {tts.DOMAIN: {"platform": "demo"}})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, _, media_status_cb = get_status_callbacks(chromecast)
    media_status = MagicMock(images=None)
    media_status.player_is_idle = True
    media_status.idle_reason = "ERROR"
    media_status.content_id = "http://example.local:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media http://example.local:8123/tts.mp3 from internal_url" in caplog.text


async def test_failed_cast_external_url(hass: HomeAssistant, caplog: Any) -> None:
    """Test warning when casting from external_url fails."""
    await async_setup_component(hass, "homeassistant", {})
    await async_process_ha_core_config(hass, {"external_url": "http://example.com:8123"})
    with assert_setup_component(1, tts.DOMAIN):
        assert await async_setup_component(hass, tts.DOMAIN, {tts.DOMAIN: {"platform": "demo"}})
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, _, media_status_cb = get_status_callbacks(chromecast)
    media_status = MagicMock(images=None)
    media_status.player_is_idle = True
    media_status.idle_reason = "ERROR"
    media_status.content_id = "http://example.com:8123/tts.mp3"
    media_status_cb(media_status)
    assert "Failed to cast media http://example.com:8123/tts.mp3 from external_url" in caplog.text


async def test_disconnect_on_stop(hass: HomeAssistant) -> None:
    """Test cast device disconnects socket on stop."""
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    assert chromecast.disconnect.call_count == 1


async def test_entry_setup_no_config(hass: HomeAssistant) -> None:
    """Test deprecated empty yaml config.."""
    await async_setup_component(hass, "cast", {})
    await hass.async_block_till_done()
    assert not hass.config_entries.async_entries("cast")


@pytest.mark.no_fail_on_log_exception
async def test_invalid_cast_platform(hass: HomeAssistant, caplog: Any) -> None:
    """Test we can play media through a cast platform."""
    cast_platform_mock = Mock()
    del cast_platform_mock.async_get_media_browser_root_object
    del cast_platform_mock.async_browse_media
    del cast_platform_mock.async_play_media
    mock_platform(hass, "test.cast", cast_platform_mock)
    await async_setup_component(hass, "test", {"test": {}})
    await hass.async_block_till_done()
    info = get_fake_chromecast_info()
    await async_setup_media_player_cast(hass, info)
    assert "Invalid cast platform <Mock id" in caplog.text


async def test_cast_platform_play_media(hass: HomeAssistant, quick_play_mock: MagicMock, caplog: Any) -> None:
    """Test we can play media through a cast platform."""
    entity_id = "media_player.speaker"
    _can_play = True

    def can_play(*args: Any, **kwargs: Any) -> bool:
        return _can_play

    cast_platform_mock = Mock(
        async_get_media_browser_root_object=AsyncMock(return_value=[]),
        async_browse_media=AsyncMock(return_value=None),
        async_play_media=AsyncMock(side_effect=can_play),
    )
    mock_platform(hass, "test.cast", cast_platform_mock)
    await async_setup_component(hass, "test", {"test": {}})
    await hass.async_block_till_done()
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    assert "Invalid cast platform <Mock id" not in caplog.text
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "audio",
            media_player.ATTR_MEDIA_CONTENT_ID: "best.mp3",
            media_player.ATTR_MEDIA_EXTRA: {"metadata": {"metadatatype": 3}},
        },
        blocking=True,
    )
    cast_platform_mock.async_play_media.assert_called_once_with(hass, entity_id, chromecast, "audio", "best.mp3")
    chromecast.media_controller.play_media.assert_not_called()
    quick_play_mock.assert_not_called()
    _can_play = False
    cast_platform_mock.async_play_media.reset_mock()
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "audio",
            media_player.ATTR_MEDIA_CONTENT_ID: "http://example.com/best.mp3",
            media_player.ATTR_MEDIA_EXTRA: {"metadata": {"metadatatype": 3}},
        },
        blocking=True,
    )
    cast_platform_mock.async_play_media.assert_called_once_with(hass, entity_id, chromecast, "audio", "http://example.com/best.mp3")
    chromecast.media_controller.play_media.assert_not_called()
    quick_play_mock.assert_called()


async def test_cast_platform_browse_media(hass: HomeAssistant, hass_ws_client: Any) -> None:
    """Test we can play media through a cast platform."""
    cast_platform_mock = Mock(
        async_get_media_browser_root_object=AsyncMock(
            return_value=[
                BrowseMedia(
                    title="Spotify",
                    media_class=MediaClass.APP,
                    media_content_id="",
                    media_content_type="spotify",
                    thumbnail="https://brands.home-assistant.io/_/spotify/logo.png",
                    can_play=False,
                    can_expand=True,
                )
            ]
        ),
        async_browse_media=AsyncMock(
            return_value=BrowseMedia(
                title="Spotify Favourites",
                media_class=MediaClass.PLAYLIST,
                media_content_id="",
                media_content_type="spotify",
                can_play=True,
                can_expand=False,
            )
        ),
        async_play_media=AsyncMock(return_value=False),
    )
    mock_platform(hass, "test.cast", cast_platform_mock)
    await async_setup_component(hass, "test", {"test": {}})
    await async_setup_component(hass, "media_source", {"media_source": {}})
    await hass.async_block_till_done()
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    client = await hass_ws_client()
    await client.send_json({"id": 1, "type": "media_player/browse_media", "entity_id": "media_player.speaker"})
    response = await client.receive_json()
    assert response["success"]
    expected_child = {
        "title": "Spotify",
        "media_class": "app",
        "media_content_type": "spotify",
        "media_content_id": "",
        "can_play": False,
        "can_expand": True,
        "thumbnail": "https://brands.home-assistant.io/_/spotify/logo.png",
        "children_media_class": None,
    }
    assert expected_child in response["result"]["children"]
    client = await hass_ws_client()
    await client.send_json({"id": 2, "type": "media_player/browse_media", "entity_id": "media_player.speaker", "media_content_id": "", "media_content_type": "spotify"})
    response = await client.receive_json()
    assert response["success"]
    expected_response = {
        "title": "Spotify Favourites",
        "media_class": "playlist",
        "media_content_type": "spotify",
        "media_content_id": "",
        "can_play": True,
        "can_expand": False,
        "children_media_class": None,
        "thumbnail": None,
        "children": [],
        "not_shown": 0,
    }
    assert response["result"] == expected_response


async def test_cast_platform_play_media_local_media(hass: HomeAssistant, quick_play_mock: MagicMock, caplog: Any) -> None:
    """Test we process data when playing local media."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    _, conn_status_cb, _ = get_status_callbacks(chromecast)
    connection_status = MagicMock()
    connection_status.status = "CONNECTED"
    conn_status_cb(connection_status)
    await hass.async_block_till_done()
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "application/vnd.apple.mpegurl",
            media_player.ATTR_MEDIA_CONTENT_ID: "/api/hls/bla/master_playlist.m3u8",
        },
        blocking=True,
    )
    await hass.async_block_till_done()
    quick_play_mock.assert_called()
    app_data = quick_play_mock.call_args[0][2]
    assert not app_data["media_id"].startswith("/")
    assert "authSig" in yarl.URL(app_data["media_id"]).query
    assert app_data["media_type"] == "application/vnd.apple.mpegurl"
    assert app_data["stream_type"] == "LIVE"
    assert app_data["media_info"] == {"hlsVideoSegmentFormat": "fmp4"}
    quick_play_mock.reset_mock()
    await hass.services.async_call(
        media_player.DOMAIN,
        media_player.SERVICE_PLAY_MEDIA,
        {
            ATTR_ENTITY_ID: entity_id,
            media_player.ATTR_MEDIA_CONTENT_TYPE: "application/vnd.apple.mpegurl",
            media_player.ATTR_MEDIA_CONTENT_ID: f"{network.get_url(hass)}/api/hls/bla/master_playlist.m3u8?token=bla",
        },
        blocking=True,
    )
    await hass.async_block_till_done()
    quick_play_mock.assert_called()
    app_data = quick_play_mock.call_args[0][2]
    assert app_data["media_id"] == f"{network.get_url(hass)}/api/hls/bla/master_playlist.m3u8?token=bla"


async def test_ha_cast(hass: HomeAssistant, ha_controller_mock: MagicMock) -> None:
    """Test Home Assistant cast."""
    entity_id = "media_player.speaker"
    info = get_fake_chromecast_info()
    chromecast, _ = await async_setup_media_player_cast(hass, info)
    chromecast.cast_type = pychromecast.const.CAST_TYPE_CHROMECAST
    ha_controller = MagicMock()
    ha_controller_mock.return_value = ha_controller
    controller_data = HomeAssistantControllerData(hass_url="url", hass_uuid="12341234", client_id="client_id_1234", refresh_token="refresh_token_1234")
    async_dispatcher_send(hass, SIGNAL_HASS_CAST_SHOW_VIEW, controller_data, "media_player.other", "view_path", "url_path")
    await hass.async_block_till_done()
    ha_controller_mock.assert_not_called()
    controller_data = HomeAssistantControllerData(hass_url="url", hass_uuid="12341234", client_id="client_id_1234", refresh_token="refresh_token_1234")
    async_dispatcher_send(hass, SIGNAL_HASS_CAST_SHOW_VIEW, controller_data, entity_id, "view_path", "url_path")
    await hass.async_block_till_done()
    ha_controller_mock.assert_called_once_with(client_id="client_id_1234", hass_url="url", hass_uuid="12341234", refresh_token="refresh_token_1234", unregister=ANY)
    ha_controller.show_lovelace_view.assert_called_once_with("view_path", "url_path")
    chromecast.unregister_handler.assert_not_called()
    unregister_cb = ha_controller_mock.mock_calls[0][2]["unregister"]
    unregister_cb()
    chromecast.unregister_handler.assert_called_once_with(ha_controller)
    chromecast.unregister_handler.reset_mock()
    unregister_cb()
    chromecast.unregister_handler.assert_not_called()