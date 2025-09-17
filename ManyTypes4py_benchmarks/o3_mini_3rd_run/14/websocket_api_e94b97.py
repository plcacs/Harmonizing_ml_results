from __future__ import annotations
from collections.abc import Callable, Iterable
from functools import lru_cache, partial
import time
from typing import Any, Dict, List
from habluetooth import (
    BluetoothScanningMode,
    HaBluetoothSlotAllocations,
    HaScannerRegistration,
    HaScannerRegistrationEvent,
)
from home_assistant_bluetooth import BluetoothServiceInfoBleak
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.json import json_bytes
from .api import _get_manager, async_register_callback
from .const import DOMAIN
from .match import BluetoothCallbackMatcher
from .models import BluetoothChange


@callback
def async_setup(hass: HomeAssistant) -> None:
    websocket_api.async_register_command(hass, ws_subscribe_advertisements)
    websocket_api.async_register_command(hass, ws_subscribe_connection_allocations)
    websocket_api.async_register_command(hass, ws_subscribe_scanner_details)


@lru_cache(maxsize=1024)
def serialize_service_info(
    service_info: BluetoothServiceInfoBleak, time_diff: float
) -> Dict[str, Any]:
    return {
        "name": service_info.name,
        "address": service_info.address,
        "rssi": service_info.rssi,
        "manufacturer_data": {
            str(manufacturer_id): manufacturer_data.hex()
            for manufacturer_id, manufacturer_data in service_info.manufacturer_data.items()
        },
        "service_data": {
            service_uuid: service_data.hex()
            for service_uuid, service_data in service_info.service_data.items()
        },
        "service_uuids": service_info.service_uuids,
        "source": service_info.source,
        "connectable": service_info.connectable,
        "time": service_info.time + time_diff,
        "tx_power": service_info.tx_power,
    }


class _AdvertisementSubscription:
    def __init__(
        self,
        hass: HomeAssistant,
        connection: Any,
        ws_msg_id: int,
        match_dict: BluetoothCallbackMatcher,
    ) -> None:
        self.hass: HomeAssistant = hass
        self.match_dict: BluetoothCallbackMatcher = match_dict
        self.pending_service_infos: List[BluetoothServiceInfoBleak] = []
        self.ws_msg_id: int = ws_msg_id
        self.connection: Any = connection
        self.pending: bool = True
        self.time_diff: float = round(time.time() - time.monotonic(), 2)

    @callback
    def _async_unsubscribe(self, cancel_callbacks: Iterable[Callable[[], None]]) -> None:
        for cancel_callback in cancel_callbacks:
            cancel_callback()

    @callback
    def async_start(self) -> None:
        connection: Any = self.connection
        cancel_adv_callback = async_register_callback(
            self.hass, self._async_on_advertisement, self.match_dict, BluetoothScanningMode.PASSIVE
        )
        cancel_disappeared_callback = _get_manager(self.hass).async_register_disappeared_callback(
            self._async_removed
        )
        connection.subscriptions[self.ws_msg_id] = partial(
            self._async_unsubscribe, (cancel_adv_callback, cancel_disappeared_callback)
        )
        self.pending = False
        connection.send_message(json_bytes(websocket_api.result_message(self.ws_msg_id)))
        self._async_added(self.pending_service_infos)
        self.pending_service_infos.clear()

    def _async_event_message(self, message: Dict[str, Any]) -> None:
        self.connection.send_message(json_bytes(websocket_api.event_message(self.ws_msg_id, message)))

    def _async_added(self, service_infos: Iterable[BluetoothServiceInfoBleak]) -> None:
        self._async_event_message(
            {"add": [serialize_service_info(service_info, self.time_diff) for service_info in service_infos]}
        )

    def _async_removed(self, address: str) -> None:
        self._async_event_message({"remove": [{"address": address}]})

    @callback
    def _async_on_advertisement(
        self, service_info: BluetoothServiceInfoBleak, change: BluetoothChange
    ) -> None:
        if self.pending:
            self.pending_service_infos.append(service_info)
            return
        self._async_added((service_info,))


@websocket_api.require_admin
@websocket_api.websocket_command(
    {vol.Required("type"): "bluetooth/subscribe_advertisements"}
)
@websocket_api.async_response
async def ws_subscribe_advertisements(hass: HomeAssistant, connection: Any, msg: Dict[str, Any]) -> None:
    _AdvertisementSubscription(
        hass, connection, msg["id"], BluetoothCallbackMatcher(connectable=False)
    ).async_start()


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "bluetooth/subscribe_connection_allocations",
        vol.Optional("config_entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_subscribe_connection_allocations(
    hass: HomeAssistant, connection: Any, msg: Dict[str, Any]
) -> None:
    ws_msg_id: int = msg["id"]
    source: Any = None
    if (config_entry_id := msg.get("config_entry_id")):
        try:
            from .util import config_entry_id_to_source
            source = config_entry_id_to_source(hass, config_entry_id)
        except Exception as err:  # Covers InvalidConfigEntryID and InvalidSource
            connection.send_error(ws_msg_id, "invalid_config_entry_id", str(err))
            return

    def _async_allocations_changed(allocations: HaBluetoothSlotAllocations) -> None:
        connection.send_message(json_bytes(websocket_api.event_message(ws_msg_id, [allocations])))

    manager: Any = _get_manager(hass)
    connection.subscriptions[ws_msg_id] = manager.async_register_allocation_callback(
        _async_allocations_changed, source
    )
    connection.send_message(json_bytes(websocket_api.result_message(ws_msg_id)))
    if (current_allocations := manager.async_current_allocations(source)):
        connection.send_message(json_bytes(websocket_api.event_message(ws_msg_id, current_allocations)))


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "bluetooth/subscribe_scanner_details",
        vol.Optional("config_entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_subscribe_scanner_details(
    hass: HomeAssistant, connection: Any, msg: Dict[str, Any]
) -> None:
    ws_msg_id: int = msg["id"]
    source: Any = None
    if (config_entry_id := msg.get("config_entry_id")):
        entry = hass.config_entries.async_get_entry(config_entry_id)
        if not entry or entry.domain != DOMAIN:
            connection.send_error(ws_msg_id, "invalid_config_entry_id", f"Invalid config entry id: {config_entry_id}")
            return
        source = entry.unique_id
        assert source is not None

    def _async_event_message(message: Dict[str, Any]) -> None:
        connection.send_message(json_bytes(websocket_api.event_message(ws_msg_id, message)))

    def _async_registration_changed(registration: HaScannerRegistration) -> None:
        added_event: HaScannerRegistrationEvent = HaScannerRegistrationEvent.ADDED
        event_type: str = "add" if registration.event == added_event else "remove"
        _async_event_message({event_type: [registration.scanner.details]})

    manager: Any = _get_manager(hass)
    connection.subscriptions[ws_msg_id] = manager.async_register_scanner_registration_callback(
        _async_registration_changed, source
    )
    connection.send_message(json_bytes(websocket_api.result_message(ws_msg_id)))
    if (scanners := manager.async_current_scanners()):
        matching_scanners = [
            scanner.details for scanner in scanners if source is None or scanner.source == source
        ]
        if matching_scanners:
            _async_event_message({"add": matching_scanners})