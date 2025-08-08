from datetime import timedelta
import logging
from typing import Any
from homeassistant.const import ATTR_BATTERY_LEVEL, ATTR_VOLTAGE, CONF_MAC
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import DeviceInfo, format_mac
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import async_track_point_in_utc_time
from homeassistant.util.dt import utcnow
from .const import DOMAIN

_LOGGER: logging.Logger = logging.getLogger(__name__)
TIME_TILL_UNAVAILABLE: timedelta = timedelta(minutes=150)

class XiaomiDevice(Entity):
    _attr_should_poll: bool = False

    def __init__(self, device: dict, device_type: str, xiaomi_hub: Any, config_entry: Any) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def device_id(self) -> str:
        ...

    @property
    def device_info(self) -> DeviceInfo:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

    @callback
    def _async_set_unavailable(self, now: Any) -> None:
        ...

    @callback
    def _async_track_unavailable(self) -> bool:
        ...

    def push_data(self, data: dict, raw_data: dict) -> None:
        ...

    @callback
    def async_push_data(self, data: dict, raw_data: dict) -> None:
        ...

    def parse_voltage(self, data: dict) -> bool:
        ...

    def parse_data(self, data: dict, raw_data: dict) -> bool:
        ...
