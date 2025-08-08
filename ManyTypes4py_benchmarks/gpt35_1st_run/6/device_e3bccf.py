from __future__ import annotations
import logging
from typing import Any, Dict
from yeelight.aio import AsyncBulb
from homeassistant.core import HomeAssistant, callback
from .const import ACTIVE_COLOR_FLOWING, ACTIVE_MODE_NIGHTLIGHT, DATA_UPDATED, STATE_CHANGE_TIME, UPDATE_REQUEST_PROPERTIES

_LOGGER: logging.Logger

@callback
def async_format_model(model: str) -> str:
    return model.replace('_', ' ').title()

@callback
def async_format_id(id_: str) -> str:
    return hex(int(id_, 16)) if id_ else 'None'

@callback
def async_format_model_id(model: str, id_: str) -> str:
    return f'{async_format_model(model)} {async_format_id(id_)}'

@callback
def _async_unique_name(capabilities: Dict[str, str]) -> str:
    model_id = async_format_model_id(capabilities['model'], capabilities['id'])
    return f'Yeelight {model_id}'

def update_needs_bg_power_workaround(data: Dict[str, Any]) -> bool:
    return 'bg_power' in data

class YeelightDevice:
    def __init__(self, hass: HomeAssistant, host: str, config: Dict[str, Any], bulb: AsyncBulb) -> None:
        self._hass: HomeAssistant
        self._config: Dict[str, Any]
        self._host: str
        self._bulb_device: AsyncBulb
        self.capabilities: Dict[str, Any]
        self._device_type: Any
        self._available: bool
        self._initialized: bool
        self._name: str

    @property
    def bulb(self) -> AsyncBulb:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        ...

    @property
    def host(self) -> str:
        ...

    @property
    def available(self) -> bool:
        ...

    @callback
    def async_mark_unavailable(self) -> None:
        ...

    @property
    def model(self) -> str:
        ...

    @property
    def fw_version(self) -> str:
        ...

    @property
    def unique_id(self) -> str:
        ...

    @property
    def is_nightlight_supported(self) -> bool:
        ...

    @property
    def is_nightlight_enabled(self) -> bool:
        ...

    @property
    def is_color_flow_enabled(self) -> bool:
        ...

    @property
    def _active_mode(self) -> Any:
        ...

    @property
    def _color_flow(self) -> Any:
        ...

    @property
    def _nightlight_brightness(self) -> Any:
        ...

    @property
    def type(self) -> Any:
        ...

    async def _async_update_properties(self) -> None:
        ...

    async def async_setup(self) -> None:
        ...

    async def async_update(self, force: bool = False) -> None:
        ...

    async def _async_forced_update(self, _now: Any) -> None:
        ...

    @callback
    def async_update_callback(self, data: Dict[str, Any]) -> None:
        ...
