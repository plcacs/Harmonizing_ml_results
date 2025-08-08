from __future__ import annotations
import logging
from rfk101py.rfk101py import rfk101py
import voluptuous as vol
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType
from typing import Any, Dict, List, Union

_LOGGER: logging.Logger
DOMAIN: str
EVENT_IDTECK_PROX_KEYCARD: str
CONFIG_SCHEMA: vol.Schema

def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    conf: List[Dict[str, Union[str, int]]]
    unit: Dict[str, Union[str, int]]
    host: str
    port: int
    name: str
    reader: IdteckReader
    error: OSError
    return_value: bool

class IdteckReader:
    def __init__(self, hass: HomeAssistant, host: str, port: int, name: str) -> None:
        self.hass: HomeAssistant
        self._host: str
        self._port: int
        self._name: str
        self._connection: Any

    def connect(self) -> None:
        pass

    def _callback(self, card: str) -> None:
        pass

    def stop(self, _: Event) -> None:
        pass
