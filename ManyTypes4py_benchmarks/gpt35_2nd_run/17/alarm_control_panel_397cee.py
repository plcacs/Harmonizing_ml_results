from __future__ import annotations
from typing import Optional, List
from datetime import timedelta
import logging
import requests
import voluptuous as vol
from homeassistant.components.alarm_control_panel import PLATFORM_SCHEMA as ALARM_CONTROL_PANEL_PLATFORM_SCHEMA, AlarmControlPanelEntity, AlarmControlPanelEntityFeature, AlarmControlPanelState, CodeFormat
from homeassistant.const import CONF_CODE, CONF_HOST, CONF_MODE, CONF_NAME, CONF_PORT
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger
DEFAULT_HOST: str
DEFAULT_NAME: str
DEFAULT_PORT: int
DEFAULT_MODE: str
SCAN_INTERVAL: timedelta
PLATFORM_SCHEMA: vol.Schema

def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

class Concord232Alarm(AlarmControlPanelEntity):
    _attr_code_format: CodeFormat
    _attr_supported_features: int

    def __init__(self, url: str, name: str, code: Optional[str], mode: str) -> None:

    def update(self) -> None:

    def alarm_disarm(self, code: Optional[str] = None) -> None:

    def alarm_arm_home(self, code: Optional[str] = None) -> None:

    def alarm_arm_away(self, code: Optional[str] = None) -> None:

    def _validate_code(self, code: Optional[str], state: AlarmControlPanelState) -> bool:
