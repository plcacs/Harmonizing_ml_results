from __future__ import annotations
from http import HTTPStatus
import logging
from typing import Any, List, Dict, Optional
from rachiopy import Rachio
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from .const import DOMAIN, KEY_BASE_STATIONS, KEY_DEVICES, KEY_ENABLED, KEY_EXTERNAL_ID, KEY_FLEX_SCHEDULES, KEY_ID, KEY_MAC_ADDRESS, KEY_MODEL, KEY_NAME, KEY_SCHEDULES, KEY_SERIAL_NUMBER, KEY_STATUS, KEY_USERNAME, KEY_ZONES, LISTEN_EVENT_TYPES, MODEL_GENERATION_1, SERVICE_PAUSE_WATERING, SERVICE_RESUME_WATERING, SERVICE_STOP_WATERING, WEBHOOK_CONST_ID
from .coordinator import RachioScheduleUpdateCoordinator, RachioUpdateCoordinator

_LOGGER = logging.getLogger(__name__)

ATTR_DEVICES: str = 'devices'
ATTR_DURATION: str = 'duration'
PERMISSION_ERROR: str = '7'

PAUSE_SERVICE_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string,
    vol.Optional(ATTR_DURATION, default=60): cv.positive_int
})
RESUME_SERVICE_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string
})
STOP_SERVICE_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional(ATTR_DEVICES): cv.string
})

class RachioPerson:
    def __init__(self, rachio: Rachio, config_entry: ConfigEntry) -> None:
        self.rachio: Rachio = rachio
        self.config_entry: ConfigEntry = config_entry
        self.username: Optional[str] = None
        self._id: Optional[str] = None
        self._controllers: List[RachioIro] = []
        self._base_stations: List[RachioBaseStation] = []

    async def func_2iigis5k(self, hass: HomeAssistant) -> None:
        ...

    def func_4n8wimym(self, hass: HomeAssistant) -> None:
        ...

    @property
    def func_024j5sr1(self) -> Optional[str]:
        ...

    @property
    def func_6v7fksuq(self) -> List[RachioIro]:
        ...

    @property
    def func_xfiv9s04(self) -> List[RachioBaseStation]:
        ...

    def func_7ahhg18x(self, zones: List[str]) -> None:
        ...

class RachioIro:
    def __init__(self, hass: HomeAssistant, rachio: Rachio, data: Dict[str, Any], webhooks: Dict[str, Any]) -> None:
        ...

    def func_7a2qvrag(self) -> None:
        ...

    def func_3k8gxg69(self) -> None:
        ...

    @property
    def func_5ph3ulfe(self) -> str:
        ...

    @property
    def func_zm8577a5(self) -> Dict[str, Any]:
        ...

    @property
    def func_7eev4nr5(self) -> Dict[str, Any]:
        ...

    def func_mh6i6lwm(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        ...

    def func_j9bmf1cr(self, zone_id: str) -> Optional[Dict[str, Any]]:
        ...

    def func_wo2a7i03(self) -> List[Dict[str, Any]]:
        ...

    def func_wbctlhws(self) -> List[Dict[str, Any]]:
        ...

    def func_ck7fqjqc(self) -> None:
        ...

    def func_r7a1mvo3(self, duration: int) -> None:
        ...

    def func_688eaxo3(self) -> None:
        ...

class RachioBaseStation:
    def __init__(self, rachio: Rachio, data: Dict[str, Any], status_coordinator: RachioUpdateCoordinator, schedule_coordinator: RachioScheduleUpdateCoordinator) -> None:
        ...

    def func_kk70570v(self, valve_id: str, duration: int) -> None:
        ...

    def func_ck7fqjqc(self, valve_id: str) -> None:
        ...

    def func_b9kwbmf4(self, program_id: str, timestamp: int) -> None:
        ...

def func_tj6lczix(http_status_code: int) -> bool:
    ...
