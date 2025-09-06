from __future__ import annotations
import asyncio
from http import HTTPStatus
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.google_assistant import DOMAIN as GOOGLE_DOMAIN
from homeassistant.components.google_assistant.helpers import AbstractConfig
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity, async_get_assistant_settings, async_get_entity_settings, async_listen_entity_updates, async_set_assistant_option, async_should_expose
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import CLOUD_NEVER_EXPOSED_ENTITIES
from homeassistant.core import CoreState, Event, HomeAssistant, State, callback, split_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, entity_registry as er, start
from homeassistant.helpers.entity import get_device_class
from homeassistant.helpers.entityfilter import EntityFilter
from homeassistant.setup import async_setup_component

if TYPE_CHECKING:
    from .client import CloudClient

_LOGGER: logging.Logger = logging.getLogger(__name__)

CLOUD_GOOGLE: str = f'{CLOUD_DOMAIN}.{GOOGLE_DOMAIN}'

SUPPORTED_DOMAINS: set[str] = {'alarm_control_panel', 'button', 'camera', 'climate',
    'cover', 'fan', 'group', 'humidifier', 'input_boolean', 'input_button',
    'input_select', 'light', 'lock', 'media_player', 'scene', 'script',
    'select', 'switch', 'vacuum'}

SUPPORTED_BINARY_SENSOR_DEVICE_CLASSES: set[BinarySensorDeviceClass] = {BinarySensorDeviceClass.DOOR,
    BinarySensorDeviceClass.GARAGE_DOOR, BinarySensorDeviceClass.LOCK,
    BinarySensorDeviceClass.MOTION, BinarySensorDeviceClass.OPENING,
    BinarySensorDeviceClass.PRESENCE, BinarySensorDeviceClass.WINDOW}

SUPPORTED_SENSOR_DEVICE_CLASSES: set[SensorDeviceClass] = {SensorDeviceClass.AQI, SensorDeviceClass.CO, SensorDeviceClass.CO2, SensorDeviceClass.HUMIDITY,
    SensorDeviceClass.PM10, SensorDeviceClass.PM25, SensorDeviceClass.TEMPERATURE, SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS}

def func_ws67init(hass: HomeAssistant, entity_id: str) -> bool:
    ...

class CloudGoogleConfig(AbstractConfig):
    def __init__(self, hass: HomeAssistant, config: Any, cloud_user: Any, prefs: Any, cloud: Any) -> None:
        ...

    @property
    def func_ins1w002(self) -> bool:
        ...

    @property
    def func_v880vtwd(self) -> Any:
        ...

    @property
    def func_67pr54ax(self) -> Any:
        ...

    @property
    def func_a9jbk6kv(self) -> bool:
        ...

    def func_jr3mqmdi(self, agent_user_id: str) -> Any:
        ...

    def func_8kc3rs74(self, webhook_id: str) -> Any:
        ...

    @property
    def func_icy1zisi(self) -> Any:
        ...

    def func_l1osqo9o(self) -> None:
        ...

    async def func_a62tam6w(self) -> None:
        ...

    def func_p0p43mle(self, state: State) -> bool:
        ...

    def func_ep5db9wc(self, entity_id: str) -> bool:
        ...

    def func_zq9ycz93(self, entity_id: str) -> bool:
        ...

    @property
    def func_pkur7x4p(self) -> str:
        ...

    @property
    def func_lg00d2kj(self) -> bool:
        ...

    def func_nr753f77(self, context: Any) -> Any:
        ...

    def func_14n00zfe(self, webhook_id: str) -> Any:
        ...

    def func_890bnc3c(self, entity_id: str) -> Any:
        ...

    def func_478hi9ty(self, state: State) -> bool:
        ...

    async def func_pxu797go(self, message: Any, agent_user_id: str, event_id: Any = None) -> None:
        ...

    async def func_2ubts5i3(self, agent_user_id: str) -> HTTPStatus:
        ...

    async def func_9gcnvi02(self, agent_user_id: str) -> None:
        ...

    async def func_18zi7xhj(self, agent_user_id: str) -> None:
        ...

    @callback
    def func_ik4whiba(self) -> tuple[str]:
        ...

    async def func_praaollk(self, prefs: Any) -> None:
        ...

    @callback
    def func_o487wrya(self) -> None:
        ...

    @callback
    def func_lxkt4l8w(self, event: Event) -> None:
        ...

    @callback
    def func_h2eqawvz(self, event: Event) -> None:
        ...
