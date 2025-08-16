from __future__ import annotations
import logging
from typing import Any, cast
import voluptuous as vol
from homeassistant.components.sensor import CONF_STATE_CLASS, SensorDeviceClass
from homeassistant.components.sensor.helpers import async_parse_date_datetime
from homeassistant.const import CONF_ATTRIBUTE, CONF_DEVICE_CLASS, CONF_ICON, CONF_NAME, CONF_UNIQUE_ID, CONF_UNIT_OF_MEASUREMENT, CONF_VALUE_TEMPLATE
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import PlatformNotReady
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback, AddEntitiesCallback
from homeassistant.helpers.template import Template
from homeassistant.helpers.trigger_template_entity import CONF_AVAILABILITY, CONF_PICTURE, TEMPLATE_SENSOR_BASE_SCHEMA, ManualTriggerEntity, ManualTriggerSensorEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from . import ScrapeConfigEntry
from .const import CONF_INDEX, CONF_SELECT, DOMAIN
from .coordinator import ScrapeCoordinator

_LOGGER: logging.Logger

TRIGGER_ENTITY_OPTIONS: tuple[str, ...]

async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddEntitiesCallback) -> None:

class ScrapeSensor(CoordinatorEntity[ScrapeCoordinator], ManualTriggerSensorEntity):

    def __init__(self, hass: HomeAssistant, coordinator: ScrapeCoordinator, trigger_entity_config: dict[str, Any], select: str, attr: str, index: int, value_template: Template, yaml: bool) -> None:

    def _extract_value(self) -> Any:

    async def async_added_to_hass(self) -> None:

    def _async_update_from_rest_data(self) -> None:

    @property
    def available(self) -> bool:

    @callback
    def _handle_coordinator_update(self) -> None:
