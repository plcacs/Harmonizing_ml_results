from __future__ import annotations
import logging
from typing import Any, Optional
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
)
from homeassistant.const import CONF_MONITORED_CONDITIONS, CONF_NAME, UnitOfInformation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import ATTR_CURRENT_BANDWIDTH_USED, ATTR_PENDING_CHARGES, CONF_SUBSCRIPTION, DATA_VULTR

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = 'Vultr {} {}'
SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(
        key=ATTR_CURRENT_BANDWIDTH_USED,
        name='Current Bandwidth Used',
        native_unit_of_measurement=UnitOfInformation.GIGABYTES,
        device_class=SensorDeviceClass.DATA_SIZE,
        icon='mdi:chart-histogram',
    ),
    SensorEntityDescription(
        key=ATTR_PENDING_CHARGES,
        name='Pending Charges',
        native_unit_of_measurement='US$',
        icon='mdi:currency-usd',
    )
)

SENSOR_KEYS: list[str] = [desc.key for desc in SENSOR_TYPES]

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_SUBSCRIPTION): cv.string,
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_MONITORED_CONDITIONS, default=SENSOR_KEYS): vol.All(cv.ensure_list, [vol.In(SENSOR_KEYS)])
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    vultr: Any = hass.data[DATA_VULTR]
    subscription: str = config[CONF_SUBSCRIPTION]
    name: str = config[CONF_NAME]
    monitored_conditions: list[str] = config[CONF_MONITORED_CONDITIONS]
    if subscription not in vultr.data:
        _LOGGER.error('Subscription %s not found', subscription)
        return
    entities = [
        VultrSensor(vultr, subscription, name, description)
        for description in SENSOR_TYPES if description.key in monitored_conditions
    ]
    add_entities(entities, True)

class VultrSensor(SensorEntity):
    def __init__(self, vultr: Any, subscription: str, name: str, description: SensorEntityDescription) -> None:
        self.entity_description: SensorEntityDescription = description
        self._vultr: Any = vultr
        self._name: str = name
        self.subscription: str = subscription
        self.data: Any = None

    @property
    def name(self) -> str:
        try:
            return self._name.format(self.entity_description.name)
        except IndexError:
            try:
                return self._name.format(self.data['label'], self.entity_description.name)
            except (KeyError, TypeError):
                return self._name

    @property
    def native_value(self) -> Any:
        try:
            return round(float(self.data.get(self.entity_description.key)), 2)
        except (TypeError, ValueError):
            return self.data.get(self.entity_description.key)

    def update(self) -> None:
        self._vultr.update()
        self.data = self._vultr.data[self.subscription]