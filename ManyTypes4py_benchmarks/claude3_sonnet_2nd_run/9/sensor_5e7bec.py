"""Support for monitoring the state of Vultr Subscriptions."""
from __future__ import annotations
import logging
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity, SensorEntityDescription
from homeassistant.const import CONF_MONITORED_CONDITIONS, CONF_NAME, UnitOfInformation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from . import ATTR_CURRENT_BANDWIDTH_USED, ATTR_PENDING_CHARGES, CONF_SUBSCRIPTION, DATA_VULTR
_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME = 'Vultr {} {}'
SENSOR_TYPES = (SensorEntityDescription(key=ATTR_CURRENT_BANDWIDTH_USED, name='Current Bandwidth Used', native_unit_of_measurement=UnitOfInformation.GIGABYTES, device_class=SensorDeviceClass.DATA_SIZE, icon='mdi:chart-histogram'), SensorEntityDescription(key=ATTR_PENDING_CHARGES, name='Pending Charges', native_unit_of_measurement='US$', icon='mdi:currency-usd'))
SENSOR_KEYS: list[str] = [desc.key for desc in SENSOR_TYPES]
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({vol.Required(CONF_SUBSCRIPTION): cv.string, vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string, vol.Optional(CONF_MONITORED_CONDITIONS, default=SENSOR_KEYS): vol.All(cv.ensure_list, [vol.In(SENSOR_KEYS)])})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Vultr subscription (server) sensor."""
    vultr = hass.data[DATA_VULTR]
    subscription: str = config[CONF_SUBSCRIPTION]
    name: str = config[CONF_NAME]
    monitored_conditions: list[str] = config[CONF_MONITORED_CONDITIONS]
    if subscription not in vultr.data:
        _LOGGER.error('Subscription %s not found', subscription)
        return
    entities = [VultrSensor(vultr, subscription, name, description) for description in SENSOR_TYPES if description.key in monitored_conditions]
    add_entities(entities, True)

class VultrSensor(SensorEntity):
    """Representation of a Vultr subscription sensor."""

    def __init__(
        self, 
        vultr: any, 
        subscription: str, 
        name: str, 
        description: SensorEntityDescription
    ) -> None:
        """Initialize a new Vultr sensor."""
        self.entity_description = description
        self._vultr = vultr
        self._name = name
        self.subscription = subscription
        self.data: dict | None = None

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        try:
            return self._name.format(self.entity_description.name)
        except IndexError:
            try:
                return self._name.format(self.data['label'], self.entity_description.name)
            except (KeyError, TypeError):
                return self._name

    @property
    def native_value(self) -> float | str | None:
        """Return the value of this given sensor type."""
        try:
            return round(float(self.data.get(self.entity_description.key)), 2)
        except (TypeError, ValueError):
            return self.data.get(self.entity_description.key)

    def update(self) -> None:
        """Update state of sensor."""
        self._vultr.update()
        self.data = self._vultr.data[self.subscription]
