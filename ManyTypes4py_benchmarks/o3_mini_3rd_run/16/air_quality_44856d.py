from collections.abc import Callable
import logging
from typing import Any, Dict, Optional, List
from miio import AirQualityMonitor, AirQualityMonitorCGDN1, DeviceException
from homeassistant.components.air_quality import AirQualityEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_DEVICE, CONF_HOST, CONF_MODEL, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import CONF_FLOW_TYPE, MODEL_AIRQUALITYMONITOR_B1, MODEL_AIRQUALITYMONITOR_CGDN1, MODEL_AIRQUALITYMONITOR_S1, MODEL_AIRQUALITYMONITOR_V1
from .entity import XiaomiMiioEntity

_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME = 'Xiaomi Miio Air Quality Monitor'
ATTR_CO2E = 'carbon_dioxide_equivalent'
ATTR_TVOC = 'total_volatile_organic_compounds'
ATTR_TEMP = 'temperature'
ATTR_HUM = 'humidity'
PROP_TO_ATTR: Dict[str, str] = {
    'carbon_dioxide_equivalent': ATTR_CO2E,
    'total_volatile_organic_compounds': ATTR_TVOC,
    'temperature': ATTR_TEMP,
    'humidity': ATTR_HUM,
}


class AirMonitorB1(XiaomiMiioEntity, AirQualityEntity):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._icon: str = 'mdi:cloud'
        self._available: Optional[bool] = None
        self._air_quality_index: Optional[float] = None
        self._carbon_dioxide: Optional[float] = None
        self._carbon_dioxide_equivalent: Optional[float] = None
        self._particulate_matter_2_5: Optional[float] = None
        self._total_volatile_organic_compounds: Optional[float] = None
        self._temperature: Optional[float] = None
        self._humidity: Optional[float] = None

    async def async_update(self) -> None:
        try:
            state: Any = await self.hass.async_add_executor_job(self._device.status)
            _LOGGER.debug('Got new state: %s', state)
            self._carbon_dioxide_equivalent = state.co2e
            self._particulate_matter_2_5 = round(state.pm25, 1)
            self._total_volatile_organic_compounds = round(state.tvoc, 3)
            self._temperature = round(state.temperature, 2)
            self._humidity = round(state.humidity, 2)
            self._available = True
        except DeviceException as ex:
            self._available = False
            _LOGGER.error('Got exception while fetching the state: %s', ex)

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def available(self) -> Optional[bool]:
        return self._available

    @property
    def air_quality_index(self) -> Optional[float]:
        return self._air_quality_index

    @property
    def carbon_dioxide(self) -> Optional[float]:
        return self._carbon_dioxide

    @property
    def carbon_dioxide_equivalent(self) -> Optional[float]:
        return self._carbon_dioxide_equivalent

    @property
    def particulate_matter_2_5(self) -> Optional[float]:
        return self._particulate_matter_2_5

    @property
    def total_volatile_organic_compounds(self) -> Optional[float]:
        return self._total_volatile_organic_compounds

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    @property
    def humidity(self) -> Optional[float]:
        return self._humidity

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for prop, attr in PROP_TO_ATTR.items():
            if (value := getattr(self, prop)) is not None:
                data[attr] = value
        return data


class AirMonitorS1(AirMonitorB1):
    async def async_update(self) -> None:
        try:
            state: Any = await self.hass.async_add_executor_job(self._device.status)
            _LOGGER.debug('Got new state: %s', state)
            self._carbon_dioxide = state.co2
            self._particulate_matter_2_5 = state.pm25
            self._total_volatile_organic_compounds = state.tvoc
            self._temperature = state.temperature
            self._humidity = state.humidity
            self._available = True
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)


class AirMonitorV1(AirMonitorB1):
    async def async_update(self) -> None:
        try:
            state: Any = await self.hass.async_add_executor_job(self._device.status)
            _LOGGER.debug('Got new state: %s', state)
            self._air_quality_index = state.aqi
            self._available = True
        except DeviceException as ex:
            if self._available:
                self._available = False
                _LOGGER.error('Got exception while fetching the state: %s', ex)

    @property
    def unit_of_measurement(self) -> None:
        return None


class AirMonitorCGDN1(XiaomiMiioEntity, AirQualityEntity):
    def __init__(self, name: str, device: Any, entry: ConfigEntry, unique_id: str) -> None:
        super().__init__(name, device, entry, unique_id)
        self._icon: str = 'mdi:cloud'
        self._available: Optional[bool] = None
        self._carbon_dioxide: Optional[float] = None
        self._particulate_matter_2_5: Optional[float] = None
        self._particulate_matter_10: Optional[float] = None

    async def async_update(self) -> None:
        try:
            state: Any = await self.hass.async_add_executor_job(self._device.status)
            _LOGGER.debug('Got new state: %s', state)
            self._carbon_dioxide = state.co2
            self._particulate_matter_2_5 = round(state.pm25, 1)
            self._particulate_matter_10 = round(state.pm10, 1)
            self._available = True
        except DeviceException as ex:
            self._available = False
            _LOGGER.error('Got exception while fetching the state: %s', ex)

    @property
    def icon(self) -> str:
        return self._icon

    @property
    def available(self) -> Optional[bool]:
        return self._available

    @property
    def carbon_dioxide(self) -> Optional[float]:
        return self._carbon_dioxide

    @property
    def particulate_matter_2_5(self) -> Optional[float]:
        return self._particulate_matter_2_5

    @property
    def particulate_matter_10(self) -> Optional[float]:
        return self._particulate_matter_10


DEVICE_MAP: Dict[str, Dict[str, Any]] = {
    MODEL_AIRQUALITYMONITOR_S1: {
        'device_class': AirQualityMonitor,
        'entity_class': AirMonitorS1,
    },
    MODEL_AIRQUALITYMONITOR_B1: {
        'device_class': AirQualityMonitor,
        'entity_class': AirMonitorB1,
    },
    MODEL_AIRQUALITYMONITOR_V1: {
        'device_class': AirQualityMonitor,
        'entity_class': AirMonitorV1,
    },
    MODEL_AIRQUALITYMONITOR_CGDN1: {
        'device_class': lambda host, token, model: AirQualityMonitorCGDN1(host, token),  # type: Callable[[str, str, str], AirQualityMonitorCGDN1]
        'entity_class': AirMonitorCGDN1,
    },
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    entities: List[AirQualityEntity] = []
    if config_entry.data[CONF_FLOW_TYPE] == CONF_DEVICE:
        host: str = config_entry.data[CONF_HOST]
        token: str = config_entry.data[CONF_TOKEN]
        name: str = config_entry.title
        model: str = config_entry.data[CONF_MODEL]
        unique_id: str = config_entry.unique_id  # type: ignore
        _LOGGER.debug('Initializing with host %s (token %s...)', host, token[:5])
        if model in DEVICE_MAP:
            device_entry: Dict[str, Any] = DEVICE_MAP[model]
            device_class: Any = device_entry['device_class']
            entity_class: Any = device_entry['entity_class']
            entities.append(
                entity_class(
                    name, device_class(host, token, model=model), config_entry, unique_id
                )
            )
        else:
            _LOGGER.warning("AirQualityMonitor model '%s' is not yet supported", model)
    async_add_entities(entities, update_before_add=True)