"""Sensor component that handles additional Tomorrowio data for your location."""
from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Type, cast
from pytomorrowio.const import HealthConcernType, PollenIndex, PrecipitationType, PrimaryPollutantType, UVDescription
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONCENTRATION_MICROGRAMS_PER_CUBIC_METER, CONCENTRATION_PARTS_PER_MILLION, CONF_API_KEY, PERCENTAGE, UnitOfIrradiance, UnitOfLength, UnitOfPressure, UnitOfSpeed, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.unit_conversion import DistanceConverter, SpeedConverter
from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM, UnitSystem
from .const import DOMAIN, TMRW_ATTR_CARBON_MONOXIDE, TMRW_ATTR_CHINA_AQI, TMRW_ATTR_CHINA_HEALTH_CONCERN, TMRW_ATTR_CHINA_PRIMARY_POLLUTANT, TMRW_ATTR_CLOUD_BASE, TMRW_ATTR_CLOUD_CEILING, TMRW_ATTR_CLOUD_COVER, TMRW_ATTR_DEW_POINT, TMRW_ATTR_EPA_AQI, TMRW_ATTR_EPA_HEALTH_CONCERN, TMRW_ATTR_EPA_PRIMARY_POLLUTANT, TMRW_ATTR_FEELS_LIKE, TMRW_ATTR_FIRE_INDEX, TMRW_ATTR_NITROGEN_DIOXIDE, TMRW_ATTR_OZONE, TMRW_ATTR_PARTICULATE_MATTER_10, TMRW_ATTR_PARTICULATE_MATTER_25, TMRW_ATTR_POLLEN_GRASS, TMRW_ATTR_POLLEN_TREE, TMRW_ATTR_POLLEN_WEED, TMRW_ATTR_PRECIPITATION_TYPE, TMRW_ATTR_PRESSURE_SURFACE_LEVEL, TMRW_ATTR_SOLAR_GHI, TMRW_ATTR_SULPHUR_DIOXIDE, TMRW_ATTR_UV_HEALTH_CONCERN, TMRW_ATTR_UV_INDEX, TMRW_ATTR_WIND_GUST
from .coordinator import TomorrowioDataUpdateCoordinator
from .entity import TomorrowioEntity

_T = TypeVar("_T")

@dataclass(frozen=True)
class TomorrowioSensorEntityDescription(SensorEntityDescription):
    """Describes a Tomorrow.io sensor entity."""
    attribute: str = ''
    unit_imperial: Optional[str] = None
    unit_metric: Optional[str] = None
    multiplication_factor: Optional[Callable[[float], float]] = None
    imperial_conversion: Optional[Callable[[float], float]] = None
    value_map: Optional[Type[_T]] = None

    def __post_init__(self) -> None:
        """Handle post init."""
        if self.unit_imperial is None and self.unit_metric is not None or (self.unit_imperial is not None and self.unit_metric is None):
            raise ValueError('Entity descriptions must include both imperial and metric units or they must both be None')
        if self.value_map is not None:
            options = [item.name.lower() for item in self.value_map]
            object.__setattr__(self, 'device_class', SensorDeviceClass.ENUM)
            object.__setattr__(self, 'options', options)

def convert_ppb_to_ugm3(molecular_weight: float) -> Callable[[float], float]:
    """Return function to convert ppb to ug/m^3."""
    return lambda x: x * molecular_weight / 24.45

SENSOR_TYPES: tuple[TomorrowioSensorEntityDescription, ...] = (
    TomorrowioSensorEntityDescription(
        key='feels_like',
        translation_key='feels_like',
        attribute=TMRW_ATTR_FEELS_LIKE,
        native_unit_of_measurement=UnitOfTemperature.CELSIUS,
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT
    ),
    # ... (rest of the SENSOR_TYPES entries remain the same)
)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up a config entry."""
    coordinator = hass.data[DOMAIN][config_entry.data[CONF_API_KEY]]
    entities = [TomorrowioSensorEntity(hass, config_entry, coordinator, 4, description) for description in SENSOR_TYPES]
    async_add_entities(entities)

def handle_conversion(value: float, conversion: Callable[[float], float] | float) -> float:
    """Handle conversion of a value based on conversion type."""
    if callable(conversion):
        return round(conversion(float(value)), 2)
    return round(float(value) * conversion, 2)

class BaseTomorrowioSensorEntity(TomorrowioEntity, SensorEntity):
    """Base Tomorrow.io sensor entity."""
    _attr_entity_registry_enabled_default: bool = False

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        coordinator: TomorrowioDataUpdateCoordinator,
        api_version: int,
        description: TomorrowioSensorEntityDescription,
    ) -> None:
        """Initialize Tomorrow.io Sensor Entity."""
        super().__init__(config_entry, coordinator, api_version)
        self.entity_description = description
        self._attr_unique_id = f'{self._config_entry.unique_id}_{description.key}'
        if self.entity_description.native_unit_of_measurement is None:
            self._attr_native_unit_of_measurement = description.unit_metric
            if hass.config.units is US_CUSTOMARY_SYSTEM:
                self._attr_native_unit_of_measurement = description.unit_imperial

    @property
    @abstractmethod
    def _state(self) -> Optional[float | int | str]:
        """Return the raw state."""

    @property
    def native_value(self) -> Optional[float | int | str]:
        """Return the state."""
        state = self._state
        desc = self.entity_description
        if state is None:
            return state
        if desc.value_map is not None:
            return desc.value_map(state).name.lower()
        if desc.multiplication_factor is not None:
            state = handle_conversion(state, desc.multiplication_factor)
        if desc.imperial_conversion and desc.unit_imperial is not None and (desc.unit_imperial != desc.unit_metric) and (self.hass.config.units is US_CUSTOMARY_SYSTEM):
            return handle_conversion(state, desc.imperial_conversion)
        return state

class TomorrowioSensorEntity(BaseTomorrowioSensorEntity):
    """Sensor entity that talks to Tomorrow.io v4 API to retrieve non-weather data."""

    @property
    def _state(self) -> Optional[float | int | str]:
        """Return the raw state."""
        val = self._get_current_property(self.entity_description.attribute)
        assert not isinstance(val, str)
        return cast(Optional[float | int], val)
