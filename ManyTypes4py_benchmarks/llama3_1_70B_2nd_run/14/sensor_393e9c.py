from __future__ import annotations
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class TomorrowioSensorEntityDescription:
    """Describes a Tomorrow.io sensor entity."""
    key: str
    translation_key: str
    attribute: str
    unit_imperial: Optional[str]
    unit_metric: Optional[str]
    multiplication_factor: Optional[Callable[[float], float] | float]
    imperial_conversion: Optional[Callable[[float], float] | float]
    value_map: Optional[Any]

def convert_ppb_to_ugm3(molecular_weight: float) -> Callable[[float], float]:
    """Return function to convert ppb to ug/m^3."""
    return lambda x: x * molecular_weight / 24.45

SENSOR_TYPES: tuple[TomorrowioSensorEntityDescription, ...] = (
    # ... existing code
)

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
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

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry, coordinator: TomorrowioDataUpdateCoordinator, api_version: int, description: TomorrowioSensorEntityDescription) -> None:
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
    def _state(self) -> float | None:
        """Return the raw state."""

    @property
    def native_value(self) -> float | str | None:
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
    def _state(self) -> float | None:
        """Return the raw state."""
        val = self._get_current_property(self.entity_description.attribute)
        assert not isinstance(val, str)
        return val
