from __future__ import annotations
from collections.abc import Mapping, Sequence
import dataclasses
import functools
from homeassistant.components import recorder, sensor
from homeassistant.const import ATTR_DEVICE_CLASS, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant, callback, valid_entity_id
from . import data
from .const import DOMAIN

ENERGY_USAGE_DEVICE_CLASSES: tuple[sensor.SensorDeviceClass, ...] = (sensor.SensorDeviceClass.ENERGY,)
ENERGY_USAGE_UNITS: dict[sensor.SensorDeviceClass, tuple[UnitOfEnergy, ...]] = {sensor.SensorDeviceClass.ENERGY: (UnitOfEnergy.GIGA_JOULE, UnitOfEnergy.KILO_WATT_HOUR, UnitOfEnergy.MEGA_JOULE, UnitOfEnergy.MEGA_WATT_HOUR, UnitOfEnergy.WATT_HOUR)}
ENERGY_PRICE_UNITS: tuple[str, ...] = tuple(f'{currency}/{unit}' for unit in ENERGY_USAGE_UNITS[sensor.SensorDeviceClass.ENERGY] for currency in (hass.config.currency,))
ENERGY_UNIT_ERROR: str = 'entity_unexpected_unit_energy'
ENERGY_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_energy_price'

GAS_USAGE_DEVICE_CLASSES: tuple[sensor.SensorDeviceClass, ...] = (sensor.SensorDeviceClass.ENERGY, sensor.SensorDeviceClass.GAS)
GAS_USAGE_UNITS: dict[sensor.SensorDeviceClass, tuple[UnitOfEnergy | UnitOfVolume, ...]] = {sensor.SensorDeviceClass.ENERGY: (UnitOfEnergy.GIGA_JOULE, UnitOfEnergy.KILO_WATT_HOUR, UnitOfEnergy.MEGA_JOULE, UnitOfEnergy.MEGA_WATT_HOUR, UnitOfEnergy.WATT_HOUR), sensor.SensorDeviceClass.GAS: (UnitOfVolume.CENTUM_CUBIC_FEET, UnitOfVolume.CUBIC_FEET, UnitOfVolume.CUBIC_METERS)}
GAS_PRICE_UNITS: tuple[str, ...] = tuple(f'{currency}/{unit}' for unit in GAS_USAGE_UNITS[sensor.SensorDeviceClass.ENERGY] for currency in (hass.config.currency,))
GAS_UNIT_ERROR: str = 'entity_unexpected_unit_gas'
GAS_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_gas_price'

WATER_USAGE_DEVICE_CLASSES: tuple[sensor.SensorDeviceClass, ...] = (sensor.SensorDeviceClass.WATER,)
WATER_USAGE_UNITS: dict[sensor.SensorDeviceClass, tuple[UnitOfVolume, ...]] = {sensor.SensorDeviceClass.WATER: (UnitOfVolume.CENTUM_CUBIC_FEET, UnitOfVolume.CUBIC_FEET, UnitOfVolume.CUBIC_METERS, UnitOfVolume.GALLONS, UnitOfVolume.LITERS)}
WATER_PRICE_UNITS: tuple[str, ...] = tuple(f'{currency}/{unit}' for unit in WATER_USAGE_UNITS[sensor.SensorDeviceClass.WATER] for currency in (hass.config.currency,))
WATER_UNIT_ERROR: str = 'entity_unexpected_unit_water'
WATER_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_water_price'

@dataclasses.dataclass(slots=True)
class ValidationIssue:
    """Error or warning message."""
    affected_entities: set[tuple[str, str]]
    translation_placeholders: dict[str, str] | None

@dataclasses.dataclass(slots=True)
class ValidationIssues:
    """Container for validation issues."""
    issues: dict[str, ValidationIssue]

    def __init__(self):
        """Container for validiation issues."""
        self.issues = {}

    def add_issue(self, hass: HomeAssistant, issue_type: str, affected_entity: str, detail: str | None = None) -> None:
        """Add an issue for an entity."""
        if not (issue := self.issues.get(issue_type)):
            self.issues[issue_type] = issue = ValidationIssue(issue_type)
            issue.translation_placeholders = _get_placeholders(hass, issue_type)
        issue.affected_entities.add((affected_entity, detail))

@dataclasses.dataclass(slots=True)
class EnergyPreferencesValidation:
    """Dictionary holding validation information."""
    energy_sources: list[ValidationIssues]
    device_consumption: list[ValidationIssues]

    def as_dict(self) -> dict[str, list[dict[str, str]]]:
        """Return dictionary version."""
        return {'energy_sources': [[dataclasses.asdict(issue) for issue in issues.issues.values()] for issues in self.energy_sources], 'device_consumption': [[dataclasses.asdict(issue) for issue in issues.issues.values()] for issues in self.device_consumption]}

@callback
def _async_validate_usage_stat(hass: HomeAssistant, metadata: dict[str, str], stat_id: str, allowed_device_classes: tuple[sensor.SensorDeviceClass, ...], allowed_units: dict[sensor.SensorDeviceClass, tuple[UnitOfEnergy | UnitOfVolume, ...]], unit_error: str, issues: ValidationIssues) -> None:
    """Validate a statistic."""
    # ... rest of the function ...

@callback
def _async_validate_price_entity(hass: HomeAssistant, entity_id: str, issues: ValidationIssues, allowed_units: tuple[str, ...], unit_error: str) -> None:
    """Validate that the price entity is correct."""
    # ... rest of the function ...

@callback
def _async_validate_cost_stat(hass: HomeAssistant, metadata: dict[str, str], stat_id: str, issues: ValidationIssues) -> None:
    """Validate that the cost stat is correct."""
    # ... rest of the function ...

@callback
def _async_validate_auto_generated_cost_entity(hass: HomeAssistant, energy_entity_id: str, issues: ValidationIssues) -> None:
    """Validate that the auto generated cost entity is correct."""
    # ... rest of the function ...

async def async_validate(hass: HomeAssistant) -> EnergyPreferencesValidation:
    """Validate the energy configuration."""
    manager = await data.async_get_manager(hass)
    statistics_metadata: dict[str, str] = {}
    validate_calls: list[callable] = []
    result: EnergyPreferencesValidation = EnergyPreferencesValidation()
    # ... rest of the function ...
