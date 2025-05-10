"""Validate the energy preferences provide valid data."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from homeassistant.components import recorder, sensor
from homeassistant.const import (
    ATTR_DEVICE_CLASS,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfEnergy,
    UnitOfVolume,
)
from homeassistant.core import (
    HomeAssistant,
    callback,
    valid_entity_id,
)
from . import data
from .const import DOMAIN

ENERGY_USAGE_DEVICE_CLASSES: Tuple[sensor.SensorDeviceClass, ...] = (sensor.SensorDeviceClass.ENERGY,)
ENERGY_USAGE_UNITS: Dict[sensor.SensorDeviceClass, Tuple[str, ...]] = {
    sensor.SensorDeviceClass.ENERGY: (
        UnitOfEnergy.GIGA_JOULE,
        UnitOfEnergy.KILO_WATT_HOUR,
        UnitOfEnergy.MEGA_JOULE,
        UnitOfEnergy.MEGA_WATT_HOUR,
        UnitOfEnergy.WATT_HOUR,
    )
}
ENERGY_PRICE_UNITS: Tuple[str, ...] = tuple(
    f'/{unit}' for units in ENERGY_USAGE_UNITS.values() for unit in units
)
ENERGY_UNIT_ERROR: str = 'entity_unexpected_unit_energy'
ENERGY_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_energy_price'

GAS_USAGE_DEVICE_CLASSES: Tuple[sensor.SensorDeviceClass, ...] = (
    sensor.SensorDeviceClass.ENERGY,
    sensor.SensorDeviceClass.GAS,
)
GAS_USAGE_UNITS: Dict[sensor.SensorDeviceClass, Tuple[str, ...]] = {
    sensor.SensorDeviceClass.ENERGY: (
        UnitOfEnergy.GIGA_JOULE,
        UnitOfEnergy.KILO_WATT_HOUR,
        UnitOfEnergy.MEGA_JOULE,
        UnitOfEnergy.MEGA_WATT_HOUR,
        UnitOfEnergy.WATT_HOUR,
    ),
    sensor.SensorDeviceClass.GAS: (
        UnitOfVolume.CENTUM_CUBIC_FEET,
        UnitOfVolume.CUBIC_FEET,
        UnitOfVolume.CUBIC_METERS,
    ),
}
GAS_PRICE_UNITS: Tuple[str, ...] = tuple(
    f'/{unit}' for units in GAS_USAGE_UNITS.values() for unit in units
)
GAS_UNIT_ERROR: str = 'entity_unexpected_unit_gas'
GAS_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_gas_price'

WATER_USAGE_DEVICE_CLASSES: Tuple[sensor.SensorDeviceClass, ...] = (sensor.SensorDeviceClass.WATER,)
WATER_USAGE_UNITS: Dict[sensor.SensorDeviceClass, Tuple[str, ...]] = {
    sensor.SensorDeviceClass.WATER: (
        UnitOfVolume.CENTUM_CUBIC_FEET,
        UnitOfVolume.CUBIC_FEET,
        UnitOfVolume.CUBIC_METERS,
        UnitOfVolume.GALLONS,
        UnitOfVolume.LITERS,
    )
}
WATER_PRICE_UNITS: Tuple[str, ...] = tuple(
    f'/{unit}' for units in WATER_USAGE_UNITS.values() for unit in units
)
WATER_UNIT_ERROR: str = 'entity_unexpected_unit_water'
WATER_PRICE_UNIT_ERROR: str = 'entity_unexpected_unit_water_price'


def _get_placeholders(
    hass: HomeAssistant, issue_type: str
) -> Optional[Dict[str, str]]:
    currency: str = hass.config.currency
    if issue_type == ENERGY_UNIT_ERROR:
        return {'energy_units': ', '.join(ENERGY_USAGE_UNITS[sensor.SensorDeviceClass.ENERGY])}
    if issue_type == ENERGY_PRICE_UNIT_ERROR:
        return {
            'price_units': ', '.join(
                (f'{currency}{unit}' for unit in ENERGY_PRICE_UNITS)
            )
        }
    if issue_type == GAS_UNIT_ERROR:
        return {
            'energy_units': ', '.join(GAS_USAGE_UNITS[sensor.SensorDeviceClass.ENERGY]),
            'gas_units': ', '.join(GAS_USAGE_UNITS[sensor.SensorDeviceClass.GAS]),
        }
    if issue_type == GAS_PRICE_UNIT_ERROR:
        return {
            'price_units': ', '.join(
                (f'{currency}{unit}' for unit in GAS_PRICE_UNITS)
            )
        }
    if issue_type == WATER_UNIT_ERROR:
        return {'water_units': ', '.join(WATER_USAGE_UNITS[sensor.SensorDeviceClass.WATER])}
    if issue_type == WATER_PRICE_UNIT_ERROR:
        return {
            'price_units': ', '.join(
                (f'{currency}{unit}' for unit in WATER_PRICE_UNITS)
            )
        }
    return None


@dataclasses.dataclass(slots=True)
class ValidationIssue:
    """Error or warning message."""
    affected_entities: Set[Tuple[str, Any]] = dataclasses.field(default_factory=set)
    translation_placeholders: Optional[Dict[str, str]] = None


@dataclasses.dataclass(slots=True)
class ValidationIssues:
    """Container for validation issues."""
    issues: Dict[str, ValidationIssue] = dataclasses.field(default_factory=dict)

    def __init__(self) -> None:
        """Container for validation issues."""
        self.issues = {}

    def add_issue(
        self,
        hass: HomeAssistant,
        issue_type: str,
        affected_entity: str,
        detail: Optional[Any] = None,
    ) -> None:
        """Add an issue for an entity."""
        if not (issue := self.issues.get(issue_type)):
            self.issues[issue_type] = issue = ValidationIssue()
            issue.translation_placeholders = _get_placeholders(hass, issue_type)
        issue.affected_entities.add((affected_entity, detail))


@dataclasses.dataclass(slots=True)
class EnergyPreferencesValidation:
    """Dictionary holding validation information."""
    energy_sources: List[ValidationIssues] = dataclasses.field(default_factory=list)
    device_consumption: List[ValidationIssues] = dataclasses.field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary version."""
        return {
            'energy_sources': [
                [dataclasses.asdict(issue) for issue in issues.issues.values()]
                for issues in self.energy_sources
            ],
            'device_consumption': [
                [dataclasses.asdict(issue) for issue in issues.issues.values()]
                for issues in self.device_consumption
            ],
        }


@callback
def _async_validate_usage_stat(
    hass: HomeAssistant,
    metadata: Dict[str, Any],
    stat_id: str,
    allowed_device_classes: Tuple[sensor.SensorDeviceClass, ...],
    allowed_units: Dict[sensor.SensorDeviceClass, Tuple[str, ...]],
    unit_error: str,
    issues: ValidationIssues,
) -> None:
    """Validate a statistic."""
    if stat_id not in metadata:
        issues.add_issue(hass, 'statistics_not_defined', stat_id)
    has_entity_source: bool = valid_entity_id(stat_id)
    if not has_entity_source:
        return
    entity_id: str = stat_id
    if not recorder.is_entity_recorded(hass, entity_id):
        issues.add_issue(hass, 'recorder_untracked', entity_id)
        return
    state = hass.states.get(entity_id)
    if state is None:
        issues.add_issue(hass, 'entity_not_defined', entity_id)
        return
    if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        issues.add_issue(hass, 'entity_unavailable', entity_id, state.state)
        return
    try:
        current_value: float = float(state.state)
    except ValueError:
        issues.add_issue(hass, 'entity_state_non_numeric', entity_id, state.state)
        return
    if current_value is not None and current_value < 0:
        issues.add_issue(hass, 'entity_negative_state', entity_id, current_value)
    device_class: Optional[str] = state.attributes.get(ATTR_DEVICE_CLASS)
    if device_class not in allowed_device_classes:
        issues.add_issue(hass, 'entity_unexpected_device_class', entity_id, device_class)
    else:
        unit: Optional[str] = state.attributes.get('unit_of_measurement')
        if device_class and unit not in allowed_units.get(device_class, []):
            issues.add_issue(hass, unit_error, entity_id, unit)
    state_class: Optional[str] = state.attributes.get(sensor.ATTR_STATE_CLASS)
    allowed_state_classes: List[str] = [
        sensor.SensorStateClass.MEASUREMENT,
        sensor.SensorStateClass.TOTAL,
        sensor.SensorStateClass.TOTAL_INCREASING,
    ]
    if state_class not in allowed_state_classes:
        issues.add_issue(hass, 'entity_unexpected_state_class', entity_id, state_class)
    if (
        state_class == sensor.SensorStateClass.MEASUREMENT
        and sensor.ATTR_LAST_RESET not in state.attributes
    ):
        issues.add_issue(
            hass, 'entity_state_class_measurement_no_last_reset', entity_id
        )


@callback
def _async_validate_price_entity(
    hass: HomeAssistant,
    entity_id: str,
    issues: ValidationIssues,
    allowed_units: Tuple[str, ...],
    unit_error: str,
) -> None:
    """Validate that the price entity is correct."""
    state = hass.states.get(entity_id)
    if state is None:
        issues.add_issue(hass, 'entity_not_defined', entity_id)
        return
    try:
        float(state.state)
    except ValueError:
        issues.add_issue(
            hass, 'entity_state_non_numeric', entity_id, state.state
        )
        return
    unit: Optional[str] = state.attributes.get('unit_of_measurement')
    if unit is None or not any(unit.endswith(allowed_unit) for allowed_unit in allowed_units):
        issues.add_issue(hass, unit_error, entity_id, unit)


@callback
def _async_validate_cost_stat(
    hass: HomeAssistant,
    metadata: Dict[str, Any],
    stat_id: str,
    issues: ValidationIssues,
) -> None:
    """Validate that the cost stat is correct."""
    if stat_id not in metadata:
        issues.add_issue(hass, 'statistics_not_defined', stat_id)
    has_entity: bool = valid_entity_id(stat_id)
    if not has_entity:
        return
    if not recorder.is_entity_recorded(hass, stat_id):
        issues.add_issue(hass, 'recorder_untracked', stat_id)
    state = hass.states.get(stat_id)
    if state is None:
        issues.add_issue(hass, 'entity_not_defined', stat_id)
        return
    state_class: Optional[str] = state.attributes.get('state_class')
    supported_state_classes: List[str] = [
        sensor.SensorStateClass.MEASUREMENT,
        sensor.SensorStateClass.TOTAL,
        sensor.SensorStateClass.TOTAL_INCREASING,
    ]
    if state_class not in supported_state_classes:
        issues.add_issue(hass, 'entity_unexpected_state_class', stat_id, state_class)
    if (
        state_class == sensor.SensorStateClass.MEASUREMENT
        and sensor.ATTR_LAST_RESET not in state.attributes
    ):
        issues.add_issue(
            hass, 'entity_state_class_measurement_no_last_reset', stat_id
        )


@callback
def _async_validate_auto_generated_cost_entity(
    hass: HomeAssistant,
    energy_entity_id: str,
    issues: ValidationIssues,
) -> None:
    """Validate that the auto generated cost entity is correct."""
    cost_sensors: Dict[str, str] = hass.data[DOMAIN].get('cost_sensors', {})
    if energy_entity_id not in cost_sensors:
        return
    cost_entity_id: str = cost_sensors[energy_entity_id]
    if not recorder.is_entity_recorded(hass, cost_entity_id):
        issues.add_issue(hass, 'recorder_untracked', cost_entity_id)


async def async_validate(hass: HomeAssistant) -> EnergyPreferencesValidation:
    """Validate the energy configuration."""
    manager = await data.async_get_manager(hass)
    statistics_metadata: Dict[str, Any] = {}
    validate_calls: List[Callable[[], None]] = []
    wanted_statistics_metadata: Set[str] = set()
    result = EnergyPreferencesValidation()
    if manager.data is None:
        return result
    for source in manager.data['energy_sources']:
        source_result = ValidationIssues()
        result.energy_sources.append(source_result)
        source_type: str = source['type']
        if source_type == 'grid':
            for flow in source['flow_from']:
                stat_energy_from: str = flow['stat_energy_from']
                wanted_statistics_metadata.add(stat_energy_from)
                validate_calls.append(
                    functools.partial(
                        _async_validate_usage_stat,
                        hass,
                        statistics_metadata,
                        stat_energy_from,
                        ENERGY_USAGE_DEVICE_CLASSES,
                        ENERGY_USAGE_UNITS,
                        ENERGY_UNIT_ERROR,
                        source_result,
                    )
                )
                stat_cost: Optional[str] = flow.get('stat_cost')
                entity_energy_price: Optional[str] = flow.get('entity_energy_price')
                if stat_cost is not None:
                    wanted_statistics_metadata.add(stat_cost)
                    validate_calls.append(
                        functools.partial(
                            _async_validate_cost_stat,
                            hass,
                            statistics_metadata,
                            stat_cost,
                            source_result,
                        )
                    )
                elif entity_energy_price is not None:
                    validate_calls.append(
                        functools.partial(
                            _async_validate_price_entity,
                            hass,
                            entity_energy_price,
                            source_result,
                            ENERGY_PRICE_UNITS,
                            ENERGY_PRICE_UNIT_ERROR,
                        )
                    )
                if flow.get('entity_energy_price') is not None or flow.get(
                    'number_energy_price'
                ) is not None:
                    validate_calls.append(
                        functools.partial(
                            _async_validate_auto_generated_cost_entity,
                            hass,
                            flow['stat_energy_from'],
                            source_result,
                        )
                    )
            for flow in source['flow_to']:
                stat_energy_to: str = flow['stat_energy_to']
                wanted_statistics_metadata.add(stat_energy_to)
                validate_calls.append(
                    functools.partial(
                        _async_validate_usage_stat,
                        hass,
                        statistics_metadata,
                        stat_energy_to,
                        ENERGY_USAGE_DEVICE_CLASSES,
                        ENERGY_USAGE_UNITS,
                        ENERGY_UNIT_ERROR,
                        source_result,
                    )
                )
                stat_compensation: Optional[str] = flow.get('stat_compensation')
                entity_energy_price: Optional[str] = flow.get('entity_energy_price')
                if stat_compensation is not None:
                    wanted_statistics_metadata.add(stat_compensation)
                    validate_calls.append(
                        functools.partial(
                            _async_validate_cost_stat,
                            hass,
                            statistics_metadata,
                            stat_compensation,
                            source_result,
                        )
                    )
                elif entity_energy_price is not None:
                    validate_calls.append(
                        functools.partial(
                            _async_validate_price_entity,
                            hass,
                            entity_energy_price,
                            source_result,
                            ENERGY_PRICE_UNITS,
                            ENERGY_PRICE_UNIT_ERROR,
                        )
                    )
                if flow.get('entity_energy_price') is not None or flow.get(
                    'number_energy_price'
                ) is not None:
                    validate_calls.append(
                        functools.partial(
                            _async_validate_auto_generated_cost_entity,
                            hass,
                            flow['stat_energy_to'],
                            source_result,
                        )
                    )
        elif source_type == 'gas':
            stat_energy_from: str = source['stat_energy_from']
            wanted_statistics_metadata.add(stat_energy_from)
            validate_calls.append(
                functools.partial(
                    _async_validate_usage_stat,
                    hass,
                    statistics_metadata,
                    stat_energy_from,
                    GAS_USAGE_DEVICE_CLASSES,
                    GAS_USAGE_UNITS,
                    GAS_UNIT_ERROR,
                    source_result,
                )
            )
            stat_cost: Optional[str] = source.get('stat_cost')
            entity_energy_price: Optional[str] = source.get('entity_energy_price')
            if stat_cost is not None:
                wanted_statistics_metadata.add(stat_cost)
                validate_calls.append(
                    functools.partial(
                        _async_validate_cost_stat,
                        hass,
                        statistics_metadata,
                        stat_cost,
                        source_result,
                    )
                )
            elif entity_energy_price is not None:
                validate_calls.append(
                    functools.partial(
                        _async_validate_price_entity,
                        hass,
                        entity_energy_price,
                        source_result,
                        GAS_PRICE_UNITS,
                        GAS_PRICE_UNIT_ERROR,
                    )
                )
            if source.get('entity_energy_price') is not None or source.get(
                'number_energy_price'
            ) is not None:
                validate_calls.append(
                    functools.partial(
                        _async_validate_auto_generated_cost_entity,
                        hass,
                        source['stat_energy_from'],
                        source_result,
                    )
                )
        elif source_type == 'water':
            stat_energy_from: str = source['stat_energy_from']
            wanted_statistics_metadata.add(stat_energy_from)
            validate_calls.append(
                functools.partial(
                    _async_validate_usage_stat,
                    hass,
                    statistics_metadata,
                    stat_energy_from,
                    WATER_USAGE_DEVICE_CLASSES,
                    WATER_USAGE_UNITS,
                    WATER_UNIT_ERROR,
                    source_result,
                )
            )
            stat_cost: Optional[str] = source.get('stat_cost')
            entity_energy_price: Optional[str] = source.get('entity_energy_price')
            if stat_cost is not None:
                wanted_statistics_metadata.add(stat_cost)
                validate_calls.append(
                    functools.partial(
                        _async_validate_cost_stat,
                        hass,
                        statistics_metadata,
                        stat_cost,
                        source_result,
                    )
                )
            elif entity_energy_price is not None:
                validate_calls.append(
                    functools.partial(
                        _async_validate_price_entity,
                        hass,
                        entity_energy_price,
                        source_result,
                        WATER_PRICE_UNITS,
                        WATER_PRICE_UNIT_ERROR,
                    )
                )
            if source.get('entity_energy_price') is not None or source.get(
                'number_energy_price'
            ) is not None:
                validate_calls.append(
                    functools.partial(
                        _async_validate_auto_generated_cost_entity,
                        hass,
                        source['stat_energy_from'],
                        source_result,
                    )
                )
        elif source_type == 'solar':
            stat_energy_from: str = source['stat_energy_from']
            wanted_statistics_metadata.add(stat_energy_from)
            validate_calls.append(
                functools.partial(
                    _async_validate_usage_stat,
                    hass,
                    statistics_metadata,
                    stat_energy_from,
                    ENERGY_USAGE_DEVICE_CLASSES,
                    ENERGY_USAGE_UNITS,
                    ENERGY_UNIT_ERROR,
                    source_result,
                )
            )
        elif source_type == 'battery':
            stat_energy_from: str = source['stat_energy_from']
            wanted_statistics_metadata.add(stat_energy_from)
            validate_calls.append(
                functools.partial(
                    _async_validate_usage_stat,
                    hass,
                    statistics_metadata,
                    stat_energy_from,
                    ENERGY_USAGE_DEVICE_CLASSES,
                    ENERGY_USAGE_UNITS,
                    ENERGY_UNIT_ERROR,
                    source_result,
                )
            )
            stat_energy_to: str = source['stat_energy_to']
            wanted_statistics_metadata.add(stat_energy_to)
            validate_calls.append(
                functools.partial(
                    _async_validate_usage_stat,
                    hass,
                    statistics_metadata,
                    stat_energy_to,
                    ENERGY_USAGE_DEVICE_CLASSES,
                    ENERGY_USAGE_UNITS,
                    ENERGY_UNIT_ERROR,
                    source_result,
                )
            )
    for device in manager.data['device_consumption']:
        device_result = ValidationIssues()
        result.device_consumption.append(device_result)
        stat_consumption: str = device['stat_consumption']
        wanted_statistics_metadata.add(stat_consumption)
        validate_calls.append(
            functools.partial(
                _async_validate_usage_stat,
                hass,
                statistics_metadata,
                stat_consumption,
                ENERGY_USAGE_DEVICE_CLASSES,
                ENERGY_USAGE_UNITS,
                ENERGY_UNIT_ERROR,
                device_result,
            )
        )
    statistics_metadata.update(
        await recorder.get_instance(hass).async_add_executor_job(
            functools.partial(
                recorder.statistics.get_metadata, hass, statistic_ids=wanted_statistics_metadata
            )
        )
    )
    for call in validate_calls:
        call()
    return result
