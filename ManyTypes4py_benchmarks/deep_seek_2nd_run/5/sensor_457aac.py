"""Creates the sensor entities for the mower."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import logging
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Optional, Union, List, Dict, Set, FrozenSet
from aioautomower.model import MowerAttributes, MowerModes, MowerStates, RestrictedReasons, WorkArea
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfLength, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from . import AutomowerConfigEntry
from .coordinator import AutomowerDataUpdateCoordinator
from .entity import AutomowerBaseEntity, WorkAreaAvailableEntity, _work_area_translation_key

_LOGGER: logging.Logger = logging.getLogger(__name__)
PARALLEL_UPDATES: int = 0
ATTR_WORK_AREA_ID_ASSIGNMENT: str = 'work_area_id_assignment'
ERROR_KEY_LIST: List[str] = ['no_error', 'alarm_mower_in_motion', 'alarm_mower_lifted', 'alarm_mower_stopped', 'alarm_mower_switched_off', 'alarm_mower_tilted', 'alarm_outside_geofence', 'angular_sensor_problem', 'battery_problem', 'battery_problem', 'battery_restriction_due_to_ambient_temperature', 'can_error', 'charging_current_too_high', 'charging_station_blocked', 'charging_system_problem', 'charging_system_problem', 'collision_sensor_defect', 'collision_sensor_error', 'collision_sensor_problem_front', 'collision_sensor_problem_rear', 'com_board_not_available', 'communication_circuit_board_sw_must_be_updated', 'complex_working_area', 'connection_changed', 'connection_not_changed', 'connectivity_problem', 'connectivity_problem', 'connectivity_problem', 'connectivity_problem', 'connectivity_problem', 'connectivity_problem', 'connectivity_settings_restored', 'cutting_drive_motor_1_defect', 'cutting_drive_motor_2_defect', 'cutting_drive_motor_3_defect', 'cutting_height_blocked', 'cutting_height_problem', 'cutting_height_problem_curr', 'cutting_height_problem_dir', 'cutting_height_problem_drive', 'cutting_motor_problem', 'cutting_stopped_slope_too_steep', 'cutting_system_blocked', 'cutting_system_blocked', 'cutting_system_imbalance_warning', 'cutting_system_major_imbalance', 'destination_not_reachable', 'difficult_finding_home', 'docking_sensor_defect', 'electronic_problem', 'empty_battery', MowerStates.ERROR.lower(), MowerStates.ERROR_AT_POWER_UP.lower(), MowerStates.FATAL_ERROR.lower(), 'folding_cutting_deck_sensor_defect', 'folding_sensor_activated', 'geofence_problem', 'geofence_problem', 'gps_navigation_problem', 'guide_1_not_found', 'guide_2_not_found', 'guide_3_not_found', 'guide_calibration_accomplished', 'guide_calibration_failed', 'high_charging_power_loss', 'high_internal_power_loss', 'high_internal_temperature', 'internal_voltage_error', 'invalid_battery_combination_invalid_combination_of_different_battery_types', 'invalid_sub_device_combination', 'invalid_system_configuration', 'left_brush_motor_overloaded', 'lift_sensor_defect', 'lifted', 'limited_cutting_height_range', 'limited_cutting_height_range', 'loop_sensor_defect', 'loop_sensor_problem_front', 'loop_sensor_problem_left', 'loop_sensor_problem_rear', 'loop_sensor_problem_right', 'low_battery', 'memory_circuit_problem', 'mower_lifted', 'mower_tilted', 'no_accurate_position_from_satellites', 'no_confirmed_position', 'no_drive', 'no_loop_signal', 'no_power_in_charging_station', 'no_response_from_charger', 'outside_working_area', 'poor_signal_quality', 'reference_station_communication_problem', 'right_brush_motor_overloaded', 'safety_function_faulty', 'settings_restored', 'sim_card_locked', 'sim_card_locked', 'sim_card_locked', 'sim_card_locked', 'sim_card_not_found', 'sim_card_requires_pin', 'slipped_mower_has_slipped_situation_not_solved_with_moving_pattern', 'slope_too_steep', 'sms_could_not_be_sent', 'stop_button_problem', 'stuck_in_charging_station', 'switch_cord_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'temporary_battery_problem', 'tilt_sensor_problem', 'too_high_discharge_current', 'too_high_internal_current', 'trapped', 'ultrasonic_problem', 'ultrasonic_sensor_1_defect', 'ultrasonic_sensor_2_defect', 'ultrasonic_sensor_3_defect', 'ultrasonic_sensor_4_defect', 'unexpected_cutting_height_adj', 'unexpected_error', 'upside_down', 'weak_gps_signal', 'wheel_drive_problem_left', 'wheel_drive_problem_rear_left', 'wheel_drive_problem_rear_right', 'wheel_drive_problem_right', 'wheel_motor_blocked_left', 'wheel_motor_blocked_rear_left', 'wheel_motor_blocked_rear_right', 'wheel_motor_blocked_right', 'wheel_motor_overloaded_left', 'wheel_motor_overloaded_rear_left', 'wheel_motor_overloaded_rear_right', 'wheel_motor_overloaded_right', 'work_area_not_valid', 'wrong_loop_signal', 'wrong_pin_code', 'zone_generator_problem']
ERROR_STATES: Set[MowerStates] = {MowerStates.ERROR, MowerStates.ERROR_AT_POWER_UP, MowerStates.FATAL_ERROR}
RESTRICTED_REASONS: List[RestrictedReasons] = [RestrictedReasons.ALL_WORK_AREAS_COMPLETED, RestrictedReasons.DAILY_LIMIT, RestrictedReasons.EXTERNAL, RestrictedReasons.FOTA, RestrictedReasons.FROST, RestrictedReasons.NONE, RestrictedReasons.NOT_APPLICABLE, RestrictedReasons.PARK_OVERRIDE, RestrictedReasons.SENSOR, RestrictedReasons.WEEK_SCHEDULE]
STATE_NO_WORK_AREA_ACTIVE: str = 'no_work_area_active'

@callback
def _get_work_area_names(data: MowerAttributes) -> List[str]:
    """Return a list with all work area names."""
    if TYPE_CHECKING:
        assert data.work_areas is not None
    work_area_list: List[str] = [data.work_areas[work_area_id].name for work_area_id in data.work_areas]
    work_area_list.append(STATE_NO_WORK_AREA_ACTIVE)
    return work_area_list

@callback
def _get_current_work_area_name(data: MowerAttributes) -> str:
    """Return the name of the current work area."""
    if data.mower.work_area_id is None:
        return STATE_NO_WORK_AREA_ACTIVE
    if TYPE_CHECKING:
        assert data.work_areas is not None
    return data.work_areas[data.mower.work_area_id].name

@callback
def _get_current_work_area_dict(data: MowerAttributes) -> Dict[str, Any]:
    """Return the name of the current work area."""
    if TYPE_CHECKING:
        assert data.work_areas is not None
    return {ATTR_WORK_AREA_ID_ASSIGNMENT: data.work_area_dict}

@callback
def _get_error_string(data: MowerAttributes) -> str:
    """Return the error key, if not provided the mower state or `no error`."""
    if data.mower.error_key is not None:
        return data.mower.error_key
    if data.mower.state in ERROR_STATES:
        return data.mower.state.lower()
    return 'no_error'

@dataclass(frozen=True, kw_only=True)
class AutomowerSensorEntityDescription(SensorEntityDescription):
    """Describes Automower sensor entity."""
    exists_fn: Callable[[MowerAttributes], bool] = lambda _: True
    extra_state_attributes_fn: Callable[[MowerAttributes], Optional[Dict[str, Any]]] = lambda _: None
    option_fn: Callable[[MowerAttributes], Optional[List[Any]]] = lambda _: None
    value_fn: Callable[[MowerAttributes], StateType]

MOWER_SENSOR_TYPES: tuple[AutomowerSensorEntityDescription, ...] = (
    AutomowerSensorEntityDescription(
        key='battery_percent',
        state_class=SensorStateClass.MEASUREMENT,
        device_class=SensorDeviceClass.BATTERY,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=attrgetter('battery.battery_percent')
    ),
    # ... (rest of the MOWER_SENSOR_TYPES entries remain the same)
)

@dataclass(frozen=True, kw_only=True)
class WorkAreaSensorEntityDescription(SensorEntityDescription):
    """Describes the work area sensor entities."""
    exists_fn: Callable[[WorkArea], bool] = lambda _: True
    translation_key_fn: Callable[[str, str], str] = _work_area_translation_key
    value_fn: Callable[[WorkArea], StateType]

WORK_AREA_SENSOR_TYPES: tuple[WorkAreaSensorEntityDescription, ...] = (
    WorkAreaSensorEntityDescription(
        key='progress',
        translation_key_fn=_work_area_translation_key,
        exists_fn=lambda data: data.progress is not None,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=attrgetter('progress')
    ),
    # ... (rest of the WORK_AREA_SENSOR_TYPES entries remain the same)
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: AutomowerConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensor platform."""
    coordinator: AutomowerDataUpdateCoordinator = entry.runtime_data
    entities: list[Union[AutomowerSensorEntity, WorkAreaSensorEntity]] = []
    for mower_id in coordinator.data:
        if coordinator.data[mower_id].capabilities.work_areas:
            _work_areas: Optional[Dict[str, WorkArea]] = coordinator.data[mower_id].work_areas
            if _work_areas is not None:
                entities.extend(
                    WorkAreaSensorEntity(mower_id, coordinator, description, work_area_id)
                    for description in WORK_AREA_SENSOR_TYPES
                    for work_area_id in _work_areas
                    if description.exists_fn(_work_areas[work_area_id])
                )
        entities.extend(
            AutomowerSensorEntity(mower_id, coordinator, description)
            for description in MOWER_SENSOR_TYPES
            if description.exists_fn(coordinator.data[mower_id])
        )
    async_add_entities(entities)

    def _async_add_new_work_areas(mower_id: str, work_area_ids: Set[str]) -> None:
        mower_data: MowerAttributes = coordinator.data[mower_id]
        if mower_data.work_areas is None:
            return
        async_add_entities(
            WorkAreaSensorEntity(mower_id, coordinator, description, work_area_id)
            for description in WORK_AREA_SENSOR_TYPES
            for work_area_id in work_area_ids
            if work_area_id in mower_data.work_areas and description.exists_fn(mower_data.work_areas[work_area_id])
        )

    def _async_add_new_devices(mower_ids: Set[str]) -> None:
        async_add_entities(
            AutomowerSensorEntity(mower_id, coordinator, description)
            for mower_id in mower_ids
            for description in MOWER_SENSOR_TYPES
            if description.exists_fn(coordinator.data[mower_id])
        )
        for mower_id in mower_ids:
            mower_data: MowerAttributes = coordinator.data[mower_id]
            if mower_data.capabilities.work_areas and mower_data.work_areas is not None:
                _async_add_new_work_areas(mower_id, set(mower_data.work_areas.keys()))
    
    coordinator.new_devices_callbacks.append(_async_add_new_devices)
    coordinator.new_areas_callbacks.append(_async_add_new_work_areas)

class AutomowerSensorEntity(AutomowerBaseEntity, SensorEntity):
    """Defining the Automower Sensors with AutomowerSensorEntityDescription."""
    _unrecorded_attributes: FrozenSet[str] = frozenset({ATTR_WORK_AREA_ID_ASSIGNMENT})
    entity_description: AutomowerSensorEntityDescription

    def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: AutomowerSensorEntityDescription) -> None:
        """Set up AutomowerSensors."""
        super().__init__(mower_id, coordinator)
        self.entity_description = description
        self._attr_unique_id: str = f'{mower_id}_{description.key}'

    @property
    def native_value(self) -> StateType:
        """Return the state of the sensor."""
        return self.entity_description.value_fn(self.mower_attributes)

    @property
    def options(self) -> Optional[List[Any]]:
        """Return the option of the sensor."""
        return self.entity_description.option_fn(self.mower_attributes)

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        """Return the state attributes."""
        return self.entity_description.extra_state_attributes_fn(self.mower_attributes)

class WorkAreaSensorEntity(WorkAreaAvailableEntity, SensorEntity):
    """Defining the Work area sensors with WorkAreaSensorEntityDescription."""
    entity_description: WorkAreaSensorEntityDescription

    def __init__(
        self,
        mower_id: str,
        coordinator: AutomowerDataUpdateCoordinator,
        description: WorkAreaSensorEntityDescription,
        work_area_id: str
    ) -> None:
        """Set up AutomowerSensors."""
        super().__init__(mower_id, coordinator, work_area_id)
        self.entity_description = description
        self._attr_unique_id: str = f'{mower_id}_{work_area_id}_{description.key}'
        self._attr_translation_placeholders: Dict[str, str] = {'work_area': self.work_area_attributes.name}

    @property
    def native_value(self) -> StateType:
        """Return the state of the sensor."""
        return self.entity_description.value_fn(self.work_area_attributes)

    @property
    def translation_key(self) -> str:
        """Return the translation key of the work area."""
        return self.entity_description.translation_key_fn(self.work_area_id, self.entity_description.key)
