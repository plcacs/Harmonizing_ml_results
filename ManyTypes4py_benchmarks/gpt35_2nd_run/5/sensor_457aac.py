from typing import Any, List, Dict, Set, Union, Callable, Tuple

def _get_work_area_names(data: Any) -> List[str]:
def _get_current_work_area_name(data: Any) -> str:
def _get_current_work_area_dict(data: Any) -> Dict[str, Any]:
def _get_error_string(data: Any) -> str:
async def async_setup_entry(hass: HomeAssistant, entry: AutomowerConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback):
class AutomowerSensorEntityDescription(SensorEntityDescription):
MOWER_SENSOR_TYPES: Tuple[AutomowerSensorEntityDescription, ...]
class WorkAreaSensorEntityDescription(SensorEntityDescription):
WORK_AREA_SENSOR_TYPES: Tuple[WorkAreaSensorEntityDescription, ...]
class AutomowerSensorEntity(AutomowerBaseEntity, SensorEntity):
def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: AutomowerSensorEntityDescription):
@property
def native_value(self) -> Any:
@property
def options(self) -> Union[List[str], None]:
@property
def extra_state_attributes(self) -> Dict[str, Any]:
class WorkAreaSensorEntity(WorkAreaAvailableEntity, SensorEntity):
def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: WorkAreaSensorEntityDescription, work_area_id: str):
@property
def native_value(self) -> Any:
@property
def translation_key(self) -> str:
