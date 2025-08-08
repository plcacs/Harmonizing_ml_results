from typing import Any, List, Dict, Union

def _get_work_area_names(data: Any) -> List[str]:
    ...

def _get_current_work_area_name(data: Any) -> str:
    ...

def _get_current_work_area_dict(data: Any) -> Dict[str, Any]:
    ...

def _get_error_string(data: Any) -> str:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: AutomowerConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class AutomowerSensorEntityDescription(SensorEntityDescription):
    exists_fn: Callable[[Any], bool]
    extra_state_attributes_fn: Callable[[Any], Any]
    option_fn: Callable[[Any], Any]

class WorkAreaSensorEntityDescription(SensorEntityDescription):
    exists_fn: Callable[[Any], bool]

class AutomowerSensorEntity(AutomowerBaseEntity, SensorEntity):
    def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: AutomowerSensorEntityDescription) -> None:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def options(self) -> Any:
        ...

    @property
    def extra_state_attributes(self) -> Any:
        ...

class WorkAreaSensorEntity(WorkAreaAvailableEntity, SensorEntity):
    def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: WorkAreaSensorEntityDescription, work_area_id: str) -> None:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def translation_key(self) -> str:
        ...
