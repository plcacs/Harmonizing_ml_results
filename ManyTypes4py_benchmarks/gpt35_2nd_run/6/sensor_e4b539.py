from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: config_entries.ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class QNAPSensor(CoordinatorEntity[QnapCoordinator], SensorEntity):

    def __init__(self, coordinator: QnapCoordinator, description: SensorEntityDescription, unique_id: str, monitor_device: Any = None) -> None:

class QNAPCPUSensor(QNAPSensor):

    @property
    def native_value(self) -> Any:

class QNAPMemorySensor(QNAPSensor):

    @property
    def native_value(self) -> Any:

class QNAPNetworkSensor(QNAPSensor):

    @property
    def native_value(self) -> Any:

class QNAPSystemSensor(QNAPSensor):

    @property
    def native_value(self) -> Any:

class QNAPDriveSensor(QNAPSensor):

    @property
    def native_value(self) -> Any:

    @property
    def extra_state_attributes(self) -> Any:

class QNAPVolumeSensor(QNAPSensor):

    @property
    def native_value(self) -> Any:
