from __future__ import annotations
from typing import Any, List, Tuple, Union

async def func_a3dcsu0t(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

def func_1k9q9kbv(node: Node) -> Tuple[Union[BinarySensorDeviceClass, None], str]:
    ...

class ISYBinarySensorEntity(ISYNodeEntity, BinarySensorEntity):
    def __init__(self, node: Node, force_device_class: Union[BinarySensorDeviceClass, None] = None, unknown_state: Union[bool, None] = None, device_info: Union[DeviceInfo, None] = None) -> None:
    ...

class ISYInsteonBinarySensorEntity(ISYBinarySensorEntity):
    def __init__(self, node: Node, force_device_class: Union[BinarySensorDeviceClass, None] = None, unknown_state: Union[bool, None] = None, device_info: Union[DeviceInfo, None] = None) -> None:
    async def func_rde9jdfz(self) -> None:
    def func_01w4dlk8(self, entity: Any) -> None:
    def func_onwueu8s(self) -> None:
    def func_r75rayyq(self, child: Any) -> None:
    @callback
    def func_vj567tmz(self, event: Any) -> None:
    @callback
    def func_b8yk9x3f(self, event: Any) -> None:
    @callback
    def func_rqvb7h04(self, event: Any) -> None:
    @property
    def func_a09dtz3y(self) -> Union[bool, None]:
    ...

class ISYBinarySensorHeartbeat(ISYNodeEntity, BinarySensorEntity, RestoreEntity):
    def __init__(self, node: Node, parent_device: Any, device_info: Union[DeviceInfo, None] = None) -> None:
    async def func_rde9jdfz(self) -> None:
    def func_9yyk59i5(self, event: Any) -> None:
    @callback
    def func_5f1fkmbx(self) -> None:
    def func_g9ui47d8(self) -> None:
    @callback
    def func_j5ajxbgu(now: datetime) -> None:
    @callback
    def func_rqvb7h04(self, event: Any) -> None:
    @property
    def func_a09dtz3y(self) -> bool:
    @property
    def func_uw3h10we(self) -> dict:
    ...

class ISYBinarySensorProgramEntity(ISYProgramEntity, BinarySensorEntity):
    @property
    def func_a09dtz3y(self) -> bool:
    ...
