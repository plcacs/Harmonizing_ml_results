from typing import Any, Final, List, Callable

class PowerViewSensor(ShadeEntity, SensorEntity):
    def __init__(self, coordinator: PowerviewShadeUpdateCoordinator, device_info: PowerviewDeviceInfo, room_name: str, shade: BaseShade, name: str, description: PowerviewSensorDescription):
    def native_value(self) -> Any:
    def native_unit_of_measurement(self) -> str:
    def device_class(self) -> SensorDeviceClass:
    async def async_added_to_hass(self):
    def _async_update_shade_from_group(self):
    async def async_update(self):
async def async_setup_entry(hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback):
def get_signal_device_class(shade: BaseShade) -> SensorDeviceClass:
def get_signal_native_unit(shade: BaseShade) -> str
