    def __init__(self, gateway_id: GatewayId, gateway: BaseAsyncGateway, node_id: int) -> None:
    def _node(self) -> Sensor:
    def sketch_name(self) -> str:
    def sketch_version(self) -> str:
    def node_name(self) -> str:
    def device_info(self) -> DeviceInfo:
    def extra_state_attributes(self) -> dict[str, Any]:
    def _async_update_callback(self) -> None:
    async def async_update_callback(self) -> None:
    async def async_added_to_hass(self) -> None:
def get_mysensors_devices(hass: HomeAssistant, domain: str) -> dict[str, Any]:
    def __init__(self, gateway_id: GatewayId, gateway: BaseAsyncGateway, node_id: int, child_id: int, value_type: int) -> None:
    def dev_id(self) -> DevId:
    def _child(self) -> ChildSensor:
    def unique_id(self) -> str:
    def name(self) -> str:
    async def async_will_remove_from_hass(self) -> None:
    def available(self) -> bool:
    def extra_state_attributes(self) -> dict[str, Any]:
    def _async_update(self) -> None:
    def _async_update_callback(self) -> None:
    async def async_added_to_hass(self) -> None:
