def _detect_device_type_and_class(node: Node) -> tuple:
    try:
        device_type: str = node.type
    except AttributeError:
        return (None, None)
    if node.protocol == PROTO_ZWAVE:
        device_type = f'Z{node.zwave_props.category}'
        for device_class, values in BINARY_SENSOR_DEVICE_TYPES_ZWAVE.items():
            if node.zwave_props.category in values:
                return (device_class, device_type)
        return (None, device_type)
    for device_class, values in BINARY_SENSOR_DEVICE_TYPES_ISY.items():
        if any((device_type.startswith(t) for t in values)):
            return (device_class, device_type)
    return (None, device_type)

class ISYBinarySensorEntity(ISYNodeEntity, BinarySensorEntity):
    def __init__(self, node: Node, force_device_class: BinarySensorDeviceClass = None, unknown_state: Any = None, device_info: DeviceInfo = None) -> None:
        ...

class ISYInsteonBinarySensorEntity(ISYBinarySensorEntity):
    def __init__(self, node: Node, force_device_class: BinarySensorDeviceClass = None, unknown_state: Any = None, device_info: DeviceInfo = None) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def add_heartbeat_device(self, entity: ISYBinarySensorEntity) -> None:
        ...

    def _async_heartbeat(self) -> None:
        ...

    def add_negative_node(self, child: Node) -> None:
        ...

    @callback
    def _async_negative_node_control_handler(self, event: Any) -> None:
        ...

    @callback
    def _async_positive_node_control_handler(self, event: Any) -> None:
        ...

    @callback
    def async_on_update(self, event: Any) -> None:
        ...

    @property
    def is_on(self) -> Any:
        ...

class ISYBinarySensorHeartbeat(ISYNodeEntity, BinarySensorEntity, RestoreEntity):
    def __init__(self, node: Node, parent_device: ISYBinarySensorEntity, device_info: DeviceInfo = None) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    def _heartbeat_node_control_handler(self, event: Any) -> None:
        ...

    @callback
    def async_heartbeat(self) -> None:
        ...

    def _restart_timer(self) -> None:
        ...

    @callback
    def async_on_update(self, event: Any) -> None:
        ...

    @property
    def is_on(self) -> Any:
        ...

    @property
    def extra_state_attributes(self) -> dict:
        ...

class ISYBinarySensorProgramEntity(ISYProgramEntity, BinarySensorEntity):
    @property
    def is_on(self) -> Any:
        ...
