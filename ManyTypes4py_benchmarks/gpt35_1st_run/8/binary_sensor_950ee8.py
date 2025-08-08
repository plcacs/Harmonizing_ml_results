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
        super().__init__(node, device_info=device_info)
        self._attr_device_class = force_device_class

    @property
    def is_on(self) -> Any:
        if self._node.status == ISY_VALUE_UNKNOWN:
            return None
        return bool(self._node.status)

class ISYInsteonBinarySensorEntity(ISYBinarySensorEntity):
    def __init__(self, node: Node, force_device_class: BinarySensorDeviceClass = None, unknown_state: Any = None, device_info: DeviceInfo = None) -> None:
        super().__init__(node, force_device_class, device_info=device_info)
        self._negative_node: Node = None
        self._heartbeat_device: ISYBinarySensorHeartbeat = None
        self._computed_state: Any = None
        self._status_was_unknown: bool = False

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._node.control_events.subscribe(self._async_positive_node_control_handler)
        if self._negative_node is not None:
            self._negative_node.control_events.subscribe(self._async_negative_node_control_handler)

    def add_heartbeat_device(self, entity: ISYBinarySensorHeartbeat) -> None:
        self._heartbeat_device = entity

    def _async_heartbeat(self) -> None:
        if self._heartbeat_device is not None:
            self._heartbeat_device.async_heartbeat()

    def add_negative_node(self, child: Node) -> None:
        self._negative_node = child
        if self._negative_node.status != ISY_VALUE_UNKNOWN and self._negative_node.status == self._node.status:
            self._computed_state = None

    @callback
    def _async_negative_node_control_handler(self, event: Any) -> None:
        if event.control == CMD_ON:
            _LOGGER.debug('Sensor %s turning Off via the Negative node sending a DON command', self.name)
            self._computed_state = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @callback
    def _async_positive_node_control_handler(self, event: Any) -> None:
        if event.control == CMD_ON:
            _LOGGER.debug('Sensor %s turning On via the Primary node sending a DON command', self.name)
            self._computed_state = True
            self.async_write_ha_state()
            self._async_heartbeat()
        if event.control == CMD_OFF:
            _LOGGER.debug('Sensor %s turning Off via the Primary node sending a DOF command', self.name)
            self._computed_state = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @callback
    def async_on_update(self, event: Any) -> None:
        if self._status_was_unknown and self._computed_state is None:
            self._computed_state = bool(self._node.status)
            self._status_was_unknown = False
            self.async_write_ha_state()
            self._async_heartbeat()

    @property
    def is_on(self) -> Any:
        if self._computed_state is None:
            return None
        if self.device_class in (BinarySensorDeviceClass.LIGHT, BinarySensorDeviceClass.MOISTURE):
            return not self._computed_state
        return self._computed_state

class ISYBinarySensorHeartbeat(ISYNodeEntity, BinarySensorEntity, RestoreEntity):
    def __init__(self, node: Node, parent_device: ISYInsteonBinarySensorEntity, device_info: DeviceInfo = None) -> None:
        super().__init__(node, device_info=device_info)
        self._parent_device = parent_device
        self._heartbeat_timer: CALLBACK_TYPE = None
        self._computed_state: Any = None
        if self.state is None:
            self._computed_state = False

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._node.control_events.subscribe(self._heartbeat_node_control_handler)
        self._restart_timer()
        if (last_state := (await self.async_get_last_state())) is not None:
            if last_state.state == STATE_ON:
                self._computed_state = True

    def _heartbeat_node_control_handler(self, event: Any) -> None:
        if event.control in (CMD_ON, CMD_OFF):
            self.async_heartbeat()

    @callback
    def async_heartbeat(self) -> None:
        self._computed_state = False
        self._restart_timer()
        self.async_write_ha_state()

    def _restart_timer(self) -> None:
        if self._heartbeat_timer is not None:
            self._heartbeat_timer()
            self._heartbeat_timer = None

        @callback
        def timer_elapsed(now: datetime) -> None:
            self._computed_state = True
            self._heartbeat_timer = None
            self.async_write_ha_state()
        self._heartbeat_timer = async_call_later(self.hass, timedelta(hours=25), timer_elapsed)

    @callback
    def async_on_update(self, event: Any) -> None:
        pass

    @property
    def is_on(self) -> Any:
        return bool(self._computed_state)

    @property
    def extra_state_attributes(self) -> dict:
        attr: dict = super().extra_state_attributes
        attr['parent_entity_id'] = self._parent_device.entity_id
        return attr

class ISYBinarySensorProgramEntity(ISYProgramEntity, BinarySensorEntity):
    @property
    def is_on(self) -> bool:
        return bool(self._node.status)
