def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ISYSensorEntity(ISYNodeEntity, SensorEntity):
    @property
    def target(self) -> Node:
        ...

    @property
    def target_value(self) -> Any:
        ...

    @property
    def raw_unit_of_measurement(self) -> Any:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def native_unit_of_measurement(self) -> Any:
        ...

class ISYAuxSensorEntity(ISYSensorEntity):
    def __init__(self, node: Node, control: str, enabled_default: bool, unique_id: str, device_info: DeviceInfo = None) -> None:
        ...

    @property
    def target(self) -> NodeProperty:
        ...

    @property
    def target_value(self) -> Any:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def async_on_update(self, event: NodeChangedEvent) -> None:
        ...

    @property
    def available(self) -> bool:
        ...
