async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class ISYAuxControlNumberEntity(ISYAuxControlEntity, NumberEntity):
    _attr_mode: NumberMode = NumberMode.SLIDER

    @property
    def native_value(self) -> Any:
        ...

    async def async_set_native_value(self, value: Any) -> None:
        ...

class ISYVariableNumberEntity(NumberEntity):
    _attr_has_entity_name: bool = False
    _attr_should_poll: bool = False

    def __init__(self, node: Node, unique_id: str, description: NumberEntityDescription, device_info: DeviceInfo, init_entity: bool = False) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def async_on_update(self, event: NodeChangedEvent) -> None:
        ...

    @property
    def native_value(self) -> Any:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    async def async_set_native_value(self, value: Any) -> None:
        ...

class ISYBacklightNumberEntity(ISYAuxControlEntity, RestoreNumber):
    _assumed_state: bool = True

    def __init__(self, node: Node, control: str, unique_id: str, description: NumberEntityDescription, device_info: DeviceInfo) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @callback
    def async_on_memory_write(self, event: NodeChangedEvent, key: str) -> None:
        ...

    async def async_set_native_value(self, value: Any) -> None:
        ...
