async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class FreeboxSensor(SensorEntity):
    def __init__(self, router: FreeboxRouter, description: SensorEntityDescription) -> None:
        ...

    @callback
    def async_update_state(self) -> None:
        ...

    @callback
    def async_on_demand_update(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

class FreeboxCallSensor(FreeboxSensor):
    def __init__(self, router: FreeboxRouter, description: SensorEntityDescription) -> None:
        ...

    @callback
    def async_update_state(self) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

class FreeboxDiskSensor(FreeboxSensor):
    def __init__(self, router: FreeboxRouter, disk: dict, partition: dict, description: SensorEntityDescription) -> None:
        ...

    @callback
    def async_update_state(self) -> None:
        ...

class FreeboxBatterySensor(FreeboxHomeEntity, SensorEntity):
    @property
    def native_value(self) -> Any:
        ...
