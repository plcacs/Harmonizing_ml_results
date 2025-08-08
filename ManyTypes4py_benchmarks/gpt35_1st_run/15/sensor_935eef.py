def parse_datetime(value: Any) -> Any:
    ...

async def async_setup_entry(hass: HomeAssistant, entry: AzureDevOpsConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class AzureDevOpsBuildSensor(AzureDevOpsEntity, SensorEntity):
    def __init__(self, coordinator: AzureDevOpsDataUpdateCoordinator, description: AzureDevOpsBuildSensorEntityDescription, item_key: int) -> None:
        ...

    @property
    def build(self) -> Build:
        ...

    @property
    def native_value(self) -> StateType:
        ...

    @property
    def extra_state_attributes(self) -> Mapping[str, Any]:
        ...

class AzureDevOpsWorkItemSensor(AzureDevOpsEntity, SensorEntity):
    def __init__(self, coordinator: AzureDevOpsDataUpdateCoordinator, description: AzureDevOpsWorkItemSensorEntityDescription, wits_key: int, state_key: int) -> None:
        ...

    @property
    def work_item_type(self) -> WorkItemTypeAndState:
        ...

    @property
    def work_item_state(self) -> WorkItemState:
        ...

    @property
    def native_value(self) -> StateType:
        ...
