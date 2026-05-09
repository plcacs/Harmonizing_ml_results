async def async_setup_entry(hass: HomeAssistant, entry: Control4ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    # ...

    async def async_update_data_non_dimmer() -> dict[int, dict[str, Any]]:
        # ...

    async def async_update_data_dimmer() -> dict[int, dict[str, Any]]:
        # ...

    non_dimmer_coordinator: DataUpdateCoordinator[dict[int, dict[str, Any]]] = DataUpdateCoordinator[dict[int, dict[str, Any]]](hass, _LOGGER, name='light', update_method=async_update_data_non_dimmer, update_interval=timedelta(seconds=runtime_data.scan_interval))
    dimmer_coordinator: DataUpdateCoordinator[dict[int, dict[str, Any]]] = DataUpdateCoordinator[dict[int, dict[str, Any]]](hass, _LOGGER, name='light', update_method=async_update_data_dimmer, update_interval=timedelta(seconds=runtime_data.scan_interval))

    # ...

    entity_list: list[Control4Light] = []

    for item in items_of_category:
        # ...

        if item_id in dimmer_coordinator.data:
            item_is_dimmer: bool = True
            item_coordinator: DataUpdateCoordinator[dict[int, dict[str, Any]]] = dimmer_coordinator
        elif item_id in non_dimmer_coordinator.data:
            item_is_dimmer: bool = False
            item_coordinator: DataUpdateCoordinator[dict[int, dict[str, Any]]] = non_dimmer_coordinator
        else:
            # ...

        entity_list.append(Control4Light(runtime_data, item_coordinator, item_name, item_id, item_device_name, item_manufacturer, item_model, item_parent_id, item_is_dimmer))

    async_add_entities(entity_list, True)

class Control4Light(Control4Entity, LightEntity):
    """Control4 light entity."""

    _attr_has_entity_name: bool = True

    def __init__(self, runtime_data: Control4RuntimeData, coordinator: DataUpdateCoordinator[dict[int, dict[str, Any]]], name: str, idx: int, device_name: str, device_manufacturer: str, device_model: str, device_id: str, is_dimmer: bool) -> None:
        # ...

    def _create_api_object(self) -> C4Light:
        # ...

    @property
    def is_on(self) -> bool:
        # ...

    @property
    def brightness(self) -> int | None:
        # ...

    @property
    def supported_features(self) -> LightEntityFeature:
        # ...

    async def async_turn_on(self, **kwargs: Any) -> None:
        # ...

    async def async_turn_off(self, **kwargs: Any) -> None:
        # ...
