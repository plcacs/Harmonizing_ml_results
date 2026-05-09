async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType | None = None) -> None:
    # ...

async def async_setup_entry(hass: HomeAssistant, entry: ScrapeConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    # ...

class ScrapeSensor(CoordinatorEntity[ScrapeCoordinator], ManualTriggerSensorEntity):
    """Representation of a web scrape sensor."""

    def __init__(self, hass: HomeAssistant, coordinator: ScrapeCoordinator, trigger_entity_config: dict, select: str, attr: str | None, index: str, value_template: Template | None, yaml: bool) -> None:
        # ...

    def _extract_value(self) -> str | None:
        # ...

    async def async_added_to_hass(self) -> None:
        # ...

    def _async_update_from_rest_data(self) -> None:
        # ...

    @property
    def available(self) -> bool:
        # ...

    @callback
    def _handle_coordinator_update(self) -> None:
        # ...
