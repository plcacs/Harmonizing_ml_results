    def __init__(self, *, hass: HomeAssistant, logger: Logger, domain: str, platform_name: str, platform: Any, scan_interval: timedelta, entity_namespace: str) -> None:

    async def async_setup_platform(self, hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:

    def setup_platform(self, hass: HomeAssistant, config: ConfigType, add_entities: Callable, discovery_info: DiscoveryInfoType = None) -> None:

    async def async_setup_entry(self, hass: HomeAssistant, entry: Any, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

    async def async_setup(self, platform_config: ConfigType, discovery_info: DiscoveryInfoType = None) -> None:

    def _get_parallel_updates_semaphore(self, entity_has_sync_update: bool) -> asyncio.Semaphore:

    async def async_shutdown(self) -> None:

    async def async_cancel_retry_setup(self) -> None:

    async def async_load_translations(self) -> None:

    def _schedule_add_entities(self, new_entities: Iterable, update_before_add: bool = False) -> None:

    async def _async_schedule_add_entities(self, new_entities: Iterable, update_before_add: bool = False) -> None:

    @callback
    def _async_schedule_add_entities_for_entry(self, new_entities: Iterable, update_before_add: bool = False, *, config_subentry_id: str = None) -> None:

    def add_entities(self, new_entities: Iterable, update_before_add: bool = False) -> None:

    async def _async_add_and_update_entities(self, coros: List[Coroutine], entities: List[Entity], timeout: int) -> None:

    async def _async_add_entities(self, coros: List[Coroutine], entities: List[Entity], timeout: int) -> None:

    async def async_add_entities(self, new_entities: Iterable, update_before_add: bool = False, *, config_subentry_id: str = None) -> None:

    async def async_reset(self) -> None:

    @callback
    def async_unsub_polling(self) -> None:

    @callback
    def async_prepare(self) -> None:

    async def async_destroy(self) -> None:

    async def async_remove_entity(self, entity_id: str) -> None:

    async def async_extract_from_service(self, service_call: ServiceCall, expand_group: bool = True) -> List[Entity]:

    @callback
    def async_register_entity_service(self, name: str, schema: VolSchemaType, func: Callable, required_features: int = None, supports_response: SupportsResponse = SupportsResponse.NONE) -> None:

    async def _async_update_entity_states(self) -> None:

@callback
def async_get_current_platform() -> EntityPlatform:

@callback
def async_get_platforms(hass: HomeAssistant, integration_name: str) -> List[EntityPlatform]:
