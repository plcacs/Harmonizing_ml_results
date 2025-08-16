async def _async_create_entities(hass: HomeAssistant, config: ConfigType) -> List[SwitchTemplate]:
async def async_setup_platform(hass: HomeAssistant, config: ConfigType, async_add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
def async_create_preview_switch(hass: HomeAssistant, name: str, config: ConfigType) -> SwitchTemplate:
class SwitchTemplate(TemplateEntity, SwitchEntity, RestoreEntity):
    def __init__(self, hass: HomeAssistant, object_id: str, config: ConfigType, unique_id: str) -> None:
    async def async_added_to_hass(self) -> None:
    def _update_state(self, result: Any) -> None:
    def _async_setup_templates(self) -> None:
    @property
    def is_on(self) -> bool:
    async def async_turn_on(self, **kwargs) -> None:
    async def async_turn_off(self, **kwargs) -> None:
