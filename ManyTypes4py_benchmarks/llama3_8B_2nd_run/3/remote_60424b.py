async def async_setup_entry(hass: HomeAssistant, entry: HarmonyConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HarmonyRemote(HarmonyEntity, RemoteEntity, RestoreEntity):
    """Remote representation used to control a Harmony device."""
    _attr_supported_features: RemoteEntityFeature = RemoteEntityFeature.ACTIVITY
    _attr_name: str | None = None

    def __init__(self, data: HarmonyData, activity: str, delay_secs: int, out_path: str) -> None:
        ...

    async def _async_update_options(self, data: VolDictType) -> None:
        ...

    def _setup_callbacks(self) -> None:
        ...

    @callback
    def async_new_activity_finished(self, activity_info: str) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    @property
    def current_activity(self) -> str | None:
        ...

    @property
    def activity_list(self) -> list[str]:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @callback
    def async_new_activity(self, activity_info: tuple[str, str]) -> None:
        ...

    async def async_new_config(self, _=None) -> None:
        ...

    async def async_turn_on(self, **kwargs: VolDictType) -> None:
        ...

    async def async_turn_off(self, **kwargs: VolDictType) -> None:
        ...

    async def async_send_command(self, command: str, **kwargs: VolDictType) -> None:
        ...

    async def change_channel(self, channel: int) -> None:
        ...

    async def sync(self) -> None:
        ...
