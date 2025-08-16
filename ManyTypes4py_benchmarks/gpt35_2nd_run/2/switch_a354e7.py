from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class LiteJetSwitch(SwitchEntity):
    def __init__(self, entry_id: str, system: LiteJet, i: int, name: str) -> None:

    async def async_added_to_hass(self) -> None:

    async def async_will_remove_from_hass(self) -> None:

    def _on_switch_pressed(self) -> None:

    def _on_switch_released(self) -> None:

    def _on_connected_changed(self, connected: bool, reason: str) -> None:

    @property
    def extra_state_attributes(self) -> dict[str, Any]:

    async def async_turn_on(self, **kwargs: Any) -> None:

    async def async_turn_off(self, **kwargs: Any) -> None:
