from __future__ import annotations
from typing import Any

def _component_to_unique_id(server_id: str, component: str, instance_num: int) -> str:
def _component_to_translation_key(component: str) -> str:
async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

    def __init__(self, server_id: str, instance_num: int, instance_name: str, component_name: str, hyperion_client: client.HyperionClient) -> None:
    def is_on(self) -> bool:
    def available(self) -> bool:
    async def _async_send_set_component(self, value: bool) -> None:
    async def async_turn_on(self, **kwargs: Any) -> None:
    async def async_turn_off(self, **kwargs: Any) -> None:
    def _update_components(self, _=None) -> None:
    async def async_added_to_hass(self) -> None:
    async def async_will_remove_from_hass(self) -> None:
