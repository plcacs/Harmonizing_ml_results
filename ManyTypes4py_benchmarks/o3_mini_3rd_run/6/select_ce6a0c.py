from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Optional, Dict, List
from aioairzone.common import GrilleAngle, OperationMode, SleepTimeout
from aioairzone.const import (
    API_COLD_ANGLE,
    API_HEAT_ANGLE,
    API_MODE,
    API_SLEEP,
    AZD_COLD_ANGLE,
    AZD_HEAT_ANGLE,
    AZD_MASTER,
    AZD_MODE,
    AZD_MODES,
    AZD_SLEEP,
    AZD_ZONES,
)
from homeassistant.components.select import SelectEntity, SelectEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import AirzoneConfigEntry, AirzoneUpdateCoordinator
from .entity import AirzoneEntity, AirzoneZoneEntity

@dataclass(frozen=True, kw_only=True)
class AirzoneSelectDescription(SelectEntityDescription):
    options_fn: Callable[[Dict[str, Any], Any], List[str]] = lambda zone_data, value: list(value)

GRILLE_ANGLE_DICT: Final[Dict[str, GrilleAngle]] = {
    '90deg': GrilleAngle.DEG_90,
    '50deg': GrilleAngle.DEG_50,
    '45deg': GrilleAngle.DEG_45,
    '40deg': GrilleAngle.DEG_40,
}
MODE_DICT: Final[Dict[str, OperationMode]] = {
    'cool': OperationMode.COOLING,
    'dry': OperationMode.DRY,
    'fan': OperationMode.FAN,
    'heat': OperationMode.HEATING,
    'heat_cool': OperationMode.AUTO,
    'stop': OperationMode.STOP,
}
SLEEP_DICT: Final[Dict[str, SleepTimeout]] = {
    'off': SleepTimeout.SLEEP_OFF,
    '30m': SleepTimeout.SLEEP_30,
    '60m': SleepTimeout.SLEEP_60,
    '90m': SleepTimeout.SLEEP_90,
}

def main_zone_options(zone_data: Dict[str, Any], options: Dict[str, Any]) -> List[str]:
    modes = zone_data.get(AZD_MODES, [])
    return [k for k, v in options.items() if v in modes]

MAIN_ZONE_SELECT_TYPES: Final[tuple[AirzoneSelectDescription, ...]] = (
    AirzoneSelectDescription(
        api_param=API_MODE,
        key=AZD_MODE,
        options_dict=MODE_DICT,
        options_fn=main_zone_options,
        translation_key='modes'
    ),
)
ZONE_SELECT_TYPES: Final[tuple[AirzoneSelectDescription, ...]] = (
    AirzoneSelectDescription(
        api_param=API_COLD_ANGLE,
        entity_category=EntityCategory.CONFIG,
        key=AZD_COLD_ANGLE,
        options=list(GRILLE_ANGLE_DICT),
        options_dict=GRILLE_ANGLE_DICT,
        translation_key='grille_angles'
    ),
    AirzoneSelectDescription(
        api_param=API_HEAT_ANGLE,
        entity_category=EntityCategory.CONFIG,
        key=AZD_HEAT_ANGLE,
        options=list(GRILLE_ANGLE_DICT),
        options_dict=GRILLE_ANGLE_DICT,
        translation_key='heat_angles'
    ),
    AirzoneSelectDescription(
        api_param=API_SLEEP,
        entity_category=EntityCategory.CONFIG,
        key=AZD_SLEEP,
        options=list(SLEEP_DICT),
        options_dict=SLEEP_DICT,
        translation_key='sleep_times'
    ),
)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    coordinator: AirzoneUpdateCoordinator = entry.runtime_data
    added_zones: set[str] = set()

    def _async_entity_listener() -> None:
        zones_data: Dict[str, Any] = coordinator.data.get(AZD_ZONES, {})
        received_zones: set[str] = set(zones_data)
        new_zones: set[str] = received_zones - added_zones
        if new_zones:
            entities = [
                AirzoneZoneSelect(coordinator, description, entry, system_zone_id, zones_data.get(system_zone_id))
                for system_zone_id in new_zones
                for description in MAIN_ZONE_SELECT_TYPES
                if (description.key in zones_data.get(system_zone_id, {})) and (zones_data.get(system_zone_id, {}).get(AZD_MASTER) is True)
            ]
            entities += [
                AirzoneZoneSelect(coordinator, description, entry, system_zone_id, zones_data.get(system_zone_id))
                for system_zone_id in new_zones
                for description in ZONE_SELECT_TYPES
                if description.key in zones_data.get(system_zone_id, {})
            ]
            async_add_entities(entities)
            added_zones.update(new_zones)

    entry.async_on_unload(coordinator.async_add_listener(_async_entity_listener))
    _async_entity_listener()

class AirzoneBaseSelect(AirzoneEntity, SelectEntity):
    @callback
    def _handle_coordinator_update(self) -> None:
        self._async_update_attrs()
        super()._handle_coordinator_update()

    def _get_current_option(self) -> Optional[str]:
        value: Any = self.get_airzone_value(self.entity_description.key)
        return self.values_dict.get(value)

    @callback
    def _async_update_attrs(self) -> None:
        self._attr_current_option = self._get_current_option()

class AirzoneZoneSelect(AirzoneZoneEntity, AirzoneBaseSelect):
    def __init__(self, coordinator: AirzoneUpdateCoordinator, description: AirzoneSelectDescription, entry: ConfigEntry, system_zone_id: str, zone_data: Dict[str, Any]) -> None:
        super().__init__(coordinator, entry, system_zone_id, zone_data)
        self._attr_unique_id = f'{self._attr_unique_id}_{system_zone_id}_{description.key}'
        self.entity_description = description
        self._attr_options = self.entity_description.options_fn(zone_data, description.options_dict)
        self.values_dict: Dict[Any, str] = {v: k for k, v in description.options_dict.items()}
        self._async_update_attrs()

    async def async_select_option(self, option: str) -> None:
        param: Any = self.entity_description.api_param
        value: Any = self.entity_description.options_dict[option]
        await self._async_update_hvac_params({param: value})