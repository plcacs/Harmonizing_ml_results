"""Support for the Airzone Cloud select."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Dict, List, Optional, Tuple
from aioairzone_cloud.common import AirQualityMode, OperationMode
from aioairzone_cloud.const import (
    API_AQ_MODE_CONF,
    API_MODE,
    API_VALUE,
    AZD_AQ_MODE_CONF,
    AZD_MASTER,
    AZD_MODE,
    AZD_MODES,
    AZD_ZONES,
)
from homeassistant.components.select import SelectEntity, SelectEntityDescription
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import AirzoneCloudConfigEntry, AirzoneUpdateCoordinator
from .entity import AirzoneEntity, AirzoneZoneEntity

@dataclass(frozen=True, kw_only=True)
class AirzoneSelectDescription(SelectEntityDescription):
    """Class to describe an Airzone select entity."""
    options_fn: Callable[[Dict[str, Any], Dict[str, Any]], List[str]] = lambda zone_data, value: list(value)

AIR_QUALITY_MAP: Final[Dict[str, AirQualityMode]] = {
    'off': AirQualityMode.OFF,
    'on': AirQualityMode.ON,
    'auto': AirQualityMode.AUTO,
}
MODE_MAP: Final[Dict[str, OperationMode]] = {
    'cool': OperationMode.COOLING,
    'dry': OperationMode.DRY,
    'fan': OperationMode.VENTILATION,
    'heat': OperationMode.HEATING,
    'heat_cool': OperationMode.AUTO,
    'stop': OperationMode.STOP,
}

def main_zone_options(zone_data: Dict[str, Any], options: Dict[str, OperationMode]) -> List[str]:
    """Filter available modes."""
    modes = zone_data.get(AZD_MODES, [])
    return [k for k, v in options.items() if v in modes]

MAIN_ZONE_SELECT_TYPES: Tuple[AirzoneSelectDescription, ...] = (
    AirzoneSelectDescription(
        api_param=API_MODE,
        key=AZD_MODE,
        options_dict=MODE_MAP,
        options_fn=main_zone_options,
        translation_key='modes',
    ),
)
ZONE_SELECT_TYPES: Tuple[AirzoneSelectDescription, ...] = (
    AirzoneSelectDescription(
        api_param=API_AQ_MODE_CONF,
        entity_category=EntityCategory.CONFIG,
        key=AZD_AQ_MODE_CONF,
        options=list(AIR_QUALITY_MAP),
        options_dict=AIR_QUALITY_MAP,
        translation_key='air_quality',
    ),
)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: AirzoneCloudConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Add Airzone Cloud select from a config_entry."""
    coordinator: AirzoneUpdateCoordinator = entry.runtime_data
    zones: Dict[str, Dict[str, Any]] = coordinator.data.get(AZD_ZONES, {})
    entities: List[AirzoneZoneSelect] = [
        AirzoneZoneSelect(coordinator, description, zone_id, zone_data)
        for description in MAIN_ZONE_SELECT_TYPES
        for zone_id, zone_data in zones.items()
        if description.key in zone_data and zone_data.get(AZD_MASTER)
    ]
    entities.extend(
        AirzoneZoneSelect(coordinator, description, zone_id, zone_data)
        for description in ZONE_SELECT_TYPES
        for zone_id, zone_data in zones.items()
        if description.key in zone_data
    )
    async_add_entities(entities)

class AirzoneBaseSelect(AirzoneEntity, SelectEntity):
    """Define an Airzone Cloud select."""

    @callback
    def _handle_coordinator_update(self) -> None:
        """Update attributes when the coordinator updates."""
        self._async_update_attrs()
        super()._handle_coordinator_update()

    def _get_current_option(self) -> Optional[str]:
        """Get current selected option."""
        value: Any = self.get_airzone_value(self.entity_description.key)
        return self.values_dict.get(value)

    @callback
    def _async_update_attrs(self) -> None:
        """Update select attributes."""
        self._attr_current_option = self._get_current_option()

class AirzoneZoneSelect(AirzoneZoneEntity, AirzoneBaseSelect):
    """Define an Airzone Cloud Zone select."""

    def __init__(
        self,
        coordinator: AirzoneUpdateCoordinator,
        description: AirzoneSelectDescription,
        zone_id: str,
        zone_data: Dict[str, Any],
    ) -> None:
        """Initialize."""
        super().__init__(coordinator, zone_id, zone_data)
        self._attr_unique_id: str = f'{zone_id}_{description.key}'
        self.entity_description: AirzoneSelectDescription = description
        self._attr_options: List[str] = self.entity_description.options_fn(zone_data, description.options_dict)
        self.values_dict: Dict[Any, str] = {v: k for k, v in description.options_dict.items()}
        self._async_update_attrs()

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        param: str = self.entity_description.api_param
        value: Any = self.entity_description.options_dict[option]
        params: Dict[str, Dict[str, Any]] = {
            param: {API_VALUE: value}
        }
        await self._async_update_params(params)
