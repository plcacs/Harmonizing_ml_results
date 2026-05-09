from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple
from aioautomower.model import MowerAttributes, WorkArea
from aioautomower.session import AutomowerSession
from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import AutomowerConfigEntry
from .coordinator import AutomowerDataUpdateCoordinator
from .entity import AutomowerControlEntity, WorkAreaControlEntity, _work_area_translation_key, handle_sending_exception

@callback
def _async_get_cutting_height(data: AutomowerAttributes) -> Optional[int]:
    """Return the cutting height."""
    return data.settings.cutting_height

async def async_set_work_area_cutting_height(coordinator: AutomowerDataUpdateCoordinator, mower_id: str, cheight: int, work_area_id: str) -> Awaitable[None]:
    """Set cutting height for work area."""
    await coordinator.api.commands.workarea_settings(mower_id, int(cheight), work_area_id)

async def async_set_cutting_height(session: AutomowerSession, mower_id: str, cheight: int) -> Awaitable[None]:
    """Set cutting height."""
    await session.commands.set_cutting_height(mower_id, int(cheight))

@dataclass(frozen=True, kw_only=True)
class AutomowerNumberEntityDescription(NumberEntityDescription):
    """Describes Automower number entity."""
    exists_fn: Callable[[AutomowerAttributes], bool]
    value_fn: Callable[[AutomowerAttributes], int]
    set_value_fn: Callable[[AutomowerSession, str, int], Awaitable[None]]

MOWER_NUMBER_TYPES = (
    AutomowerNumberEntityDescription(
        key='cutting_height',
        translation_key='cutting_height',
        entity_registry_enabled_default=False,
        entity_category=EntityCategory.CONFIG,
        native_min_value=1,
        native_max_value=9,
        exists_fn=lambda data: data.settings.cutting_height is not None,
        value_fn=_async_get_cutting_height,
        set_value_fn=async_set_cutting_height,
    ),
)

@dataclass(frozen=True, kw_only=True)
class WorkAreaNumberEntityDescription(NumberEntityDescription):
    """Describes Automower work area number entity."""
    translation_key_fn: Callable[[str, str], str]
    value_fn: Callable[[WorkArea], int]
    set_value_fn: Callable[[AutomowerDataUpdateCoordinator, str, int, str], Awaitable[None]]

WORK_AREA_NUMBER_TYPES = (
    WorkAreaNumberEntityDescription(
        key='cutting_height_work_area',
        translation_key_fn=_work_area_translation_key,
        entity_category=EntityCategory.CONFIG,
        native_unit_of_measurement=PERCENTAGE,
        value_fn=lambda data: data.cutting_height,
        set_value_fn=async_set_work_area_cutting_height,
    ),
)

class AutomowerNumberEntity(AutomowerControlEntity, NumberEntity):
    """Defining the AutomowerNumberEntity with AutomowerNumberEntityDescription."""

    def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: AutomowerNumberEntityDescription):
        """Set up AutomowerNumberEntity."""
        super().__init__(mower_id, coordinator)
        self.entity_description = description
        self._attr_unique_id = f'{mower_id}_{description.key}'

    @property
    def native_value(self) -> int:
        """Return the state of the number."""
        return self.entity_description.value_fn(self.mower_attributes)

    @handle_sending_exception()
    async def async_set_native_value(self, value: int) -> None:
        """Change to new number value."""
        await self.entity_description.set_value_fn(self.coordinator.api, self.mower_id, value)

class WorkAreaNumberEntity(WorkAreaControlEntity, NumberEntity):
    """Defining the WorkAreaNumberEntity with WorkAreaNumberEntityDescription."""

    def __init__(self, mower_id: str, coordinator: AutomowerDataUpdateCoordinator, description: WorkAreaNumberEntityDescription, work_area_id: str):
        """Set up AutomowerNumberEntity."""
        super().__init__(mower_id, coordinator, work_area_id)
        self.entity_description = description
        self._attr_unique_id = f'{mower_id}_{work_area_id}_{description.key}'
        self._attr_translation_placeholders = {'work_area': self.work_area_attributes.name}

    @property
    def translation_key(self) -> str:
        """Return the translation key of the work area."""
        return self.entity_description.translation_key_fn(self.work_area_id, self.entity_description.key)

    @property
    def native_value(self) -> int:
        """Return the state of the number."""
        return self.entity_description.value_fn(self.work_area_attributes)

    @handle_sending_exception(poll_after_sending=True)
    async def async_set_native_value(self, value: int) -> None:
        """Change to new number value."""
        await self.entity_description.set_value_fn(self.coordinator, self.mower_id, value, self.work_area_id)
