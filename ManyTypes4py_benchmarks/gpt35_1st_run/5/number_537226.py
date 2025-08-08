from aioautomower.model import MowerAttributes, WorkArea
from aioautomower.session import AutomowerSession
from homeassistant.components.number import NumberEntity, NumberEntityDescription
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .coordinator import AutomowerDataUpdateCoordinator
from .entity import AutomowerControlEntity, WorkAreaControlEntity, _work_area_translation_key, handle_sending_exception
from typing import TYPE_CHECKING, Any

@callback
def _async_get_cutting_height(data: MowerAttributes) -> int:
    ...

async def async_set_work_area_cutting_height(coordinator: AutomowerDataUpdateCoordinator, mower_id: str, cheight: int, work_area_id: int) -> None:
    ...

async def async_set_cutting_height(session: AutomowerSession, mower_id: str, cheight: int) -> None:
    ...

class AutomowerNumberEntityDescription(NumberEntityDescription):
    ...

class WorkAreaNumberEntityDescription(NumberEntityDescription):
    ...

async def async_setup_entry(hass: HomeAssistant, entry: AutomowerConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class AutomowerNumberEntity(AutomowerControlEntity, NumberEntity):
    ...

class WorkAreaNumberEntity(WorkAreaControlEntity, NumberEntity):
    ...
