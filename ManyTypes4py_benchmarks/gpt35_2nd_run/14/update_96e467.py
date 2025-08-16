from __future__ import annotations
from typing import Any, Dict, List, Union, Callable

async def async_setup_entry(hass: HomeAssistant, entry: EzvizConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class EzvizUpdateEntity(EzvizEntity, UpdateEntity):
    def __init__(self, coordinator: EzvizDataUpdateCoordinator, serial: str, sensor: str, description: UpdateEntityDescription) -> None:
        ...

    @property
    def installed_version(self) -> str:
        ...

    @property
    def in_progress(self) -> bool:
        ...

    @property
    def latest_version(self) -> str:
        ...

    def release_notes(self) -> Union[str, None]:
        ...

    @property
    def update_percentage(self) -> Union[int, None]:
        ...

    async def async_install(self, version: str, backup: bool, **kwargs: Any) -> None:
        ...
