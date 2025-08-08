from __future__ import annotations
from typing import Any

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

class HomematicipBlindModule(HomematicipGenericEntity, CoverEntity):

    @property
    def current_cover_position(self) -> int:

    @property
    def current_cover_tilt_position(self) -> int:

    async def async_set_cover_position(self, **kwargs: Any) -> None:

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:

    @property
    def is_closed(self) -> bool:

    async def async_open_cover(self, **kwargs: Any) -> None:

    async def async_close_cover(self, **kwargs: Any) -> None:

    async def async_stop_cover(self, **kwargs: Any) -> None:

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:

class HomematicipMultiCoverShutter(HomematicipGenericEntity, CoverEntity):

    @property
    def current_cover_position(self) -> int:

    async def async_set_cover_position(self, **kwargs: Any) -> None:

    @property
    def is_closed(self) -> bool:

    async def async_open_cover(self, **kwargs: Any) -> None:

    async def async_close_cover(self, **kwargs: Any) -> None:

    async def async_stop_cover(self, **kwargs: Any) -> None:

class HomematicipCoverShutter(HomematicipMultiCoverShutter, CoverEntity):

class HomematicipMultiCoverSlats(HomematicipMultiCoverShutter, CoverEntity):

    @property
    def current_cover_tilt_position(self) -> int:

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:

class HomematicipCoverSlats(HomematicipMultiCoverSlats, CoverEntity):

class HomematicipGarageDoorModule(HomematicipGenericEntity, CoverEntity):

    @property
    def current_cover_position(self) -> int:

    @property
    def is_closed(self) -> bool:

    async def async_open_cover(self, **kwargs: Any) -> None:

    async def async_close_cover(self, **kwargs: Any) -> None:

    async def async_stop_cover(self, **kwargs: Any) -> None:

class HomematicipCoverShutterGroup(HomematicipGenericEntity, CoverEntity):

    @property
    def current_cover_position(self) -> int:

    @property
    def current_cover_tilt_position(self) -> int:

    @property
    def is_closed(self) -> bool:

    async def async_set_cover_position(self, **kwargs: Any) -> None:

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:

    async def async_open_cover(self, **kwargs: Any) -> None:

    async def async_close_cover(self, **kwargs: Any) -> None:

    async def async_stop_cover(self, **kwargs: Any) -> None:

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:
