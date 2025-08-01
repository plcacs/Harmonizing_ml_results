"""Support for Brunt Blind Engine covers."""
from __future__ import annotations
from typing import Any, Optional, Dict, Coroutine
from aiohttp.client_exceptions import ClientResponseError
from brunt import Thing
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.cover import (
    ATTR_POSITION,
    CoverDeviceClass,
    CoverEntity,
    CoverEntityFeature,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import (
    ATTR_REQUEST_POSITION,
    ATTRIBUTION,
    CLOSED_POSITION,
    DOMAIN,
    FAST_INTERVAL,
    OPEN_POSITION,
    REGULAR_INTERVAL,
)
from .coordinator import BruntConfigEntry, BruntCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the brunt platform."""
    coordinator: BruntCoordinator = entry.runtime_data
    async_add_entities(
        (
            BruntDevice(coordinator, serial, thing, entry.entry_id)
            for serial, thing in coordinator.data.items()
        )
    )

class BruntDevice(CoordinatorEntity[BruntCoordinator], CoverEntity):
    """Representation of a Brunt cover device.

    Contains the common logic for all Brunt devices.
    """
    _attr_has_entity_name: bool = True
    _attr_name: Optional[str] = None
    _attr_device_class: CoverDeviceClass = CoverDeviceClass.BLIND
    _attr_attribution: str = ATTRIBUTION
    _attr_supported_features: int = CoverEntityFeature.OPEN | CoverEntityFeature.CLOSE | CoverEntityFeature.SET_POSITION

    def __init__(
        self,
        coordinator: BruntCoordinator,
        serial: str,
        thing: Thing,
        entry_id: str,
    ) -> None:
        """Initialize the Brunt device."""
        super().__init__(coordinator)
        self._attr_unique_id: str = serial
        self._thing: Thing = thing
        self._entry_id: str = entry_id
        self._remove_update_listener: Optional[Any] = None
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, self._attr_unique_id)},
            name=self._thing.name,
            via_device=(DOMAIN, self._entry_id),
            manufacturer="Brunt",
            sw_version=self._thing.fw_version,
            model=self._thing.model,
        )

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        self.async_on_remove(self.coordinator.async_add_listener(self._brunt_update_listener))

    @property
    def current_cover_position(self) -> Optional[int]:
        """Return current position of cover.

        None is unknown, 0 is closed, 100 is fully open.
        """
        return self.coordinator.data[self.unique_id].current_position

    @property
    def request_cover_position(self) -> Optional[int]:
        """Return request position of cover.

        The request position is the position of the last request
        to Brunt, at times there is a diff of 1 to current.
        None is unknown, 0 is closed, 100 is fully open.
        """
        return self.coordinator.data[self.unique_id].request_position

    @property
    def move_state(self) -> Optional[int]:
        """Return current moving state of cover.

        None is unknown, 0 when stopped, 1 when opening, 2 when closing.
        """
        return self.coordinator.data[self.unique_id].move_state

    @property
    def is_opening(self) -> bool:
        """Return if the cover is opening or not."""
        return self.move_state == 1

    @property
    def is_closing(self) -> bool:
        """Return if the cover is closing or not."""
        return self.move_state == 2

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the detailed device state attributes."""
        return {ATTR_REQUEST_POSITION: self.request_cover_position}

    @property
    def is_closed(self) -> bool:
        """Return true if cover is closed, else False."""
        return self.current_cover_position == CLOSED_POSITION

    async def async_open_cover(self, **kwargs: Any) -> None:
        """Set the cover to the open position."""
        await self._async_update_cover(OPEN_POSITION)

    async def async_close_cover(self, **kwargs: Any) -> None:
        """Set the cover to the closed position."""
        await self._async_update_cover(CLOSED_POSITION)

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        """Set the cover to a specific position."""
        await self._async_update_cover(int(kwargs[ATTR_POSITION]))

    async def _async_update_cover(self, position: int) -> None:
        """Set the cover to the new position and wait for the update to be reflected."""
        try:
            await self.coordinator.bapi.async_change_request_position(
                position, thing_uri=self._thing.thing_uri
            )
        except ClientResponseError as exc:
            raise HomeAssistantError(f"Unable to reposition {self._thing.name}") from exc
        self.coordinator.update_interval = FAST_INTERVAL
        await self.coordinator.async_request_refresh()

    @callback
    def _brunt_update_listener(self) -> None:
        """Update the update interval after each refresh."""
        if (
            self.request_cover_position
            == self.coordinator.bapi.last_requested_positions[self._thing.thing_uri]
            and self.move_state == 0
        ):
            self.coordinator.update_interval = REGULAR_INTERVAL
        else:
            self.coordinator.update_interval = FAST_INTERVAL
