"""Support for hunter douglas shades."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import datetime, timedelta
import logging
from math import ceil
from typing import Any

from aiopvapi.helpers.constants import (
    ATTR_NAME,
    CLOSED_POSITION,
    MAX_POSITION,
    MIN_POSITION,
    MOTION_STOP,
)
from aiopvapi.resources.shade import BaseShade, ShadePosition

from homeassistant.components.cover import (
    ATTR_POSITION,
    ATTR_TILT_POSITION,
    CoverDeviceClass,
    CoverEntity,
    CoverEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later

from .const import STATE_ATTRIBUTE_ROOM_NAME
from .coordinator import PowerviewShadeUpdateCoordinator
from .entity import ShadeEntity
from .model import PowerviewConfigEntry, PowerviewDeviceInfo

_LOGGER = logging.getLogger(__name__)

# Estimated time it takes to complete a transition
# from one state to another
TRANSITION_COMPLETE_DURATION = 40

PARALLEL_UPDATES = 1

RESYNC_DELAY = 60

SCAN_INTERVAL = timedelta(minutes=10)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PowerviewConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the hunter douglas shades."""
    pv_entry = entry.runtime_data
    coordinator = pv_entry.coordinator

    async def _async_initial_refresh() -> None:
        """Force position refresh shortly after adding.

        Legacy shades can become out of sync with hub when moved
        using physical remotes. This also allows reducing speed
        of calls to older generation hubs in an effort to
        prevent hub crashes.
        """

        for shade in pv_entry.shade_data.values():
            _LOGGER.debug("Initial refresh of shade: %s", shade.name)
            async with coordinator.radio_operation_lock:
                await shade.refresh(suppress_timeout=True)  # default 15 second timeout

    entities: list[ShadeEntity] = []
    for shade in pv_entry.shade_data.values():
        room_name = getattr(pv_entry.room_data.get(shade.room_id), ATTR_NAME, "")
        entities.extend(
            create_powerview_shade_entity(
                coordinator, pv_entry.device_info, room_name, shade, shade.name
            )
        )

    async_add_entities(entities)

    # background the fetching of state for initial launch
    entry.async_create_background_task(
        hass,
        _async_initial_refresh(),
        f"powerview {entry.title} initial shade refresh",
    )


class PowerViewShadeBase(ShadeEntity, CoverEntity):
    """Representation of a powerview shade."""

    _attr_device_class = CoverDeviceClass.SHADE
    _attr_supported_features = (
        CoverEntityFeature.OPEN
        | CoverEntityFeature.CLOSE
        | CoverEntityFeature.SET_POSITION
    )

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        """Initialize the shade."""
        super().__init__(coordinator, device_info, room_name, shade, name)
        self._shade: BaseShade = shade
        self._scheduled_transition_update: CALLBACK_TYPE | None = None
        if self._shade.is_supported(MOTION_STOP):
            self._attr_supported_features |= CoverEntityFeature.STOP
        self._forced_resync: Callable[[], None] | None = None

    @property
    def assumed_state(self) -> bool:
        """If the device is hard wired we are polling state.

        The hub will frequently provide the wrong state
        for battery power devices so we set assumed
        state in this case.
        """
        return not self._is_hard_wired

    @property
    def should_poll(self) -> bool:
        """Only poll if the device is hard wired.

        We cannot poll battery powered devices
        as it would drain their batteries in a matter
        of days.
        """
        return self._is_hard_wired

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Return the state attributes."""
        return {STATE_ATTRIBUTE_ROOM_NAME: self._room_name}

    @property
    def is_closed(self) -> bool:
        """Return if the cover is closed."""
        return self.positions.primary <= CLOSED_POSITION

    @property
    def current_cover_position(self) -> int:
        """Return the current position of cover."""
        return self.positions.primary

    @property
    def transition_steps(self) -> int:
        """Return the steps to make a move."""
        return self.positions.primary

    @property
    def open_position(self) -> ShadePosition:
        """Return the open position and required additional positions."""
        return replace(self._shade.open_position, velocity=self.positions.velocity)

    @property
    def close_position(self) -> ShadePosition:
        """Return the close position and required additional positions."""
        return replace(self._shade.close_position, velocity=self.positions.velocity)

    async def async_close_cover(self, **kwargs: Any) -> None:
        """Close the cover."""
        self._async_schedule_update_for_transition(self.transition_steps)
        await self._async_execute_move(self.close_position)
        self._attr_is_opening = False
        self._attr_is_closing = True
        self.async_write_ha_state()

    async def async_open_cover(self, **kwargs: Any) -> None:
        """Open the cover."""
        self._async_schedule_update_for_transition(100 - self.transition_steps)
        await self._async_execute_move(self.open_position)
        self._attr_is_opening = True
        self._attr_is_closing = False
        self.async_write_ha_state()

    async def async_stop_cover(self, **kwargs: Any) -> None:
        """Stop the cover."""
        self._async_cancel_scheduled_transition_update()
        await self._shade.stop()
        await self._async_force_refresh_state()

    @callback
    def _clamp_cover_limit(self, target_hass_position: int) -> int:
        """Don't allow a cover to go into an impossbile position."""
        # no override required in base
        return target_hass_position

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        """Move the shade to a specific position."""
        await self._async_set_cover_position(kwargs[ATTR_POSITION])

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        """Return a ShadePosition."""
        return ShadePosition(
            primary=target_hass_position,
            velocity=self.positions.velocity,
        )

    async def _async_execute_move(self, move: ShadePosition) -> None:
        """Execute a move that can affect multiple positions."""
        _LOGGER.debug("Move request %s: %s", self.name, move)
        async with self.coordinator.radio_operation_lock:
            response = await self._shade.move(move)
        _LOGGER.debug("Move response %s: %s", self.name, response)

        # Process the response from the hub (including new positions)
        self.data.update_shade_position(self._shade.id, response)

    async def _async_set_cover_position(self, target_hass_position: int) -> None:
        """Move the shade to a position."""
        target_hass_position = self._clamp_cover_limit(target_hass_position)
        current_hass_position = self.current_cover_position
        self._async_schedule_update_for_transition(
            abs(current_hass_position - target_hass_position)
        )
        await self._async_execute_move(self._get_shade_move(target_hass_position))
        self._attr_is_opening = target_hass_position > current_hass_position
        self._attr_is_closing = target_hass_position < current_hass_position
        self.async_write_ha_state()

    @callback
    def _async_update_shade_data(self, shade_data: dict[str, Any]) -> None:
        """Update the current cover position from the data."""
        self.data.update_shade_position(self._shade.id, shade_data)
        self._attr_is_opening = False
        self._attr_is_closing = False

    @callback
    def _async_cancel_scheduled_transition_update(self) -> None:
        """Cancel any previous updates."""
        if self._scheduled_transition_update:
            self._scheduled_transition_update()
            self._scheduled_transition_update = None
        if self._forced_resync:
            self._forced_resync()
            self._forced_resync = None

    @callback
    def _async_schedule_update_for_transition(self, steps: int) -> None:
        # Cancel any previous updates
        self._async_cancel_scheduled_transition_update()

        est_time_to_complete_transition = 1 + int(
            TRANSITION_COMPLETE_DURATION * (steps / 100)
        )

        _LOGGER.debug(
            "Estimated time to complete transition of %s steps for %s: %s",
            steps,
            self.name,
            est_time_to_complete_transition,
        )

        # Schedule a forced update for when we expect the transition
        # to be completed.
        self._scheduled_transition_update = async_call_later(
            self.hass,
            est_time_to_complete_transition,
            self._async_complete_schedule_update,
        )

    async def _async_complete_schedule_update(self, _: datetime) -> None:
        """Update status of the cover."""
        _LOGGER.debug("Processing scheduled update for %s", self.name)
        self._scheduled_transition_update = None
        await self._async_force_refresh_state()
        self._forced_resync = async_call_later(
            self.hass, RESYNC_DELAY, self._async_force_resync
        )

    async def _async_force_resync(self, *_: Any) -> None:
        """Force a resync after an update since the hub may have stale state."""
        self._forced_resync = None
        _LOGGER.debug("Force resync of shade %s", self.name)
        await self._async_force_refresh_state()

    async def _async_force_refresh_state(self) -> None:
        """Refresh the cover state and force the device cache to be bypassed."""
        await self.async_update()
        self.async_write_ha_state()

    # pylint: disable-next=hass-missing-super-call
    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        self.async_on_remove(
            self.coordinator.async_add_listener(self._async_update_shade_from_group)
        )

    async def async_will_remove_from_hass(self) -> None:
        """Cancel any pending refreshes."""
        self._async_cancel_scheduled_transition_update()

    @property
    def _update_in_progress(self) -> bool:
        """Check if an update is already in progress."""
        return bool(self._scheduled_transition_update or self._forced_resync)

    @callback
    def _async_update_shade_from_group(self) -> None:
        """Update with new data from the coordinator."""
        if self._update_in_progress:
            # If a transition is in progress the data will be wrong
            return
        self.data.update_from_group_data(self._shade.id)
        self.async_write_ha_state()

    async def async_update(self) -> None:
        """Refresh shade position."""
        if self._update_in_progress:
            # The update will likely timeout and
            # error if are already have one in flight
            return
        # suppress timeouts caused by hub nightly reboot
        async with self.coordinator.radio_operation_lock:
            await self._shade.refresh(
                suppress_timeout=True
            )  # default 15 second timeout
        _LOGGER.debug("Process update %s: %s", self.name, self._shade.current_position)
        self._async_update_shade_data(self._shade.current_position)


class PowerViewShade(PowerViewShadeBase):
    """Represent a standard shade."""

    _attr_name = None


class PowerViewShadeWithTiltBase(PowerViewShadeBase):
    """Representation for PowerView shades with tilt capabilities."""

    _attr_name = None

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        """Initialize the shade."""
        super().__init__(coordinator, device_info, room_name, shade, name)
        self._attr_supported_features |= (
            CoverEntityFeature.OPEN_TILT
            | CoverEntityFeature.CLOSE_TILT
            | CoverEntityFeature.SET_TILT_POSITION
        )
        if self._shade.is_supported(MOTION_STOP):
            self._attr_supported_features |= CoverEntityFeature.STOP_TILT
        self._max_tilt = self._shade.shade_limits.tilt_max

    @property
    def current_cover_tilt_position(self) -> int:
        """Return the current cover tile position."""
        return self.positions.tilt

    @property
    def transition_steps(self) -> int:
        """Return the steps to make a move."""
        return self.positions.primary + self.positions.tilt

    @property
    def open_tilt_position(self) -> ShadePosition:
        """Return the open tilt position and required additional positions."""
        return replace(self._shade.open_position_tilt, velocity=self.positions.velocity)

    @property
    def close_tilt_position(self) -> ShadePosition:
        """Return the close tilt position and required additional positions."""
        return replace(
            self._shade.close_position_tilt, velocity=self.positions.velocity
        )

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:
        """Close the cover tilt."""
        self._async_schedule_update_for_transition(self.transition_steps)
        await self._async_execute_move(self.close_tilt_position)
        self.async_write_ha_state()

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:
        """Open the cover tilt."""
        self._async_schedule_update_for_transition(100 - self.transition_steps)
        await self._async_execute_move(self.open_tilt_position)
        self.async_write_ha_state()

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:
        """Move the tilt to a specific position."""
        await self._async_set_cover_tilt_position(kwargs[ATTR_TILT_POSITION])

    async def _async_set_cover_tilt_position(
        self, target_hass_tilt_position: int
    ) -> None:
        """Move the tilt to a specific position."""
        final_position = self.current_cover_position + target_hass_tilt_position
        self._async_schedule_update_for_transition(
            abs(self.transition_steps - final_position)
        )
        await self._async_execute_move(self._get_shade_tilt(target_hass_tilt_position))
        self.async_write_ha_state()

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        """Return a ShadePosition."""
        return ShadePosition(
            primary=target_hass_position,
            velocity=self.positions.velocity,
        )

    @callback
    def _get_shade_tilt(self, target_hass_tilt_position: int) -> ShadePosition:
        """Return a ShadePosition."""
        return ShadePosition(
            tilt=target_hass_tilt_position,
            velocity=self.positions.velocity,
        )

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:
        """Stop the cover tilting."""
        await self.async_stop_cover()


class PowerViewShadeWithTiltOnClosed(PowerViewShadeWithTiltBase):
    """Representation of a PowerView shade with tilt when closed capabilities.

    API Class: ShadeBottomUpTiltOnClosed + ShadeBottomUpTiltOnClosed90

    Type 1 - Bottom Up w/ 90° Tilt
    Shade 44 - a shade thought to have been a firmware issue (type 0 usually don't tilt)
    """

    _attr_name = None

    @property
    def open_position(self) -> ShadePosition:
        """Return the open position and required additional positions."""
        return replace(self._shade.open_position, velocity=self.positions.velocity)

    @property
    def close_position(self) -> ShadePosition:
        """Return the close position and required additional positions."""
        return replace(self._shade.close_position, velocity=self.positions.velocity)

    @property
    def open_tilt_position(self) -> ShadePosition:
        """Return the open tilt position and required additional positions."""
        return replace(self._shade.open_position_tilt, velocity=self.positions.velocity)

    @property
    def close_tilt_position(self) -> ShadePosition:
        """Return the close tilt position and required additional positions."""
        return replace(
            self._shade.close_position_tilt, velocity=self.positions.velocity
        )


class PowerViewShadeWithTiltAnywhere(PowerViewShadeWithTiltBase):
    """Representation of a PowerView shade with tilt anywhere capabilities.

    API Class: ShadeBottomUpTiltAnywhere, ShadeVerticalTiltAnywhere

    Type 2 - Bottom Up w/ 180° Tilt
    Type 4 - Vertical (Traversing) w/ 180° Tilt
    """

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
       