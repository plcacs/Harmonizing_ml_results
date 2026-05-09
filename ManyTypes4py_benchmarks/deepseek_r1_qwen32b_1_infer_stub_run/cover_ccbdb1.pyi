"""Support for hunter douglas shades."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from math import ceil
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)
from homeassistant.components.cover import (
    CoverDeviceClass,
    CoverEntity,
    CoverEntityFeature,
)
from homeassistant.core import HomeAssistant, CALLBACK_TYPE
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from .const import STATE_ATTRIBUTE_ROOM_NAME
from .entity import ShadeEntity
from .model import PowerviewDeviceInfo

_LOGGER = logging.getLogger(__name__)
TRANSITION_COMPLETE_DURATION = 40
PARALLEL_UPDATES = 1
RESYNC_DELAY = 60
SCAN_INTERVAL = timedelta(minutes=10)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    ...

class PowerViewShadeBase(ShadeEntity, CoverEntity):
    _attr_device_class: CoverDeviceClass
    _attr_supported_features: CoverEntityFeature
    _attr_is_opening: bool
    _attr_is_closing: bool
    _shade: BaseShade
    _scheduled_transition_update: Optional[asyncio.TimerHandle]
    _forced_resync: Optional[asyncio.TimerHandle]
    _is_hard_wired: bool

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def assumed_state(self) -> bool:
        ...

    @property
    def should_poll(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> Dict[str, str]:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @property
    def transition_steps(self) -> int:
        ...

    @property
    def open_position(self) -> ShadePosition:
        ...

    @property
    def close_position(self) -> ShadePosition:
        ...

    async def async_close_cover(self, **kwargs: Any) -> None:
        ...

    async def async_open_cover(self, **kwargs: Any) -> None:
        ...

    async def async_stop_cover(self, **kwargs: Any) -> None:
        ...

    @callback
    def _clamp_cover_limit(self, target_hass_position: int) -> int:
        ...

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

    async def _async_execute_move(self, move: ShadePosition) -> None:
        ...

    async def _async_set_cover_position(self, target_hass_position: int) -> None:
        ...

    @callback
    def _async_update_shade_data(self, shade_data: ShadePosition) -> None:
        ...

    @callback
    def _async_cancel_scheduled_transition_update(self) -> None:
        ...

    @callback
    def _async_schedule_update_for_transition(self, steps: int) -> None:
        ...

    async def _async_complete_schedule_update(self, _) -> None:
        ...

    async def _async_force_resync(self, *_) -> None:
        ...

    async def _async_force_refresh_state(self) -> None:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_will_remove_from_hass(self) -> None:
        ...

    @property
    def _update_in_progress(self) -> bool:
        ...

    @callback
    def _async_update_shade_from_group(self) -> None:
        ...

    async def async_update(self) -> None:
        ...

class PowerViewShade(PowerViewShadeBase):
    _attr_name: Optional[str]

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

class PowerViewShadeWithTiltBase(PowerViewShadeBase):
    _attr_name: Optional[str]
    _max_tilt: int

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def current_cover_tilt_position(self) -> int:
        ...

    @property
    def transition_steps(self) -> int:
        ...

    @property
    def open_tilt_position(self) -> ShadePosition:
        ...

    @property
    def close_tilt_position(self) -> ShadePosition:
        ...

    async def async_close_cover_tilt(self, **kwargs: Any) -> None:
        ...

    async def async_open_cover_tilt(self, **kwargs: Any) -> None:
        ...

    async def async_set_cover_tilt_position(self, **kwargs: Any) -> None:
        ...

    async def _async_set_cover_tilt_position(self, target_hass_tilt_position: int) -> None:
        ...

    @callback
    def _get_shade_tilt(self, target_hass_tilt_position: int) -> ShadePosition:
        ...

    async def async_stop_cover_tilt(self, **kwargs: Any) -> None:
        ...

class PowerViewShadeWithTiltOnClosed(PowerViewShadeWithTiltBase):
    _attr_name: Optional[str]

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

class PowerViewShadeWithTiltAnywhere(PowerViewShadeWithTiltBase):
    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

    @callback
    def _get_shade_tilt(self, target_hass_tilt_position: int) -> ShadePosition:
        ...

class PowerViewShadeTiltOnly(PowerViewShadeWithTiltBase):
    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @property
    def transition_steps(self) -> int:
        ...

    @property
    def is_closed(self) -> bool:
        ...

class PowerViewShadeTopDown(PowerViewShadeBase):
    _attr_name: Optional[str]

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    async def async_set_cover_position(self, **kwargs: Any) -> None:
        ...

    @property
    def is_closed(self) -> bool:
        ...

class PowerViewShadeDualRailBase(PowerViewShadeBase):
    @property
    def transition_steps(self) -> int:
        ...

class PowerViewShadeTDBUBottom(PowerViewShadeDualRailBase):
    _attr_translation_key: str

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @callback
    def _clamp_cover_limit(self, target_hass_position: int) -> int:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

class PowerViewShadeTDBUTop(PowerViewShadeDualRailBase):
    _attr_translation_key: str

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def should_poll(self) -> bool:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @property
    def open_position(self) -> ShadePosition:
        ...

    @callback
    def _clamp_cover_limit(self, target_hass_position: int) -> int:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

class PowerViewShadeDualOverlappedBase(PowerViewShadeBase):
    @property
    def transition_steps(self) -> int:
        ...

    @property
    def open_position(self) -> ShadePosition:
        ...

    @property
    def close_position(self) -> ShadePosition:
        ...

class PowerViewShadeDualOverlappedCombined(PowerViewShadeDualOverlappedBase):
    _attr_translation_key: str

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

class PowerViewShadeDualOverlappedFront(PowerViewShadeDualOverlappedBase):
    _attr_translation_key: str

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def should_poll(self) -> bool:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

    @property
    def close_position(self) -> ShadePosition:
        ...

class PowerViewShadeDualOverlappedRear(PowerViewShadeDualOverlappedBase):
    _attr_translation_key: str

    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def should_poll(self) -> bool:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def current_cover_position(self) -> int:
        ...

    @callback
    def _get_shade_move(self, target_hass_position: int) -> ShadePosition:
        ...

    @property
    def open_position(self) -> ShadePosition:
        ...

class PowerViewShadeDualOverlappedCombinedTilt(PowerViewShadeDualOverlappedCombined):
    def __init__(
        self,
        coordinator: PowerviewShadeUpdateCoordinator,
        device_info: PowerviewDeviceInfo,
        room_name: str,
        shade: BaseShade,
        name: str,
    ) -> None:
        ...

    @property
    def transition_steps(self) -> int:
        ...

    @callback
    def _get_shade_tilt(self, target_hass_tilt_position: int) -> ShadePosition:
        ...

    @property
    def open_tilt_position(self) -> ShadePosition:
        ...

    @property
    def close_tilt_position(self) -> ShadePosition:
        ...

TYPE_TO_CLASSES: Dict[int, Tuple[Type[PowerViewShade], ...]] = ...

def create_powerview_shade_entity(
    coordinator: PowerviewShadeUpdateCoordinator,
    device_info: PowerviewDeviceInfo,
    room_name: str,
    shade: BaseShade,
    name_before_refresh: str,
) -> List[PowerViewShade]:
    ...