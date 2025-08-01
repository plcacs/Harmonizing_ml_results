"""Cover integration microBees."""
from typing import Any, Optional, Generator
from microBeesPy import Actuator
from homeassistant.components.cover import (
    CoverDeviceClass, 
    CoverEntity, 
    CoverEntityFeature
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from .const import DOMAIN
from .coordinator import MicroBeesUpdateCoordinator
from .entity import MicroBeesEntity

COVER_IDS: dict[int, str] = {47: 'roller_shutter'}

async def async_setup_entry(
    hass: HomeAssistant, 
    entry: ConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the microBees cover platform."""
    coordinator: MicroBeesUpdateCoordinator = hass.data[DOMAIN][entry.entry_id].coordinator
    async_add_entities(
        (
            MBCover(
                coordinator, 
                bee_id, 
                next((actuator.id for actuator in bee.actuators if actuator.deviceID == 551), None), 
                next((actuator.id for actuator in bee.actuators if actuator.deviceID == 552), None)
            ) 
            for bee_id, bee in coordinator.data.bees.items() 
            if bee.productID in COVER_IDS
        )
    )

class MBCover(MicroBeesEntity, CoverEntity):
    """Representation of a microBees cover."""
    _attr_device_class: CoverDeviceClass = CoverDeviceClass.SHUTTER
    _attr_supported_features: int = CoverEntityFeature.OPEN | CoverEntityFeature.STOP | CoverEntityFeature.CLOSE

    def __init__(
        self, 
        coordinator: MicroBeesUpdateCoordinator, 
        bee_id: int, 
        actuator_up_id: Optional[int], 
        actuator_down_id: Optional[int]
    ) -> None:
        """Initialize the microBees cover."""
        super().__init__(coordinator, bee_id)
        self.actuator_up_id: Optional[int] = actuator_up_id
        self.actuator_down_id: Optional[int] = actuator_down_id
        self._attr_is_closed: Optional[bool] = None

    @property
    def name(self) -> str:
        """Name of the cover."""
        return self.bee.name

    @property
    def actuator_up(self) -> Actuator:
        """Return the rolling up actuator."""
        return self.coordinator.data.actuators[self.actuator_up_id]  # type: ignore

    @property
    def actuator_down(self) -> Actuator:
        """Return the rolling down actuator."""
        return self.coordinator.data.actuators[self.actuator_down_id]  # type: ignore

    def _reset_open_close(self, *_: Any) -> None:
        """Reset the opening and closing state."""
        self._attr_is_opening = False
        self._attr_is_closing = False
        self.async_write_ha_state()

    async def async_open_cover(self, **kwargs: Any) -> None:
        """Open the cover."""
        sendCommand: bool = await self.coordinator.microbees.sendCommand(
            self.actuator_up_id,  # type: ignore
            self.actuator_up.configuration.actuator_timing * 1000
        )
        if not sendCommand:
            raise HomeAssistantError(f'Failed to open {self.name}')
        self._attr_is_opening = True
        async_call_later(
            self.hass, 
            self.actuator_down.configuration.actuator_timing, 
            self._reset_open_close
        )

    async def async_close_cover(self, **kwargs: Any) -> None:
        """Close the cover."""
        sendCommand: bool = await self.coordinator.microbees.sendCommand(
            self.actuator_down_id,  # type: ignore
            self.actuator_down.configuration.actuator_timing * 1000
        )
        if not sendCommand:
            raise HomeAssistantError(f'Failed to close {self.name}')
        self._attr_is_closing = True
        async_call_later(
            self.hass, 
            self.actuator_down.configuration.actuator_timing, 
            self._reset_open_close
        )

    async def async_stop_cover(self, **kwargs: Any) -> None:
        """Stop the cover."""
        if self.is_opening:
            await self.coordinator.microbees.sendCommand(self.actuator_up_id, 0)  # type: ignore
        if self.is_closing:
            await self.coordinator.microbees.sendCommand(self.actuator_down_id, 0)  # type: ignore
