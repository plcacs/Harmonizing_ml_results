#!/usr/bin/env python3
"""Support for interfacing with WS66i 6 zone home audio controller."""
from typing import Any, List
from pyws66i import WS66i, ZoneStatus
from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from .const import DOMAIN, MAX_VOL
from .coordinator import Ws66iDataUpdateCoordinator
from .models import Ws66iData

PARALLEL_UPDATES: int = 1

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    """Set up the WS66i 6-zone amplifier platform from a config entry."""
    ws66i_data: Ws66iData = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        (
            Ws66iZone(
                device=ws66i_data.device,
                ws66i_data=ws66i_data,
                entry_id=config_entry.entry_id,
                zone_id=zone_id,
                data_idx=idx,
                coordinator=ws66i_data.coordinator,
            )
            for idx, zone_id in enumerate(ws66i_data.zones)
        )
    )

class Ws66iZone(CoordinatorEntity[Ws66iDataUpdateCoordinator], MediaPlayerEntity):
    """Representation of a WS66i amplifier zone."""
    _attr_has_entity_name: bool = True
    _attr_name: Any = None
    _attr_supported_features: int = (
        MediaPlayerEntityFeature.VOLUME_MUTE
        | MediaPlayerEntityFeature.VOLUME_SET
        | MediaPlayerEntityFeature.VOLUME_STEP
        | MediaPlayerEntityFeature.TURN_ON
        | MediaPlayerEntityFeature.TURN_OFF
        | MediaPlayerEntityFeature.SELECT_SOURCE
    )

    def __init__(
        self,
        device: WS66i,
        ws66i_data: Ws66iData,
        entry_id: str,
        zone_id: str,
        data_idx: int,
        coordinator: Ws66iDataUpdateCoordinator,
    ) -> None:
        """Initialize a zone entity."""
        super().__init__(coordinator)
        self._ws66i: WS66i = device
        self._ws66i_data: Ws66iData = ws66i_data
        self._zone_id: str = zone_id
        self._zone_id_idx: int = data_idx
        self._status: ZoneStatus = coordinator.data[data_idx]
        self._attr_source_list: List[str] = ws66i_data.sources.name_list
        self._attr_unique_id: str = f'{entry_id}_{zone_id}'
        self._attr_device_info: DeviceInfo = DeviceInfo(
            identifiers={(DOMAIN, str(self.unique_id))},
            name=f'Zone {zone_id}',
            manufacturer='Soundavo',
            model='WS66i 6-Zone Amplifier',
        )
        self._set_attrs_from_status()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        self._status = self.coordinator.data[self._zone_id_idx]
        self._set_attrs_from_status()
        super()._handle_coordinator_update()

    @callback
    def _set_attrs_from_status(self) -> None:
        sources: Any = self._ws66i_data.sources.id_name
        self._attr_state: MediaPlayerState = MediaPlayerState.ON if self._status.power else MediaPlayerState.OFF
        self._attr_volume_level: float = self._status.volume / float(MAX_VOL)
        self._attr_is_volume_muted: bool = self._status.mute
        self._attr_source = self._attr_media_title = sources[self._status.source]

    @callback
    def _async_update_attrs_write_ha_state(self) -> None:
        self._set_attrs_from_status()
        self.async_write_ha_state()

    async def async_select_source(self, source: str) -> None:
        """Set input source."""
        idx: int = self._ws66i_data.sources.name_id[source]
        await self.hass.async_add_executor_job(self._ws66i.set_source, self._zone_id, idx)
        self._status.source = idx
        self._async_update_attrs_write_ha_state()

    async def async_turn_on(self) -> None:
        """Turn the media player on."""
        await self.hass.async_add_executor_job(self._ws66i.set_power, self._zone_id, True)
        self._status.power = True
        self._async_update_attrs_write_ha_state()

    async def async_turn_off(self) -> None:
        """Turn the media player off."""
        await self.hass.async_add_executor_job(self._ws66i.set_power, self._zone_id, False)
        self._status.power = False
        self._async_update_attrs_write_ha_state()

    async def async_mute_volume(self, mute: bool) -> None:
        """Mute (true) or unmute (false) media player."""
        await self.hass.async_add_executor_job(self._ws66i.set_mute, self._zone_id, mute)
        self._status.mute = bool(mute)
        self._async_update_attrs_write_ha_state()

    async def async_set_volume_level(self, volume: float) -> None:
        """Set volume level, range 0..1."""
        await self.hass.async_add_executor_job(self._set_volume, int(volume * MAX_VOL))
        self._async_update_attrs_write_ha_state()

    async def async_volume_up(self) -> None:
        """Volume up the media player."""
        await self.hass.async_add_executor_job(
            self._set_volume, min(self._status.volume + 1, MAX_VOL)
        )
        self._async_update_attrs_write_ha_state()

    async def async_volume_down(self) -> None:
        """Volume down media player."""
        await self.hass.async_add_executor_job(
            self._set_volume, max(self._status.volume - 1, 0)
        )
        self._async_update_attrs_write_ha_state()

    def _set_volume(self, volume: int) -> None:
        """Set the volume of the media player."""
        if self._status.mute:
            self._ws66i.set_mute(self._zone_id, False)
            self._status.mute = False
        self._ws66i.set_volume(self._zone_id, volume)
        self._status.volume = volume
