from __future__ import annotations
from typing import Any, Dict, Optional, Callable
from homeassistant.components.lock import LockEntity, LockState
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later
from .const import DOMAIN, GATEWAYS_KEY
from .entity import XiaomiDevice

FINGER_KEY = 'fing_verified'
PASSWORD_KEY = 'psw_verified'
CARD_KEY = 'card_verified'
VERIFIED_WRONG_KEY = 'verified_wrong'
ATTR_VERIFIED_WRONG_TIMES = 'verified_wrong_times'
UNLOCK_MAINTAIN_TIME = 5

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Perform the setup for Xiaomi devices."""
    gateway = hass.data[DOMAIN][GATEWAYS_KEY][config_entry.entry_id]
    async_add_entities(
        (
            XiaomiAqaraLock(device, 'Lock', gateway, config_entry)
            for device in gateway.devices['lock']
            if device['model'] == 'lock.aq1'
        )
    )

class XiaomiAqaraLock(LockEntity, XiaomiDevice):
    """Representation of a XiaomiAqaraLock."""

    def __init__(self, device: Dict[str, Any], name: str, xiaomi_hub: Any, config_entry: ConfigEntry) -> None:
        """Initialize the XiaomiAqaraLock."""
        self._changed_by: int = 0
        self._verified_wrong_times: int = 0
        super().__init__(device, name, xiaomi_hub, config_entry)

    @property
    def is_locked(self) -> Optional[bool]:
        """Return true if lock is locked."""
        if self._state is not None:
            return self._state == LockState.LOCKED
        return None

    @property
    def changed_by(self) -> int:
        """Last change triggered by."""
        return self._changed_by

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return {ATTR_VERIFIED_WRONG_TIMES: self._verified_wrong_times}

    @callback
    def clear_unlock_state(self, _: Any) -> None:
        """Clear unlock state automatically."""
        self._state = LockState.LOCKED
        self.async_write_ha_state()

    def parse_data(self, data: Dict[str, Any], raw_data: Any) -> bool:
        """Parse data sent by gateway."""
        if (value := data.get(VERIFIED_WRONG_KEY)) is not None:
            self._verified_wrong_times = int(value)
            return True
        for key in (FINGER_KEY, PASSWORD_KEY, CARD_KEY):
            if (value := data.get(key)) is not None:
                self._changed_by = int(value)
                self._verified_wrong_times = 0
                self._state = LockState.UNLOCKED
                async_call_later(self.hass, UNLOCK_MAINTAIN_TIME, self.clear_unlock_state)
                return True
        return False