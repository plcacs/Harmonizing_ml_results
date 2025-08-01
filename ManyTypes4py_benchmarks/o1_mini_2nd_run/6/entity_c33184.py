"""Support for 1-Wire entities."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Optional
from pyownet import protocol
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity import Entity, EntityDescription
from homeassistant.helpers.typing import StateType
from .const import READ_MODE_BOOL, READ_MODE_INT

@dataclass(frozen=True)
class OneWireEntityDescription(EntityDescription):
    """Class describing OneWire entities."""
    read_mode: Optional[str] = None

_LOGGER: logging.Logger = logging.getLogger(__name__)

class OneWireEntity(Entity):
    """Implementation of a 1-Wire entity."""
    _attr_has_entity_name: bool = True

    def __init__(
        self,
        description: OneWireEntityDescription,
        device_id: str,
        device_info: DeviceInfo,
        device_file: str,
        owproxy: protocol.Protocol
    ) -> None:
        """Initialize the entity."""
        self.entity_description: OneWireEntityDescription = description
        self._last_update_success: bool = True
        self._attr_unique_id: str = f'/{device_id}/{description.key}'
        self._attr_device_info: DeviceInfo = device_info
        self._device_file: str = device_file
        self._state: Optional[StateType] = None
        self._value_raw: Optional[float] = None
        self._owproxy: protocol.Protocol = owproxy

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the entity."""
        return {'device_file': self._device_file, 'raw_value': self._value_raw}

    def _read_value(self) -> str:
        """Read a value from the server."""
        read_bytes: bytes = self._owproxy.read(self._device_file)
        return read_bytes.decode().lstrip()

    def _write_value(self, value: str) -> None:
        """Write a value to the server."""
        self._owproxy.write(self._device_file, value)

    def update(self) -> None:
        """Get the latest data from the device."""
        try:
            self._value_raw = float(self._read_value())
        except protocol.Error as exc:
            if self._last_update_success:
                _LOGGER.error('Error fetching %s data: %s', self.name, exc)
                self._last_update_success = False
            self._state = None
        else:
            if not self._last_update_success:
                self._last_update_success = True
                _LOGGER.debug('Fetching %s data recovered', self.name)
            if self.entity_description.read_mode == READ_MODE_INT:
                self._state = int(self._value_raw)
            elif self.entity_description.read_mode == READ_MODE_BOOL:
                self._state = int(self._value_raw) == 1
            else:
                self._state = self._value_raw

    @property
    def state(self) -> Optional[StateType]:
        """Return the state of the entity."""
        return self._state
