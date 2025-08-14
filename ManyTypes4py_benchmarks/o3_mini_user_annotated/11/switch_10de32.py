"""Support for VersaSense actuator peripheral."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.switch import SwitchEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from . import DOMAIN
from .const import (
    KEY_CONSUMER,
    KEY_IDENTIFIER,
    KEY_MEASUREMENT,
    KEY_PARENT_MAC,
    KEY_PARENT_NAME,
    KEY_UNIT,
    PERIPHERAL_STATE_OFF,
    PERIPHERAL_STATE_ON,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up actuator platform."""
    if discovery_info is None:
        return

    consumer: Any = hass.data[DOMAIN][KEY_CONSUMER]

    actuator_list: list[VActuator] = []

    for entity_info in discovery_info.values():
        peripheral: Any = hass.data[DOMAIN][entity_info[KEY_PARENT_MAC]][
            entity_info[KEY_IDENTIFIER]
        ]
        parent_name: str = entity_info[KEY_PARENT_NAME]
        unit: str = entity_info[KEY_UNIT]
        measurement: str = entity_info[KEY_MEASUREMENT]

        actuator_list.append(
            VActuator(peripheral, parent_name, unit, measurement, consumer)
        )

    async_add_entities(actuator_list)


class VActuator(SwitchEntity):
    """Representation of an Actuator."""

    def __init__(
        self,
        peripheral: Any,
        parent_name: str,
        unit: str,
        measurement: str,
        consumer: Any,
    ) -> None:
        """Initialize the actuator."""
        self._is_on: bool | None = False
        self._available: bool = True
        self._name: str = f"{parent_name} {measurement}"
        self._parent_mac: str = peripheral.parentMac
        self._identifier: str = peripheral.identifier
        self._unit: str = unit
        self._measurement: str = measurement
        self.consumer: Any = consumer

    @property
    def unique_id(self) -> str:
        """Return the unique id of the actuator."""
        return f"{self._parent_mac}/{self._identifier}/{self._measurement}"

    @property
    def name(self) -> str:
        """Return the name of the actuator."""
        return self._name

    @property
    def is_on(self) -> bool | None:
        """Return the state of the actuator."""
        return self._is_on

    @property
    def available(self) -> bool:
        """Return if the actuator is available."""
        return self._available

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the actuator."""
        await self.update_state(0)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the actuator."""
        await self.update_state(1)

    async def update_state(self, state: int) -> None:
        """Update the state of the actuator."""
        payload: dict[str, Any] = {"id": "state-num", "value": state}

        await self.consumer.actuatePeripheral(
            None, self._identifier, self._parent_mac, payload
        )

    async def async_update(self) -> None:
        """Fetch state data from the actuator."""
        samples: Any = await self.consumer.fetchPeripheralSample(
            None, self._identifier, self._parent_mac
        )

        if samples is not None:
            for sample in samples:
                if sample.measurement == self._measurement:
                    self._available = True
                    if sample.value == PERIPHERAL_STATE_OFF:
                        self._is_on = False
                    elif sample.value == PERIPHERAL_STATE_ON:
                        self._is_on = True
                    break
        else:
            _LOGGER.error("Sample unavailable")
            self._available = False
            self._is_on = None
