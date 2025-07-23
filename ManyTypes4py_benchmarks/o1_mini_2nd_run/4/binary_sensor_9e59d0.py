"""Support for RFXtrx binary sensors."""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
import RFXtrx as rfxtrxmod
from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_COMMAND_OFF, CONF_COMMAND_ON, STATE_ON
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers import event as evt
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import DeviceTuple, async_setup_platform_entry, get_pt2262_cmd
from .const import (
    COMMAND_OFF_LIST,
    COMMAND_ON_LIST,
    CONF_DATA_BITS,
    CONF_OFF_DELAY,
    DEVICE_PACKET_TYPE_LIGHTING4,
)
from .entity import RfxtrxEntity

_LOGGER = logging.getLogger(__name__)

SENSOR_STATUS_ON: List[str] = [
    "Panic",
    "Motion",
    "Motion Tamper",
    "Light Detected",
    "Alarm",
    "Alarm Tamper",
]
SENSOR_STATUS_OFF: List[str] = [
    "End Panic",
    "No Motion",
    "No Motion Tamper",
    "Dark Detected",
    "Normal",
    "Normal Tamper",
]
SENSOR_TYPES: tuple[BinarySensorEntityDescription, ...] = (
    BinarySensorEntityDescription(
        key="X10 Security Motion Detector",
        device_class=BinarySensorDeviceClass.MOTION,
    ),
    BinarySensorEntityDescription(
        key="KD101 Smoke Detector", device_class=BinarySensorDeviceClass.SMOKE
    ),
    BinarySensorEntityDescription(
        key="Visonic Powercode Motion Detector",
        device_class=BinarySensorDeviceClass.MOTION,
    ),
    BinarySensorEntityDescription(
        key="Alecto SA30 Smoke Detector", device_class=BinarySensorDeviceClass.SMOKE
    ),
    BinarySensorEntityDescription(
        key="RM174RF Smoke Detector", device_class=BinarySensorDeviceClass.SMOKE
    ),
)
SENSOR_TYPES_DICT: Dict[str, BinarySensorEntityDescription] = {
    desc.key: desc for desc in SENSOR_TYPES
}


def supported(event: rfxtrxmod.BaseEvent) -> bool:
    """Return whether an event supports binary_sensor."""
    if isinstance(event, rfxtrxmod.ControlEvent):
        return True
    if isinstance(event, rfxtrxmod.SensorEvent):
        return event.values.get("Sensor Status") in [*SENSOR_STATUS_ON, *SENSOR_STATUS_OFF]
    return False


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up config entry."""

    def get_sensor_description(type_string: str) -> BinarySensorEntityDescription:
        if (description := SENSOR_TYPES_DICT.get(type_string)) is None:
            return BinarySensorEntityDescription(key=type_string)
        return description

    def _constructor(
        event: rfxtrxmod.BaseEvent,
        auto: bool,
        device_id: str,
        entity_info: Dict[str, Any],
    ) -> List[RfxtrxBinarySensor]:
        return [
            RfxtrxBinarySensor(
                device=event.device,
                device_id=device_id,
                entity_description=get_sensor_description(event.device.type_string),
                off_delay=entity_info.get(CONF_OFF_DELAY),
                data_bits=entity_info.get(CONF_DATA_BITS),
                cmd_on=entity_info.get(CONF_COMMAND_ON),
                cmd_off=entity_info.get(CONF_COMMAND_OFF),
                event=event if auto else None,
            )
        ]

    await async_setup_platform_entry(
        hass, config_entry, async_add_entities, supported, _constructor
    )


class RfxtrxBinarySensor(RfxtrxEntity, BinarySensorEntity):
    """A representation of a RFXtrx binary sensor.

    Since all repeated events have meaning, these types of sensors
    need to have force update enabled.
    """

    _attr_force_update: bool = True
    _attr_name: Optional[str] = None

    def __init__(
        self,
        device: rfxtrxmod.BaseDevice,
        device_id: str,
        entity_description: BinarySensorEntityDescription,
        off_delay: Optional[float] = None,
        data_bits: Optional[int] = None,
        cmd_on: Optional[int] = None,
        cmd_off: Optional[int] = None,
        event: Optional[rfxtrxmod.BaseEvent] = None,
    ) -> None:
        """Initialize the RFXtrx sensor."""
        super().__init__(device, device_id, event=event)
        self.entity_description: BinarySensorEntityDescription = entity_description
        self._data_bits: Optional[int] = data_bits
        self._off_delay: Optional[float] = off_delay
        self._delay_listener: Optional[Callable[[], None]] = None
        self._cmd_on: Optional[int] = cmd_on
        self._cmd_off: Optional[int] = cmd_off

    async def async_added_to_hass(self) -> None:
        """Restore device state."""
        await super().async_added_to_hass()
        if self._event is None:
            old_state = await self.async_get_last_state()
            if old_state is not None:
                self._attr_is_on: bool = old_state.state == STATE_ON
        if self.is_on and self._off_delay is not None:
            self._attr_is_on = False

    def _apply_event_lighting4(self, event: rfxtrxmod.BaseEvent) -> None:
        """Apply event for a lighting 4 device."""
        if self._data_bits is not None:
            cmdstr: Optional[str] = get_pt2262_cmd(event.device.id_string, self._data_bits)
            assert cmdstr is not None
            cmd: int = int(cmdstr, 16)
            if cmd == self._cmd_on:
                self._attr_is_on = True
            elif cmd == self._cmd_off:
                self._attr_is_on = False
        else:
            self._attr_is_on = True

    def _apply_event_standard(self, event: rfxtrxmod.BaseEvent) -> None:
        """Apply standard event."""
        assert isinstance(
            event, (rfxtrxmod.SensorEvent, rfxtrxmod.ControlEvent)
        )
        command: Optional[str] = event.values.get("Command")
        sensor_status: Optional[str] = event.values.get("Sensor Status")

        if command in COMMAND_ON_LIST:
            self._attr_is_on = True
        elif command in COMMAND_OFF_LIST:
            self._attr_is_on = False
        elif sensor_status in SENSOR_STATUS_ON:
            self._attr_is_on = True
        elif sensor_status in SENSOR_STATUS_OFF:
            self._attr_is_on = False

    def _apply_event(self, event: rfxtrxmod.BaseEvent) -> None:
        """Apply command from rfxtrx."""
        super()._apply_event(event)
        if event.device.packettype == DEVICE_PACKET_TYPE_LIGHTING4:
            self._apply_event_lighting4(event)
        else:
            self._apply_event_standard(event)

    @callback
    def _handle_event(
        self, event: rfxtrxmod.BaseEvent, device_id: str
    ) -> None:
        """Check if event applies to me and update."""
        if not self._event_applies(event, device_id):
            return
        _LOGGER.debug(
            "Binary sensor update (Device ID: %s Class: %s Sub: %s)",
            event.device.id_string,
            event.device.__class__.__name__,
            event.device.subtype,
        )
        self._apply_event(event)
        self.async_write_ha_state()
        if self._delay_listener:
            self._delay_listener()
            self._delay_listener = None
        if self.is_on and self._off_delay is not None:

            @callback
            def off_delay_listener(now: float) -> None:
                """Switch device off after a delay."""
                self._delay_listener = None
                self._attr_is_on = False
                self.async_write_ha_state()

            self._delay_listener = evt.async_call_later(
                self.hass, self._off_delay, off_delay_listener
            )
