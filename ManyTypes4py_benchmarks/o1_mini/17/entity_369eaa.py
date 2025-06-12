"""Homematic base entity."""
from __future__ import annotations
from abc import abstractmethod
from datetime import timedelta, datetime
import logging
from typing import Any, Callable, Dict, Optional, Union

from pyhomematic import HMConnection
from pyhomematic.devicetypes.generic import HMGeneric
from homeassistant.const import ATTR_NAME
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import Entity, EntityDescription
from homeassistant.helpers.event import track_time_interval

from .const import (
    ATTR_ADDRESS,
    ATTR_CHANNEL,
    ATTR_INTERFACE,
    ATTR_PARAM,
    ATTR_UNIQUE_ID,
    DATA_HOMEMATIC,
    DOMAIN,
    HM_ATTRIBUTE_SUPPORT,
)

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL_HUB = timedelta(seconds=300)
SCAN_INTERVAL_VARIABLES = timedelta(seconds=30)


class HMDevice(Entity):
    """The HomeMatic device base object."""

    _attr_should_poll: bool = False

    def __init__(
        self,
        config: Dict[str, Any],
        entity_description: Optional[EntityDescription] = None,
    ) -> None:
        """Initialize a generic HomeMatic device."""
        self._name: str = config.get(ATTR_NAME, "")
        self._address: str = config.get(ATTR_ADDRESS, "")
        self._interface: str = config.get(ATTR_INTERFACE, "")
        self._channel: Union[str, int] = config.get(ATTR_CHANNEL, "")
        self._state: Optional[str] = config.get(ATTR_PARAM)
        self._unique_id: str = config.get(ATTR_UNIQUE_ID, "")
        self._data: Dict[str, Any] = {}
        self._connected: bool = False
        self._available: bool = False
        self._channel_map: Dict[str, str] = {}
        if entity_description is not None:
            self.entity_description = entity_description
        if self._state:
            self._state = self._state.upper()

    async def async_added_to_hass(self) -> None:
        """Load data init callbacks."""
        self._subscribe_homematic_events()

    @property
    def unique_id(self) -> str:
        """Return unique ID. HomeMatic entity IDs are unique by default."""
        return self._unique_id.replace(" ", "_")

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def available(self) -> bool:
        """Return true if device is available."""
        return self._available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return device specific state attributes."""
        attr: Dict[str, Any] = {"id": self._hmdevice.ADDRESS, "interface": self._interface}
        for node, data in HM_ATTRIBUTE_SUPPORT.items():
            if node in self._data:
                value = data[1].get(self._data[node], self._data[node])
                attr[data[0]] = value
        return attr

    def update(self) -> bool:
        """Connect to HomeMatic init values."""
        if self._connected:
            return True
        self._homematic: HMConnection = self.hass.data[DATA_HOMEMATIC]
        self._hmdevice: HMGeneric = self._homematic.devices[self._interface][self._address]
        self._connected = True
        try:
            self._init_data()
            self._load_data_from_hm()
            self._available = not self._hmdevice.UNREACH
            return True
        except Exception as err:
            self._connected = False
            _LOGGER.error("Exception while linking %s: %s", self._address, str(err))
            return False

    def _hm_event_callback(
        self,
        device: str,
        caller: str,
        attribute: str,
        value: Any,
    ) -> None:
        """Handle all pyhomematic device events."""
        has_changed = False
        device_partition = device.partition(":")[2]
        mapped_channel = self._channel_map.get(attribute)
        if device_partition == mapped_channel:
            self._data[attribute] = value
            has_changed = True
        if self.available != (not self._hmdevice.UNREACH):
            self._available = not self._hmdevice.UNREACH
            has_changed = True
        if has_changed:
            self.schedule_update_ha_state()

    def _subscribe_homematic_events(self) -> None:
        """Subscribe all required events to handle job."""
        for metadata in (
            self._hmdevice.ACTIONNODE,
            self._hmdevice.EVENTNODE,
            self._hmdevice.WRITENODE,
            self._hmdevice.ATTRIBUTENODE,
            self._hmdevice.BINARYNODE,
            self._hmdevice.SENSORNODE,
        ):
            for node, channels in metadata.items():
                if node in self._data:
                    if len(channels) == 1:
                        channel = channels[0]
                    else:
                        channel = self._channel
                    self._channel_map[node] = str(channel)
        _LOGGER.debug("Channel map for %s: %s", self._address, str(self._channel_map))
        self._hmdevice.setEventCallback(callback=self._hm_event_callback, bequeath=True)

    def _load_data_from_hm(self) -> bool:
        """Load first value from pyhomematic."""
        if not self._connected:
            return False
        for metadata, funct in (
            (self._hmdevice.ATTRIBUTENODE, self._hmdevice.getAttributeData),
            (self._hmdevice.WRITENODE, self._hmdevice.getWriteData),
            (self._hmdevice.SENSORNODE, self._hmdevice.getSensorData),
            (self._hmdevice.BINARYNODE, self._hmdevice.getBinaryData),
        ):
            for node in metadata:
                if metadata[node] and node in self._data:
                    self._data[node] = funct(name=node, channel=self._channel)
        return True

    def _hm_set_state(self, value: Any) -> None:
        """Set data to main datapoint."""
        if self._state in self._data:
            self._data[self._state] = value

    def _hm_get_state(self) -> Optional[Any]:
        """Get data from main datapoint."""
        if self._state in self._data:
            return self._data[self._state]
        return None

    def _init_data(self) -> None:
        """Generate a data dict (self._data) from the HomeMatic metadata."""
        for data_note in self._hmdevice.ATTRIBUTENODE:
            self._data.update({data_note: None})
        self._init_data_struct()

    @abstractmethod
    def _init_data_struct(self) -> None:
        """Generate a data dictionary from the HomeMatic device metadata."""


class HMHub(Entity):
    """The HomeMatic hub. (CCU2/HomeGear)."""

    _attr_should_poll: bool = False

    def __init__(
        self,
        hass: Any,
        homematic: HMConnection,
        name: str,
    ) -> None:
        """Initialize HomeMatic hub."""
        self.hass = hass
        self.entity_id: str = f"{DOMAIN}.{name.lower()}"
        self._homematic: HMConnection = homematic
        self._variables: Dict[str, Any] = {}
        self._name: str = name
        self._state: Optional[int] = None
        track_time_interval(self.hass, self._update_hub, SCAN_INTERVAL_HUB)
        self.hass.add_job(self._update_hub, None)
        track_time_interval(self.hass, self._update_variables, SCAN_INTERVAL_VARIABLES)
        self.hass.add_job(self._update_variables, None)

    @property
    def name(self) -> str:
        """Return the name of the device."""
        return self._name

    @property
    def state(self) -> Optional[int]:
        """Return the state of the entity."""
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return self._variables.copy()

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend, if any."""
        return "mdi:gradient-vertical"

    def _update_hub(self, now: datetime) -> None:
        """Retrieve latest state."""
        service_message = self._homematic.getServiceMessages(self._name)
        state: Optional[int] = None if service_message is None else len(service_message)
        if self._state != state:
            self._state = state
            self.schedule_update_ha_state()

    def _update_variables(self, now: datetime) -> None:
        """Retrieve all variable data and update hmvariable states."""
        variables: Optional[Dict[str, Any]] = self._homematic.getAllSystemVariables(self._name)
        if variables is None:
            return
        state_change = False
        for key, value in variables.items():
            if key in self._variables and value == self._variables[key]:
                continue
            state_change = True
            self._variables.update({key: value})
        if state_change:
            self.schedule_update_ha_state()

    def hm_set_variable(self, name: str, value: Any) -> None:
        """Set variable value on CCU/Homegear."""
        if name not in self._variables:
            _LOGGER.error("Variable %s not found on %s", name, self.name)
            return
        old_value: Any = self._variables.get(name)
        if isinstance(old_value, bool):
            value_converted: bool = cv.boolean(value)
        else:
            value_converted: float = float(value)
        self._homematic.setSystemVariable(self.name, name, value_converted)
        self._variables.update({name: value_converted})
        self.schedule_update_ha_state()
