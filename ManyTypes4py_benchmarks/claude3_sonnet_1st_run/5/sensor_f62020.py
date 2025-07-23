"""Support for Etherscan sensors."""
from __future__ import annotations
from datetime import timedelta
from typing import Any, Optional

from pyetherscan import get_balance
import voluptuous as vol

from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_ADDRESS, CONF_NAME, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

CONF_TOKEN_ADDRESS = 'token_address'
SCAN_INTERVAL = timedelta(minutes=5)
PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_ADDRESS): cv.string,
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_TOKEN): cv.string,
    vol.Optional(CONF_TOKEN_ADDRESS): cv.string
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Etherscan.io sensors."""
    address: str = config.get(CONF_ADDRESS)
    name: Optional[str] = config.get(CONF_NAME)
    token: Optional[str] = config.get(CONF_TOKEN)
    token_address: Optional[str] = config.get(CONF_TOKEN_ADDRESS)
    
    if token:
        token = token.upper()
        if not name:
            name = f'{token} Balance'
    if not name:
        name = 'ETH Balance'
    
    add_entities([EtherscanSensor(name, address, token, token_address)], True)

class EtherscanSensor(SensorEntity):
    """Representation of an Etherscan.io sensor."""
    _attr_attribution: str = 'Data provided by etherscan.io'

    def __init__(
        self,
        name: str,
        address: str,
        token: Optional[str],
        token_address: Optional[str]
    ) -> None:
        """Initialize the sensor."""
        self._name: str = name
        self._address: str = address
        self._token_address: Optional[str] = token_address
        self._token: Optional[str] = token
        self._state: Optional[float] = None
        self._unit_of_measurement: str = self._token or 'ETH'

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return self._name

    @property
    def native_value(self) -> Optional[float]:
        """Return the state of the sensor."""
        return self._state

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement this sensor expresses itself in."""
        return self._unit_of_measurement

    def update(self) -> None:
        """Get the latest state of the sensor."""
        if self._token_address:
            self._state = get_balance(self._address, self._token_address)
        elif self._token:
            self._state = get_balance(self._address, self._token)
        else:
            self._state = get_balance(self._address)
