from __future__ import annotations
from typing import Any, Dict, List, Optional
from raspyrfm_client import RaspyRFMClient
from raspyrfm_client.device_implementations.controlunit.actions import Action
from raspyrfm_client.device_implementations.controlunit.controlunit_constants import ControlUnitModel
from raspyrfm_client.device_implementations.gateway.manufacturer.gateway_constants import GatewayModel
from raspyrfm_client.device_implementations.manufacturer_constants import Manufacturer
import voluptuous as vol
from homeassistant.components.switch import PLATFORM_SCHEMA as SWITCH_PLATFORM_SCHEMA, SwitchEntity
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT, CONF_SWITCHES, DEVICE_DEFAULT_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

CONF_GATEWAY_MANUFACTURER = 'gateway_manufacturer'
CONF_GATEWAY_MODEL = 'gateway_model'
CONF_CONTROLUNIT_MANUFACTURER = 'controlunit_manufacturer'
CONF_CONTROLUNIT_MODEL = 'controlunit_model'
CONF_CHANNEL_CONFIG = 'channel_config'
DEFAULT_HOST = '127.0.0.1'

PLATFORM_SCHEMA = SWITCH_PLATFORM_SCHEMA.extend({
    vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
    vol.Optional(CONF_PORT): cv.port,
    vol.Optional(CONF_GATEWAY_MANUFACTURER): cv.string,
    vol.Optional(CONF_GATEWAY_MODEL): cv.string,
    vol.Required(CONF_SWITCHES): vol.Schema([{
        vol.Optional(CONF_NAME, default=DEVICE_DEFAULT_NAME): cv.string,
        vol.Required(CONF_CONTROLUNIT_MANUFACTURER): cv.string,
        vol.Required(CONF_CONTROLUNIT_MODEL): cv.string,
        vol.Required(CONF_CHANNEL_CONFIG): {cv.string: cv.match_all}
    }])
}, extra=vol.ALLOW_EXTRA)


def setup_platform(hass: HomeAssistant,
                   config: ConfigType,
                   add_entities: AddEntitiesCallback,
                   discovery_info: Optional[DiscoveryInfoType] = None) -> None:
    """Set up the RaspyRFM switch."""
    gateway_manufacturer: str = config.get(CONF_GATEWAY_MANUFACTURER, Manufacturer.SEEGEL_SYSTEME.value)
    gateway_model: str = config.get(CONF_GATEWAY_MODEL, GatewayModel.RASPYRFM.value)
    host: str = config[CONF_HOST]
    port: Optional[int] = config.get(CONF_PORT)
    switches: List[Dict[str, Any]] = config[CONF_SWITCHES]
    raspyrfm_client: RaspyRFMClient = RaspyRFMClient()
    gateway: Any = raspyrfm_client.get_gateway(Manufacturer(gateway_manufacturer),
                                                 GatewayModel(gateway_model),
                                                 host,
                                                 port)
    switch_entities: List[RaspyRFMSwitch] = []
    for switch in switches:
        name: str = switch[CONF_NAME]
        controlunit_manufacturer: str = switch[CONF_CONTROLUNIT_MANUFACTURER]
        controlunit_model: str = switch[CONF_CONTROLUNIT_MODEL]
        channel_config: Dict[str, Any] = switch[CONF_CHANNEL_CONFIG]
        controlunit: Any = raspyrfm_client.get_controlunit(Manufacturer(controlunit_manufacturer),
                                                             ControlUnitModel(controlunit_model))
        controlunit.set_channel_config(**channel_config)
        switch_entity = RaspyRFMSwitch(raspyrfm_client, name, gateway, controlunit)
        switch_entities.append(switch_entity)
    add_entities(switch_entities)


class RaspyRFMSwitch(SwitchEntity):
    """Representation of a RaspyRFM switch."""
    _attr_should_poll: bool = False

    def __init__(self, raspyrfm_client: RaspyRFMClient, name: str, gateway: Any, controlunit: Any) -> None:
        """Initialize the switch."""
        self._raspyrfm_client: RaspyRFMClient = raspyrfm_client
        self._name: str = name
        self._gateway: Any = gateway
        self._controlunit: Any = controlunit
        self._state: Optional[bool] = None

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return self._name

    @property
    def assumed_state(self) -> bool:
        """Return True when the current state cannot be queried."""
        return True

    @property
    def is_on(self) -> Optional[bool]:
        """Return true if switch is on."""
        return self._state

    def turn_on(self, **kwargs: Any) -> None:
        """Turn the switch on."""
        self._raspyrfm_client.send(self._gateway, self._controlunit, Action.ON)
        self._state = True
        self.schedule_update_ha_state()

    def turn_off(self, **kwargs: Any) -> None:
        """Turn the switch off."""
        if Action.OFF in self._controlunit.get_supported_actions():
            self._raspyrfm_client.send(self._gateway, self._controlunit, Action.OFF)
        else:
            self._raspyrfm_client.send(self._gateway, self._controlunit, Action.ON)
        self._state = False
        self.schedule_update_ha_state()