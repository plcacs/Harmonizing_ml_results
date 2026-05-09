"""Support for switches that can be controlled using the RaspyRFM rc module."""
from __future__ import annotations
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

CONF_GATEWAY_MANUFACTURER: str = 'gateway_manufacturer'
CONF_GATEWAY_MODEL: str = 'gateway_model'
CONF_CONTROLUNIT_MANUFACTURER: str = 'controlunit_manufacturer'
CONF_CONTROLUNIT_MODEL: str = 'controlunit_model'
CONF_CHANNEL_CONFIG: str = 'channel_config'
DEFAULT_HOST: str = '127.0.0.1'
PLATFORM_SCHEMA: vol.Schema = SWITCH_PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_HOST, default=DEFAULT_HOST): cv.string,
        vol.Optional(CONF_PORT): cv.port,
        vol.Optional(CONF_GATEWAY_MANUFACTURER): cv.string,
        vol.Optional(CONF_GATEWAY_MODEL): cv.string,
        vol.Required(CONF_SWITCHES): vol.Schema(
            [
                {
                    vol.Optional(CONF_NAME, default=DEVICE_DEFAULT_NAME): cv.string,
                    vol.Required(CONF_CONTROLUNIT_MANUFACTURER): cv.string,
                    vol.Required(CONF_CONTROLUNIT_MODEL): cv.string,
                    vol.Required(CONF_CHANNEL_CONFIG): {cv.string: cv.match_all},
                }
            ]
        ),
    ],
    extra=vol.ALLOW_EXTRA,
)

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the RaspyRFM switch."""
    gateway_manufacturer: str = config.get(CONF_GATEWAY_MANUFACTURER, Manufacturer.SEEGEL_SYSTEME.value)
    gateway_model: str = config.get(CONF_GATEWAY_MODEL, GatewayModel.RASPYRFM.value)
    host: str = config[CONF_HOST]
    port: int | None = config.get(CONF_PORT)
    switches: list[dict] = config[CONF_SWITCHES]
    raspyrfm_client: RaspyRFMClient = RaspyRFMClient()
    gateway: object = raspyrfm_client.get_gateway(
        Manufacturer(gateway_manufacturer), GatewayModel(gateway_model), host, port
    )
    switch_entities: list[RaspyRFMSwitch] = []
    for switch in switches:
        name: str = switch[CONF_NAME]
        controlunit_manufacturer: str = switch[CONF_CONTROLUNIT_MANUFACTURER]
        controlunit_model: str = switch[CONF_CONTROLUNIT_MODEL]
        channel_config: dict = switch[CONF_CHANNEL_CONFIG]
        controlunit: object = raspyrfm_client.get_controlunit(
            Manufacturer(controlunit_manufacturer), ControlUnitModel(controlunit_model)
        )
        controlunit.set_channel_config(**channel_config)
        switch_entity: RaspyRFMSwitch = RaspyRFMSwitch(raspyrfm_client, name, gateway, controlunit)
        switch_entities.append(switch_entity)
    add_entities(switch_entities)

class RaspyRFMSwitch(SwitchEntity):
    """Representation of a RaspyRFM switch."""

    _attr_should_poll: bool = False

    def __init__(self, raspyrfm_client: RaspyRFMClient, name: str, gateway: object, controlunit: object) -> None:
        """Initialize the switch."""
        self._raspyrfm_client: RaspyRFMClient = raspyrfm_client
        self._name: str = name
        self._gateway: object = gateway
        self._controlunit: object = controlunit
        self._state: bool | None = None

    @property
    def name(self) -> str:
        """Return the name of the device if any."""
        return self._name

    @property
    def assumed_state(self) -> bool:
        """Return True when the current state cannot be queried."""
        return True

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        return self._state

    def turn_on(self, **kwargs: dict) -> None:
        """Turn the switch on."""
        self._raspyrfm_client.send(self._gateway, self._controlunit, Action.ON)
        self._state = True
        self.schedule_update_ha_state()

    def turn_off(self, **kwargs: dict) -> None:
        """Turn the switch off."""
        if Action.OFF in self._controlunit.get_supported_actions():
            self._raspyrfm_client.send(self._gateway, self._controlunit, Action.OFF)
        else:
            self._raspyrfm_client.send(self._gateway, self._controlunit, Action.ON)
        self._state = False
        self.schedule_update_ha_state()
