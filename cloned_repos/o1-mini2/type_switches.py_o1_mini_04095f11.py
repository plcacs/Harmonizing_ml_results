"""Class to hold all switch accessories."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Final, NamedTuple, Set

from pyhap.characteristic import Characteristic
from pyhap.const import (
    CATEGORY_FAUCET,
    CATEGORY_OUTLET,
    CATEGORY_SHOWER_HEAD,
    CATEGORY_SPRINKLER,
    CATEGORY_SWITCH,
)

from homeassistant.components import button, input_button
from homeassistant.components.input_select import ATTR_OPTIONS, SERVICE_SELECT_OPTION
from homeassistant.components.switch import DOMAIN as SWITCH_DOMAIN
from homeassistant.components.vacuum import (
    DOMAIN as VACUUM_DOMAIN,
    SERVICE_RETURN_TO_BASE,
    SERVICE_START,
    VacuumActivity,
    VacuumEntityFeature,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_SUPPORTED_FEATURES,
    CONF_TYPE,
    SERVICE_CLOSE_VALVE,
    SERVICE_OPEN_VALVE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_CLOSING,
    STATE_ON,
    STATE_OPEN,
    STATE_OPENING,
)
from homeassistant.core import HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers.event import async_call_later

from .accessories import TYPES, HomeAccessory, HomeDriver
from .const import (
    CHAR_ACTIVE,
    CHAR_IN_USE,
    CHAR_NAME,
    CHAR_ON,
    CHAR_OUTLET_IN_USE,
    CHAR_VALVE_TYPE,
    SERV_OUTLET,
    SERV_SWITCH,
    SERV_VALVE,
    TYPE_FAUCET,
    TYPE_SHOWER,
    TYPE_SPRINKLER,
    TYPE_VALVE,
)
from .util import cleanup_name_for_homekit

_LOGGER: logging.Logger = logging.getLogger(__name__)

VALVE_OPEN_STATES: Final[Set[str]] = {STATE_OPEN, STATE_OPENING, STATE_CLOSING}


class ValveInfo(NamedTuple):
    """Category and type information for valve."""

    category: int
    valve_type: int


VALVE_TYPE: Dict[str, ValveInfo] = {
    TYPE_FAUCET: ValveInfo(CATEGORY_FAUCET, 3),
    TYPE_SHOWER: ValveInfo(CATEGORY_SHOWER_HEAD, 2),
    TYPE_SPRINKLER: ValveInfo(CATEGORY_SPRINKLER, 1),
    TYPE_VALVE: ValveInfo(CATEGORY_FAUCET, 0),
}

ACTIVATE_ONLY_SWITCH_DOMAINS: Set[str] = {"button", "input_button", "scene", "script"}

ACTIVATE_ONLY_RESET_SECONDS: int = 10


@TYPES.register("Outlet")
class Outlet(HomeAccessory):
    """Generate an Outlet accessory."""

    def __init__(self, *args: Any) -> None:
        """Initialize an Outlet accessory object."""
        super().__init__(*args, category=CATEGORY_OUTLET)
        state: State = self.hass.states.get(self.entity_id)
        assert state

        serv_outlet = self.add_preload_service(SERV_OUTLET)
        self.char_on: Characteristic = serv_outlet.configure_char(
            CHAR_ON, value=False, setter_callback=self.set_state
        )
        self.char_outlet_in_use: Characteristic = serv_outlet.configure_char(
            CHAR_OUTLET_IN_USE, value=True
        )
        # Set the state so it is in sync on initial
        # GET to avoid an event storm after homekit startup
        self.async_update_state(state)

    def set_state(self, value: bool) -> None:
        """Move switch state to value if call came from HomeKit."""
        _LOGGER.debug("%s: Set switch state to %s", self.entity_id, value)
        params: Dict[str, Any] = {ATTR_ENTITY_ID: self.entity_id}
        service: str = SERVICE_TURN_ON if value else SERVICE_TURN_OFF
        self.async_call_service(SWITCH_DOMAIN, service, params)

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Update switch state after state changed."""
        current_state: bool = new_state.state == STATE_ON
        _LOGGER.debug("%s: Set current state to %s", self.entity_id, current_state)
        self.char_on.set_value(current_state)


@TYPES.register("Switch")
class Switch(HomeAccessory):
    """Generate a Switch accessory."""

    def __init__(self, *args: Any) -> None:
        """Initialize a Switch accessory object."""
        super().__init__(*args, category=CATEGORY_SWITCH)
        self._domain: str
        self._object_id: str
        self._domain, self._object_id = split_entity_id(self.entity_id)
        state: State = self.hass.states.get(self.entity_id)
        assert state

        self.activate_only: bool = self.is_activate(state)

        serv_switch = self.add_preload_service(SERV_SWITCH)
        self.char_on: Characteristic = serv_switch.configure_char(
            CHAR_ON, value=False, setter_callback=self.set_state
        )
        # Set the state so it is in sync on initial
        # GET to avoid an event storm after homekit startup
        self.async_update_state(state)

    def is_activate(self, state: State) -> bool:
        """Check if entity is activate only."""
        return self._domain in ACTIVATE_ONLY_SWITCH_DOMAINS

    def reset_switch(self, *args: Any) -> None:
        """Reset switch to emulate activate click."""
        _LOGGER.debug("%s: Reset switch to off", self.entity_id)
        self.char_on.set_value(False)

    def set_state(self, value: bool) -> None:
        """Move switch state to value if call came from HomeKit."""
        _LOGGER.debug("%s: Set switch state to %s", self.entity_id, value)
        if self.activate_only and not value:
            _LOGGER.debug("%s: Ignoring turn_off call", self.entity_id)
            return

        params: Dict[str, Any] = {ATTR_ENTITY_ID: self.entity_id}
        service: str
        if self._domain == "script":
            service = self._object_id
            params = {}
        elif self._domain == button.DOMAIN:
            service = button.SERVICE_PRESS
        elif self._domain == input_button.DOMAIN:
            service = input_button.SERVICE_PRESS
        else:
            service = SERVICE_TURN_ON if value else SERVICE_TURN_OFF

        self.async_call_service(self._domain, service, params)

        if self.activate_only:
            async_call_later(self.hass, ACTIVATE_ONLY_RESET_SECONDS, self.reset_switch)

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Update switch state after state changed."""
        self.activate_only = self.is_activate(new_state)
        if self.activate_only:
            _LOGGER.debug(
                "%s: Ignore state change, entity is activate only", self.entity_id
            )
            return

        current_state: bool = new_state.state == STATE_ON
        _LOGGER.debug("%s: Set current state to %s", self.entity_id, current_state)
        self.char_on.set_value(current_state)


@TYPES.register("Vacuum")
class Vacuum(Switch):
    """Generate a Switch accessory."""

    def set_state(self, value: bool) -> None:
        """Move switch state to value if call came from HomeKit."""
        _LOGGER.debug("%s: Set switch state to %s", self.entity_id, value)
        state: State = self.hass.states.get(self.entity_id)
        assert state

        features: int = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)

        if value:
            sup_start = features & VacuumEntityFeature.START
            service = SERVICE_START if sup_start else SERVICE_TURN_ON
        else:
            sup_return_home = features & VacuumEntityFeature.RETURN_HOME
            service = SERVICE_RETURN_TO_BASE if sup_return_home else SERVICE_TURN_OFF

        self.async_call_service(
            VACUUM_DOMAIN, service, {ATTR_ENTITY_ID: self.entity_id}
        )

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Update switch state after state changed."""
        current_state: bool = new_state.state in (VacuumActivity.CLEANING, STATE_ON)
        _LOGGER.debug("%s: Set current state to %s", self.entity_id, current_state)
        self.char_on.set_value(current_state)


class ValveBase(HomeAccessory):
    """Valve base class."""

    def __init__(
        self,
        valve_type: str,
        open_states: Set[str],
        on_service: str,
        off_service: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a Valve accessory object."""
        super().__init__(*args, **kwargs)
        self.domain: str = split_entity_id(self.entity_id)[0]
        state: State = self.hass.states.get(self.entity_id)
        assert state

        self.category: int = VALVE_TYPE[valve_type].category
        self.open_states: Set[str] = open_states
        self.on_service: str = on_service
        self.off_service: str = off_service

        serv_valve = self.add_preload_service(SERV_VALVE)
        self.char_active: Characteristic = serv_valve.configure_char(
            CHAR_ACTIVE, value=False, setter_callback=self.set_state
        )
        self.char_in_use: Characteristic = serv_valve.configure_char(CHAR_IN_USE, value=False)
        self.char_valve_type: Characteristic = serv_valve.configure_char(
            CHAR_VALVE_TYPE, value=VALVE_TYPE[valve_type].valve_type
        )
        # Set the state so it is in sync on initial
        # GET to avoid an event storm after homekit startup
        self.async_update_state(state)

    def set_state(self, value: bool) -> None:
        """Move value state to value if call came from HomeKit."""
        _LOGGER.debug("%s: Set switch state to %s", self.entity_id, value)
        self.char_in_use.set_value(value)
        params: Dict[str, Any] = {ATTR_ENTITY_ID: self.entity_id}
        service: str = self.on_service if value else self.off_service
        self.async_call_service(self.domain, service, params)

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Update switch state after state changed."""
        current_state: int = 1 if new_state.state in self.open_states else 0
        _LOGGER.debug("%s: Set active state to %s", self.entity_id, current_state)
        self.char_active.set_value(current_state)
        _LOGGER.debug("%s: Set in_use state to %s", self.entity_id, current_state)
        self.char_in_use.set_value(current_state)


@TYPES.register("ValveSwitch")
class ValveSwitch(ValveBase):
    """Generate a Valve accessory from a HomeAssistant switch."""

    def __init__(
        self,
        hass: HomeAssistant,
        driver: HomeDriver,
        name: str,
        entity_id: str,
        aid: int,
        config: Dict[str, Any],
        *args: Any,
    ) -> None:
        """Initialize a Valve accessory object."""
        super().__init__(
            valve_type=config[CONF_TYPE],
            open_states={STATE_ON},
            on_service=SERVICE_TURN_ON,
            off_service=SERVICE_TURN_OFF,
            hass=hass,
            driver=driver,
            name=name,
            entity_id=entity_id,
            aid=aid,
            config=config,
            *args,
        )


@TYPES.register("Valve")
class Valve(ValveBase):
    """Generate a Valve accessory from a HomeAssistant valve."""

    def __init__(self, *args: Any) -> None:
        """Initialize a Valve accessory object."""
        super().__init__(
            valve_type=TYPE_VALVE,
            open_states=VALVE_OPEN_STATES,
            on_service=SERVICE_OPEN_VALVE,
            off_service=SERVICE_CLOSE_VALVE,
            *args,
        )


@TYPES.register("SelectSwitch")
class SelectSwitch(HomeAccessory):
    """Generate a Switch accessory that contains multiple switches."""

    def __init__(self, *args: Any) -> None:
        """Initialize a Switch accessory object."""
        super().__init__(*args, category=CATEGORY_SWITCH)
        self.domain: str = split_entity_id(self.entity_id)[0]
        state: State = self.hass.states.get(self.entity_id)
        assert state

        self.select_chars: Dict[str, Characteristic] = {}
        options: list[str] = state.attributes[ATTR_OPTIONS]
        for option in options:
            serv_option = self.add_preload_service(
                SERV_OUTLET, [CHAR_NAME, CHAR_IN_USE], unique_id=option
            )
            serv_option.configure_char(
                CHAR_NAME, value=cleanup_name_for_homekit(option)
            )
            serv_option.configure_char(CHAR_IN_USE, value=False)
            self.select_chars[option] = serv_option.configure_char(
                CHAR_ON,
                value=False,
                setter_callback=self._create_setter(option),
            )
        self.set_primary_service(next(iter(self.select_chars.values())))
        # Set the state so it is in sync on initial
        # GET to avoid an event storm after homekit startup
        self.async_update_state(state)

    def _create_setter(self, option: str) -> Callable[[bool], None]:
        def setter(value: bool) -> None:
            if value:
                self.select_option(option)
        return setter

    def select_option(self, option: str) -> None:
        """Set option from HomeKit."""
        _LOGGER.debug("%s: Set option to %s", self.entity_id, option)
        params: Dict[str, Any] = {ATTR_ENTITY_ID: self.entity_id, "option": option}
        self.async_call_service(self.domain, SERVICE_SELECT_OPTION, params)

    @callback
    def async_update_state(self, new_state: State) -> None:
        """Update switch state after state changed."""
        current_option: str = cleanup_name_for_homekit(new_state.state)
        for option, char in self.select_chars.items():
            char.set_value(option == current_option)
