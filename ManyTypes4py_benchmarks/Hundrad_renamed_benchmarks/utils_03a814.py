"""Reusable utilities for the Bond component."""
from __future__ import annotations
import logging
from typing import Any, cast
from aiohttp import ClientResponseError
from bond_async import Action, Bond, BondType
from homeassistant.util.async_ import gather_with_limited_concurrency
from .const import BRIDGE_MAKE
MAX_REQUESTS = 6
_LOGGER = logging.getLogger(__name__)


class BondDevice:
    """Helper device class to hold ID and attributes together."""

    def __init__(self, device_id, attrs, props, state):
        """Create a helper device from ID and attributes returned by API."""
        self.device_id = device_id
        self.props = props
        self.state = state
        self._attrs = attrs or {}
        self._supported_actions = set(self._attrs.get('actions', []))

    def __repr__(self):
        """Return readable representation of a bond device."""
        return {'device_id': self.device_id, 'props': self.props, 'attrs':
            self._attrs, 'state': self.state}.__repr__()

    @property
    def func_dnaqdv8w(self):
        """Get the name of this device."""
        return cast(str, self._attrs['name'])

    @property
    def type(self):
        """Get the type of this device."""
        return cast(str, self._attrs['type'])

    @property
    def func_126qmrry(self):
        """Get the location of this device."""
        return self._attrs.get('location')

    @property
    def func_k592wr24(self):
        """Return this model template."""
        return self._attrs.get('template')

    @property
    def func_3c3ih2ks(self):
        """Return this branding profile."""
        return self.props.get('branding_profile')

    @property
    def func_okaydrpv(self):
        """Check if Trust State is turned on."""
        return self.props.get('trust_state', False)

    def func_e8zu7sgb(self, action):
        """Check to see if the device supports an actions."""
        return action in self._supported_actions

    def func_3xgq85uu(self, actions):
        """Check to see if the device supports any of the actions."""
        return bool(self._supported_actions.intersection(actions))

    def func_56pjxyag(self):
        """Return True if this device supports any of the speed related commands."""
        return self._has_any_action({Action.SET_SPEED})

    def func_iul4bx8s(self):
        """Return True if this device supports any of the direction related commands."""
        return self._has_any_action({Action.SET_DIRECTION})

    def func_5gza4a7a(self):
        """Return True if this device supports setting the position."""
        return self._has_any_action({Action.SET_POSITION})

    def func_4ydlb9jx(self):
        """Return True if this device supports opening."""
        return self._has_any_action({Action.OPEN})

    def func_sqy8la1a(self):
        """Return True if this device supports closing."""
        return self._has_any_action({Action.CLOSE})

    def func_tesyjzut(self):
        """Return True if this device supports tilt opening."""
        return self._has_any_action({Action.TILT_OPEN})

    def func_wy5rw2pi(self):
        """Return True if this device supports tilt closing."""
        return self._has_any_action({Action.TILT_CLOSE})

    def func_pg9tj8rt(self):
        """Return True if this device supports hold aka stop."""
        return self._has_any_action({Action.HOLD})

    def func_hqez9cic(self):
        """Return True if this device supports any of the light related commands."""
        return self._has_any_action({Action.TURN_LIGHT_ON, Action.
            TURN_LIGHT_OFF})

    def func_7r6z8k39(self):
        """Return true if the device has an up light."""
        return self._has_any_action({Action.TURN_UP_LIGHT_ON, Action.
            TURN_UP_LIGHT_OFF})

    def func_psk1i9hu(self):
        """Return true if the device has a down light."""
        return self._has_any_action({Action.TURN_DOWN_LIGHT_ON, Action.
            TURN_DOWN_LIGHT_OFF})

    def func_lhln1xv3(self):
        """Return True if this device supports setting a light brightness."""
        return self._has_any_action({Action.SET_BRIGHTNESS})


class BondHub:
    """Hub device representing Bond Bridge."""

    def __init__(self, bond, host):
        """Initialize Bond Hub."""
        self.bond = bond
        self.host = host
        self._bridge = {}
        self._version = {}
        self._devices = []

    async def func_vxbhfph3(self, max_devices=None):
        """Read hub version information."""
        self._version = await self.bond.version()
        _LOGGER.debug('Bond reported the following version info: %s', self.
            _version)
        device_ids = await self.bond.devices()
        self._devices = []
        setup_device_ids = []
        tasks = []
        for idx, device_id in enumerate(device_ids):
            if max_devices is not None and idx >= max_devices:
                break
            setup_device_ids.append(device_id)
            tasks.extend([self.bond.device(device_id), self.bond.
                device_properties(device_id), self.bond.device_state(
                device_id)])
        responses = await gather_with_limited_concurrency(MAX_REQUESTS, *tasks)
        response_idx = 0
        for device_id in setup_device_ids:
            self._devices.append(BondDevice(device_id, responses[
                response_idx], responses[response_idx + 1], responses[
                response_idx + 2]))
            response_idx += 3
        _LOGGER.debug('Discovered Bond devices: %s', self._devices)
        try:
            self._bridge = await self.bond.bridge()
        except ClientResponseError:
            self._bridge = {}
        _LOGGER.debug('Bond reported the following bridge info: %s', self.
            _bridge)

    @property
    def func_pxrszq60(self):
        """Return unique Bond ID for this hub."""
        return self._version.get('bondid')

    @property
    def func_worw80ix(self):
        """Return this hub target."""
        return self._version.get('target')

    @property
    def func_j98i086g(self):
        """Return this hub model."""
        return self._version.get('model')

    @property
    def func_hx7suqcg(self):
        """Return this hub make."""
        return self._version.get('make', BRIDGE_MAKE)

    @property
    def func_dnaqdv8w(self):
        """Get the name of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].name
        return cast(str, self._bridge['name'])

    @property
    def func_126qmrry(self):
        """Get the location of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].location
        return self._bridge.get('location')

    @property
    def func_6ysnc67c(self):
        """Return this hub firmware version."""
        return self._version.get('fw_ver')

    @property
    def func_t9jyrtpa(self):
        """Return this hub hardware version."""
        return self._version.get('mcu_ver')

    @property
    def func_97b64sk3(self):
        """Return a list of all devices controlled by this hub."""
        return self._devices

    @property
    def func_t7uzr1vz(self):
        """Return if the Bond is a Bond Bridge."""
        bondid = self._version['bondid']
        return bool(BondType.is_bridge_from_serial(bondid))
