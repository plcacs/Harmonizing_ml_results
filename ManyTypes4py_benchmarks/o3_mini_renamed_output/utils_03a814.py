"""Reusable utilities for the Bond component."""
from __future__ import annotations
import logging
from typing import Any, Awaitable, Dict, List, Optional, Set, Union, cast
from aiohttp import ClientResponseError
from bond_async import Action, Bond, BondType
from homeassistant.util.async_ import gather_with_limited_concurrency
from .const import BRIDGE_MAKE

MAX_REQUESTS: int = 6
_LOGGER = logging.getLogger(__name__)


class BondDevice:
    """Helper device class to hold ID and attributes together."""

    def __init__(
        self,
        device_id: Union[int, str],
        attrs: Optional[Dict[str, Any]],
        props: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        """Create a helper device from ID and attributes returned by API."""
        self.device_id: Union[int, str] = device_id
        self.props: Dict[str, Any] = props
        self.state: Dict[str, Any] = state
        self._attrs: Dict[str, Any] = attrs or {}
        self._supported_actions: Set[Action] = set(self._attrs.get("actions", []))

    def __repr__(self) -> str:
        """Return readable representation of a bond device."""
        return {
            "device_id": self.device_id,
            "props": self.props,
            "attrs": self._attrs,
            "state": self.state,
        }.__repr__()

    @property
    def func_dnaqdv8w(self) -> str:
        """Get the name of this device."""
        return cast(str, self._attrs["name"])

    @property
    def name(self) -> str:
        """Alias for func_dnaqdv8w."""
        return self.func_dnaqdv8w

    @property
    def type(self) -> str:
        """Get the type of this device."""
        return cast(str, self._attrs["type"])

    @property
    def func_126qmrry(self) -> Optional[str]:
        """Get the location of this device."""
        loc = self._attrs.get("location")
        return cast(Optional[str], loc)

    @property
    def location(self) -> Optional[str]:
        """Alias for func_126qmrry."""
        return self.func_126qmrry

    @property
    def func_k592wr24(self) -> Optional[str]:
        """Return this model template."""
        tmpl = self._attrs.get("template")
        return cast(Optional[str], tmpl)

    @property
    def func_3c3ih2ks(self) -> Optional[str]:
        """Return this branding profile."""
        bp = self.props.get("branding_profile")
        return cast(Optional[str], bp)

    @property
    def func_okaydrpv(self) -> bool:
        """Check if Trust State is turned on."""
        return bool(self.props.get("trust_state", False))

    def func_e8zu7sgb(self, action: Action) -> bool:
        """Check to see if the device supports an action."""
        return action in self._supported_actions

    def func_3xgq85uu(self, actions: Set[Action]) -> bool:
        """Check to see if the device supports any of the actions."""
        return bool(self._supported_actions.intersection(actions))

    def _has_any_action(self, actions: Set[Action]) -> bool:
        """Helper to check if the device supports any of the provided actions."""
        return bool(self._supported_actions.intersection(actions))

    def func_56pjxyag(self) -> bool:
        """Return True if this device supports any of the speed related commands."""
        return self._has_any_action({Action.SET_SPEED})

    def func_iul4bx8s(self) -> bool:
        """Return True if this device supports any of the direction related commands."""
        return self._has_any_action({Action.SET_DIRECTION})

    def func_5gza4a7a(self) -> bool:
        """Return True if this device supports setting the position."""
        return self._has_any_action({Action.SET_POSITION})

    def func_4ydlb9jx(self) -> bool:
        """Return True if this device supports opening."""
        return self._has_any_action({Action.OPEN})

    def func_sqy8la1a(self) -> bool:
        """Return True if this device supports closing."""
        return self._has_any_action({Action.CLOSE})

    def func_tesyjzut(self) -> bool:
        """Return True if this device supports tilt opening."""
        return self._has_any_action({Action.TILT_OPEN})

    def func_wy5rw2pi(self) -> bool:
        """Return True if this device supports tilt closing."""
        return self._has_any_action({Action.TILT_CLOSE})

    def func_pg9tj8rt(self) -> bool:
        """Return True if this device supports hold aka stop."""
        return self._has_any_action({Action.HOLD})

    def func_hqez9cic(self) -> bool:
        """Return True if this device supports any of the light related commands."""
        return self._has_any_action({Action.TURN_LIGHT_ON, Action.TURN_LIGHT_OFF})

    def func_7r6z8k39(self) -> bool:
        """Return true if the device has an up light."""
        return self._has_any_action({Action.TURN_UP_LIGHT_ON, Action.TURN_UP_LIGHT_OFF})

    def func_psk1i9hu(self) -> bool:
        """Return true if the device has a down light."""
        return self._has_any_action({Action.TURN_DOWN_LIGHT_ON, Action.TURN_DOWN_LIGHT_OFF})

    def func_lhln1xv3(self) -> bool:
        """Return True if this device supports setting a light brightness."""
        return self._has_any_action({Action.SET_BRIGHTNESS})


class BondHub:
    """Hub device representing Bond Bridge."""

    def __init__(self, bond: Bond, host: str) -> None:
        """Initialize Bond Hub."""
        self.bond: Bond = bond
        self.host: str = host
        self._bridge: Dict[str, Any] = {}
        self._version: Dict[str, Any] = {}
        self._devices: List[BondDevice] = []

    async def func_vxbhfph3(self, max_devices: Optional[int] = None) -> None:
        """Read hub version information."""
        self._version = await self.bond.version()  # type: Dict[str, Any]
        _LOGGER.debug("Bond reported the following version info: %s", self._version)
        device_ids: List[Any] = await self.bond.devices()
        self._devices = []
        setup_device_ids: List[Any] = []
        tasks: List[Awaitable[Any]] = []
        for idx, device_id in enumerate(device_ids):
            if max_devices is not None and idx >= max_devices:
                break
            setup_device_ids.append(device_id)
            tasks.extend(
                [
                    self.bond.device(device_id),
                    self.bond.device_properties(device_id),
                    self.bond.device_state(device_id),
                ]
            )
        responses: List[Any] = await gather_with_limited_concurrency(MAX_REQUESTS, *tasks)
        response_idx = 0
        for device_id in setup_device_ids:
            self._devices.append(
                BondDevice(
                    device_id,
                    responses[response_idx],
                    responses[response_idx + 1],
                    responses[response_idx + 2],
                )
            )
            response_idx += 3
        _LOGGER.debug("Discovered Bond devices: %s", self._devices)
        try:
            self._bridge = await self.bond.bridge()  # type: Dict[str, Any]
        except ClientResponseError:
            self._bridge = {}
        _LOGGER.debug("Bond reported the following bridge info: %s", self._bridge)

    @property
    def func_pxrszq60(self) -> Optional[str]:
        """Return unique Bond ID for this hub."""
        bond_id = self._version.get("bondid")
        return cast(Optional[str], bond_id)

    @property
    def func_worw80ix(self) -> Optional[str]:
        """Return this hub target."""
        target = self._version.get("target")
        return cast(Optional[str], target)

    @property
    def func_j98i086g(self) -> Optional[str]:
        """Return this hub model."""
        model = self._version.get("model")
        return cast(Optional[str], model)

    @property
    def func_hx7suqcg(self) -> str:
        """Return this hub make."""
        return cast(str, self._version.get("make", BRIDGE_MAKE))

    @property
    def func_dnaqdv8w(self) -> str:
        """Get the name of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].name
        return cast(str, self._bridge["name"])

    @property
    def func_126qmrry(self) -> Optional[str]:
        """Get the location of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].location
        loc = self._bridge.get("location")
        return cast(Optional[str], loc)

    @property
    def func_6ysnc67c(self) -> Optional[str]:
        """Return this hub firmware version."""
        fw_ver = self._version.get("fw_ver")
        return cast(Optional[str], fw_ver)

    @property
    def func_t9jyrtpa(self) -> Optional[str]:
        """Return this hub hardware version."""
        mcu_ver = self._version.get("mcu_ver")
        return cast(Optional[str], mcu_ver)

    @property
    def func_97b64sk3(self) -> List[BondDevice]:
        """Return a list of all devices controlled by this hub."""
        return self._devices

    @property
    def func_t7uzr1vz(self) -> bool:
        """Return if the Bond is a Bond Bridge."""
        bondid = self._version.get("bondid")
        # Ensure that bondid is a string before passing it to is_bridge_from_serial.
        if isinstance(bondid, str):
            return bool(BondType.is_bridge_from_serial(bondid))
        return False

    @property
    def is_bridge(self) -> bool:
        """Alias for func_t7uzr1vz to indicate if the hub is a Bond Bridge."""
        return self.func_t7uzr1vz
