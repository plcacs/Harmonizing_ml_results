from __future__ import annotations
import logging
from typing import Any, cast, Dict, List, Optional, Union
from aiohttp import ClientResponseError
from bond_async import Action, Bond, BondType
from homeassistant.util.async_ import gather_with_limited_concurrency
from .const import BRIDGE_MAKE

MAX_REQUESTS: int = 6
_LOGGER: logging.Logger = logging.getLogger(__name__)


class BondDevice:
    """Helper device class to hold ID and attributes together."""

    def __init__(
        self,
        device_id: Union[str, int],
        attrs: Optional[Dict[str, Any]],
        props: Dict[str, Any],
        state: Dict[str, Any],
    ) -> None:
        """Create a helper device from ID and attributes returned by API."""
        self.device_id: Union[str, int] = device_id
        self.props: Dict[str, Any] = props
        self.state: Dict[str, Any] = state
        self._attrs: Dict[str, Any] = attrs or {}
        self._supported_actions: set = set(self._attrs.get("actions", []))

    def __repr__(self) -> str:
        """Return readable representation of a bond device."""
        return {
            "device_id": self.device_id,
            "props": self.props,
            "attrs": self._attrs,
            "state": self.state,
        }.__repr__()

    @property
    def name(self) -> str:
        """Get the name of this device."""
        return cast(str, self._attrs["name"])

    @property
    def type(self) -> str:
        """Get the type of this device."""
        return cast(str, self._attrs["type"])

    @property
    def location(self) -> Optional[str]:
        """Get the location of this device."""
        return cast(Optional[str], self._attrs.get("location"))

    @property
    def template(self) -> Optional[str]:
        """Return this model template."""
        return cast(Optional[str], self._attrs.get("template"))

    @property
    def branding_profile(self) -> Optional[Any]:
        """Return this branding profile."""
        return self.props.get("branding_profile")

    @property
    def trust_state(self) -> bool:
        """Check if Trust State is turned on."""
        return bool(self.props.get("trust_state", False))

    def has_action(self, action: Action) -> bool:
        """Check to see if the device supports an actions."""
        return action in self._supported_actions

    def _has_any_action(self, actions: set[Action]) -> bool:
        """Check to see if the device supports any of the actions."""
        return bool(self._supported_actions.intersection(actions))

    def supports_speed(self) -> bool:
        """Return True if this device supports any of the speed related commands."""
        return self._has_any_action({Action.SET_SPEED})

    def supports_direction(self) -> bool:
        """Return True if this device supports any of the direction related commands."""
        return self._has_any_action({Action.SET_DIRECTION})

    def supports_set_position(self) -> bool:
        """Return True if this device supports setting the position."""
        return self._has_any_action({Action.SET_POSITION})

    def supports_open(self) -> bool:
        """Return True if this device supports opening."""
        return self._has_any_action({Action.OPEN})

    def supports_close(self) -> bool:
        """Return True if this device supports closing."""
        return self._has_any_action({Action.CLOSE})

    def supports_tilt_open(self) -> bool:
        """Return True if this device supports tilt opening."""
        return self._has_any_action({Action.TILT_OPEN})

    def supports_tilt_close(self) -> bool:
        """Return True if this device supports tilt closing."""
        return self._has_any_action({Action.TILT_CLOSE})

    def supports_hold(self) -> bool:
        """Return True if this device supports hold aka stop."""
        return self._has_any_action({Action.HOLD})

    def supports_light(self) -> bool:
        """Return True if this device supports any of the light related commands."""
        return self._has_any_action({Action.TURN_LIGHT_ON, Action.TURN_LIGHT_OFF})

    def supports_up_light(self) -> bool:
        """Return true if the device has an up light."""
        return self._has_any_action({Action.TURN_UP_LIGHT_ON, Action.TURN_UP_LIGHT_OFF})

    def supports_down_light(self) -> bool:
        """Return true if the device has a down light."""
        return self._has_any_action({Action.TURN_DOWN_LIGHT_ON, Action.TURN_DOWN_LIGHT_OFF})

    def supports_set_brightness(self) -> bool:
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

    async def setup(self, max_devices: Optional[int] = None) -> None:
        """Read hub version information."""
        self._version = await self.bond.version()
        _LOGGER.debug("Bond reported the following version info: %s", self._version)
        device_ids: List[Union[str, int]] = await self.bond.devices()
        self._devices = []
        setup_device_ids: List[Union[str, int]] = []
        tasks: List[Any] = []
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
        response_idx: int = 0
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
            self._bridge = await self.bond.bridge()
        except ClientResponseError:
            self._bridge = {}
        _LOGGER.debug("Bond reported the following bridge info: %s", self._bridge)

    @property
    def bond_id(self) -> Optional[str]:
        """Return unique Bond ID for this hub."""
        bondid = self._version.get("bondid")
        return bondid if isinstance(bondid, str) else None

    @property
    def target(self) -> Optional[str]:
        """Return this hub target."""
        target = self._version.get("target")
        return target if isinstance(target, str) else None

    @property
    def model(self) -> Optional[str]:
        """Return this hub model."""
        model = self._version.get("model")
        return model if isinstance(model, str) else None

    @property
    def make(self) -> str:
        """Return this hub make."""
        return cast(str, self._version.get("make", BRIDGE_MAKE))

    @property
    def name(self) -> str:
        """Get the name of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].name
        return cast(str, self._bridge["name"])

    @property
    def location(self) -> Optional[str]:
        """Get the location of this bridge."""
        if not self.is_bridge and self._devices:
            return self._devices[0].location
        location = self._bridge.get("location")
        return location if isinstance(location, str) else None

    @property
    def fw_ver(self) -> Optional[str]:
        """Return this hub firmware version."""
        fw_ver = self._version.get("fw_ver")
        return fw_ver if isinstance(fw_ver, str) else None

    @property
    def mcu_ver(self) -> Optional[str]:
        """Return this hub hardware version."""
        mcu_ver = self._version.get("mcu_ver")
        return mcu_ver if isinstance(mcu_ver, str) else None

    @property
    def devices(self) -> List[BondDevice]:
        """Return a list of all devices controlled by this hub."""
        return self._devices

    @property
    def is_bridge(self) -> bool:
        """Return if the Bond is a Bond Bridge."""
        bondid = self._version.get("bondid", "")
        return bool(BondType.is_bridge_from_serial(bondid))
