from __future__ import annotations
import logging
from typing import Any, cast, List, Optional, Set
from aiohttp import ClientResponseError
from bond_async import Action, Bond, BondType
from homeassistant.util.async_ import gather_with_limited_concurrency
from .const import BRIDGE_MAKE

MAX_REQUESTS = 6
_LOGGER = logging.getLogger(__name__)

class BondDevice:
    """Helper device class to hold ID and attributes together."""

    def __init__(self, device_id: str, attrs: dict, props: dict, state: Any) -> None:
        """Create a helper device from ID and attributes returned by API."""
        self.device_id: str = device_id
        self.props: dict = props
        self.state: Any = state
        self._attrs: dict = attrs or {}
        self._supported_actions: Set[Action] = set(self._attrs.get('actions', []))

    # ... (rest of the class remains the same)

class BondHub:
    """Hub device representing Bond Bridge."""

    def __init__(self, bond: Bond, host: str) -> None:
        """Initialize Bond Hub."""
        self.bond: Bond = bond
        self.host: str = host
        self._bridge: dict = {}
        self._version: dict = {}
        self._devices: List[BondDevice] = []

    async def setup(self, max_devices: Optional[int] = None) -> None:
        """Read hub version information."""
        self._version = await self.bond.version()
        _LOGGER.debug('Bond reported the following version info: %s', self._version)
        device_ids: List[str] = await self.bond.devices()
        self._devices = []
        setup_device_ids: List[str] = []
        tasks: List[asyncio.Task] = []
        for idx, device_id in enumerate(device_ids):
            if max_devices is not None and idx >= max_devices:
                break
            setup_device_ids.append(device_id)
            tasks.extend([self.bond.device(device_id), self.bond.device_properties(device_id), self.bond.device_state(device_id)])
        responses: List[Any] = await gather_with_limited_concurrency(MAX_REQUESTS, *tasks)
        response_idx: int = 0
        for device_id in setup_device_ids:
            self._devices.append(BondDevice(device_id, responses[response_idx], responses[response_idx + 1], responses[response_idx + 2]))
            response_idx += 3
        _LOGGER.debug('Discovered Bond devices: %s', self._devices)
        try:
            self._bridge = await self.bond.bridge()
        except ClientResponseError:
            self._bridge = {}
        _LOGGER.debug('Bond reported the following bridge info: %s', self._bridge)

    # ... (rest of the class remains the same)
