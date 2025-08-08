from __future__ import annotations
import logging
from typing import Any, cast, Set, Dict, List, Union
from aiohttp import ClientResponseError
from bond_async import Action, Bond, BondType
from homeassistant.util.async_ import gather_with_limited_concurrency
from .const import BRIDGE_MAKE

MAX_REQUESTS: int = 6
_LOGGER: logging.Logger = logging.getLogger(__name__)

class BondDevice:
    def __init__(self, device_id: str, attrs: Dict[str, Any], props: Dict[str, Any], state: Any) -> None:
    def __repr__(self) -> str:
    @property
    def name(self) -> str:
    @property
    def type(self) -> str:
    @property
    def location(self) -> Union[str, None]:
    @property
    def template(self) -> Union[str, None]:
    @property
    def branding_profile(self) -> Any:
    @property
    def trust_state(self) -> bool:
    def has_action(self, action: Action) -> bool:
    def _has_any_action(self, actions: Set[Action]) -> bool:
    def supports_speed(self) -> bool:
    def supports_direction(self) -> bool:
    def supports_set_position(self) -> bool:
    def supports_open(self) -> bool:
    def supports_close(self) -> bool:
    def supports_tilt_open(self) -> bool:
    def supports_tilt_close(self) -> bool:
    def supports_hold(self) -> bool:
    def supports_light(self) -> bool:
    def supports_up_light(self) -> bool:
    def supports_down_light(self) -> bool:
    def supports_set_brightness(self) -> bool:

class BondHub:
    def __init__(self, bond: Bond, host: str) -> None:
    async def setup(self, max_devices: Union[int, None] = None) -> None:
    @property
    def bond_id(self) -> Union[str, None]:
    @property
    def target(self) -> Union[str, None]:
    @property
    def model(self) -> Union[str, None]:
    @property
    def make(self) -> str:
    @property
    def name(self) -> str:
    @property
    def location(self) -> Union[str, None]:
    @property
    def fw_ver(self) -> Union[str, None]:
    @property
    def mcu_ver(self) -> Union[str, None]:
    @property
    def devices(self) -> List[BondDevice]:
    @property
    def is_bridge(self) -> bool:
