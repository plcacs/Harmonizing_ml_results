from typing import Optional, Set

@dataclass
class RainbirdDeviceState:
    zones: Optional[Set[int]]
    active_zones: Optional[Set[int]]
    rain: Optional[bool]
    rain_delay: Optional[int]
