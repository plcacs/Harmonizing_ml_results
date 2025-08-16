    def __init__(self, device_id: str, attrs: dict, props: dict, state: dict) -> None:
    def __repr__(self) -> str:
    @property
    def name(self) -> str:
    @property
    def type(self) -> str:
    @property
    def location(self) -> str:
    @property
    def template(self) -> str:
    @property
    def branding_profile(self) -> Any:
    @property
    def trust_state(self) -> bool:
    def has_action(self, action: Action) -> bool:
    def _has_any_action(self, actions: set) -> bool:
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

    def __init__(self, bond: Bond, host: str) -> None:
    async def setup(self, max_devices: int = None) -> None:
    @property
    def bond_id(self) -> str:
    @property
    def target(self) -> str:
    @property
    def model(self) -> str:
    @property
    def make(self) -> str:
    @property
    def name(self) -> str:
    @property
    def location(self) -> str:
    @property
    def fw_ver(self) -> str:
    @property
    def mcu_ver(self) -> str:
    @property
    def devices(self) -> list[BondDevice]:
    @property
    def is_bridge(self) -> bool:
