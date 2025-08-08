def modbus_create_issue(hass: HomeAssistant, key: str, subs: list[str], err: str) -> None:
def struct_validator(config: dict[str, Any]) -> dict[str, Any]:
def hvac_fixedsize_reglist_validator(value: Union[int, list[int]]) -> list[int]:
def nan_validator(value: Union[int, str]) -> int:
def duplicate_fan_mode_validator(config: dict[str, Any]) -> dict[str, Any]:
def duplicate_swing_mode_validator(config: dict[str, Any]) -> dict[str, Any]:
def register_int_list_validator(value: Union[int, list[int]]) -> Union[int, list[int]]:
def validate_modbus(hass: HomeAssistant, hosts: set[str], hub_names: set[str], hub: dict[str, Any], hub_name_inx: int) -> bool:
def validate_entity(hass: HomeAssistant, hub_name: str, component: str, entity: dict[str, Any], minimum_scan_interval: int, ent_names: set[str], ent_addr: set[int]) -> bool:
def check_config(hass: HomeAssistant, config: list[dict[str, Any]]) -> list[dict[str, Any]]:
