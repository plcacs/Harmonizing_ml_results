from typing import Any

# === Internal dependency: homeassistant.components.fan ===
class FanEntityFeature(IntFlag): ...
class FanEntity(ToggleEntity):
    def is_on(self) -> bool | None: ...
    def percentage_step(self) -> float: ...
    def capability_attributes(self) -> dict[str, list[str] | None]: ...
    def state_attributes(self) -> dict[str, float | str | None]: ...
    def supported_features_compat(self) -> FanEntityFeature: ...

# === Internal dependency: homeassistant.components.xiaomi_miio.const ===
DOMAIN: str
CONF_FLOW_TYPE: str
KEY_COORDINATOR: str
KEY_DEVICE: str
MODEL_AIRPURIFIER_4: str
MODEL_AIRPURIFIER_4_LITE_RMA1: str
MODEL_AIRPURIFIER_4_LITE_RMB1: str
MODEL_AIRPURIFIER_4_PRO: str
MODEL_AIRPURIFIER_2H: str
MODEL_AIRPURIFIER_2S: str
MODEL_AIRPURIFIER_3C: str
MODEL_AIRPURIFIER_PRO: str
MODEL_AIRPURIFIER_PRO_V7: str
MODEL_AIRPURIFIER_V3: str
MODEL_AIRPURIFIER_ZA1: str
MODEL_AIRFRESH_A1: str
MODEL_AIRFRESH_T2017: str
MODEL_FAN_1C: str
MODEL_FAN_P10: str
MODEL_FAN_P11: str
MODEL_FAN_P5: str
MODEL_FAN_P9: str
MODEL_FAN_ZA5: str
MODELS_FAN_MIIO: Any
MODELS_FAN_MIOT: Any
MODELS_PURIFIER_MIOT: Any
SERVICE_RESET_FILTER: str
SERVICE_SET_EXTRA_FEATURES: str
FEATURE_RESET_FILTER: int
FEATURE_SET_EXTRA_FEATURES: int
FEATURE_FLAGS_AIRPURIFIER_MIIO: Any
FEATURE_FLAGS_AIRPURIFIER_MIOT: Any
FEATURE_FLAGS_AIRPURIFIER_4_LITE: Any
FEATURE_FLAGS_AIRPURIFIER_4: Any
FEATURE_FLAGS_AIRPURIFIER_3C: Any
FEATURE_FLAGS_AIRPURIFIER_PRO: Any
FEATURE_FLAGS_AIRPURIFIER_PRO_V7: Any
FEATURE_FLAGS_AIRPURIFIER_2S: Any
FEATURE_FLAGS_AIRPURIFIER_V3: Any
FEATURE_FLAGS_AIRPURIFIER_ZA1: Any
FEATURE_FLAGS_AIRFRESH_A1: Any
FEATURE_FLAGS_AIRFRESH: Any
FEATURE_FLAGS_AIRFRESH_T2017: Any
FEATURE_FLAGS_FAN_P5: Any
FEATURE_FLAGS_FAN: Any
FEATURE_FLAGS_FAN_ZA5: Any
FEATURE_FLAGS_FAN_1C: Any
FEATURE_FLAGS_FAN_P9: Any
FEATURE_FLAGS_FAN_P10_P11: Any

# === Internal dependency: homeassistant.components.xiaomi_miio.device ===
class XiaomiCoordinatedMiioEntity(CoordinatorEntity[_T]):
    def unique_id(self) -> Any: ...
    def device_info(self) -> DeviceInfo: ...

# === Internal dependency: homeassistant.components.xiaomi_miio.typing ===
class ServiceMethodDetails(NamedTuple): ...

# === Internal dependency: homeassistant.const ===
CONF_DEVICE: Final
CONF_MODEL: Final
ATTR_ENTITY_ID: Final

# === Internal dependency: homeassistant.core ===
def callback(func: _CallableT) -> _CallableT: ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def entity_ids(value: str | list) -> list[str]: ...
positive_int: All

# === Internal dependency: homeassistant.util.percentage ===
def ranged_value_to_percentage(low_high_range: tuple[float, float], value: float) -> int: ...
def percentage_to_ranged_value(low_high_range: tuple[float, float], percentage: float) -> float: ...

# === Unresolved dependency: miio.fan_common ===
# Used unresolved symbols: MoveDirection, OperationMode

# === Unresolved dependency: miio.integrations.airpurifier.dmaker.airfresh_t2017 ===
# Used unresolved symbols: OperationMode

# === Unresolved dependency: miio.integrations.airpurifier.zhimi.airfresh ===
# Used unresolved symbols: OperationMode

# === Unresolved dependency: miio.integrations.airpurifier.zhimi.airpurifier ===
# Used unresolved symbols: OperationMode

# === Unresolved dependency: miio.integrations.airpurifier.zhimi.airpurifier_miot ===
# Used unresolved symbols: OperationMode

# === Unresolved dependency: miio.integrations.fan.zhimi.zhimi_miot ===
# Used unresolved symbols: OperationModeFanZA5

# === Third-party dependency: voluptuous ===
# Used symbols: All, Optional, Required, Schema