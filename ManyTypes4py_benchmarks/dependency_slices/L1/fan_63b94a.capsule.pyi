# === Internal dependency: homeassistant.components.fan ===
class FanEntityFeature(IntFlag): ...
class FanEntity(ToggleEntity):
    def is_on(self): ...
    def percentage_step(self): ...
    def capability_attributes(self): ...
    def state_attributes(self): ...
    def supported_features_compat(self): ...

# === Internal dependency: homeassistant.components.xiaomi_miio.const ===
DOMAIN = 'xiaomi_miio'
CONF_FLOW_TYPE = 'config_flow_device'
KEY_COORDINATOR = 'coordinator'
KEY_DEVICE = 'device'
MODEL_AIRPURIFIER_4 = 'zhimi.airp.mb5'
MODEL_AIRPURIFIER_4_LITE_RMA1 = 'zhimi.airpurifier.rma1'
MODEL_AIRPURIFIER_4_LITE_RMB1 = 'zhimi.airp.rmb1'
MODEL_AIRPURIFIER_4_PRO = 'zhimi.airp.vb4'
MODEL_AIRPURIFIER_2H = 'zhimi.airpurifier.mc2'
MODEL_AIRPURIFIER_2S = 'zhimi.airpurifier.mc1'
MODEL_AIRPURIFIER_3 = 'zhimi.airpurifier.ma4'
MODEL_AIRPURIFIER_3C = 'zhimi.airpurifier.mb4'
MODEL_AIRPURIFIER_3H = 'zhimi.airpurifier.mb3'
MODEL_AIRPURIFIER_PRO = 'zhimi.airpurifier.v6'
MODEL_AIRPURIFIER_PROH = 'zhimi.airpurifier.va1'
MODEL_AIRPURIFIER_PROH_EU = 'zhimi.airpurifier.vb2'
MODEL_AIRPURIFIER_PRO_V7 = 'zhimi.airpurifier.v7'
MODEL_AIRPURIFIER_V3 = 'zhimi.airpurifier.v3'
MODEL_AIRPURIFIER_ZA1 = 'zhimi.airpurifier.za1'
MODEL_AIRFRESH_A1 = 'dmaker.airfresh.a1'
MODEL_AIRFRESH_T2017 = 'dmaker.airfresh.t2017'
MODEL_FAN_1C = 'dmaker.fan.1c'
MODEL_FAN_P10 = 'dmaker.fan.p10'
MODEL_FAN_P11 = 'dmaker.fan.p11'
MODEL_FAN_P5 = 'dmaker.fan.p5'
MODEL_FAN_P9 = 'dmaker.fan.p9'
MODEL_FAN_SA1 = 'zhimi.fan.sa1'
MODEL_FAN_V2 = 'zhimi.fan.v2'
MODEL_FAN_V3 = 'zhimi.fan.v3'
MODEL_FAN_ZA1 = 'zhimi.fan.za1'
MODEL_FAN_ZA3 = 'zhimi.fan.za3'
MODEL_FAN_ZA4 = 'zhimi.fan.za4'
MODEL_FAN_ZA5 = 'zhimi.fan.za5'
MODELS_FAN_MIIO = [MODEL_FAN_P5, MODEL_FAN_SA1, MODEL_FAN_V2, MODEL_FAN_V3, MODEL_FAN_ZA1, MODEL_FAN_ZA3, MODEL_FAN_ZA4]
MODELS_FAN_MIOT = [MODEL_FAN_1C, MODEL_FAN_P10, MODEL_FAN_P11, MODEL_FAN_P9, MODEL_FAN_ZA5]
MODELS_PURIFIER_MIOT = [MODEL_AIRPURIFIER_3, MODEL_AIRPURIFIER_3C, MODEL_AIRPURIFIER_3H, MODEL_AIRPURIFIER_PROH, MODEL_AIRPURIFIER_PROH_EU, MODEL_AIRPURIFIER_4_LITE_RMA1, MODEL_AIRPURIFIER_4_LITE_RMB1, MODEL_AIRPURIFIER_4, ...]
SERVICE_RESET_FILTER = 'fan_reset_filter'
SERVICE_SET_EXTRA_FEATURES = 'fan_set_extra_features'
FEATURE_SET_BUZZER = 1
FEATURE_SET_LED = 2
FEATURE_SET_CHILD_LOCK = 4
FEATURE_SET_LED_BRIGHTNESS = 8
FEATURE_SET_FAVORITE_LEVEL = 16
FEATURE_SET_LEARN_MODE = 64
FEATURE_SET_VOLUME = 128
FEATURE_RESET_FILTER = 256
FEATURE_SET_EXTRA_FEATURES = 512
FEATURE_SET_FAN_LEVEL = 4096
FEATURE_SET_OSCILLATION_ANGLE = 32768
FEATURE_SET_DELAY_OFF_COUNTDOWN = 65536
FEATURE_SET_LED_BRIGHTNESS_LEVEL = 131072
FEATURE_SET_FAVORITE_RPM = 262144
FEATURE_SET_IONIZER = 524288
FEATURE_SET_DISPLAY = 1048576
FEATURE_SET_PTC = 2097152
FEATURE_SET_ANION = 4194304
FEATURE_FLAGS_AIRPURIFIER_MIIO = ...
FEATURE_FLAGS_AIRPURIFIER_MIOT = ...
FEATURE_FLAGS_AIRPURIFIER_4_LITE = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED_BRIGHTNESS
FEATURE_FLAGS_AIRPURIFIER_4 = ...
FEATURE_FLAGS_AIRPURIFIER_3C = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED_BRIGHTNESS_LEVEL | FEATURE_SET_FAVORITE_RPM
FEATURE_FLAGS_AIRPURIFIER_PRO = FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED | FEATURE_SET_FAVORITE_LEVEL | FEATURE_SET_VOLUME
FEATURE_FLAGS_AIRPURIFIER_PRO_V7 = FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED | FEATURE_SET_FAVORITE_LEVEL | FEATURE_SET_VOLUME
FEATURE_FLAGS_AIRPURIFIER_2S = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED | FEATURE_SET_FAVORITE_LEVEL
FEATURE_FLAGS_AIRPURIFIER_V3 = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED
FEATURE_FLAGS_AIRPURIFIER_ZA1 = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_FAVORITE_LEVEL
FEATURE_FLAGS_AIRFRESH_A1 = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_DISPLAY | FEATURE_SET_PTC
FEATURE_FLAGS_AIRFRESH = ...
FEATURE_FLAGS_AIRFRESH_T2017 = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_DISPLAY | FEATURE_SET_PTC
FEATURE_FLAGS_FAN_P5 = ...
FEATURE_FLAGS_FAN = ...
FEATURE_FLAGS_FAN_ZA5 = ...
FEATURE_FLAGS_FAN_1C = FEATURE_SET_BUZZER | FEATURE_SET_CHILD_LOCK | FEATURE_SET_LED | FEATURE_SET_DELAY_OFF_COUNTDOWN
FEATURE_FLAGS_FAN_P9 = ...
FEATURE_FLAGS_FAN_P10_P11 = ...

# === Internal dependency: homeassistant.components.xiaomi_miio.device ===
class XiaomiCoordinatedMiioEntity(CoordinatorEntity[_T]):
    def unique_id(self): ...
    def device_info(self): ...

# === Internal dependency: homeassistant.components.xiaomi_miio.typing ===
class ServiceMethodDetails(NamedTuple): ...

# === Internal dependency: homeassistant.const ===
CONF_DEVICE = 'device'
CONF_MODEL = 'model'
ATTR_ENTITY_ID = 'entity_id'

# === Internal dependency: homeassistant.core ===
def callback(func): ...

# === Internal dependency: homeassistant.helpers.config_validation ===
def entity_ids(value): ...
positive_int = vol.All(...)

# === Internal dependency: homeassistant.util.percentage ===
def ranged_value_to_percentage(low_high_range, value): ...
def percentage_to_ranged_value(low_high_range, percentage): ...

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