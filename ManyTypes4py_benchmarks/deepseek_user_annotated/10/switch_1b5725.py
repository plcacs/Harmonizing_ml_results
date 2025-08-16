"""Support for Xiaomi Smart WiFi Socket and Smart Power Strip."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from miio import AirConditioningCompanionV3, ChuangmiPlug, DeviceException, PowerStrip
from miio.powerstrip import PowerMode
import voluptuous as vol

from homeassistant.components.switch import (
    SwitchDeviceClass,
    SwitchEntity,
    SwitchEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_MODE,
    ATTR_TEMPERATURE,
    CONF_DEVICE,
    CONF_HOST,
    CONF_MODEL,
    CONF_TOKEN,
    EntityCategory,
)
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback, AddEntityCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import (
    CONF_FLOW_TYPE,
    CONF_GATEWAY,
    DOMAIN,
    FEATURE_FLAGS_AIRFRESH,
    FEATURE_FLAGS_AIRFRESH_A1,
    FEATURE_FLAGS_AIRFRESH_T2017,
    FEATURE_FLAGS_AIRFRESH_VA4,
    FEATURE_FLAGS_AIRHUMIDIFIER,
    FEATURE_FLAGS_AIRHUMIDIFIER_CA4,
    FEATURE_FLAGS_AIRHUMIDIFIER_CA_AND_CB,
    FEATURE_FLAGS_AIRHUMIDIFIER_MJSSQ,
    FEATURE_FLAGS_AIRPURIFIER_2S,
    FEATURE_FLAGS_AIRPURIFIER_3C,
    FEATURE_FLAGS_AIRPURIFIER_4,
    FEATURE_FLAGS_AIRPURIFIER_4_LITE,
    FEATURE_FLAGS_AIRPURIFIER_MIIO,
    FEATURE_FLAGS_AIRPURIFIER_MIOT,
    FEATURE_FLAGS_AIRPURIFIER_PRO,
    FEATURE_FLAGS_AIRPURIFIER_PRO_V7,
    FEATURE_FLAGS_AIRPURIFIER_V1,
    FEATURE_FLAGS_AIRPURIFIER_V3,
    FEATURE_FLAGS_AIRPURIFIER_ZA1,
    FEATURE_FLAGS_FAN,
    FEATURE_FLAGS_FAN_1C,
    FEATURE_FLAGS_FAN_P5,
    FEATURE_FLAGS_FAN_P9,
    FEATURE_FLAGS_FAN_P10_P11_P18,
    FEATURE_FLAGS_FAN_ZA5,
    FEATURE_SET_ANION,
    FEATURE_SET_AUTO_DETECT,
    FEATURE_SET_BUZZER,
    FEATURE_SET_CHILD_LOCK,
    FEATURE_SET_CLEAN,
    FEATURE_SET_DISPLAY,
    FEATURE_SET_DRY,
    FEATURE_SET_IONIZER,
    FEATURE_SET_LEARN_MODE,
    FEATURE_SET_LED,
    FEATURE_SET_PTC,
    KEY_COORDINATOR,
    KEY_DEVICE,
    MODEL_AIRFRESH_A1,
    MODEL_AIRFRESH_T2017,
    MODEL_AIRFRESH_VA2,
    MODEL_AIRFRESH_VA4,
    MODEL_AIRHUMIDIFIER_CA1,
    MODEL_AIRHUMIDIFIER_CA4,
    MODEL_AIRHUMIDIFIER_CB1,
    MODEL_AIRPURIFIER_2H,
    MODEL_AIRPURIFIER_2S,
    MODEL_AIRPURIFIER_3C,
    MODEL_AIRPURIFIER_3C_REV_A,
    MODEL_AIRPURIFIER_4,
    MODEL_AIRPURIFIER_4_LITE_RMA1,
    MODEL_AIRPURIFIER_4_LITE_RMB1,
    MODEL_AIRPURIFIER_4_PRO,
    MODEL_AIRPURIFIER_PRO,
    MODEL_AIRPURIFIER_PRO_V7,
    MODEL_AIRPURIFIER_V1,
    MODEL_AIRPURIFIER_V3,
    MODEL_AIRPURIFIER_ZA1,
    MODEL_FAN_1C,
    MODEL_FAN_P5,
    MODEL_FAN_P9,
    MODEL_FAN_P10,
    MODEL_FAN_P11,
    MODEL_FAN_P18,
    MODEL_FAN_ZA1,
    MODEL_FAN_ZA3,
    MODEL_FAN_ZA4,
    MODEL_FAN_ZA5,
    MODELS_FAN,
    MODELS_HUMIDIFIER,
    MODELS_HUMIDIFIER_MJJSQ,
    MODELS_PURIFIER_MIIO,
    MODELS_PURIFIER_MIOT,
    SERVICE_SET_POWER_MODE,
    SERVICE_SET_POWER_PRICE,
    SERVICE_SET_WIFI_LED_OFF,
    SERVICE_SET_WIFI_LED_ON,
    SUCCESS,
)
from .entity import XiaomiCoordinatedMiioEntity, XiaomiGatewayDevice, XiaomiMiioEntity
from .typing import ServiceMethodDetails

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME: str = "Xiaomi Miio Switch"
DATA_KEY: str = "switch.xiaomi_miio"

MODEL_POWER_STRIP_V2: str = "zimi.powerstrip.v2"
MODEL_PLUG_V3: str = "chuangmi.plug.v3"

KEY_CHANNEL: str = "channel"
GATEWAY_SWITCH_VARS: Dict[str, Dict[str, int]] = {
    "status_ch0": {KEY_CHANNEL: 0},
    "status_ch1": {KEY_CHANNEL: 1},
    "status_ch2": {KEY_CHANNEL: 2},
}

ATTR_AUTO_DETECT: str = "auto_detect"
ATTR_BUZZER: str = "buzzer"
ATTR_CHILD_LOCK: str = "child_lock"
ATTR_CLEAN: str = "clean_mode"
ATTR_DISPLAY: str = "display"
ATTR_DRY: str = "dry"
ATTR_LEARN_MODE: str = "learn_mode"
ATTR_LED: str = "led"
ATTR_IONIZER: str = "ionizer"
ATTR_ANION: str = "anion"
ATTR_LOAD_POWER: str = "load_power"
ATTR_MODEL: str = "model"
ATTR_POWER: str = "power"
ATTR_POWER_MODE: str = "power_mode"
ATTR_POWER_PRICE: str = "power_price"
ATTR_PRICE: str = "price"
ATTR_PTC: str = "ptc"
ATTR_WIFI_LED: str = "wifi_led"

FEATURE_SET_POWER_MODE: int = 1
FEATURE_SET_WIFI_LED: int = 2
FEATURE_SET_POWER_PRICE: int = 4

FEATURE_FLAGS_GENERIC: int = 0

FEATURE_FLAGS_POWER_STRIP_V1: int = (
    FEATURE_SET_POWER_MODE | FEATURE_SET_WIFI_LED | FEATURE_SET_POWER_PRICE
)

FEATURE_FLAGS_POWER_STRIP_V2: int = FEATURE_SET_WIFI_LED | FEATURE_SET_POWER_PRICE

FEATURE_FLAGS_PLUG_V3: int = FEATURE_SET_WIFI_LED

SERVICE_SCHEMA: vol.Schema = vol.Schema({vol.Optional(ATTR_ENTITY_ID): cv.entity_ids})

SERVICE_SCHEMA_POWER_MODE: vol.Schema = SERVICE_SCHEMA.extend(
    {vol.Required(ATTR_MODE): vol.All(vol.In(["green", "normal"]))}
)

SERVICE_SCHEMA_POWER_PRICE: vol.Schema = SERVICE_SCHEMA.extend(
    {vol.Required(ATTR_PRICE): cv.positive_float}
)

SERVICE_TO_METHOD: Dict[str, ServiceMethodDetails] = {
    SERVICE_SET_WIFI_LED_ON: ServiceMethodDetails(method="async_set_wifi_led_on"),
    SERVICE_SET_WIFI_LED_OFF: ServiceMethodDetails(method="async_set_wifi_led_off"),
    SERVICE_SET_POWER_MODE: ServiceMethodDetails(
        method="async_set_power_mode",
        schema=SERVICE_SCHEMA_POWER_MODE,
    ),
    SERVICE_SET_POWER_PRICE: ServiceMethodDetails(
        method="async_set_power_price",
        schema=SERVICE_SCHEMA_POWER_PRICE,
    ),
}

MODEL_TO_FEATURES_MAP: Dict[str, int] = {
    MODEL_AIRFRESH_A1: FEATURE_FLAGS_AIRFRESH_A1,
    MODEL_AIRFRESH_VA2: FEATURE_FLAGS_AIRFRESH,
    MODEL_AIRFRESH_VA4: FEATURE_FLAGS_AIRFRESH_VA4,
    MODEL_AIRFRESH_T2017: FEATURE_FLAGS_AIRFRESH_T2017,
    MODEL_AIRHUMIDIFIER_CA1: FEATURE_FLAGS_AIRHUMIDIFIER_CA_AND_CB,
    MODEL_AIRHUMIDIFIER_CA4: FEATURE_FLAGS_AIRHUMIDIFIER_CA4,
    MODEL_AIRHUMIDIFIER_CB1: FEATURE_FLAGS_AIRHUMIDIFIER_CA_AND_CB,
    MODEL_AIRPURIFIER_2H: FEATURE_FLAGS_AIRPURIFIER_2S,
    MODEL_AIRPURIFIER_2S: FEATURE_FLAGS_AIRPURIFIER_2S,
    MODEL_AIRPURIFIER_3C: FEATURE_FLAGS_AIRPURIFIER_3C,
    MODEL_AIRPURIFIER_3C_REV_A: FEATURE_FLAGS_AIRPURIFIER_3C,
    MODEL_AIRPURIFIER_PRO: FEATURE_FLAGS_AIRPURIFIER_PRO,
    MODEL_AIRPURIFIER_PRO_V7: FEATURE_FLAGS_AIRPURIFIER_PRO_V7,
    MODEL_AIRPURIFIER_V1: FEATURE_FLAGS_AIRPURIFIER_V1,
    MODEL_AIRPURIFIER_V3: FEATURE_FLAGS_AIRPURIFIER_V3,
    MODEL_AIRPURIFIER_4_LITE_RMA1: FEATURE_FLAGS_AIRPURIFIER_4_LITE,
    MODEL_AIRPURIFIER_4_LITE_RMB1: FEATURE_FLAGS_AIRPURIFIER_4_LITE,
    MODEL_AIRPURIFIER_4: FEATURE_FLAGS_AIRPURIFIER_4,
    MODEL_AIRPURIFIER_4_PRO: FEATURE_FLAGS_AIRPURIFIER_4,
    MODEL_AIRPURIFIER_ZA1: FEATURE_FLAGS_AIRPURIFIER_ZA1,
    MODEL_FAN_1C: FEATURE_FLAGS_FAN_1C,
    MODEL_FAN_P10: FEATURE_FLAGS_FAN_P10_P11_P18,
    MODEL_FAN_P11: FEATURE_FLAGS_FAN_P10_P11_P18,
    MODEL_FAN_P18: FEATURE_FLAGS_FAN_P10_P11_P18,
    MODEL_FAN_P5: FEATURE_FLAGS_FAN_P5,
    MODEL_FAN_P9: FEATURE_FLAGS_FAN_P9,
    MODEL_FAN_ZA1: FEATURE_FLAGS_FAN,
    MODEL_FAN_ZA3: FEATURE_FLAGS_FAN,
    MODEL_FAN_ZA4: FEATURE_FLAGS_FAN,
    MODEL_FAN_ZA5: FEATURE_FLAGS_FAN_ZA5,
}

@dataclass(frozen=True, kw_only=True)
class XiaomiMiioSwitchDescription(SwitchEntityDescription):
    """A class that describes switch entities."""

    feature: int
    method_on: str
    method_off: str
    available_with_device_off: bool = True

SWITCH_TYPES: Tuple[XiaomiMiioSwitchDescription, ...] = (
    XiaomiMiioSwitchDescription(
        key=ATTR_BUZZER,
        feature=FEATURE_SET_BUZZER,
        translation_key=ATTR_BUZZER,
        icon="mdi:volume-high",
        method_on="async_set_buzzer_on",
        method_off="async_set_buzzer_off",
        entity_category=EntityCategory.CONFIG,
    ),
    # ... (rest of the SWITCH_TYPES tuple remains the same)
)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the switch from a config entry."""
    model: str = config_entry.data[CONF_MODEL]
    if model in (*MODELS_HUMIDIFIER, *MODELS_FAN):
        await async_setup_coordinated_entry(hass, config_entry, async_add_entities)
    else:
        await async_setup_other_entry(hass, config_entry, async_add_entities)

async def async_setup_coordinated_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the coordinated switch from a config entry."""
    model: str = config_entry.data[CONF_MODEL]
    unique_id: str = config_entry.unique_id
    device = hass.data[DOMAIN][config_entry.entry_id][KEY_DEVICE]
    coordinator = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR]

    if DATA_KEY not in hass.data:
        hass.data[DATA_KEY] = {}

    device_features: int = 0

    if model in MODEL_TO_FEATURES_MAP:
        device_features = MODEL_TO_FEATURES_MAP[model]
    elif model in MODELS_HUMIDIFIER_MJJSQ:
        device_features = FEATURE_FLAGS_AIRHUMIDIFIER_MJSSQ
    elif model in MODELS_HUMIDIFIER:
        device_features = FEATURE_FLAGS_AIRHUMIDIFIER
    elif model in MODELS_PURIFIER_MIIO:
        device_features = FEATURE_FLAGS_AIRPURIFIER_MIIO
    elif model in MODELS_PURIFIER_MIOT:
        device_features = FEATURE_FLAGS_AIRPURIFIER_MIOT

    async_add_entities(
        XiaomiGenericCoordinatedSwitch(
            device,
            config_entry,
            f"{description.key}_{unique_id}",
            coordinator,
            description,
        )
        for description in SWITCH_TYPES
        if description.feature & device_features
    )

async def async_setup_other_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the other type switch from a config entry."""
    entities: List[SwitchEntity] = []
    host: str = config_entry.data[CONF_HOST]
    token: str = config_entry.data[CONF_TOKEN]
    name: str = config_entry.title
    model: str = config_entry.data[CONF_MODEL]
    unique_id: str = config_entry.unique_id
    
    if config_entry.data[CONF_FLOW_TYPE] == CONF_GATEWAY:
        gateway = hass.data[DOMAIN][config_entry.entry_id][CONF_GATEWAY]
        sub_devices = gateway.devices
        for sub_device in sub_devices.values():
            if sub_device.device_type != "Switch":
                continue
            coordinator = hass.data[DOMAIN][config_entry.entry_id][KEY_COORDINATOR][
                sub_device.sid
            ]
            switch_variables: Set[str] = set(sub_device.status) & set(GATEWAY_SWITCH_VARS)
            if switch_variables:
                entities.extend(
                    [
                        XiaomiGatewaySwitch(
                            coordinator, sub_device, config_entry, variable
                        )
                        for variable in switch_variables
                    ]
                )

    if config_entry.data[CONF_FLOW_TYPE] == CONF_DEVICE or (
        config_entry.data[CONF_FLOW_TYPE] == CONF_GATEWAY
        and model == "lumi.acpartner.v3"
    ):
        if DATA_KEY not in hass.data:
            hass.data[DATA_KEY] = {}

        _LOGGER.debug("Initializing with host %s (token %s...)", host, token[:5])

        if model in ["chuangmi.plug.v1", "chuangmi.plug.v3", "chuangmi.plug.hmi208"]:
            plug = ChuangmiPlug(host, token, model=model)

            for channel_usb in (True, False):
                if channel_usb:
                    unique_id_ch = f"{unique_id}-USB"
                else:
                    unique_id_ch = f"{unique_id}-mains"
                device = ChuangMiPlugSwitch(
                    name, plug, config_entry, unique_id_ch, channel_usb
                )
                entities.append(device)
                hass.data[DATA_KEY][host] = device
        elif model in ["qmi.powerstrip.v1", "zimi.powerstrip.v2"]:
            plug = PowerStrip(host, token, model=model)
            device = XiaomiPowerStripSwitch(name, plug, config_entry, unique_id)
            entities.append(device)
            hass.data[DATA_KEY][host] = device
        elif model in [
            "chuangmi.plug.m1",
            "chuangmi.plug.m3",
            "chuangmi.plug.v2",
            "chuangmi.plug.hmi205",
            "chuangmi.plug.hmi206",
        ]:
            plug = ChuangmiPlug(host, token, model=model)
            device = XiaomiPlugGenericSwitch(name, plug, config_entry, unique_id)
            entities.append(device)
            hass.data[DATA_KEY][host] = device
        elif model in ["lumi.acpartner.v3"]:
            plug = AirConditioningCompanionV3(host, token)
            device = XiaomiAirConditioningCompanionSwitch(
                name, plug, config_entry, unique_id
            )
            entities.append(device)
            hass.data[DATA_KEY][host] = device
        else:
            _LOGGER.error(
                "Unsupported device found! Please create an issue at "
                "https://github.com