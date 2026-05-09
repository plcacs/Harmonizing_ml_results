from __future__ import annotations
import asyncio
from dataclasses import dataclass, kw_only
from functools import partial
import logging
from typing import Any, Dict, List, Optional, Tuple
from miio import AirConditioningCompanionV3, ChuangmiPlug, DeviceException, PowerStrip
from miio.powerstrip import PowerMode
import voluptuous as vol
from homeassistant.components.switch import (
    SwitchDeviceClass,
    SwitchEntity,
    SwitchEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_ENTITY_ID, CONF_DEVICE, CONF_HOST, CONF_MODEL, CONF_TOKEN, EntityCategory
from homeassistant.core import HomeAssistant, ServiceCall, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import (
    CONF_FLOW_TYPE,
    CONF_GATEWAY,
    DEFAULT_NAME,
    DATA_KEY,
    FEATURE_FLAGS_AIRFRESH,
    FEATURE_FLAGS_AIRFRESH_A1,
    FEATURE_FLAGS_AIRFRESH_T2017,
    FEATURE_FLAGS_AIRHUMIDIFIER,
    FEATURE_FLAGS_AIRHUMIDIFIER_CA1,
    FEATURE_FLAGS_AIRHUMIDIFIER_CA4,
    FEATURE_FLAGS_AIRHUMIDIFIER_CB1,
    FEATURE_FLAGS_AIRPURIFIER_2H,
    FEATURE_FLAGS_AIRPURIFIER_2S,
    FEATURE_FLAGS_AIRPURIFIER_3C,
    FEATURE_FLAGS_AIRPURIFIER_3C_REV_A,
    FEATURE_FLAGS_AIRPURIFIER_4,
    FEATURE_FLAGS_AIRPURIFIER_4_LITE,
    FEATURE_FLAGS_AIRPURIFIER_4_PRO,
    FEATURE_FLAGS_AIRPURIFIER_4_LITE_RMA1,
    FEATURE_FLAGS_AIRPURIFIER_4_LITE_RMB1,
    FEATURE_FLAGS_AIRPURIFIER_ZA1,
    FEATURE_FLAGS_FAN,
    FEATURE_FLAGS_FAN_1C,
    FEATURE_FLAGS_FAN_P10,
    FEATURE_FLAGS_FAN_P11,
    FEATURE_FLAGS_FAN_P18,
    FEATURE_FLAGS_FAN_P5,
    FEATURE_FLAGS_FAN_P9,
    FEATURE_FLAGS_FAN_ZA1,
    FEATURE_FLAGS_FAN_ZA3,
    FEATURE_FLAGS_FAN_ZA4,
    FEATURE_FLAGS_FAN_ZA5,
    GATEWAY_SWITCH_VARS,
    KEY_CHANNEL,
    KEY_DEVICE,
    KEY_COORDINATOR,
    MODEL_AIRFRESH_A1,
    MODEL_AIRFRESH_VA2,
    MODEL_AIRFRESH_VA4,
    MODEL_AIRFRESH_T2017,
    MODEL_AIRHUMIDIFIER_CA1,
    MODEL_AIRHUMIDIFIER_CA4,
    MODEL_AIRHUMIDIFIER_CB1,
    MODEL_AIRPURIFIER_2H,
    MODEL_AIRPURIFIER_2S,
    MODEL_AIRPURIFIER_3C,
    MODEL_AIRPURIFIER_3C_REV_A,
    MODEL_AIRPURIFIER