from __future__ import annotations
import io
import ipaddress
import logging
import os
import re
import secrets
import socket
from typing import Any, cast

from pyhap.accessory import Accessory
import pyqrcode
import voluptuous as vol
from homeassistant.components import binary_sensor, media_player, persistent_notification, sensor
from homeassistant.components.camera import DOMAIN as CAMERA_DOMAIN
from homeassistant.components.event import DOMAIN as EVENT_DOMAIN
from homeassistant.components.lock import DOMAIN as LOCK_DOMAIN
from homeassistant.components.media_player import DOMAIN as MEDIA_PLAYER_DOMAIN, MediaPlayerDeviceClass, MediaPlayerEntityFeature
from homeassistant.components.remote import DOMAIN as REMOTE_DOMAIN, RemoteEntityFeature
from homeassistant.const import ATTR_CODE, ATTR_DEVICE_CLASS, ATTR_SUPPORTED_FEATURES, CONF_NAME, CONF_PORT, CONF_TYPE, UnitOfTemperature
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State, callback, split_entity_id
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.storage import STORAGE_DIR
from homeassistant.util.unit_conversion import TemperatureConverter
from .const import AUDIO_CODEC_COPY, AUDIO_CODEC_OPUS, CONF_AUDIO_CODEC, CONF_AUDIO_MAP, CONF_AUDIO_PACKET_SIZE, CONF_FEATURE, CONF_FEATURE_LIST, CONF_LINKED_BATTERY_CHARGING_SENSOR, CONF_LINKED_BATTERY_SENSOR, CONF_LINKED_DOORBELL_SENSOR, CONF_LINKED_HUMIDITY_SENSOR, CONF_LINKED_MOTION_SENSOR, CONF_LINKED_OBSTRUCTION_SENSOR, CONF_LOW_BATTERY_THRESHOLD, CONF_MAX_FPS, CONF_MAX_HEIGHT, CONF_MAX_WIDTH, CONF_STREAM_ADDRESS, CONF_STREAM_COUNT, CONF_STREAM_SOURCE, CONF_SUPPORT_AUDIO, CONF_THRESHOLD_CO, CONF_THRESHOLD_CO2, CONF_VIDEO_CODEC, CONF_VIDEO_MAP, CONF_VIDEO_PACKET_SIZE, CONF_VIDEO_PROFILE_NAMES, DEFAULT_AUDIO_CODEC, DEFAULT_AUDIO_MAP, DEFAULT_AUDIO_PACKET_SIZE, DEFAULT_LOW_BATTERY_THRESHOLD, DEFAULT_MAX_FPS, DEFAULT_MAX_HEIGHT, DEFAULT_MAX_WIDTH, DEFAULT_STREAM_COUNT, DEFAULT_SUPPORT_AUDIO, DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_MAP, DEFAULT_VIDEO_PACKET_SIZE, DEFAULT_VIDEO_PROFILE_NAMES, DOMAIN, FEATURE_ON_OFF, FEATURE_PLAY_PAUSE, FEATURE_PLAY_STOP, FEATURE_TOGGLE_MUTE, MAX_NAME_LENGTH, TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE, VIDEO_CODEC_COPY, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, VIDEO_CODEC_LIBX264
from .models import HomeKitConfigEntry
_LOGGER = logging.getLogger(__name__)
NUMBERS_ONLY_RE = re.compile('[^\\d.]+')
VERSION_RE = re.compile('([0-9]+)(\\.[0-9]+)?(\\.[0-9]+)?')
INVALID_END_CHARS = '-_ '
MAX_VERSION_PART = 2 ** 32 - 1
MAX_PORT = 65535
VALID_VIDEO_CODECS = [VIDEO_CODEC_LIBX264, VIDEO_CODEC_H264_OMX, VIDEO_CODEC_H264_V4L2M2M, AUDIO_CODEC_COPY]
VALID_AUDIO_CODECS = [AUDIO_CODEC_OPUS, VIDEO_CODEC_COPY]

def validate_entity_config(values: dict) -> dict:
    """Validate config entry for CONF_ENTITY."""
    if not isinstance(values, dict):
        raise vol.Invalid('expected a dictionary')
    entities = {}
    for entity_id, config in values.items():
        entity = cv.entity_id(entity_id)
        domain, _ = split_entity_id(entity)
        if not isinstance(config, dict):
            raise vol.Invalid(f'The configuration for {entity} must be a dictionary.')
        if domain == 'alarm_control_panel':
            config = CODE_SCHEMA(config)
        elif domain == media_player.const.DOMAIN:
            config = FEATURE_SCHEMA(config)
            feature_list = {}
            for feature in config[CONF_FEATURE_LIST]:
                params = MEDIA_PLAYER_SCHEMA(feature)
                key = params.pop(CONF_FEATURE)
                if key in feature_list:
                    raise vol.Invalid(f'A feature can be added only once for {entity}')
                feature_list[key] = params
            config[CONF_FEATURE_LIST] = feature_list
        elif domain == 'camera':
            config = CAMERA_SCHEMA(config)
        elif domain == 'lock':
            config = LOCK_SCHEMA(config)
        elif domain == 'switch':
            config = SWITCH_TYPE_SCHEMA(config)
        elif domain == 'humidifier':
            config = HUMIDIFIER_SCHEMA(config)
        elif domain == 'cover':
            config = COVER_SCHEMA(config)
        elif domain == 'sensor':
            config = SENSOR_SCHEMA(config)
        else:
            config = BASIC_INFO_SCHEMA(config)
        entities[entity] = config
    return entities

def get_media_player_features(state: State) -> list:
    """Determine features for media players."""
    features = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)
    supported_modes = []
    if features & (MediaPlayerEntityFeature.TURN_ON | MediaPlayerEntityFeature.TURN_OFF):
        supported_modes.append(FEATURE_ON_OFF)
    if features & (MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.PAUSE):
        supported_modes.append(FEATURE_PLAY_PAUSE)
    if features & (MediaPlayerEntityFeature.PLAY | MediaPlayerEntityFeature.STOP):
        supported_modes.append(FEATURE_PLAY_STOP)
    if features & MediaPlayerEntityFeature.VOLUME_MUTE:
        supported_modes.append(FEATURE_TOGGLE_MUTE)
    return supported_modes

def validate_media_player_features(state: State, feature_list: list) -> bool:
    """Validate features for media players."""
    if not (supported_modes := get_media_player_features(state)):
        _LOGGER.error('%s does not support any media_player features', state.entity_id)
        return False
    if not feature_list:
        return True
    error_list = [feature for feature in feature_list if feature not in supported_modes]
    if error_list:
        _LOGGER.error('%s does not support media_player features: %s', state.entity_id, error_list)
        return False
    return True

# ... rest of the code ...
