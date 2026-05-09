from __future__ import annotations
from typing import Any
from homematicip.aio.device import AsyncAccelerationSensor, AsyncContactInterface, AsyncDevice, AsyncFullFlushContactInterface, AsyncFullFlushContactInterface6, AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton, AsyncPluggableMainsFailureSurveillance, AsyncPresenceDetectorIndoor, AsyncRainSensor, AsyncRotaryHandleSensor, AsyncShutterContact, AsyncShutterContactMagnetic, AsyncSmokeDetector, AsyncTiltVibrationSensor, AsyncWaterSensor, AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro, AsyncWiredInput32
from homematicip.aio.group import AsyncSecurityGroup, AsyncSecurityZoneGroup
from homematicip.base.enums import SmokeDetectorAlarmType, WindowState
from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from .const import DOMAIN
from .entity import HomematicipGenericEntity
from .hap import HomematicipHAP

ATTR_ACCELERATION_SENSOR_MODE: str
ATTR_ACCELERATION_SENSOR_NEUTRAL_POSITION: str
ATTR_ACCELERATION_SENSOR_SENSITIVITY: str
ATTR_ACCELERATION_SENSOR_TRIGGER_ANGLE: str
ATTR_INTRUSION_ALARM: str
ATTR_MOISTURE_DETECTED: str
ATTR_MOTION_DETECTED: str
ATTR_POWER_MAINS_FAILURE: str
ATTR_PRESENCE_DETECTED: str
ATTR_SMOKE_DETECTOR_ALARM: str
ATTR_TODAY_SUNSHINE_DURATION: str
ATTR_WATER_LEVEL_DETECTED: str
ATTR_WINDOW_STATE: str
GROUP_ATTRIBUTES: dict[str, str]
SAM_DEVICE_ATTRIBUTES: dict[str, str]

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HomematicipCloudConnectionSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP cloud connection sensor."""

    def __init__(self, hap: HomematicipHAP) -> None:
        ...

    @property
    def name(self) -> str
    ...

    @property
    def device_info(self) -> DeviceInfo
    ...

    @property
    def icon(self) -> str
    ...

    @property
    def is_on(self) -> bool
    ...

    @property
    def available(self) -> bool
    ...

class HomematicipBaseActionSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP base action sensor."""
    _attr_device_class: BinarySensorDeviceClass

    @property
    def is_on(self) -> bool
    ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]
    ...

class HomematicipAccelerationSensor(HomematicipBaseActionSensor):
    """Representation of the HomematicIP acceleration sensor."""

class HomematicipTiltVibrationSensor(HomematicipBaseActionSensor):
    """Representation of the HomematicIP tilt vibration sensor."""

class HomematicipMultiContactInterface(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP multi room/area contact interface."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice, channel: int = 1, is_multi_channel: bool = True) -> None:
        ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipContactInterface(HomematicipMultiContactInterface, BinarySensorEntity):
    """Representation of the HomematicIP contact interface."""

    def __init__(self, hap: HomematicipHAP, device: AsyncContactInterface) -> None:
        ...

class HomematicipShutterContact(HomematicipMultiContactInterface, BinarySensorEntity):
    """Representation of the HomematicIP shutter contact."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncShutterContact, has_additional_state: bool = False) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]
    ...

class HomematicipMotionDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP motion detector."""
    _attr_device_class: BinarySensorDeviceClass

    @property
    def is_on(self) -> bool
    ...

class HomematicipPresenceDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP presence detector."""
    _attr_device_class: BinarySensorDeviceClass

    @property
    def is_on(self) -> bool
    ...

class HomematicipSmokeDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP smoke detector."""
    _attr_device_class: BinarySensorDeviceClass

    @property
    def is_on(self) -> bool
    ...

class HomematicipWaterDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP water detector."""
    _attr_device_class: BinarySensorDeviceClass

    @property
    def is_on(self) -> bool
    ...

class HomematicipStormSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP storm sensor."""

    def __init__(self, hap: HomematicipHAP, device: AsyncWeatherSensor) -> None:
        ...

    @property
    def icon(self) -> str
    ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipRainSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP rain sensor."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncWeatherSensor) -> None:
        ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipSunshineSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP sunshine sensor."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncWeatherSensor) -> None:
        ...

    @property
    def is_on(self) -> bool
    ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]
    ...

class HomematicipBatterySensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP low battery sensor."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipPluggableMainsFailureSurveillanceSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP pluggable mains failure surveillance sensor."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncPluggableMainsFailureSurveillance) -> None:
        ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipSecurityZoneSensorGroup(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP security zone sensor group."""
    _attr_device_class: BinarySensorDeviceClass

    def __init__(self, hap: HomematicipHAP, device: AsyncSecurityZoneGroup, post: str = 'SecurityZone') -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]
    ...

    @property
    def is_on(self) -> bool
    ...

class HomematicipSecuritySensorGroup(HomematicipSecurityZoneSensorGroup, BinarySensorEntity):
    """Representation of the HomematicIP security group."""

    def __init__(self, hap: HomematicipHAP, device: AsyncSecurityGroup) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]
    ...

    @property
    def is_on(self) -> bool
    ...
