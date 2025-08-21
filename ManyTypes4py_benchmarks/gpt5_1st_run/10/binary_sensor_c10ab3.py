"""Support for HomematicIP Cloud binary sensor."""
from __future__ import annotations

from typing import Any, Final

from homematicip.aio.device import (
    AsyncAccelerationSensor,
    AsyncContactInterface,
    AsyncDevice,
    AsyncFullFlushContactInterface,
    AsyncFullFlushContactInterface6,
    AsyncMotionDetectorIndoor,
    AsyncMotionDetectorOutdoor,
    AsyncMotionDetectorPushButton,
    AsyncPluggableMainsFailureSurveillance,
    AsyncPresenceDetectorIndoor,
    AsyncRainSensor,
    AsyncRotaryHandleSensor,
    AsyncShutterContact,
    AsyncShutterContactMagnetic,
    AsyncSmokeDetector,
    AsyncTiltVibrationSensor,
    AsyncWaterSensor,
    AsyncWeatherSensor,
    AsyncWeatherSensorPlus,
    AsyncWeatherSensorPro,
    AsyncWiredInput32,
)
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


ATTR_ACCELERATION_SENSOR_MODE: Final[str] = "acceleration_sensor_mode"
ATTR_ACCELERATION_SENSOR_NEUTRAL_POSITION: Final[str] = "acceleration_sensor_neutral_position"
ATTR_ACCELERATION_SENSOR_SENSITIVITY: Final[str] = "acceleration_sensor_sensitivity"
ATTR_ACCELERATION_SENSOR_TRIGGER_ANGLE: Final[str] = "acceleration_sensor_trigger_angle"
ATTR_INTRUSION_ALARM: Final[str] = "intrusion_alarm"
ATTR_MOISTURE_DETECTED: Final[str] = "moisture_detected"
ATTR_MOTION_DETECTED: Final[str] = "motion_detected"
ATTR_POWER_MAINS_FAILURE: Final[str] = "power_mains_failure"
ATTR_PRESENCE_DETECTED: Final[str] = "presence_detected"
ATTR_SMOKE_DETECTOR_ALARM: Final[str] = "smoke_detector_alarm"
ATTR_TODAY_SUNSHINE_DURATION: Final[str] = "today_sunshine_duration_in_minutes"
ATTR_WATER_LEVEL_DETECTED: Final[str] = "water_level_detected"
ATTR_WINDOW_STATE: Final[str] = "window_state"

GROUP_ATTRIBUTES: dict[str, str] = {
    "moistureDetected": ATTR_MOISTURE_DETECTED,
    "motionDetected": ATTR_MOTION_DETECTED,
    "powerMainsFailure": ATTR_POWER_MAINS_FAILURE,
    "presenceDetected": ATTR_PRESENCE_DETECTED,
    "waterlevelDetected": ATTR_WATER_LEVEL_DETECTED,
}
SAM_DEVICE_ATTRIBUTES: dict[str, str] = {
    "accelerationSensorNeutralPosition": ATTR_ACCELERATION_SENSOR_NEUTRAL_POSITION,
    "accelerationSensorMode": ATTR_ACCELERATION_SENSOR_MODE,
    "accelerationSensorSensitivity": ATTR_ACCELERATION_SENSOR_SENSITIVITY,
    "accelerationSensorTriggerAngle": ATTR_ACCELERATION_SENSOR_TRIGGER_ANGLE,
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the HomematicIP Cloud binary sensor from a config entry."""
    hap: HomematicipHAP = hass.data[DOMAIN][config_entry.unique_id]  # type: ignore[index]
    entities: list[BinarySensorEntity] = [HomematicipCloudConnectionSensor(hap)]

    for device in hap.home.devices:
        if isinstance(device, AsyncAccelerationSensor):
            entities.append(HomematicipAccelerationSensor(hap, device))
        if isinstance(device, AsyncTiltVibrationSensor):
            entities.append(HomematicipTiltVibrationSensor(hap, device))
        if isinstance(device, AsyncWiredInput32):
            entities.extend(
                (
                    HomematicipMultiContactInterface(hap, device, channel=channel)
                    for channel in range(1, 33)
                )
            )
        elif isinstance(device, AsyncFullFlushContactInterface6):
            entities.extend(
                (
                    HomematicipMultiContactInterface(hap, device, channel=channel)
                    for channel in range(1, 7)
                )
            )
        elif isinstance(device, (AsyncContactInterface, AsyncFullFlushContactInterface)):
            entities.append(HomematicipContactInterface(hap, device))
        if isinstance(device, (AsyncShutterContact, AsyncShutterContactMagnetic)):
            entities.append(HomematicipShutterContact(hap, device))
        if isinstance(device, AsyncRotaryHandleSensor):
            entities.append(HomematicipShutterContact(hap, device, True))
        if isinstance(device, (AsyncMotionDetectorIndoor, AsyncMotionDetectorOutdoor, AsyncMotionDetectorPushButton)):
            entities.append(HomematicipMotionDetector(hap, device))
        if isinstance(device, AsyncPluggableMainsFailureSurveillance):
            entities.append(HomematicipPluggableMainsFailureSurveillanceSensor(hap, device))
        if isinstance(device, AsyncPresenceDetectorIndoor):
            entities.append(HomematicipPresenceDetector(hap, device))
        if isinstance(device, AsyncSmokeDetector):
            entities.append(HomematicipSmokeDetector(hap, device))
        if isinstance(device, AsyncWaterSensor):
            entities.append(HomematicipWaterDetector(hap, device))
        if isinstance(device, (AsyncRainSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipRainSensor(hap, device))
        if isinstance(device, (AsyncWeatherSensor, AsyncWeatherSensorPlus, AsyncWeatherSensorPro)):
            entities.append(HomematicipStormSensor(hap, device))
            entities.append(HomematicipSunshineSensor(hap, device))
        if isinstance(device, AsyncDevice) and device.lowBat is not None:
            entities.append(HomematicipBatterySensor(hap, device))

    for group in hap.home.groups:
        if isinstance(group, AsyncSecurityGroup):
            entities.append(HomematicipSecuritySensorGroup(hap, device=group))
        elif isinstance(group, AsyncSecurityZoneGroup):
            entities.append(HomematicipSecurityZoneSensorGroup(hap, device=group))

    async_add_entities(entities)


class HomematicipCloudConnectionSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP cloud connection sensor."""

    def __init__(self, hap: HomematicipHAP) -> None:
        """Initialize the cloud connection sensor."""
        super().__init__(hap, hap.home)

    @property
    def name(self) -> str:
        """Return the name cloud connection entity."""
        name = "Cloud Connection"
        return name if not self._home.name else f"{self._home.name} {name}"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device specific attributes."""
        return DeviceInfo(identifiers={(DOMAIN, self._home.id)})

    @property
    def icon(self) -> str:
        """Return the icon of the access point entity."""
        return "mdi:access-point-network" if self._home.connected else "mdi:access-point-network-off"

    @property
    def is_on(self) -> bool:
        """Return true if hap is connected to cloud."""
        return self._home.connected

    @property
    def available(self) -> bool:
        """Sensor is always available."""
        return True


class HomematicipBaseActionSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP base action sensor."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.MOVING

    @property
    def is_on(self) -> bool:
        """Return true if acceleration is detected."""
        return self._device.accelerationSensorTriggered  # type: ignore[attr-defined]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the acceleration sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes  # type: ignore[assignment]
        for attr, attr_key in SAM_DEVICE_ATTRIBUTES.items():
            if (attr_value := getattr(self._device, attr, None)) is not None:
                state_attr[attr_key] = attr_value
        return state_attr


class HomematicipAccelerationSensor(HomematicipBaseActionSensor):
    """Representation of the HomematicIP acceleration sensor."""

    def __init__(self, hap: HomematicipHAP, device: AsyncAccelerationSensor) -> None:
        super().__init__(hap, device)


class HomematicipTiltVibrationSensor(HomematicipBaseActionSensor):
    """Representation of the HomematicIP tilt vibration sensor."""

    def __init__(self, hap: HomematicipHAP, device: AsyncTiltVibrationSensor) -> None:
        super().__init__(hap, device)


class HomematicipMultiContactInterface(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP multi room/area contact interface."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.OPENING

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncWiredInput32 | AsyncFullFlushContactInterface6 | AsyncContactInterface | AsyncFullFlushContactInterface,
        channel: int = 1,
        is_multi_channel: bool = True,
    ) -> None:
        """Initialize the multi contact entity."""
        super().__init__(hap, device, channel=channel, is_multi_channel=is_multi_channel)

    @property
    def is_on(self) -> bool | None:
        """Return true if the contact interface is on/open."""
        if self._device.functionalChannels[self._channel].windowState is None:  # type: ignore[attr-defined]
            return None
        return self._device.functionalChannels[self._channel].windowState != WindowState.CLOSED  # type: ignore[attr-defined]


class HomematicipContactInterface(HomematicipMultiContactInterface, BinarySensorEntity):
    """Representation of the HomematicIP contact interface."""

    def __init__(self, hap: HomematicipHAP, device: AsyncContactInterface | AsyncFullFlushContactInterface) -> None:
        """Initialize the multi contact entity."""
        super().__init__(hap, device, is_multi_channel=False)


class HomematicipShutterContact(HomematicipMultiContactInterface, BinarySensorEntity):
    """Representation of the HomematicIP shutter contact."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.DOOR

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncShutterContact | AsyncShutterContactMagnetic | AsyncRotaryHandleSensor,
        has_additional_state: bool = False,
    ) -> None:
        """Initialize the shutter contact."""
        super().__init__(hap, device, is_multi_channel=False)
        self.has_additional_state: bool = has_additional_state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the Shutter Contact."""
        state_attr: dict[str, Any] = super().extra_state_attributes  # type: ignore[assignment]
        if self.has_additional_state:
            window_state = getattr(self._device, "windowState", None)
            if window_state and window_state != WindowState.CLOSED:
                state_attr[ATTR_WINDOW_STATE] = window_state
        return state_attr


class HomematicipMotionDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP motion detector."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.MOTION

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncMotionDetectorIndoor | AsyncMotionDetectorOutdoor | AsyncMotionDetectorPushButton,
    ) -> None:
        super().__init__(hap, device)

    @property
    def is_on(self) -> bool:
        """Return true if motion is detected."""
        return self._device.motionDetected  # type: ignore[attr-defined]


class HomematicipPresenceDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP presence detector."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.PRESENCE

    def __init__(self, hap: HomematicipHAP, device: AsyncPresenceDetectorIndoor) -> None:
        super().__init__(hap, device)

    @property
    def is_on(self) -> bool:
        """Return true if presence is detected."""
        return self._device.presenceDetected  # type: ignore[attr-defined]


class HomematicipSmokeDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP smoke detector."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.SMOKE

    def __init__(self, hap: HomematicipHAP, device: AsyncSmokeDetector) -> None:
        super().__init__(hap, device)

    @property
    def is_on(self) -> bool:
        """Return true if smoke is detected."""
        if self._device.smokeDetectorAlarmType:  # type: ignore[attr-defined]
            return self._device.smokeDetectorAlarmType == SmokeDetectorAlarmType.PRIMARY_ALARM  # type: ignore[attr-defined]
        return False


class HomematicipWaterDetector(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP water detector."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.MOISTURE

    def __init__(self, hap: HomematicipHAP, device: AsyncWaterSensor) -> None:
        super().__init__(hap, device)

    @property
    def is_on(self) -> bool:
        """Return true, if moisture or waterlevel is detected."""
        return self._device.moistureDetected or self._device.waterlevelDetected  # type: ignore[attr-defined]


class HomematicipStormSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP storm sensor."""

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncWeatherSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro,
    ) -> None:
        """Initialize storm sensor."""
        super().__init__(hap, device, "Storm")

    @property
    def icon(self) -> str:
        """Return the icon."""
        return "mdi:weather-windy" if self.is_on else "mdi:pinwheel-outline"

    @property
    def is_on(self) -> bool:
        """Return true, if storm is detected."""
        return self._device.storm  # type: ignore[attr-defined]


class HomematicipRainSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP rain sensor."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.MOISTURE

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncRainSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro,
    ) -> None:
        """Initialize rain sensor."""
        super().__init__(hap, device, "Raining")

    @property
    def is_on(self) -> bool:
        """Return true, if it is raining."""
        return self._device.raining  # type: ignore[attr-defined]


class HomematicipSunshineSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP sunshine sensor."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.LIGHT

    def __init__(
        self,
        hap: HomematicipHAP,
        device: AsyncWeatherSensor | AsyncWeatherSensorPlus | AsyncWeatherSensorPro,
    ) -> None:
        """Initialize sunshine sensor."""
        super().__init__(hap, device, post="Sunshine")

    @property
    def is_on(self) -> bool:
        """Return true if sun is shining."""
        return self._device.sunshine  # type: ignore[attr-defined]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the illuminance sensor."""
        state_attr: dict[str, Any] = super().extra_state_attributes  # type: ignore[assignment]
        today_sunshine_duration = getattr(self._device, "todaySunshineDuration", None)
        if today_sunshine_duration:
            state_attr[ATTR_TODAY_SUNSHINE_DURATION] = today_sunshine_duration
        return state_attr


class HomematicipBatterySensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP low battery sensor."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.BATTERY

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        """Initialize battery sensor."""
        super().__init__(hap, device, post="Battery")

    @property
    def is_on(self) -> bool:
        """Return true if battery is low."""
        return self._device.lowBat  # type: ignore[attr-defined]


class HomematicipPluggableMainsFailureSurveillanceSensor(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP pluggable mains failure surveillance sensor."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.POWER

    def __init__(self, hap: HomematicipHAP, device: AsyncPluggableMainsFailureSurveillance) -> None:
        """Initialize pluggable mains failure surveillance sensor."""
        super().__init__(hap, device)

    @property
    def is_on(self) -> bool:
        """Return true if power mains fails."""
        return not self._device.powerMainsFailure  # type: ignore[attr-defined]


class HomematicipSecurityZoneSensorGroup(HomematicipGenericEntity, BinarySensorEntity):
    """Representation of the HomematicIP security zone sensor group."""

    _attr_device_class: BinarySensorDeviceClass | None = BinarySensorDeviceClass.SAFETY

    def __init__(self, hap: HomematicipHAP, device: AsyncSecurityZoneGroup, post: str = "SecurityZone") -> None:
        """Initialize security zone group."""
        device.modelType = f"HmIP-{post}"  # type: ignore[attr-defined]
        super().__init__(hap, device, post=post)

    @property
    def available(self) -> bool:
        """Security-Group available."""
        return True

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the security zone group."""
        state_attr: dict[str, Any] = super().extra_state_attributes  # type: ignore[assignment]
        for attr, attr_key in GROUP_ATTRIBUTES.items():
            if (attr_value := getattr(self._device, attr, None)) is not None:
                state_attr[attr_key] = attr_value
        window_state = getattr(self._device, "windowState", None)
        if window_state and window_state != WindowState.CLOSED:
            state_attr[ATTR_WINDOW_STATE] = str(window_state)
        return state_attr

    @property
    def is_on(self) -> bool:
        """Return true if security issue detected."""
        if (
            self._device.motionDetected  # type: ignore[attr-defined]
            or self._device.presenceDetected  # type: ignore[attr-defined]
            or self._device.unreach  # type: ignore[attr-defined]
            or self._device.sabotage  # type: ignore[attr-defined]
        ):
            return True
        if self._device.windowState is not None and self._device.windowState != WindowState.CLOSED:  # type: ignore[attr-defined]
            return True
        return False


class HomematicipSecuritySensorGroup(HomematicipSecurityZoneSensorGroup, BinarySensorEntity):
    """Representation of the HomematicIP security group."""

    def __init__(self, hap: HomematicipHAP, device: AsyncSecurityGroup) -> None:
        """Initialize security group."""
        super().__init__(hap, device, post="Sensors")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes of the security group."""
        state_attr: dict[str, Any] = super().extra_state_attributes  # type: ignore[assignment]
        smoke_detector_at = getattr(self._device, "smokeDetectorAlarmType", None)
        if smoke_detector_at:
            if smoke_detector_at == SmokeDetectorAlarmType.PRIMARY_ALARM:
                state_attr[ATTR_SMOKE_DETECTOR_ALARM] = str(smoke_detector_at)
            if smoke_detector_at == SmokeDetectorAlarmType.INTRUSION_ALARM:
                state_attr[ATTR_INTRUSION_ALARM] = str(smoke_detector_at)
        return state_attr

    @property
    def is_on(self) -> bool:
        """Return true if safety issue detected."""
        if super().is_on:
            return True
        if (
            self._device.powerMainsFailure  # type: ignore[attr-defined]
            or self._device.moistureDetected  # type: ignore[attr-defined]
            or self._device.waterlevelDetected  # type: ignore[attr-defined]
            or self._device.lowBat  # type: ignore[attr-defined]
            or self._device.dutyCycle  # type: ignore[attr-defined]
        ):
            return True
        if (
            self._device.smokeDetectorAlarmType is not None  # type: ignore[attr-defined]
            and self._device.smokeDetectorAlarmType != SmokeDetectorAlarmType.IDLE_OFF  # type: ignore[attr-defined]
        ):
            return True
        return False