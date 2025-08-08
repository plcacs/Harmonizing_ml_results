async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HomematicipCloudConnectionSensor(HomematicipGenericEntity, BinarySensorEntity):
    def __init__(self, hap: HomematicipHAP) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def device_info(self) -> DeviceInfo:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def available(self) -> bool:
        ...

class HomematicipBaseActionSensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

class HomematicipAccelerationSensor(HomematicipBaseActionSensor):
    ...

class HomematicipTiltVibrationSensor(HomematicipBaseActionSensor):
    ...

class HomematicipMultiContactInterface(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice, channel: int = 1, is_multi_channel: bool = True) -> None:
        ...

    @property
    def is_on(self) -> Optional[bool]:
        ...

class HomematicipContactInterface(HomematicipMultiContactInterface, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

class HomematicipShutterContact(HomematicipMultiContactInterface, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice, has_additional_state: bool = False) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

class HomematicipMotionDetector(HomematicipGenericEntity, BinarySensorEntity):
    ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipPresenceDetector(HomematicipGenericEntity, BinarySensorEntity):
    ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipSmokeDetector(HomematicipGenericEntity, BinarySensorEntity):
    ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipWaterDetector(HomematicipGenericEntity, BinarySensorEntity):
    ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipStormSensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def icon(self) -> str:
        ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipRainSensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipSunshineSensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

class HomematicipBatterySensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipPluggableMainsFailureSurveillanceSensor(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipSecurityZoneSensorGroup(HomematicipGenericEntity, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice, post: str = 'SecurityZone') -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def is_on(self) -> bool:
        ...

class HomematicipSecuritySensorGroup(HomematicipSecurityZoneSensorGroup, BinarySensorEntity):
    ...

    def __init__(self, hap: HomematicipHAP, device: AsyncDevice) -> None:
        ...

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        ...

    @property
    def is_on(self) -> bool:
        ...
