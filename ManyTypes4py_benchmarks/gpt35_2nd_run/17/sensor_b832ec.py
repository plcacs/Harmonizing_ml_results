async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class HomematicipFloorTerminalBlockMechanicChannelValve(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any, channel: int, is_multi_channel: bool = True) -> None:
        ...

class HomematicipAccesspointDutyCycle(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipHeatingThermostat(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipHumiditySensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipTemperatureSensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipIlluminanceSensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipPowerSensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipEnergySensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipWindspeedSensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipTodayRainSensor(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicpTemperatureExternalSensorCh1(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicpTemperatureExternalSensorCh2(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicpTemperatureExternalSensorDelta(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiSensorEntity(HomematicipGenericEntity, SensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any, key: str, value_fn: Callable, type_fn: Callable) -> None:
        ...

class HmipEsiIecPowerConsumption(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiIecEnergyCounterHighTariff(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiIecEnergyCounterLowTariff(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiIecEnergyCounterInputSingleTariff(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiGasCurrentGasFlow(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiGasGasVolume(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiLedCurrentPowerConsumption(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HmipEsiLedEnergyCounterHighTariff(HmipEsiSensorEntity):
    def __init__(self, hap: HomematicipHAP, device: Any) -> None:
        ...

class HomematicipPassageDetectorDeltaCounter(HomematicipGenericEntity, SensorEntity):
    ...
