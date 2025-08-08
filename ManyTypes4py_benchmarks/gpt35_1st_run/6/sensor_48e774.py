from typing import List, Optional

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.const import REVOLUTIONS_PER_MINUTE, Platform, UnitOfElectricCurrent, UnitOfEnergy, UnitOfFrequency, UnitOfPower, UnitOfPressure, UnitOfTemperature, UnitOfTime, UnitOfVolumeFlowRate
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from .const import F_SERIES
from .coordinator import MyUplinkConfigEntry, MyUplinkDataCoordinator
from .entity import MyUplinkEntity
from .helpers import find_matching_platform, skip_entity, transform_model_series

DEVICE_POINT_UNIT_DESCRIPTIONS: dict[str, SensorEntityDescription] = {'°C': SensorEntityDescription(key='celsius', device_class=SensorDeviceClass.TEMPERATURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTemperature.CELSIUS), '°F': SensorEntityDescription(key='fahrenheit', device_class=SensorDeviceClass.TEMPERATURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTemperature.FAHRENHEIT), 'A': SensorEntityDescription(key='ampere', device_class=SensorDeviceClass.CURRENT, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfElectricCurrent.AMPERE), 'bar': SensorEntityDescription(key='pressure', device_class=SensorDeviceClass.PRESSURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfPressure.BAR), 'days': SensorEntityDescription(key='days', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.DAYS, suggested_display_precision=0), 'h': SensorEntityDescription(key='hours', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.HOURS, suggested_display_precision=1), 'hrs': SensorEntityDescription(key='hours_hrs', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.HOURS, suggested_display_precision=1), 'Hz': SensorEntityDescription(key='hertz', device_class=SensorDeviceClass.FREQUENCY, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfFrequency.HERTZ), 'kW': SensorEntityDescription(key='power', device_class=SensorDeviceClass.POWER, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfPower.KILO_WATT), 'kWh': SensorEntityDescription(key='energy', device_class=SensorDeviceClass.ENERGY, state_class=SensorStateClass.TOTAL_INCREASING, native_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR), 'm3/h': SensorEntityDescription(key='airflow', translation_key='airflow', device_class=SensorDeviceClass.VOLUME_FLOW_RATE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfVolumeFlowRate.CUBIC_METERS_PER_HOUR), 'min': SensorEntityDescription(key='minutes', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.MINUTES, suggested_display_precision=0), 'Pa': SensorEntityDescription(key='pressure_pa', device_class=SensorDeviceClass.PRESSURE, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfPressure.PA, suggested_display_precision=0), 'rpm': SensorEntityDescription(key='rpm', translation_key='rpm', state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=REVOLUTIONS_PER_MINUTE, suggested_display_precision=0), 's': SensorEntityDescription(key='seconds', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.SECONDS, suggested_display_precision=0), 'sec': SensorEntityDescription(key='seconds_sec', device_class=SensorDeviceClass.DURATION, state_class=SensorStateClass.MEASUREMENT, native_unit_of_measurement=UnitOfTime.SECONDS, suggested_display_precision=0)}
MARKER_FOR_UNKNOWN_VALUE: int = -32768
CATEGORY_BASED_DESCRIPTIONS: dict[str, dict[str, SensorEntityDescription]] = {F_SERIES: {'43108': SensorEntityDescription(key='fan_mode', translation_key='fan_mode'), '43427': SensorEntityDescription(key='status_compressor', translation_key='status_compressor', device_class=SensorDeviceClass.ENUM), '49993': SensorEntityDescription(key='elect_add', translation_key='elect_add', device_class=SensorDeviceClass.ENUM), '49994': SensorEntityDescription(key='priority', translation_key='priority', device_class=SensorDeviceClass.ENUM), '50095': SensorEntityDescription(key='status', translation_key='status', device_class=SensorDeviceClass.ENUM)}, 'NIBEF': {'43108': SensorEntityDescription(key='fan_mode', translation_key='fan_mode'), '43427': SensorEntityDescription(key='status_compressor', translation_key='status_compressor', device_class=SensorDeviceClass.ENUM), '49993': SensorEntityDescription(key='elect_add', translation_key='elect_add', device_class=SensorDeviceClass.ENUM), '49994': SensorEntityDescription(key='priority', translation_key='priority', device_class=SensorDeviceClass.ENUM)}, 'NIBE': {}}

def get_description(device_point: DevicePoint) -> Optional[SensorEntityDescription]:
    ...

async def async_setup_entry(hass: HomeAssistant, config_entry: MyUplinkConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    ...

class MyUplinkDevicePointSensor(MyUplinkEntity, SensorEntity):
    ...

class MyUplinkEnumSensor(MyUplinkDevicePointSensor):
    ...

class MyUplinkEnumRawSensor(MyUplinkDevicePointSensor):
    ...
