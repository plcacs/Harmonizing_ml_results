"""Support for EDL21 Smart Meters."""
from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Callable, Optional, Union
from sml import SmlGetListResponse
from sml.asyncio import SmlProtocol
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE, UnitOfElectricCurrent, UnitOfElectricPotential, UnitOfEnergy, UnitOfFrequency, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.dt import utcnow
from .const import CONF_SERIAL_PORT, DEFAULT_DEVICE_NAME, DOMAIN, LOGGER, SIGNAL_EDL21_TELEGRAM

MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=60)
SENSOR_TYPES: tuple[SensorEntityDescription, ...] = (
    SensorEntityDescription(key='1-0:0.0.0*255', translation_key='ownership_id', entity_registry_enabled_default=False),
    SensorEntityDescription(key='1-0:0.0.9*255', translation_key='electricity_id'),
    SensorEntityDescription(key='1-0:0.2.0*0', translation_key='configuration_program_version_number'),
    SensorEntityDescription(key='1-0:0.2.0*1', translation_key='firmware_version_number'),
    SensorEntityDescription(key='1-0:1.7.0*255', translation_key='positive_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:1.8.0*255', translation_key='positive_active_energy_total', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:1.8.1*255', translation_key='positive_active_energy_tariff_t1', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:1.8.2*255', translation_key='positive_active_energy_tariff_t2', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:1.17.0*255', translation_key='last_signed_positive_active_energy_total'),
    SensorEntityDescription(key='1-0:2.8.0*255', translation_key='negative_active_energy_total', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:2.8.1*255', translation_key='negative_active_energy_tariff_t1', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:2.8.2*255', translation_key='negative_active_energy_tariff_t2', state_class=SensorStateClass.TOTAL_INCREASING, device_class=SensorDeviceClass.ENERGY),
    SensorEntityDescription(key='1-0:14.7.0*255', translation_key='supply_frequency'),
    SensorEntityDescription(key='1-0:15.7.0*255', translation_key='absolute_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:16.7.0*255', translation_key='sum_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:31.7.0*255', translation_key='l1_active_instantaneous_amperage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.CURRENT),
    SensorEntityDescription(key='1-0:32.7.0*255', translation_key='l1_active_instantaneous_voltage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.VOLTAGE),
    SensorEntityDescription(key='1-0:36.7.0*255', translation_key='l1_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:51.7.0*255', translation_key='l2_active_instantaneous_amperage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.CURRENT),
    SensorEntityDescription(key='1-0:52.7.0*255', translation_key='l2_active_instantaneous_voltage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.VOLTAGE),
    SensorEntityDescription(key='1-0:56.7.0*255', translation_key='l2_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:71.7.0*255', translation_key='l3_active_instantaneous_amperage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.CURRENT),
    SensorEntityDescription(key='1-0:72.7.0*255', translation_key='l3_active_instantaneous_voltage', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.VOLTAGE),
    SensorEntityDescription(key='1-0:76.7.0*255', translation_key='l3_active_instantaneous_power', state_class=SensorStateClass.MEASUREMENT, device_class=SensorDeviceClass.POWER),
    SensorEntityDescription(key='1-0:81.7.1*255', translation_key='u_l2_u_l1_phase_angle'),
    SensorEntityDescription(key='1-0:81.7.2*255', translation_key='u_l3_u_l1_phase_angle'),
    SensorEntityDescription(key='1-0:81.7.4*255', translation_key='u_l1_i_l1_phase_angle'),
    SensorEntityDescription(key='1-0:81.7.15*255', translation_key='u_l2_i_l2_phase_angle'),
    SensorEntityDescription(key='1-0:81.7.26*255', translation_key='u_l3_i_l3_phase_angle'),
    SensorEntityDescription(key='1-0:96.1.0*255', translation_key='metering_point_id_1'),
    SensorEntityDescription(key='1-0:96.5.0*255', translation_key='internal_operating_status')
)
SENSORS: dict[str, SensorEntityDescription] = {desc.key: desc for desc in SENSOR_TYPES}
SENSOR_UNIT_MAPPING: dict[str, str] = {
    'Wh': UnitOfEnergy.WATT_HOUR,
    'kWh': UnitOfEnergy.KILO_WATT_HOUR,
    'W': UnitOfPower.WATT,
    'A': UnitOfElectricCurrent.AMPERE,
    'V': UnitOfElectricPotential.VOLT,
    '°': DEGREE,
    'Hz': UnitOfFrequency.HERTZ
}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the EDL21 sensor."""
    api = EDL21(hass, config_entry.data, async_add_entities)
    await api.connect()

class EDL21:
    """EDL21 handles telegrams sent by a compatible smart meter."""
    _OBIS_BLACKLIST: set[str] = {'1-0:96.50.1*1', '1-0:96.50.1*4', '1-0:96.50.4*4', '1-0:96.90.2*1', '1-0:96.90.2*2', '1-0:97.97.0*0', '129-129:199.130.3*255', '129-129:199.130.5*255'}

    def __init__(self, hass: HomeAssistant, config: Mapping[str, Any], async_add_entities: AddConfigEntryEntitiesCallback) -> None:
        """Initialize an EDL21 object."""
        self._registered_obis: set[tuple[str, str]] = set()
        self._hass: HomeAssistant = hass
        self._async_add_entities: AddConfigEntryEntitiesCallback = async_add_entities
        self._serial_port: str = config[CONF_SERIAL_PORT]
        self._proto: SmlProtocol = SmlProtocol(config[CONF_SERIAL_PORT])
        self._proto.add_listener(self.event, ['SmlGetListResponse'])
        LOGGER.debug('Initialized EDL21 on %s', config[CONF_SERIAL_PORT])

    async def connect(self) -> None:
        """Connect to an EDL21 reader."""
        await self._proto.connect(self._hass.loop)

    def event(self, message_body: SmlGetListResponse) -> None:
        """Handle events from pysml."""
        assert isinstance(message_body, SmlGetListResponse)
        LOGGER.debug('Received sml message on %s: %s', self._serial_port, message_body)
        electricity_id: Optional[str] = message_body['serverId']
        if electricity_id is None:
            LOGGER.debug('No electricity id found in sml message on %s', self._serial_port)
            return
        electricity_id = electricity_id.replace(' ', '')
        new_entities: list[EDL21Entity] = []
        for telegram in message_body.get('valList', []):
            if not (obis := telegram.get('objName')):
                continue
            if (electricity_id, obis) in self._registered_obis:
                async_dispatcher_send(self._hass, SIGNAL_EDL21_TELEGRAM, electricity_id, telegram)
            else:
                entity_description = SENSORS.get(obis)
                if entity_description:
                    new_entities.append(EDL21Entity(electricity_id, obis, entity_description, telegram))
                    self._registered_obis.add((electricity_id, obis))
                elif obis not in self._OBIS_BLACKLIST:
                    LOGGER.warning('Unhandled sensor %s detected. Please report at %s', obis, 'https://github.com/home-assistant/core/issues?q=is%3Aopen+is%3Aissue+label%3A%22integration%3A+edl21%22')
                    self._OBIS_BLACKLIST.add(obis)
        if new_entities:
            self._async_add_entities(new_entities, update_before_add=True)

class EDL21Entity(SensorEntity):
    """Entity reading values from EDL21 telegram."""
    _attr_should_poll: bool = False
    _attr_has_entity_name: bool = True

    def __init__(self, electricity_id: str, obis: str, entity_description: SensorEntityDescription, telegram: dict[str, Any]) -> None:
        """Initialize an EDL21Entity."""
        self._electricity_id: str = electricity_id
        self._obis: str = obis
        self._telegram: dict[str, Any] = telegram
        self._min_time: timedelta = MIN_TIME_BETWEEN_UPDATES
        self._last_update: datetime = utcnow()
        self._async_remove_dispatcher: Optional[Callable[[], None]] = None
        self.entity_description: SensorEntityDescription = entity_description
        self._attr_unique_id: str = f'{electricity_id}_{obis}'
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, self._electricity_id)}, name=DEFAULT_DEVICE_NAME)

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""

        @callback
        def handle_telegram(electricity_id: str, telegram: dict[str, Any]) -> None:
            """Update attributes from last received telegram for this object."""
            if self._electricity_id != electricity_id:
                return
            if self._obis != telegram.get('objName'):
                return
            if self._telegram == telegram:
                return
            now = utcnow()
            if now - self._last_update < self._min_time:
                return
            self._telegram = telegram
            self._last_update = now
            self.async_write_ha_state()
        self._async_remove_dispatcher = async_dispatcher_connect(self.hass, SIGNAL_EDL21_TELEGRAM, handle_telegram)

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        if self._async_remove_dispatcher:
            self._async_remove_dispatcher()

    @property
    def native_value(self) -> Any:
        """Return the value of the last received telegram."""
        return self._telegram.get('value')

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        """Return the unit of measurement."""
        if (unit := self._telegram.get('unit')) is None or unit == 0:
            return None
        return SENSOR_UNIT_MAPPING[unit]
