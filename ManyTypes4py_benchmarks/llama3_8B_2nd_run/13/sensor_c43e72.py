from __future__ import annotations
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Dict, List, Optional

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

MIN_TIME_BETWEEN_UPDATES = timedelta(seconds=60)
SENSOR_TYPES: List[SensorEntityDescription] = [
    SensorEntityDescription(key='1-0:0.0.0*255', translation_key='ownership_id', entity_registry_enabled_default=False),
    # ... rest of the sensor types
]

SENSORS: Dict[str, SensorEntityDescription] = {desc.key: desc for desc in SENSOR_TYPES}
SENSOR_UNIT_MAPPING: Dict[str, UnitOfMeasurement] = {'Wh': UnitOfEnergy.WATT_HOUR, 'kWh': UnitOfEnergy.KILO_WATT_HOUR, 'W': UnitOfPower.WATT, 'A': UnitOfElectricCurrent.AMPERE, 'V': UnitOfElectricPotential.VOLT, '°': DEGREE, 'Hz': UnitOfFrequency.HERTZ}

async def async_setup_entry(hass: HomeAssistant, config_entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the EDL21 sensor."""
    api = EDL21(hass, config_entry.data, async_add_entities)
    await api.connect()

class EDL21:
    """EDL21 handles telegrams sent by a compatible smart meter."""

    def __init__(self, hass: HomeAssistant, config: Dict[str, Any], async_add_entities: AddConfigEntryEntitiesCallback) -> None:
        """Initialize an EDL21 object."""
        self._registered_obis: set = set()
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
        new_entities: List[EDL21Entity] = []
        for telegram in message_body.get('valList', []):
            if not (obis := telegram.get('objName')):
                continue
            if (electricity_id, obis) in self._registered_obis:
                async_dispatcher_send(self._hass, SIGNAL_EDL21_TELEGRAM, electricity_id, telegram)
            else:
                entity_description: Optional[SensorEntityDescription] = SENSORS.get(obis)
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

    def __init__(self, electricity_id: str, obis: str, entity_description: SensorEntityDescription, telegram: Any) -> None:
        """Initialize an EDL21Entity."""
        self._electricity_id: str = electricity_id
        self._obis: str = obis
        self._telegram: Any = telegram
        self._min_time: timedelta = MIN_TIME_BETWEEN_UPDATES
        self._last_update: datetime = utcnow()
        self._async_remove_dispatcher: Optional[Callable[[], None]] = None
        self.entity_description: SensorEntityDescription = entity_description
        self._attr_unique_id: str = f'{electricity_id}_{obis}'
        self._attr_device_info: DeviceInfo = DeviceInfo(identifiers={(DOMAIN, self._electricity_id)}, name=DEFAULT_DEVICE_NAME)

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""

        @callback
        def handle_telegram(electricity_id: str, telegram: Any) -> None:
            """Update attributes from last received telegram for this object."""
            if self._electricity_id != electricity_id:
                return
            if self._obis != telegram.get('objName'):
                return
            if self._telegram == telegram:
                return
            now: datetime = utcnow()
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
    def native_unit_of_measurement(self) -> Optional[UnitOfMeasurement]:
        """Return the unit of measurement."""
        if (unit := self._telegram.get('unit')) is None or unit == 0:
            return None
        return SENSOR_UNIT_MAPPING[unit]
