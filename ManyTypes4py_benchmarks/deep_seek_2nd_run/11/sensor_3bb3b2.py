"""Support for Dutch Smart Meter (also known as Smartmeter or P1 port)."""
from __future__ import annotations
import asyncio
from asyncio import CancelledError
from collections.abc import Callable, Generator
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from functools import partial
from typing import Any, Optional, Tuple, Union, cast

from dsmr_parser.clients.protocol import create_dsmr_reader, create_tcp_dsmr_reader
from dsmr_parser.clients.rfxtrx_protocol import create_rfxtrx_dsmr_reader, create_rfxtrx_tcp_dsmr_reader
from dsmr_parser.objects import DSMRObject, MbusDevice, Telegram
import serial
from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass, SensorEntity, SensorEntityDescription, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_PORT, CONF_PROTOCOL, EVENT_HOMEASSISTANT_STOP, EntityCategory, UnitOfEnergy, UnitOfVolume
from homeassistant.core import CoreState, Event, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType
from homeassistant.util import Throttle
from . import DsmrConfigEntry
from .const import CONF_DSMR_VERSION, CONF_SERIAL_ID, CONF_SERIAL_ID_GAS, CONF_TIME_BETWEEN_UPDATE, DEFAULT_PRECISION, DEFAULT_RECONNECT_INTERVAL, DEFAULT_TIME_BETWEEN_UPDATE, DEVICE_NAME_ELECTRICITY, DEVICE_NAME_GAS, DEVICE_NAME_HEAT, DEVICE_NAME_WATER, DOMAIN, DSMR_PROTOCOL, LOGGER

EVENT_FIRST_TELEGRAM = 'dsmr_first_telegram_{}'
UNIT_CONVERSION = {'m3': UnitOfVolume.CUBIC_METERS}

@dataclass(frozen=True, kw_only=True)
class DSMRSensorEntityDescription(SensorEntityDescription):
    """Represents an DSMR Sensor."""
    dsmr_versions: Optional[set[str]] = None
    is_gas: bool = False
    is_water: bool = False
    is_heat: bool = False

class MbusDeviceType(IntEnum):
    """Types of mbus devices (13757-3:2013)."""
    GAS = 3
    HEAT = 4
    WATER = 7

SENSORS = (
    DSMRSensorEntityDescription(key='timestamp', obis_reference='P1_MESSAGE_TIMESTAMP', device_class=SensorDeviceClass.TIMESTAMP, entity_category=EntityCategory.DIAGNOSTIC, entity_registry_enabled_default=False),
    # ... (rest of the SENSORS tuple remains the same)
)

SENSORS_MBUS_DEVICE_TYPE = {
    MbusDeviceType.GAS: (DSMRSensorEntityDescription(key='gas_reading', translation_key='gas_meter_reading', obis_reference='MBUS_METER_READING', is_gas=True, device_class=SensorDeviceClass.GAS, state_class=SensorStateClass.TOTAL_INCREASING),),
    MbusDeviceType.HEAT: (DSMRSensorEntityDescription(key='heat_reading', translation_key='heat_meter_reading', obis_reference='MBUS_METER_READING', is_heat=True, device_class=SensorDeviceClass.ENERGY, state_class=SensorStateClass.TOTAL_INCREASING),),
    MbusDeviceType.WATER: (DSMRSensorEntityDescription(key='water_reading', translation_key='water_meter_reading', obis_reference='MBUS_METER_READING', is_water=True, device_class=SensorDeviceClass.WATER, state_class=SensorStateClass.TOTAL_INCREASING),)
}

def device_class_and_uom(data: Telegram, entity_description: DSMRSensorEntityDescription) -> Tuple[SensorDeviceClass, Optional[str]]:
    """Get native unit of measurement from telegram."""
    dsmr_object = getattr(data, entity_description.obis_reference)
    uom = getattr(dsmr_object, 'unit') or None
    with suppress(ValueError):
        if entity_description.device_class == SensorDeviceClass.GAS and (energy_uom := UnitOfEnergy(str(uom))):
            return (SensorDeviceClass.ENERGY, energy_uom)
    if uom in UNIT_CONVERSION:
        return (entity_description.device_class, UNIT_CONVERSION[uom])
    return (entity_description.device_class, uom)

def rename_old_gas_to_mbus(hass: HomeAssistant, entry: ConfigEntry, mbus_device_id: str) -> None:
    """Rename old gas sensor to mbus variant."""
    dev_reg = dr.async_get(hass)
    for dev_id in (mbus_device_id, entry.entry_id):
        device_entry_v1 = dev_reg.async_get_device(identifiers={(DOMAIN, dev_id)})
        if device_entry_v1 is not None:
            device_id = device_entry_v1.id
            ent_reg = er.async_get(hass)
            entries = er.async_entries_for_device(ent_reg, device_id)
            for entity in entries:
                if entity.unique_id.endswith('belgium_5min_gas_meter_reading') or entity.unique_id.endswith('hourly_gas_meter_reading'):
                    if ent_reg.async_get_entity_id(SENSOR_DOMAIN, DOMAIN, mbus_device_id):
                        LOGGER.debug('Skip migration of %s because it already exists', entity.entity_id)
                        continue
                    new_device = dev_reg.async_get_or_create(config_entry_id=entry.entry_id, identifiers={(DOMAIN, mbus_device_id)})
                    ent_reg.async_update_entity(entity.entity_id, new_unique_id=mbus_device_id, device_id=new_device.id)
                    LOGGER.debug('Migrated entity %s from unique id %s to %s', entity.entity_id, entity.unique_id, mbus_device_id)
            dev_entities = er.async_entries_for_device(ent_reg, device_id, include_disabled_entities=True)
            if not dev_entities:
                dev_reg.async_remove_device(device_id)

def is_supported_description(data: Telegram, description: DSMRSensorEntityDescription, dsmr_version: str) -> bool:
    """Check if this is a supported description for this telegram."""
    return hasattr(data, description.obis_reference) and (description.dsmr_versions is None or dsmr_version in description.dsmr_versions)

def create_mbus_entities(hass: HomeAssistant, telegram: Telegram, entry: ConfigEntry, dsmr_version: str) -> Generator[DSMREntity, None, None]:
    """Create MBUS Entities."""
    mbus_devices = getattr(telegram, 'MBUS_DEVICES', [])
    for device in mbus_devices:
        if (device_type := getattr(device, 'MBUS_DEVICE_TYPE', None)) is None:
            continue
        type_ = int(device_type.value)
        if type_ not in SENSORS_MBUS_DEVICE_TYPE:
            LOGGER.warning('Unsupported MBUS_DEVICE_TYPE (%d)', type_)
            continue
        if (identifier := getattr(device, 'MBUS_EQUIPMENT_IDENTIFIER', None)):
            serial_ = identifier.value
            rename_old_gas_to_mbus(hass, entry, serial_)
        else:
            serial_ = ''
        for description in SENSORS_MBUS_DEVICE_TYPE.get(type_, ()):
            if not is_supported_description(device, description, dsmr_version):
                continue
            yield DSMREntity(description, entry, telegram, *device_class_and_uom(device, description), serial_, device.channel_id)

def get_dsmr_object(telegram: Optional[Telegram], mbus_id: int, obis_reference: str) -> Optional[DSMRObject]:
    """Extract DSMR object from telegram."""
    if not telegram:
        return None
    telegram_or_device = telegram
    if mbus_id:
        telegram_or_device = telegram.get_mbus_device_by_channel(mbus_id)
        if telegram_or_device is None:
            return None
    return getattr(telegram_or_device, obis_reference, None)

async def async_setup_entry(hass: HomeAssistant, entry: DsmrConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:
    """Set up the DSMR sensor."""
    dsmr_version = entry.data[CONF_DSMR_VERSION]
    entities: list[DSMREntity] = []
    initialized = False

    @callback
    def init_async_add_entities(telegram: Telegram) -> None:
        """Add the sensor entities after the first telegram was received."""
        nonlocal add_entities_handler
        assert add_entities_handler is not None
        add_entities_handler()
        add_entities_handler = None
        entities.extend(create_mbus_entities(hass, telegram, entry, dsmr_version))
        entities.extend([DSMREntity(description, entry, telegram, *device_class_and_uom(telegram, description)) for description in SENSORS if is_supported_description(telegram, description, dsmr_version) and (not description.is_gas and (not description.is_heat) or CONF_SERIAL_ID_GAS in entry.data)])
        async_add_entities(entities)
    add_entities_handler: Optional[Callable[[], None]] = async_dispatcher_connect(hass, EVENT_FIRST_TELEGRAM.format(entry.entry_id), init_async_add_entities)
    min_time_between_updates = timedelta(seconds=entry.options.get(CONF_TIME_BETWEEN_UPDATE, DEFAULT_TIME_BETWEEN_UPDATE))

    @Throttle(min_time_between_updates)
    def update_entities_telegram(telegram: Optional[Telegram]) -> None:
        """Update entities with latest telegram and trigger state update."""
        nonlocal initialized
        for entity in entities:
            entity.update_data(telegram)
        entry.runtime_data.telegram = telegram
        if not initialized and telegram:
            initialized = True
            async_dispatcher_send(hass, EVENT_FIRST_TELEGRAM.format(entry.entry_id), telegram)
    protocol = entry.data.get(CONF_PROTOCOL, DSMR_PROTOCOL)
    if CONF_HOST in entry.data:
        if protocol == DSMR_PROTOCOL:
            create_reader = create_tcp_dsmr_reader
        else:
            create_reader = create_rfxtrx_tcp_dsmr_reader
        reader_factory = partial(create_reader, entry.data[CONF_HOST], entry.data[CONF_PORT], dsmr_version, update_entities_telegram, loop=hass.loop, keep_alive_interval=60)
    else:
        if protocol == DSMR_PROTOCOL:
            create_reader = create_dsmr_reader
        else:
            create_reader = create_rfxtrx_dsmr_reader
        reader_factory = partial(create_reader, entry.data[CONF_PORT], dsmr_version, update_entities_telegram, loop=hass.loop)

    async def connect_and_reconnect() -> None:
        """Connect to DSMR and keep reconnecting until Home Assistant stops."""
        stop_listener: Optional[Callable[[], None]] = None
        transport: Optional[asyncio.Transport] = None
        protocol: Optional[Any] = None
        while hass.state is CoreState.not_running or hass.is_running:
            update_entities_telegram({})
            try:
                transport, protocol = await hass.loop.create_task(reader_factory())
                if transport:

                    @callback
                    def close_transport(_event: Event) -> None:
                        """Close the transport on HA shutdown."""
                        if not transport:
                            return
                        transport.close()
                    stop_listener = hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, close_transport)
                    await protocol.wait_closed()
                    if hass.state is CoreState.not_running or hass.is_running:
                        if stop_listener:
                            stop_listener()
                transport = None
                protocol = None
                update_entities_telegram(None)
                await asyncio.sleep(DEFAULT_RECONNECT_INTERVAL)
            except (serial.SerialException, OSError):
                LOGGER.exception('Error connecting to DSMR')
                transport = None
                protocol = None
                update_entities_telegram(None)
                await asyncio.sleep(DEFAULT_RECONNECT_INTERVAL)
            except CancelledError:
                update_entities_telegram(None)
                if stop_listener and (hass.state is CoreState.not_running or hass.is_running):
                    stop_listener()
                if transport:
                    transport.close()
                if protocol:
                    await protocol.wait_closed()
                return
    task = asyncio.create_task(connect_and_reconnect())

    @callback
    def _async_stop(_: Event) -> None:
        if add_entities_handler is not None:
            add_entities_handler()
        task.cancel()
    entry.async_on_unload(hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _async_stop))
    entry.runtime_data.task = task

class DSMREntity(SensorEntity):
    """Entity reading values from DSMR telegram."""
    _attr_has_entity_name = True
    _attr_should_poll = False

    def __init__(
        self,
        entity_description: DSMRSensorEntityDescription,
        entry: ConfigEntry,
        telegram: Optional[Telegram],
        device_class: SensorDeviceClass,
        native_unit_of_measurement: Optional[str],
        serial_id: str = '',
        mbus_id: int = 0
    ) -> None:
        """Initialize entity."""
        self.entity_description = entity_description
        self._attr_device_class = device_class
        self._attr_native_unit_of_measurement = native_unit_of_measurement
        self._entry = entry
        self.telegram = telegram
        device_serial = entry.data[CONF_SERIAL_ID]
        device_name = DEVICE_NAME_ELECTRICITY
        if entity_description.is_gas:
            if serial_id:
                device_serial = serial_id
            else:
                device_serial = entry.data[CONF_SERIAL_ID_GAS]
            device_name = DEVICE_NAME_GAS
        if entity_description.is_water:
            if serial_id:
                device_serial = serial_id
            device_name = DEVICE_NAME_WATER
        if entity_description.is_heat:
            if serial_id:
                device_serial = serial_id
            device_name = DEVICE_NAME_HEAT
        if device_serial is None:
            device_serial = entry.entry_id
        self._attr_device_info = DeviceInfo(identifiers={(DOMAIN, device_serial)}, name=device_name)
        self._mbus_id = mbus_id
        if mbus_id != 0:
            if serial_id:
                self._attr_unique_id = f'{device_serial}'
            else:
                self._attr_unique_id = f'{device_serial}_{mbus_id}'
        else:
            self._attr_unique_id = f'{device_serial}_{entity_description.key}'

    @callback
    def update_data(self, telegram: Optional[Telegram]) -> None:
        """Update data."""
        self.telegram = telegram
        if self.hass and (telegram is None or get_dsmr_object(telegram, self._mbus_id, self.entity_description.obis_reference)):
            self.async_write_ha_state()

    def get_dsmr_object_attr(self, attribute: str) -> Any:
        """Read attribute from last received telegram for this DSMR object."""
        dsmr_object = get_dsmr_object(self.telegram, self._mbus_id, self.entity_description.obis_reference)
        if dsmr_object is None:
            return None
        attr = getattr(dsmr_object, attribute)
        return attr

    @property
    def available(self) -> bool:
        """Entity is only available if there is a telegram."""
        return self.telegram is not None

    @property
    def native_value(self) -> Optional[Union[str, float, int]]:
        """Return the state of sensor, if available, translate if needed."""
        if (value := self.get_dsmr_object_attr('value')) is None:
            return None
        if self.entity_description.obis_reference == 'ELECTRICITY_ACTIVE_TARIFF':
            return self.translate_tariff(value, self._entry.data[CONF_DSMR_VERSION])
        with suppress(TypeError):
            value = round(float(value), DEFAULT_PRECISION)
        if not value and self.state_class == SensorStateClass.TOTAL_INCREASING:
            return None
        return value

    @staticmethod
    def translate_tariff(value: str, dsmr_version: str) -> Optional[str]:
        """Convert 2/1 to normal/low depending on DSMR version."""
        if dsmr_version in ('5B', '5EONHU'):
            if value == '0001':
                value = '0002'
            elif value == '0002':
                value = '0001'
        if value == '0002':
            return 'normal'
        if value == '0001':
            return 'low'
        return None
