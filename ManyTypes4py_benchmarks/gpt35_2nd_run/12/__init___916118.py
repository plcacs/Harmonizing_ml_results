from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import logging
import pyrepetierng as pyrepetier
import voluptuous as vol
from homeassistant.components.sensor import SensorDeviceClass, SensorEntityDescription
from homeassistant.const import CONF_API_KEY, CONF_HOST, CONF_MONITORED_CONDITIONS, CONF_NAME, CONF_PORT, CONF_SENSORS, PERCENTAGE, UnitOfTemperature
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.discovery import load_platform
from homeassistant.helpers.dispatcher import dispatcher_send
from homeassistant.helpers.event import track_time_interval
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import slugify as util_slugify

_LOGGER: logging.Logger = logging.getLogger(__name__)
DEFAULT_NAME: str = 'RepetierServer'
DOMAIN: str = 'repetier'
REPETIER_API: str = 'repetier_api'
SCAN_INTERVAL: timedelta = timedelta(seconds=10)
UPDATE_SIGNAL: str = 'repetier_update_signal'
TEMP_DATA: dict[str, str] = {'tempset': 'temp_set', 'tempread': 'state', 'output': 'output'}

@dataclass
class APIMethods:
    temp_data: dict[str, str] = None
    attribute: str = None

API_PRINTER_METHODS: dict[str, APIMethods] = {
    'bed_temperature': APIMethods(offline={'heatedbeds': None, 'state': 'off'}, state={'heatedbeds': 'temp_data'}, temp_data=TEMP_DATA, attribute='heatedbeds'),
    'extruder_temperature': APIMethods(offline={'extruder': None, 'state': 'off'}, state={'extruder': 'temp_data'}, temp_data=TEMP_DATA, attribute='extruder'),
    'chamber_temperature': APIMethods(offline={'heatedchambers': None, 'state': 'off'}, state={'heatedchambers': 'temp_data'}, temp_data=TEMP_DATA, attribute='heatedchambers'),
    'current_state': APIMethods(offline={'state': None}, state={'state': 'state', 'activeextruder': 'active_extruder', 'hasxhome': 'x_homed', 'hasyhome': 'y_homed', 'haszhome': 'z_homed', 'firmware': 'firmware', 'firmwareurl': 'firmware_url'}),
    'current_job': APIMethods(offline={'job': None, 'state': 'off'}, state={'done': 'state', 'job': 'job_name', 'jobid': 'job_id', 'totallines': 'total_lines', 'linessent': 'lines_sent', 'oflayer': 'total_layers', 'layer': 'current_layer', 'speedmultiply': 'feed_rate', 'flowmultiply': 'flow', 'x': 'x', 'y': 'y', 'z': 'z'}),
    'job_end': APIMethods(offline={'job': None, 'state': 'off', 'start': None, 'printtime': None}, state={'job': 'job_name', 'start': 'start', 'printtime': 'print_time', 'printedtimecomp': 'from_start'}),
    'job_start': APIMethods(offline={'job': None, 'state': 'off', 'start': None, 'printedtimecomp': None}, state={'job': 'job_name', 'start': 'start', 'printedtimecomp': 'from_start'})
}

def has_all_unique_names(value: list[dict[str, str]]) -> list[dict[str, str]]:
    names: list[str] = [util_slugify(printer[CONF_NAME]) for printer in value]
    vol.Schema(vol.Unique())(names)
    return value

@dataclass(frozen=True)
class RepetierRequiredKeysMixin:
    pass

@dataclass(frozen=True)
class RepetierSensorEntityDescription(SensorEntityDescription, RepetierRequiredKeysMixin):
    key: str
    type: str
    native_unit_of_measurement: str
    name: str
    device_class: str

SENSOR_TYPES: dict[str, RepetierSensorEntityDescription] = {
    'bed_temperature': RepetierSensorEntityDescription(key='bed_temperature', type='temperature', native_unit_of_measurement=UnitOfTemperature.CELSIUS, name='_bed_', device_class=SensorDeviceClass.TEMPERATURE),
    'extruder_temperature': RepetierSensorEntityDescription(key='extruder_temperature', type='temperature', native_unit_of_measurement=UnitOfTemperature.CELSIUS, name='_extruder_', device_class=SensorDeviceClass.TEMPERATURE),
    'chamber_temperature': RepetierSensorEntityDescription(key='chamber_temperature', type='temperature', native_unit_of_measurement=UnitOfTemperature.CELSIUS, name='_chamber_', device_class=SensorDeviceClass.TEMPERATURE),
    'current_state': RepetierSensorEntityDescription(key='current_state', type='state', icon='mdi:printer-3d'),
    'current_job': RepetierSensorEntityDescription(key='current_job', type='progress', native_unit_of_measurement=PERCENTAGE, icon='mdi:file-percent', name='_current_job'),
    'job_end': RepetierSensorEntityDescription(key='job_end', type='progress', icon='mdi:clock-end', name='_job_end'),
    'job_start': RepetierSensorEntityDescription(key='job_start', type='progress', icon='mdi:clock-start', name='_job_start')
}

SENSOR_SCHEMA: vol.Schema = vol.Schema({
    vol.Optional(CONF_MONITORED_CONDITIONS, default=list(SENSOR_TYPES)): vol.All(cv.ensure_list, [vol.In(SENSOR_TYPES)]),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string
})

CONFIG_SCHEMA: vol.Schema = vol.Schema({
    DOMAIN: vol.All(cv.ensure_list, [
        vol.Schema({
            vol.Required(CONF_API_KEY): cv.string,
            vol.Required(CONF_HOST): cv.string,
            vol.Optional(CONF_PORT, default=3344): cv.port,
            vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
            vol.Optional(CONF_SENSORS, default={}): SENSOR_SCHEMA
        }, has_all_unique_names)
    ], extra=vol.ALLOW_EXTRA)
})

def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    hass.data[REPETIER_API] = {}
    for repetier in config[DOMAIN]:
        _LOGGER.debug('Repetier server config %s', repetier[CONF_HOST])
        url: str = f'http://{repetier[CONF_HOST]}'
        port: int = repetier[CONF_PORT]
        api_key: str = repetier[CONF_API_KEY]
        client = pyrepetier.Repetier(url=url, port=port, apikey=api_key)
        printers = client.getprinters()
        if not printers:
            return False
        sensors = repetier[CONF_SENSORS][CONF_MONITORED_CONDITIONS]
        api = PrinterAPI(hass, client, printers, sensors, repetier[CONF_NAME], config)
        api.update()
        track_time_interval(hass, api.update, SCAN_INTERVAL)
        hass.data[REPETIER_API][repetier[CONF_NAME]] = api
    return True

class PrinterAPI:
    def __init__(self, hass: HomeAssistant, client, printers, sensors, conf_name: str, config: ConfigType):
        self._hass: HomeAssistant = hass
        self._client = client
        self.printers = printers
        self.sensors = sensors
        self.conf_name = conf_name
        self.config = config
        self._known_entities: set[str] = set()

    def get_data(self, printer_id: int, sensor_type: str, temp_id: str) -> dict[str, str]:
        printer = self.printers[printer_id]
        methods = API_PRINTER_METHODS[sensor_type]
        for prop, offline in methods.offline.items():
            if getattr(printer, prop) == offline:
                return None
        data: dict[str, str] = {}
        for prop, attr in methods.state.items():
            prop_data = getattr(printer, prop)
            if attr == 'temp_data':
                temp_methods = methods.temp_data or {}
                for temp_prop, temp_attr in temp_methods.items():
                    data[temp_attr] = getattr(prop_data[temp_id], temp_prop)
            else:
                data[attr] = prop_data
        return data

    def update(self, now=None):
        for printer in self.printers:
            printer.get_data()
        self._load_entities()
        dispatcher_send(self._hass, UPDATE_SIGNAL)

    def _load_entities(self):
        sensor_info: list[dict[str, str]] = []
        for pidx, printer in enumerate(self.printers):
            for sensor_type in self.sensors:
                info: dict[str, str] = {}
                info['sensor_type'] = sensor_type
                info['printer_id'] = pidx
                info['name'] = printer.slug
                info['printer_name'] = self.conf_name
                known: str = f'{printer.slug}-{sensor_type}'
                if known in self._known_entities:
                    continue
                methods = API_PRINTER_METHODS[sensor_type]
                if 'temp_data' in methods.state.values():
                    prop_data = getattr(printer, methods.attribute or '')
                    if prop_data is None:
                        continue
                    for idx, _ in enumerate(prop_data):
                        prop_info = info.copy()
                        prop_info['temp_id'] = idx
                        sensor_info.append(prop_info)
                else:
                    info['temp_id'] = None
                    sensor_info.append(info)
                self._known_entities.add(known)
        if not sensor_info:
            return
        load_platform(self._hass, 'sensor', DOMAIN, {'sensors': sensor_info}, self.config)
