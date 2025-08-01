from datetime import datetime
from functools import partial
import logging
from typing import Any, Optional, List, Dict, Tuple
from pyhomematic import HMConnection
import voluptuous as vol
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_MODE,
    ATTR_NAME,
    ATTR_TIME,
    CONF_HOST,
    CONF_HOSTS,
    CONF_PASSWORD,
    CONF_PATH,
    CONF_PLATFORM,
    CONF_PORT,
    CONF_SSL,
    CONF_USERNAME,
    CONF_VERIFY_SSL,
    EVENT_HOMEASSISTANT_STOP,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers import config_validation as cv, discovery
from homeassistant.helpers.typing import ConfigType
from .const import (
    ATTR_ADDRESS,
    ATTR_CHANNEL,
    ATTR_DEVICE_TYPE,
    ATTR_DISCOVER_DEVICES,
    ATTR_DISCOVERY_TYPE,
    ATTR_ERRORCODE,
    ATTR_INTERFACE,
    ATTR_LOW_BAT,
    ATTR_LOWBAT,
    ATTR_MESSAGE,
    ATTR_PARAM,
    ATTR_PARAMSET,
    ATTR_PARAMSET_KEY,
    ATTR_RX_MODE,
    ATTR_UNIQUE_ID,
    ATTR_VALUE,
    ATTR_VALUE_TYPE,
    CONF_CALLBACK_IP,
    CONF_CALLBACK_PORT,
    CONF_INTERFACES,
    CONF_JSONPORT,
    CONF_LOCAL_IP,
    CONF_LOCAL_PORT,
    CONF_RESOLVENAMES,
    CONF_RESOLVENAMES_OPTIONS,
    DATA_CONF,
    DATA_HOMEMATIC,
    DATA_STORE,
    DISCOVER_BATTERY,
    DISCOVER_BINARY_SENSORS,
    DISCOVER_CLIMATE,
    DISCOVER_COVER,
    DISCOVER_LIGHTS,
    DISCOVER_LOCKS,
    DISCOVER_SENSORS,
    DISCOVER_SWITCHES,
    DOMAIN,
    EVENT_ERROR,
    EVENT_IMPULSE,
    EVENT_KEYPRESS,
    HM_DEVICE_TYPES,
    HM_IGNORE_DISCOVERY_NODE,
    HM_IGNORE_DISCOVERY_NODE_EXCEPTIONS,
    HM_IMPULSE_EVENTS,
    HM_PRESS_EVENTS,
    SERVICE_PUT_PARAMSET,
    SERVICE_RECONNECT,
    SERVICE_SET_DEVICE_VALUE,
    SERVICE_SET_INSTALL_MODE,
    SERVICE_SET_VARIABLE_VALUE,
    SERVICE_VIRTUALKEY,
)
from .entity import HMHub

_LOGGER = logging.getLogger(__name__)
DEFAULT_LOCAL_IP: str = '0.0.0.0'
DEFAULT_LOCAL_PORT: int = 0
DEFAULT_RESOLVENAMES: bool = False
DEFAULT_JSONPORT: int = 80
DEFAULT_PORT: int = 2001
DEFAULT_PATH: str = ''
DEFAULT_USERNAME: str = 'Admin'
DEFAULT_PASSWORD: str = ''
DEFAULT_SSL: bool = False
DEFAULT_VERIFY_SSL: bool = False
DEFAULT_CHANNEL: int = 1

DEVICE_SCHEMA = vol.Schema({
    vol.Required(CONF_PLATFORM): 'homematic',
    vol.Required(ATTR_NAME): cv.string,
    vol.Required(ATTR_ADDRESS): cv.string,
    vol.Required(ATTR_INTERFACE): cv.string,
    vol.Optional(ATTR_DEVICE_TYPE): cv.string,
    vol.Optional(ATTR_CHANNEL, default=DEFAULT_CHANNEL): vol.Coerce(int),
    vol.Optional(ATTR_PARAM): cv.string,
    vol.Optional(ATTR_UNIQUE_ID): cv.string,
})

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Optional(CONF_INTERFACES, default={}): {
            cv.match_all: {
                vol.Required(CONF_HOST): cv.string,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
                vol.Optional(CONF_PATH, default=DEFAULT_PATH): cv.string,
                vol.Optional(CONF_RESOLVENAMES, default=DEFAULT_RESOLVENAMES): vol.In(CONF_RESOLVENAMES_OPTIONS),
                vol.Optional(CONF_JSONPORT, default=DEFAULT_JSONPORT): cv.port,
                vol.Optional(CONF_USERNAME, default=DEFAULT_USERNAME): cv.string,
                vol.Optional(CONF_PASSWORD, default=DEFAULT_PASSWORD): cv.string,
                vol.Optional(CONF_CALLBACK_IP): cv.string,
                vol.Optional(CONF_CALLBACK_PORT): cv.port,
                vol.Optional(CONF_SSL, default=DEFAULT_SSL): cv.boolean,
                vol.Optional(CONF_VERIFY_SSL, default=DEFAULT_VERIFY_SSL): cv.boolean
            }
        },
        vol.Optional(CONF_HOSTS, default={}): {
            cv.match_all: {
                vol.Required(CONF_HOST): cv.string,
                vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
                vol.Optional(CONF_USERNAME, default=DEFAULT_USERNAME): cv.string,
                vol.Optional(CONF_PASSWORD, default=DEFAULT_PASSWORD): cv.string
            }
        },
        vol.Optional(CONF_LOCAL_IP, default=DEFAULT_LOCAL_IP): cv.string,
        vol.Optional(CONF_LOCAL_PORT): cv.port
    })
}, extra=vol.ALLOW_EXTRA)

SCHEMA_SERVICE_VIRTUALKEY = vol.Schema({
    vol.Required(ATTR_ADDRESS): vol.All(cv.string, vol.Upper),
    vol.Required(ATTR_CHANNEL): vol.Coerce(int),
    vol.Required(ATTR_PARAM): cv.string,
    vol.Optional(ATTR_INTERFACE): cv.string
})
SCHEMA_SERVICE_SET_VARIABLE_VALUE = vol.Schema({
    vol.Required(ATTR_NAME): cv.string,
    vol.Required(ATTR_VALUE): cv.match_all,
    vol.Optional(ATTR_ENTITY_ID): cv.entity_ids
})
SCHEMA_SERVICE_SET_DEVICE_VALUE = vol.Schema({
    vol.Required(ATTR_ADDRESS): vol.All(cv.string, vol.Upper),
    vol.Required(ATTR_CHANNEL): vol.Coerce(int),
    vol.Required(ATTR_PARAM): vol.All(cv.string, vol.Upper),
    vol.Required(ATTR_VALUE): cv.match_all,
    vol.Optional(ATTR_VALUE_TYPE): vol.In(['boolean', 'dateTime.iso8601', 'double', 'int', 'string']),
    vol.Optional(ATTR_INTERFACE): cv.string
})
SCHEMA_SERVICE_RECONNECT = vol.Schema({})
SCHEMA_SERVICE_SET_INSTALL_MODE = vol.Schema({
    vol.Required(ATTR_INTERFACE): cv.string,
    vol.Optional(ATTR_TIME, default=60): cv.positive_int,
    vol.Optional(ATTR_MODE, default=1): vol.All(vol.Coerce(int), vol.In([1, 2])),
    vol.Optional(ATTR_ADDRESS): vol.All(cv.string, vol.Upper)
})
SCHEMA_SERVICE_PUT_PARAMSET = vol.Schema({
    vol.Required(ATTR_INTERFACE): cv.string,
    vol.Required(ATTR_ADDRESS): vol.All(cv.string, vol.Upper),
    vol.Required(ATTR_PARAMSET_KEY): vol.All(cv.string, vol.Upper),
    vol.Required(ATTR_PARAMSET): dict,
    vol.Optional(ATTR_RX_MODE): vol.All(cv.string, vol.Upper)
})


def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the Homematic component."""
    conf = config[DOMAIN]
    hass.data[DATA_CONF] = remotes: Dict[str, Any] = {}
    hass.data[DATA_STORE] = set()
    for rname, rconfig in conf[CONF_INTERFACES].items():
        remotes[rname] = {
            'ip': rconfig.get(CONF_HOST),
            'port': rconfig.get(CONF_PORT),
            'path': rconfig.get(CONF_PATH),
            'resolvenames': rconfig.get(CONF_RESOLVENAMES),
            'jsonport': rconfig.get(CONF_JSONPORT),
            'username': rconfig.get(CONF_USERNAME),
            'password': rconfig.get(CONF_PASSWORD),
            'callbackip': rconfig.get(CONF_CALLBACK_IP),
            'callbackport': rconfig.get(CONF_CALLBACK_PORT),
            'ssl': rconfig[CONF_SSL],
            'verify_ssl': rconfig.get(CONF_VERIFY_SSL),
            'connect': True
        }
    for sname, sconfig in conf[CONF_HOSTS].items():
        remotes[sname] = {
            'ip': sconfig.get(CONF_HOST),
            'port': sconfig[CONF_PORT],
            'username': sconfig.get(CONF_USERNAME),
            'password': sconfig.get(CONF_PASSWORD),
            'connect': False
        }
    bound_system_callback = partial(_system_callback_handler, hass, config)
    hass.data[DATA_HOMEMATIC] = homematic = HMConnection(
        local=config[DOMAIN].get(CONF_LOCAL_IP),
        localport=config[DOMAIN].get(CONF_LOCAL_PORT, DEFAULT_LOCAL_PORT),
        remotes=remotes,
        systemcallback=bound_system_callback,
        interface_id='homeassistant'
    )
    homematic.start()
    hass.bus.listen_once(EVENT_HOMEASSISTANT_STOP, hass.data[DATA_HOMEMATIC].stop)
    entity_hubs: List[HMHub] = [HMHub(hass, homematic, hub_name) for hub_name in conf[CONF_HOSTS]]

    def _hm_service_virtualkey(service: ServiceCall) -> None:
        """Service to handle virtualkey servicecalls."""
        address: str = service.data.get(ATTR_ADDRESS)
        channel: int = service.data.get(ATTR_CHANNEL)
        param: str = service.data.get(ATTR_PARAM)
        hmdevice = _device_from_servicecall(hass, service)
        if hmdevice is None:
            _LOGGER.error('%s not found for service virtualkey!', address)
            return
        if param not in hmdevice.ACTIONNODE:
            _LOGGER.error('%s not datapoint in hm device %s', param, address)
            return
        if channel not in hmdevice.ACTIONNODE[param]:
            _LOGGER.error('%i is not a channel in hm device %s', channel, address)
            return
        hmdevice.actionNodeData(param, True, channel)

    hass.services.register(DOMAIN, SERVICE_VIRTUALKEY, _hm_service_virtualkey, schema=SCHEMA_SERVICE_VIRTUALKEY)

    def _service_handle_value(service: ServiceCall) -> None:
        """Service to call setValue method for HomeMatic system variable."""
        entity_ids = service.data.get(ATTR_ENTITY_ID)
        name: str = service.data[ATTR_NAME]
        value: Any = service.data[ATTR_VALUE]
        if entity_ids:
            entities: List[HMHub] = [entity for entity in entity_hubs if entity.entity_id in entity_ids]
        else:
            entities = entity_hubs
        if not entities:
            _LOGGER.error('No HomeMatic hubs available')
            return
        for hub in entities:
            hub.hm_set_variable(name, value)

    hass.services.register(DOMAIN, SERVICE_SET_VARIABLE_VALUE, _service_handle_value, schema=SCHEMA_SERVICE_SET_VARIABLE_VALUE)

    def _service_handle_reconnect(service: ServiceCall) -> None:
        """Service to reconnect all HomeMatic hubs."""
        homematic.reconnect()

    hass.services.register(DOMAIN, SERVICE_RECONNECT, _service_handle_reconnect, schema=SCHEMA_SERVICE_RECONNECT)

    def _service_handle_device(service: ServiceCall) -> None:
        """Service to call setValue method for HomeMatic devices."""
        address: str = service.data[ATTR_ADDRESS]
        channel: int = service.data[ATTR_CHANNEL]
        param: str = service.data[ATTR_PARAM]
        value: Any = service.data[ATTR_VALUE]
        value_type: Optional[str] = service.data.get(ATTR_VALUE_TYPE)
        if value_type:
            if value_type == 'int':
                value = int(value)
            elif value_type == 'double':
                value = float(value)
            elif value_type == 'boolean':
                value = bool(value)
            elif value_type == 'dateTime.iso8601':
                value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')
            else:
                value = str(value)
        hmdevice = _device_from_servicecall(hass, service)
        if hmdevice is None:
            _LOGGER.error('%s not found!', address)
            return
        hmdevice.setValue(param, value, channel)

    hass.services.register(DOMAIN, SERVICE_SET_DEVICE_VALUE, _service_handle_device, schema=SCHEMA_SERVICE_SET_DEVICE_VALUE)

    def _service_handle_install_mode(service: ServiceCall) -> None:
        """Service to set interface into install mode."""
        interface: str = service.data.get(ATTR_INTERFACE)
        mode: int = service.data.get(ATTR_MODE)
        time: int = service.data.get(ATTR_TIME)
        address: Optional[str] = service.data.get(ATTR_ADDRESS)
        homematic.setInstallMode(interface, t=time, mode=mode, address=address)

    hass.services.register(DOMAIN, SERVICE_SET_INSTALL_MODE, _service_handle_install_mode, schema=SCHEMA_SERVICE_SET_INSTALL_MODE)

    def _service_put_paramset(service: ServiceCall) -> None:
        """Service to call the putParamset method on a HomeMatic connection."""
        interface: str = service.data[ATTR_INTERFACE]
        address: str = service.data[ATTR_ADDRESS]
        paramset_key: str = service.data[ATTR_PARAMSET_KEY]
        paramset: Dict[Any, Any] = dict(service.data[ATTR_PARAMSET])
        rx_mode: Optional[str] = service.data.get(ATTR_RX_MODE)
        _LOGGER.debug('Calling putParamset: %s, %s, %s, %s, %s', interface, address, paramset_key, paramset, rx_mode)
        homematic.putParamset(interface, address, paramset_key, paramset, rx_mode)

    hass.services.register(DOMAIN, SERVICE_PUT_PARAMSET, _service_put_paramset, schema=SCHEMA_SERVICE_PUT_PARAMSET)
    return True


def _system_callback_handler(hass: HomeAssistant, config: ConfigType, src: str, *args: Any) -> None:
    """System callback handler."""
    if src == 'newDevices':
        interface_id: str = args[0]
        dev_descriptions: List[Dict[str, Any]] = args[1]
        interface: str = interface_id.split('-')[-1]
        if not hass.data[DATA_CONF][interface]['connect']:
            return
        addresses: List[str] = []
        for dev in dev_descriptions:
            address: str = dev['ADDRESS'].split(':')[0]
            if address not in hass.data[DATA_STORE]:
                hass.data[DATA_STORE].add(address)
                addresses.append(address)
        bound_event_callback = partial(_hm_event_handler, hass, interface)
        for dev in addresses:
            hmdevice = hass.data[DATA_HOMEMATIC].devices[interface].get(dev)
            if hmdevice.EVENTNODE:
                hmdevice.setEventCallback(callback=bound_event_callback, bequeath=True)
        if addresses:
            for component_name, discovery_type in (
                ('switch', DISCOVER_SWITCHES),
                ('light', DISCOVER_LIGHTS),
                ('cover', DISCOVER_COVER),
                ('binary_sensor', DISCOVER_BINARY_SENSORS),
                ('sensor', DISCOVER_SENSORS),
                ('climate', DISCOVER_CLIMATE),
                ('lock', DISCOVER_LOCKS),
                ('binary_sensor', DISCOVER_BATTERY),
            ):
                found_devices = _get_devices(hass, discovery_type, addresses, interface)
                if found_devices:
                    discovery.load_platform(hass, component_name, DOMAIN, {ATTR_DISCOVER_DEVICES: found_devices, ATTR_DISCOVERY_TYPE: discovery_type}, config)
    elif src == 'error':
        _LOGGER.error('Error: %s', args)
        interface_id, errorcode, message = args
        hass.bus.fire(EVENT_ERROR, {ATTR_ERRORCODE: errorcode, ATTR_MESSAGE: message})


def _get_devices(hass: HomeAssistant, discovery_type: str, keys: List[str], interface: str) -> List[Dict[str, Any]]:
    """Get the HomeMatic devices for given discovery_type."""
    device_arr: List[Dict[str, Any]] = []
    for key in keys:
        device = hass.data[DATA_HOMEMATIC].devices[interface][key]
        class_name: str = device.__class__.__name__
        metadata: Dict[Any, Any] = {}
        if discovery_type != DISCOVER_BATTERY and class_name not in HM_DEVICE_TYPES[discovery_type]:
            continue
        if discovery_type == DISCOVER_SENSORS:
            metadata.update(device.SENSORNODE)
        elif discovery_type == DISCOVER_BINARY_SENSORS:
            metadata.update(device.BINARYNODE)
        elif discovery_type == DISCOVER_BATTERY:
            if ATTR_LOWBAT in device.ATTRIBUTENODE:
                metadata.update({ATTR_LOWBAT: device.ATTRIBUTENODE[ATTR_LOWBAT]})
            elif ATTR_LOW_BAT in device.ATTRIBUTENODE:
                metadata.update({ATTR_LOW_BAT: device.ATTRIBUTENODE[ATTR_LOW_BAT]})
            else:
                continue
        else:
            metadata.update({None: device.ELEMENT})
        for param, channels in metadata.items():
            if param in HM_IGNORE_DISCOVERY_NODE and class_name not in HM_IGNORE_DISCOVERY_NODE_EXCEPTIONS.get(param, []):
                continue
            if discovery_type == DISCOVER_SWITCHES and class_name == 'IPKeySwitchLevel':
                channels.remove(8)
                channels.remove(12)
            if discovery_type == DISCOVER_LIGHTS and class_name == 'IPKeySwitchLevel':
                channels.remove(4)
            _LOGGER.debug('%s: Handling %s: %s: %s', discovery_type, key, param, channels)
            for channel in channels:
                name: str = _create_ha_id(name=device.NAME, channel=channel, param=param, count=len(channels))
                unique_id: str = _create_ha_id(name=key, channel=channel, param=param, count=len(channels))
                device_dict: Dict[str, Any] = {
                    CONF_PLATFORM: 'homematic',
                    ATTR_ADDRESS: key,
                    ATTR_INTERFACE: interface,
                    ATTR_NAME: name,
                    ATTR_DEVICE_TYPE: class_name,
                    ATTR_CHANNEL: channel,
                    ATTR_UNIQUE_ID: unique_id
                }
                if param is not None:
                    device_dict[ATTR_PARAM] = param
                try:
                    DEVICE_SCHEMA(device_dict)
                    device_arr.append(device_dict)
                except vol.MultipleInvalid as err:
                    _LOGGER.error('Invalid device config: %s', str(err))
    return device_arr


def _create_ha_id(name: str, channel: int, param: Optional[str], count: int) -> str:
    """Generate a unique entity id."""
    if count == 1 and param is None:
        return name
    if count > 1 and param is None:
        return f'{name} {channel}'
    if count == 1 and param is not None:
        return f'{name} {param}'
    if count > 1 and param is not None:
        return f'{name} {channel} {param}'
    raise ValueError(f'Unable to create unique id for count:{count} and param:{param}')


def _hm_event_handler(hass: HomeAssistant, interface: str, device: str, caller: Any, attribute: str, value: Any) -> None:
    """Handle all pyhomematic device events."""
    try:
        channel: int = int(device.split(':')[1])
        address: str = device.split(':')[0]
        hmdevice = hass.data[DATA_HOMEMATIC].devices[interface].get(address)
    except (TypeError, ValueError):
        _LOGGER.error('Event handling channel convert error!')
        return
    if attribute not in hmdevice.EVENTNODE:
        return
    _LOGGER.debug('Event %s for %s channel %i', attribute, hmdevice.NAME, channel)
    if attribute in HM_PRESS_EVENTS:
        hass.bus.fire(EVENT_KEYPRESS, {ATTR_NAME: hmdevice.NAME, ATTR_PARAM: attribute, ATTR_CHANNEL: channel})
        return
    if attribute in HM_IMPULSE_EVENTS:
        hass.bus.fire(EVENT_IMPULSE, {ATTR_NAME: hmdevice.NAME, ATTR_CHANNEL: channel})
        return
    _LOGGER.warning('Event is unknown and not forwarded')


def _device_from_servicecall(hass: HomeAssistant, service: ServiceCall) -> Optional[Any]:
    """Extract HomeMatic device from service call."""
    address: str = service.data.get(ATTR_ADDRESS)
    interface: Optional[str] = service.data.get(ATTR_INTERFACE)
    if address == 'BIDCOS-RF':
        address = 'BidCoS-RF'
    if address == 'HMIP-RCV-1':
        address = 'HmIP-RCV-1'
    if interface:
        return hass.data[DATA_HOMEMATIC].devices[interface].get(address)
    for devices in hass.data[DATA_HOMEMATIC].devices.values():
        if address in devices:
            return devices[address]
    return None