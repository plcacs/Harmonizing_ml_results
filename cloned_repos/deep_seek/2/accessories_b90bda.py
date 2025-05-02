"""Extend the basic Accessory and Bridge functions."""
from __future__ import annotations
import logging
from typing import Any, cast, Optional, Dict, List, Tuple, Union
from uuid import UUID
from pyhap.accessory import Accessory, Bridge
from pyhap.accessory_driver import AccessoryDriver
from pyhap.characteristic import Characteristic
from pyhap.const import CATEGORY_OTHER
from pyhap.iid_manager import IIDManager
from pyhap.service import Service
from pyhap.util import callback as pyhap_callback
from homeassistant.components.cover import CoverDeviceClass, CoverEntityFeature
from homeassistant.components.media_player import MediaPlayerDeviceClass
from homeassistant.components.remote import RemoteEntityFeature
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.switch import SwitchDeviceClass
from homeassistant.const import ATTR_BATTERY_CHARGING, ATTR_BATTERY_LEVEL, ATTR_DEVICE_CLASS, ATTR_ENTITY_ID, ATTR_HW_VERSION, ATTR_MANUFACTURER, ATTR_MODEL, ATTR_SERVICE, ATTR_SUPPORTED_FEATURES, ATTR_SW_VERSION, ATTR_UNIT_OF_MEASUREMENT, CONF_NAME, CONF_TYPE, LIGHT_LUX, PERCENTAGE, STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN, UnitOfTemperature, __version__
from homeassistant.core import CALLBACK_TYPE, Context, Event, EventStateChangedData, HassJobType, HomeAssistant, State, callback as ha_callback, split_entity_id
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util.decorator import Registry
from .const import ATTR_DISPLAY_NAME, ATTR_INTEGRATION, ATTR_VALUE, BRIDGE_MODEL, BRIDGE_SERIAL_NUMBER, CHAR_BATTERY_LEVEL, CHAR_CHARGING_STATE, CHAR_HARDWARE_REVISION, CHAR_STATUS_LOW_BATTERY, CONF_FEATURE_LIST, CONF_LINKED_BATTERY_CHARGING_SENSOR, CONF_LINKED_BATTERY_SENSOR, CONF_LOW_BATTERY_THRESHOLD, DEFAULT_LOW_BATTERY_THRESHOLD, EMPTY_MAC, EVENT_HOMEKIT_CHANGED, HK_CHARGING, HK_NOT_CHARGABLE, HK_NOT_CHARGING, MANUFACTURER, MAX_MANUFACTURER_LENGTH, MAX_MODEL_LENGTH, MAX_SERIAL_LENGTH, MAX_VERSION_LENGTH, SERV_ACCESSORY_INFO, SERV_BATTERY_SERVICE, SIGNAL_RELOAD_ENTITIES, TYPE_FAUCET, TYPE_OUTLET, TYPE_SHOWER, TYPE_SPRINKLER, TYPE_SWITCH, TYPE_VALVE
from .iidmanager import AccessoryIIDStorage
from .util import accessory_friendly_name, async_dismiss_setup_message, async_show_setup_message, cleanup_name_for_homekit, convert_to_float, format_version, validate_media_player_features

_LOGGER = logging.getLogger(__name__)
SWITCH_TYPES: Dict[str, str] = {TYPE_FAUCET: 'ValveSwitch', TYPE_OUTLET: 'Outlet', TYPE_SHOWER: 'ValveSwitch', TYPE_SPRINKLER: 'ValveSwitch', TYPE_SWITCH: 'Switch', TYPE_VALVE: 'ValveSwitch'}
TYPES: Registry = Registry()
RELOAD_ON_CHANGE_ATTRS: Tuple[str, str, str] = (ATTR_SUPPORTED_FEATURES, ATTR_DEVICE_CLASS, ATTR_UNIT_OF_MEASUREMENT)

def get_accessory(hass: HomeAssistant, driver: HomeDriver, state: State, aid: int, config: Dict[str, Any]) -> Optional[Accessory]:
    """Take state and return an accessory object if supported."""
    if not aid:
        _LOGGER.warning('The entity "%s" is not supported, since it generates an invalid aid, please change it', state.entity_id)
        return None
    a_type: Optional[str] = None
    name: str = config.get(CONF_NAME, state.name)
    features: int = state.attributes.get(ATTR_SUPPORTED_FEATURES, 0)
    
    if state.domain == 'alarm_control_panel':
        a_type = 'SecuritySystem'
    elif state.domain in ('binary_sensor', 'device_tracker', 'person'):
        a_type = 'BinarySensor'
    elif state.domain == 'climate':
        a_type = 'Thermostat'
    elif state.domain == 'cover':
        device_class = state.attributes.get(ATTR_DEVICE_CLASS)
        if device_class in (CoverDeviceClass.GARAGE, CoverDeviceClass.GATE) and features & (CoverEntityFeature.OPEN | CoverEntityFeature.CLOSE):
            a_type = 'GarageDoorOpener'
        elif device_class == CoverDeviceClass.WINDOW and features & CoverEntityFeature.SET_POSITION:
            a_type = 'Window'
        elif device_class == CoverDeviceClass.DOOR and features & CoverEntityFeature.SET_POSITION:
            a_type = 'Door'
        elif features & CoverEntityFeature.SET_POSITION:
            a_type = 'WindowCovering'
        elif features & (CoverEntityFeature.OPEN | CoverEntityFeature.CLOSE):
            a_type = 'WindowCoveringBasic'
        elif features & CoverEntityFeature.SET_TILT_POSITION:
            a_type = 'WindowCovering'
    elif state.domain == 'fan':
        a_type = 'Fan'
    elif state.domain == 'humidifier':
        a_type = 'HumidifierDehumidifier'
    elif state.domain == 'light':
        a_type = 'Light'
    elif state.domain == 'lock':
        a_type = 'Lock'
    elif state.domain == 'media_player':
        device_class = state.attributes.get(ATTR_DEVICE_CLASS)
        feature_list = config.get(CONF_FEATURE_LIST, [])
        if device_class == MediaPlayerDeviceClass.RECEIVER:
            a_type = 'ReceiverMediaPlayer'
        elif device_class == MediaPlayerDeviceClass.TV:
            a_type = 'TelevisionMediaPlayer'
        elif validate_media_player_features(state, feature_list):
            a_type = 'MediaPlayer'
    elif state.domain == 'sensor':
        device_class = state.attributes.get(ATTR_DEVICE_CLASS)
        unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        if device_class == SensorDeviceClass.TEMPERATURE or unit in (UnitOfTemperature.CELSIUS, UnitOfTemperature.FAHRENHEIT):
            a_type = 'TemperatureSensor'
        elif device_class == SensorDeviceClass.HUMIDITY and unit == PERCENTAGE:
            a_type = 'HumiditySensor'
        elif device_class == SensorDeviceClass.PM10 or SensorDeviceClass.PM10 in state.entity_id:
            a_type = 'PM10Sensor'
        elif device_class == SensorDeviceClass.PM25 or SensorDeviceClass.PM25 in state.entity_id:
            a_type = 'PM25Sensor'
        elif device_class == SensorDeviceClass.NITROGEN_DIOXIDE:
            a_type = 'NitrogenDioxideSensor'
        elif device_class == SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS:
            a_type = 'VolatileOrganicCompoundsSensor'
        elif device_class == SensorDeviceClass.GAS or SensorDeviceClass.GAS in state.entity_id:
            a_type = 'AirQualitySensor'
        elif device_class == SensorDeviceClass.CO:
            a_type = 'CarbonMonoxideSensor'
        elif device_class == SensorDeviceClass.CO2 or 'co2' in state.entity_id:
            a_type = 'CarbonDioxideSensor'
        elif device_class == SensorDeviceClass.ILLUMINANCE or unit == LIGHT_LUX:
            a_type = 'LightSensor'
    elif state.domain == 'switch':
        if (switch_type := config.get(CONF_TYPE)):
            a_type = SWITCH_TYPES[switch_type]
        elif state.attributes.get(ATTR_DEVICE_CLASS) == SwitchDeviceClass.OUTLET:
            a_type = 'Outlet'
        else:
            a_type = 'Switch'
    elif state.domain == 'valve':
        a_type = 'Valve'
    elif state.domain == 'vacuum':
        a_type = 'Vacuum'
    elif state.domain == 'remote' and features & RemoteEntityFeature.ACTIVITY:
        a_type = 'ActivityRemote'
    elif state.domain in ('automation', 'button', 'input_boolean', 'input_button', 'remote', 'scene', 'script'):
        a_type = 'Switch'
    elif state.domain in ('input_select', 'select'):
        a_type = 'SelectSwitch'
    elif state.domain == 'water_heater':
        a_type = 'WaterHeater'
    elif state.domain == 'camera':
        a_type = 'Camera'
    if a_type is None:
        return None
    _LOGGER.debug('Add "%s" as "%s"', state.entity_id, a_type)
    return TYPES[a_type](hass, driver, name, state.entity_id, aid, config)

class HomeAccessory(Accessory):
    """Adapter class for Accessory."""

    def __init__(
        self,
        hass: HomeAssistant,
        driver: HomeDriver,
        name: str,
        entity_id: str,
        aid: int,
        config: Dict[str, Any],
        *args: Any,
        category: int = CATEGORY_OTHER,
        device_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize a Accessory object."""
        super().__init__(*args, driver=driver, display_name=cleanup_name_for_homekit(name), aid=aid, iid_manager=HomeIIDManager(driver.iid_storage), **kwargs)
        self._reload_on_change_attrs: List[str] = list(RELOAD_ON_CHANGE_ATTRS)
        self.config: Dict[str, Any] = config or {}
        if device_id:
            self.device_id: Optional[str] = device_id
            serial_number: str = device_id
            domain: Optional[str] = None
        else:
            self.device_id: Optional[str] = None
            serial_number: str = entity_id
            domain: Optional[str] = split_entity_id(entity_id)[0].replace('_', ' ')
        
        manufacturer: str
        if self.config.get(ATTR_MANUFACTURER) is not None:
            manufacturer = str(self.config[ATTR_MANUFACTURER])
        elif self.config.get(ATTR_INTEGRATION) is not None:
            manufacturer = self.config[ATTR_INTEGRATION].replace('_', ' ').title()
        elif domain:
            manufacturer = f'{MANUFACTURER} {domain}'.title()
        else:
            manufacturer = MANUFACTURER

        model: str
        if self.config.get(ATTR_MODEL) is not None:
            model = str(self.config[ATTR_MODEL])
        elif domain:
            model = domain.title()
        else:
            model = MANUFACTURER

        sw_version: Optional[str] = None
        if self.config.get(ATTR_SW_VERSION) is not None:
            sw_version = format_version(self.config[ATTR_SW_VERSION])
        if sw_version is None:
            sw_version = format_version(__version__)
            assert sw_version is not None

        hw_version: Optional[str] = None
        if self.config.get(ATTR_HW_VERSION) is not None:
            hw_version = format_version(self.config[ATTR_HW_VERSION])

        self.set_info_service(
            manufacturer=manufacturer[:MAX_MANUFACTURER_LENGTH],
            model=model[:MAX_MODEL_LENGTH],
            serial_number=serial_number[:MAX_SERIAL_LENGTH],
            firmware_revision=sw_version[:MAX_VERSION_LENGTH]
        )

        if hw_version:
            serv_info = self.get_service(SERV_ACCESSORY_INFO)
            char = self.driver.loader.get_char(CHAR_HARDWARE_REVISION)
            serv_info.add_characteristic(char)
            serv_info.configure_char(CHAR_HARDWARE_REVISION, value=hw_version[:MAX_VERSION_LENGTH])
            char.broker = self
            self.iid_manager.assign(char)

        self.category: int = category
        self.entity_id: str = entity_id
        self.hass: HomeAssistant = hass
        self._subscriptions: List[CALLBACK_TYPE] = []
        self._available: bool = False
        self._char_battery: Optional[Characteristic] = None
        self._char_charging: Optional[Characteristic] = None
        self._char_low_battery: Optional[Characteristic] = None
        self.linked_battery_sensor: Optional[str] = self.config.get(CONF_LINKED_BATTERY_SENSOR)
        self.linked_battery_charging_sensor: Optional[str] = self.config.get(CONF_LINKED_BATTERY_CHARGING_SENSOR)
        self.low_battery_threshold: int = self.config.get(CONF_LOW_BATTERY_THRESHOLD, DEFAULT_LOW_BATTERY_THRESHOLD)

        if device_id:
            return

        state: Optional[State] = self.hass.states.get(self.entity_id)
        self._update_available_from_state(state)
        assert state is not None
        entity_attributes = state.attributes
        battery_found: Union[bool, str, None] = entity_attributes.get(ATTR_BATTERY_LEVEL)

        if self.linked_battery_sensor:
            state = self.hass.states.get(self.linked_battery_sensor)
            if state is not None:
                battery_found = state.state
            else:
                _LOGGER.warning('%s: Battery sensor state missing: %s', self.entity_id, self.linked_battery_sensor)
                self.linked_battery_sensor = None

        if not battery_found:
            return

        _LOGGER.debug('%s: Found battery level', self.entity_id)
        if self.linked_battery_charging_sensor:
            state = self.hass.states.get(self.linked_battery_charging_sensor)
            if state is None:
                self.linked_battery_charging_sensor = None
                _LOGGER.warning('%s: Battery charging binary_sensor state missing: %s', self.entity_id, self.linked_battery_charging_sensor)
            else:
                _LOGGER.debug('%s: Found battery charging', self.entity_id)

        serv_battery = self.add_preload_service(SERV_BATTERY_SERVICE)
        self._char_battery = serv_battery.configure_char(CHAR_BATTERY_LEVEL, value=0)
        self._char_charging = serv_battery.configure_char(CHAR_CHARGING_STATE, value=HK_NOT_CHARGABLE)
        self._char_low_battery = serv_battery.configure_char(CHAR_STATUS_LOW_BATTERY, value=0)

    def _update_available_from_state(self, new_state: Optional[State]) -> None:
        """Update the available property based on the state."""
        self._available = new_state is not None and new_state.state != STATE_UNAVAILABLE

    @property
    def available(self) -> bool:
        """Return if accessory is available."""
        return self._available

    @ha_callback
    @pyhap_callback
    def run(self) -> None:
        """Handle accessory driver started event."""
        state: Optional[State] = self.hass.states.get(self.entity_id)
        if state:
            self.async_update_state_callback(state)
        self._update_available_from_state(state)
        self._subscriptions.append(
            async_track_state_change_event(
                self.hass,
                [self.entity_id],
                self.async_update_event_state_callback,
                job_type=HassJobType.Callback
            )
        )

        battery_charging_state: Optional[bool] = None
        battery_state: Optional[str] = None

        if self.linked_battery_sensor and (linked_battery_sensor_state := self.hass.states.get(self.linked_battery_sensor)):
            battery_state = linked_battery_sensor_state.state
            battery_charging_state = linked_battery_sensor_state.attributes.get(ATTR_BATTERY_CHARGING)
            self._subscriptions.append(
                async_track_state_change_event(
                    self.hass,
                    [self.linked_battery_sensor],
                    self.async_update_linked_battery_callback,
                    job_type=HassJobType.Callback
                )
            )
        elif state is not None:
            battery_state = state.attributes.get(ATTR_BATTERY_LEVEL)

        if self.linked_battery_charging_sensor:
            state = self.hass.states.get(self.linked_battery_charging_sensor)
            battery_charging_state = state and state.state == STATE_ON
            self._subscriptions.append(
                async_track_state_change_event(
                    self.hass,
                    [self.linked_battery_charging_sensor],
                    self.async_update_linked_battery_charging_callback,
                    job_type=HassJobType.Callback
                )
            )
        elif battery_charging_state is None and state is not None:
            battery_charging_state = state.attributes.get(ATTR_BATTERY_CHARGING)

        if battery_state is not None or battery_charging_state is not None:
            self.async_update_battery(battery_state, battery_charging_state)

    @ha_callback
    def async_update_event_state_callback(self, event: Event[EventStateChangedData]) -> None:
        """Handle state change event listener callback."""
        new_state: Optional[State] = event.data['new_state']
        old_state: Optional[State] = event.data['old_state']
        self._update_available_from_state(new_state)
        if new_state and old_state and (STATE_UNAVAILABLE not in (old_state.state, new_state.state)):
            old_attributes = old_state.attributes
            new_attributes = new_state.attributes
            for attr in self._reload_on_change_attrs:
                if old_attributes.get(attr) != new_attributes.get(attr):
                    _LOGGER.debug('%s: Reloading HomeKit accessory since %s has changed from %s -> %s', self.entity_id, attr, old_attributes.get(attr), new_attributes.get(