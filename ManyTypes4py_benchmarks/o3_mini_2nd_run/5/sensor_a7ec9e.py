from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Optional
import hpilo
import voluptuous as vol

from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import (
    CONF_HOST,
    CONF_MONITORED_VARIABLES,
    CONF_NAME,
    CONF_PASSWORD,
    CONF_PORT,
    CONF_SENSOR_TYPE,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_USERNAME,
    CONF_VALUE_TEMPLATE,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import Throttle

_LOGGER: logging.Logger = logging.getLogger(__name__)

DEFAULT_NAME: str = 'HP ILO'
DEFAULT_PORT: int = 443
MIN_TIME_BETWEEN_UPDATES: timedelta = timedelta(seconds=300)

SENSOR_TYPES: dict[str, list[Any]] = {
    'server_name': ['Server Name', 'get_server_name'],
    'server_fqdn': ['Server FQDN', 'get_server_fqdn'],
    'server_host_data': ['Server Host Data', 'get_host_data'],
    'server_oa_info': ['Server Onboard Administrator Info', 'get_oa_info'],
    'server_power_status': ['Server Power state', 'get_host_power_status'],
    'server_power_readings': ['Server Power readings', 'get_power_readings'],
    'server_power_on_time': ['Server Power On time', 'get_server_power_on_time'],
    'server_asset_tag': ['Server Asset Tag', 'get_asset_tag'],
    'server_uid_status': ['Server UID light', 'get_uid_status'],
    'server_health': ['Server Health', 'get_embedded_health'],
    'network_settings': ['Network Settings', 'get_network_settings'],
}

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_HOST): cv.string,
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
    vol.Optional(CONF_MONITORED_VARIABLES, default=[]): vol.All(
        cv.ensure_list,
        [vol.Schema({
            vol.Required(CONF_NAME): cv.string,
            vol.Required(CONF_SENSOR_TYPE): vol.All(cv.string, vol.In(SENSOR_TYPES)),
            vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string,
            vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
        })]
    ),
    vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PORT, default=DEFAULT_PORT): cv.port,
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    hostname: str = config[CONF_HOST]
    port: int = config[CONF_PORT]
    login: str = config[CONF_USERNAME]
    password: str = config[CONF_PASSWORD]
    monitored_variables: list[dict[str, Any]] = config[CONF_MONITORED_VARIABLES]
    try:
        hp_ilo_data: HpIloData = HpIloData(hostname, port, login, password)
    except ValueError as error:
        _LOGGER.error(error)
        return
    devices: list[HpIloSensor] = []
    for monitored_variable in monitored_variables:
        new_device: HpIloSensor = HpIloSensor(
            hass=hass,
            hp_ilo_data=hp_ilo_data,
            sensor_name=f'{config[CONF_NAME]} {monitored_variable[CONF_NAME]}',
            sensor_type=monitored_variable[CONF_SENSOR_TYPE],
            sensor_value_template=monitored_variable.get(CONF_VALUE_TEMPLATE),
            unit_of_measurement=monitored_variable.get(CONF_UNIT_OF_MEASUREMENT),
        )
        devices.append(new_device)
    add_entities(devices, True)

class HpIloSensor(SensorEntity):
    def __init__(
        self,
        hass: HomeAssistant,
        hp_ilo_data: HpIloData,
        sensor_type: str,
        sensor_name: str,
        sensor_value_template: Optional[Any],
        unit_of_measurement: Optional[str],
    ) -> None:
        self._hass: HomeAssistant = hass
        self._name: str = sensor_name
        self._unit_of_measurement: Optional[str] = unit_of_measurement
        self._ilo_function: str = SENSOR_TYPES[sensor_type][1]
        self.hp_ilo_data: HpIloData = hp_ilo_data
        self._sensor_value_template: Optional[Any] = sensor_value_template
        self._state: Any = None
        self._state_attributes: Any = None
        _LOGGER.debug('Created HP iLO sensor %r', self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_unit_of_measurement(self) -> Optional[str]:
        return self._unit_of_measurement

    @property
    def native_value(self) -> Any:
        return self._state

    @property
    def extra_state_attributes(self) -> Any:
        return self._state_attributes

    def update(self) -> None:
        self.hp_ilo_data.update()
        ilo_data: Any = getattr(self.hp_ilo_data.data, self._ilo_function)()
        if self._sensor_value_template is not None:
            ilo_data = self._sensor_value_template.render(ilo_data=ilo_data, parse_result=False)
        self._state = ilo_data

class HpIloData:
    def __init__(self, host: str, port: int, login: str, password: str) -> None:
        self._host: str = host
        self._port: int = port
        self._login: str = login
        self._password: str = password
        self.data: Optional[hpilo.Ilo] = None
        self.update()

    @Throttle(MIN_TIME_BETWEEN_UPDATES)
    def update(self) -> None:
        try:
            self.data = hpilo.Ilo(
                hostname=self._host,
                login=self._login,
                password=self._password,
                port=self._port,
            )
        except (hpilo.IloError, hpilo.IloCommunicationError, hpilo.IloLoginFailed) as error:
            raise ValueError(f'Unable to init HP ILO, {error}') from error