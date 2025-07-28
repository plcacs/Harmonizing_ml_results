from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import logging
import os

from rtmapi import Rtm
import voluptuous as vol

from homeassistant.components import configurator
from homeassistant.const import CONF_API_KEY, CONF_ID, CONF_NAME, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.typing import ConfigType
from .entity import RememberTheMilkEntity

_LOGGER: logging.Logger = logging.getLogger(__name__)

DOMAIN = 'remember_the_milk'
DEFAULT_NAME = DOMAIN
CONF_SHARED_SECRET = 'shared_secret'
CONF_ID_MAP = 'id_map'
CONF_LIST_ID = 'list_id'
CONF_TIMESERIES_ID = 'timeseries_id'
CONF_TASK_ID = 'task_id'

RTM_SCHEMA = vol.Schema({
    vol.Required(CONF_NAME): cv.string,
    vol.Required(CONF_API_KEY): cv.string,
    vol.Required(CONF_SHARED_SECRET): cv.string
})
CONFIG_SCHEMA = vol.Schema({DOMAIN: vol.All(cv.ensure_list, [RTM_SCHEMA])}, extra=vol.ALLOW_EXTRA)
CONFIG_FILE_NAME = '.remember_the_milk.conf'
SERVICE_CREATE_TASK = 'create_task'
SERVICE_COMPLETE_TASK = 'complete_task'
SERVICE_SCHEMA_CREATE_TASK = vol.Schema({vol.Required(CONF_NAME): cv.string, vol.Optional(CONF_ID): cv.string})
SERVICE_SCHEMA_COMPLETE_TASK = vol.Schema({vol.Required(CONF_ID): cv.string})


def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    component: EntityComponent[RememberTheMilkEntity] = EntityComponent(_LOGGER, DOMAIN, hass)
    stored_rtm_config: RememberTheMilkConfiguration = RememberTheMilkConfiguration(hass)
    for rtm_config in config[DOMAIN]:
        account_name: str = rtm_config[CONF_NAME]
        _LOGGER.debug('Adding Remember the milk account %s', account_name)
        api_key: str = rtm_config[CONF_API_KEY]
        shared_secret: str = rtm_config[CONF_SHARED_SECRET]
        token: Optional[str] = stored_rtm_config.get_token(account_name)
        if token:
            _LOGGER.debug('found token for account %s', account_name)
            _create_instance(hass, account_name, api_key, shared_secret, token, stored_rtm_config, component)
        else:
            _register_new_account(hass, account_name, api_key, shared_secret, stored_rtm_config, component)
    _LOGGER.debug('Finished adding all Remember the milk accounts')
    return True


def _create_instance(hass: HomeAssistant, account_name: str, api_key: str, shared_secret: str, token: str,
                     stored_rtm_config: "RememberTheMilkConfiguration",
                     component: EntityComponent[RememberTheMilkEntity]) -> None:
    entity: RememberTheMilkEntity = RememberTheMilkEntity(account_name, api_key, shared_secret, token, stored_rtm_config)
    component.add_entities([entity])
    hass.services.register(DOMAIN, f'{account_name}_create_task', entity.create_task, schema=SERVICE_SCHEMA_CREATE_TASK)
    hass.services.register(DOMAIN, f'{account_name}_complete_task', entity.complete_task, schema=SERVICE_SCHEMA_COMPLETE_TASK)


def _register_new_account(hass: HomeAssistant, account_name: str, api_key: str, shared_secret: str,
                          stored_rtm_config: "RememberTheMilkConfiguration",
                          component: EntityComponent[RememberTheMilkEntity]) -> None:
    request_id: Optional[str] = None
    api: Rtm = Rtm(api_key, shared_secret, 'write', None)
    url: str
    frob: str
    url, frob = api.authenticate_desktop()
    _LOGGER.debug('Sent authentication request to server')

    def register_account_callback(fields: Dict[str, Any]) -> None:
        api.retrieve_token(frob)
        token: Optional[str] = api.token
        if token is None:
            _LOGGER.error('Failed to register, please try again')
            configurator.notify_errors(hass, request_id, 'Failed to register, please try again.')
            return
        stored_rtm_config.set_token(account_name, token)
        _LOGGER.debug('Retrieved new token from server')
        _create_instance(hass, account_name, api_key, shared_secret, token, stored_rtm_config, component)
        configurator.request_done(hass, request_id)

    request_id = configurator.request_config(
        hass,
        f'{DOMAIN} - {account_name}',
        callback=register_account_callback,
        description="You need to log in to Remember The Milk to connect your account. \n\nStep 1: Click on the link 'Remember The Milk login'\n\nStep 2: Click on 'login completed'",
        link_name='Remember The Milk login',
        link_url=url,
        submit_caption='login completed'
    )


class RememberTheMilkConfiguration:
    def __init__(self, hass: HomeAssistant) -> None:
        self._config_file_path: str = hass.config.path(CONFIG_FILE_NAME)
        if not os.path.isfile(self._config_file_path):
            self._config: Dict[str, Any] = {}
            return
        try:
            _LOGGER.debug('Loading configuration from file: %s', self._config_file_path)
            with open(self._config_file_path, encoding='utf8') as config_file:
                self._config = json.load(config_file)
        except ValueError:
            _LOGGER.error('Failed to load configuration file, creating a new one: %s', self._config_file_path)
            self._config = {}

    def save_config(self) -> None:
        with open(self._config_file_path, 'w', encoding='utf8') as config_file:
            json.dump(self._config, config_file)

    def get_token(self, profile_name: str) -> Optional[str]:
        if profile_name in self._config:
            return self._config[profile_name][CONF_TOKEN]
        return None

    def set_token(self, profile_name: str, token: str) -> None:
        self._initialize_profile(profile_name)
        self._config[profile_name][CONF_TOKEN] = token
        self.save_config()

    def delete_token(self, profile_name: str) -> None:
        self._config.pop(profile_name, None)
        self.save_config()

    def _initialize_profile(self, profile_name: str) -> None:
        if profile_name not in self._config:
            self._config[profile_name] = {}
        if CONF_ID_MAP not in self._config[profile_name]:
            self._config[profile_name][CONF_ID_MAP] = {}

    def get_rtm_id(self, profile_name: str, hass_id: str) -> Optional[Tuple[Any, Any, Any]]:
        self._initialize_profile(profile_name)
        ids: Optional[Dict[str, Any]] = self._config[profile_name][CONF_ID_MAP].get(hass_id)
        if ids is None:
            return None
        return (ids[CONF_LIST_ID], ids[CONF_TIMESERIES_ID], ids[CONF_TASK_ID])

    def set_rtm_id(self, profile_name: str, hass_id: str, list_id: Any, time_series_id: Any, rtm_task_id: Any) -> None:
        self._initialize_profile(profile_name)
        id_tuple: Dict[str, Any] = {CONF_LIST_ID: list_id, CONF_TIMESERIES_ID: time_series_id, CONF_TASK_ID: rtm_task_id}
        self._config[profile_name][CONF_ID_MAP][hass_id] = id_tuple
        self.save_config()

    def delete_rtm_id(self, profile_name: str, hass_id: str) -> None:
        self._initialize_profile(profile_name)
        if hass_id in self._config[profile_name][CONF_ID_MAP]:
            del self._config[profile_name][CONF_ID_MAP][hass_id]
            self.save_config()