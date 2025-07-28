import logging
from typing import Any, Optional
from rtmapi import Rtm, RtmRequestFailedException
from homeassistant.const import CONF_ID, CONF_NAME, STATE_OK
from homeassistant.core import ServiceCall
from homeassistant.helpers.entity import Entity

_LOGGER = logging.getLogger(__name__)

class RememberTheMilkEntity(Entity):
    def __init__(self, name: str, api_key: str, shared_secret: str, token: str, rtm_config: Any) -> None:
        self._name: str = name
        self._api_key: str = api_key
        self._shared_secret: str = shared_secret
        self._token: str = token
        self._rtm_config: Any = rtm_config
        self._rtm_api: Rtm = Rtm(api_key, shared_secret, 'delete', token)
        self._token_valid: Optional[bool] = None
        self._check_token()
        _LOGGER.debug('Instance created for account %s', self._name)

    def _check_token(self) -> bool:
        valid: bool = self._rtm_api.token_valid()
        if not valid:
            _LOGGER.error('Token for account %s is invalid. You need to register again!', self.name)
            self._rtm_config.delete_token(self._name)
            self._token_valid = False
        else:
            self._token_valid = True
        return self._token_valid

    def create_task(self, call: ServiceCall) -> None:
        try:
            task_name: str = call.data[CONF_NAME]
            hass_id: Optional[Any] = call.data.get(CONF_ID)
            rtm_id: Optional[Any] = None
            if hass_id is not None:
                rtm_id = self._rtm_config.get_rtm_id(self._name, hass_id)
            result = self._rtm_api.rtm.timelines.create()
            timeline: str = result.timeline.value
            if hass_id is None or rtm_id is None:
                result = self._rtm_api.rtm.tasks.add(timeline=timeline, name=task_name, parse='1')
                _LOGGER.debug("Created new task '%s' in account %s", task_name, self.name)
                self._rtm_config.set_rtm_id(self._name, hass_id, result.list.id, result.list.taskseries.id, result.list.taskseries.task.id)
            else:
                self._rtm_api.rtm.tasks.setName(name=task_name, list_id=rtm_id[0], taskseries_id=rtm_id[1], task_id=rtm_id[2], timeline=timeline)
                _LOGGER.debug("Updated task with id '%s' in account %s to name %s", hass_id, self.name, task_name)
        except RtmRequestFailedException as rtm_exception:
            _LOGGER.error('Error creating new Remember The Milk task for account %s: %s', self._name, rtm_exception)

    def complete_task(self, call: ServiceCall) -> None:
        hass_id: Any = call.data[CONF_ID]
        rtm_id: Optional[Any] = self._rtm_config.get_rtm_id(self._name, hass_id)
        if rtm_id is None:
            _LOGGER.error('Could not find task with ID %s in account %s. So task could not be closed', hass_id, self._name)
            return
        try:
            result = self._rtm_api.rtm.timelines.create()
            timeline: str = result.timeline.value
            self._rtm_api.rtm.tasks.complete(list_id=rtm_id[0], taskseries_id=rtm_id[1], task_id=rtm_id[2], timeline=timeline)
            self._rtm_config.delete_rtm_id(self._name, hass_id)
            _LOGGER.debug('Completed task with id %s in account %s', hass_id, self._name)
        except RtmRequestFailedException as rtm_exception:
            _LOGGER.error('Error creating new Remember The Milk task for account %s: %s', self._name, rtm_exception)

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> str:
        if not self._token_valid:
            return 'API token invalid'
        return STATE_OK