from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from typing import Any, Protocol, cast
from propcache.api import cached_property
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import ATTR_ENTITY_ID, ATTR_MODE, ATTR_NAME, CONF_ALIAS, CONF_CONDITIONS, CONF_DEVICE_ID, CONF_ENTITY_ID, CONF_EVENT_DATA, CONF_ID, CONF_MODE, CONF_PATH, CONF_PLATFORM, CONF_VARIABLES, CONF_ZONE, EVENT_HOMEASSISTANT_STARTED, SERVICE_RELOAD, SERVICE_TOGGLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Context, CoreState, Event, HomeAssistant, ServiceCall, callback, split_entity_id, valid_entity_id
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, TemplateError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.entity import ToggleEntity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.script import ATTR_CUR, ATTR_MAX, CONF_MAX, CONF_MAX_EXCEEDED, Script, ScriptRunResult, script_stack_cv
from homeassistant.helpers.script_variables import ScriptVariables
from homeassistant.helpers.service import ReloadServiceHelper, async_register_admin_service
from homeassistant.helpers.trace import TraceElement, script_execution_set, trace_append_element, trace_get, trace_path
from homeassistant.helpers.trigger import async_initialize_triggers
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass
from homeassistant.util.dt import parse_datetime
from homeassistant.util.hass_dict import HassKey
from .config import AutomationConfig, ValidationStatus
from .const import CONF_ACTIONS, CONF_INITIAL_STATE, CONF_TRACE, CONF_TRIGGER_VARIABLES, CONF_TRIGGERS, DEFAULT_INITIAL_STATE, DOMAIN, LOGGER
from .helpers import async_get_blueprints
from .trace import trace_automation
DATA_COMPONENT = HassKey(DOMAIN)
ENTITY_ID_FORMAT = DOMAIN + '.{}'
CONF_SKIP_CONDITION = 'skip_condition'
CONF_STOP_ACTIONS = 'stop_actions'
DEFAULT_STOP_ACTIONS = True
EVENT_AUTOMATION_RELOADED = 'automation_reloaded'
EVENT_AUTOMATION_TRIGGERED = 'automation_triggered'
ATTR_LAST_TRIGGERED = 'last_triggered'
ATTR_SOURCE = 'source'
ATTR_VARIABLES = 'variables'
SERVICE_TRIGGER = 'trigger'

class IfAction(Protocol):
    def __call__(self, variables=None) -> None:
        ...

@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    ...

def _automations_with_x(hass: HomeAssistant, referenced_id: str, property_name: str) -> List[str]:
    ...

def _x_in_automation(hass: HomeAssistant, entity_id: str, property_name: str) -> List[Any]:
    ...

@callback
def automations_with_entity(hass: HomeAssistant, entity_id: str) -> List[str]:
    ...

@callback
def entities_in_automation(hass: HomeAssistant, entity_id: str) -> List[Any]:
    ...

@callback
def automations_with_device(hass: HomeAssistant, device_id: str) -> List[str]:
    ...

@callback
def devices_in_automation(hass: HomeAssistant, entity_id: str) -> List[Any]:
    ...

@callback
def automations_with_area(hass: HomeAssistant, area_id: str) -> List[str]:
    ...

@callback
def areas_in_automation(hass: HomeAssistant, entity_id: str) -> List[Any]:
    ...

@callback
def automations_with_floor(hass: HomeAssistant, floor_id: str) -> List[str]:
    ...

@callback
def floors_in_automation(hass: HomeAssistant, entity_id: str) -> List[Any]:
    ...

@callback
def automations_with_label(hass: HomeAssistant, label_id: str) -> List[str]:
    ...

@callback
def labels_in_automation(hass: HomeAssistant, entity_id: str) -> List[Any]:
    ...

@callback
def automations_with_blueprint(hass: HomeAssistant, blueprint_path: str) -> List[str]:
    ...

@callback
def blueprint_in_automation(hass: HomeAssistant, entity_id: str) -> Optional[str]:
    ...

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    ...

class BaseAutomationEntity(ToggleEntity, ABC):
    ...

class UnavailableAutomationEntity(BaseAutomationEntity):
    ...

class AutomationEntity(BaseAutomationEntity, RestoreEntity):
    ...

@dataclass(slots=True)
class AutomationEntityConfig:
    ...

async def _prepare_automation_config(hass: HomeAssistant, config: ConfigType, wanted_automation_id: Optional[str]) -> List[AutomationEntityConfig]:
    ...

def _automation_name(automation_config: AutomationEntityConfig) -> str:
    ...

async def _create_automation_entities(hass: HomeAssistant, automation_configs: List[AutomationEntityConfig]) -> List[AutomationEntity]:
    ...

async def _async_process_config(hass: HomeAssistant, config: ConfigType, component: EntityComponent) -> None:
    ...

def _automation_matches_config(automation: AutomationEntity, config: AutomationEntityConfig) -> bool:
    ...

async def _async_process_single_config(hass: HomeAssistant, config: ConfigType, component: EntityComponent, automation_id: str) -> None:
    ...

async def _async_process_if(hass: HomeAssistant, name: str, config: ConfigType) -> IfAction:
    ...

@callback
def _trigger_extract_devices(trigger_conf: Mapping[str, Any]) -> List[str]:
    ...

@callback
def _trigger_extract_entities(trigger_conf: Mapping[str, Any]) -> List[str]:
    ...

@websocket_api.websocket_command({'type': 'automation/config', 'entity_id': str})
def websocket_config(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Mapping[str, Any]) -> None:
    ...
