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
from homeassistant.const import ATTR_ENTITY_ID, ATTR_MODE, ATTR_NAME, CONF_ALIAS, CONF_CONDITIONS, CONF_DEVICE_ID, CONF_EVENT_DATA, CONF_ID, CONF_MODE, CONF_PATH, CONF_PLATFORM, CONF_VARIABLES, CONF_ZONE, EVENT_HOMEASSISTANT_STARTED, SERVICE_RELOAD, SERVICE_TOGGLE, SERVICE_TURN_OFF, SERVICE_TURN_ON, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Context, CoreState, Event, HomeAssistant, ServiceCall, callback, split_entity_id, valid_entity_id
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, TemplateError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.entity import ToggleEntity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
from homeassistant.helpers.script import ATTR_CUR, ATTR_MAX, CONF_MAX, CONF_MAX_EXCEEDED, Script, ScriptRunResult, script_stack_cv
from homeassistant.helpers.script_variables import ScriptVariables
from homeassistant.helpers.service import ReloadServiceHelper, async_register_admin_service
from homeassistant.helpers.trace import TraceElement, script_execution_set, trace_append_element, trace_get, trace_path
from homeassistant.loader import bind_hass
from homeassistant.util.dt import parse_datetime
from homeassistant.util.hass_dict import HassKey
from .config import AutomationConfig, ValidationStatus
from .const import CONF_ACTIONS, CONF_INITIAL_STATE, CONF_TRACE, CONF_TRIGGER_VARIABLES, CONF_TRIGGERS, DEFAULT_INITIAL_STATE, DOMAIN, LOGGER
from .helpers import async_get_blueprints

class IfAction(Protocol):
    """Define the format of if_action."""

    def __call__(self, variables: Mapping[str, Any]) -> bool:
        """AND all conditions."""

@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    """Return true if specified automation entity_id is on.

    Async friendly.
    """
    return hass.states.is_state(entity_id, STATE_ON)

class BaseAutomationEntity(ToggleEntity, ABC):
    """Base class for automation entities."""

    _entity_component_unrecorded_attributes: frozenset[str]

    @property
    def capability_attributes(self) -> Mapping[str, Any]:
        """Return capability attributes."""
        if self.unique_id is not None:
            return {CONF_ID: self.unique_id}
        return None

    @cached_property
    @abstractmethod
    def referenced_labels(self) -> set[str]:
        """Return a set of referenced labels."""

    @cached_property
    @abstractmethod
    def referenced_floors(self) -> set[str]:
        """Return a set of referenced floors."""

    @cached_property
    @abstractmethod
    def referenced_areas(self) -> set[str]:
        """Return a set of referenced areas."""

    @property
    @abstractmethod
    def referenced_blueprint(self) -> str | None:
        """Return referenced blueprint or None."""

    @cached_property
    @abstractmethod
    def referenced_devices(self) -> set[str]:
        """Return a set of referenced devices."""

    @cached_property
    @abstractmethod
    def referenced_entities(self) -> set[str]:
        """Return a set of referenced entities."""

    @abstractmethod
    async def async_trigger(self, run_variables: Mapping[str, Any], context: Context | None = None, skip_condition: bool = False) -> ScriptRunResult | None:
        """Trigger automation."""

class UnavailableAutomationEntity(BaseAutomationEntity):
    """A non-functional automation entity with its state set to unavailable.

    This class is instantiated when an automation fails to validate.
    """

    _attr_should_poll: bool
    _attr_available: bool

    def __init__(self, automation_id: str, name: str, trigger_config: Mapping[str, Any], cond_func: IfAction | None, action_script: Script, initial_state: bool, variables: ScriptVariables | None, trigger_variables: Mapping[str, Any] | None, raw_config: AutomationConfig, blueprint_inputs: Mapping[str, Any] | None, trace_config: Mapping[str, Any] | None):
        """Initialize an automation entity."""
        self._attr_name = name
        self._trigger_config = trigger_config
        self._async_detach_triggers: CALLBACK_TYPE | None
        self._cond_func = cond_func
        self.action_script = action_script
        self.action_script.change_listener = self.async_write_ha_state
        self._initial_state = initial_state
        self._is_enabled: bool
        self._logger = LOGGER
        self._variables = variables
        self._trigger_variables = trigger_variables
        self.raw_config = raw_config
        self._blueprint_inputs = blueprint_inputs
        self._trace_config = trace_config
        self._attr_unique_id = automation_id

    # ... rest of the class

class AutomationEntity(BaseAutomationEntity, RestoreEntity):
    """Entity to show status of entity."""

    _attr_should_poll: bool

    def __init__(self, automation_id: str, name: str, trigger_config: Mapping[str, Any], cond_func: IfAction | None, action_script: Script, initial_state: bool, variables: ScriptVariables | None, trigger_variables: Mapping[str, Any] | None, raw_config: AutomationConfig, blueprint_inputs: Mapping[str, Any] | None, trace_config: Mapping[str, Any] | None):
        """Initialize an automation entity."""
        # ... rest of the class

@dataclass(slots=True)
class AutomationEntityConfig:
    """Container for prepared automation entity configuration."""

async def _prepare_automation_config(hass: HomeAssistant, config: Mapping[str, Any], wanted_automation_id: str | None) -> list[AutomationEntityConfig]:
    """Parse configuration and prepare automation entity configuration."""
    # ... rest of the function

async def _create_automation_entities(hass: HomeAssistant, automation_configs: list[AutomationEntityConfig]) -> list[BaseAutomationEntity]:
    """Create automation entities from prepared configuration."""
    # ... rest of the function

async def _async_process_config(hass: HomeAssistant, config: Mapping[str, Any], component: EntityComponent):
    """Process config and add automations."""
    # ... rest of the function

async def _async_process_single_config(hass: HomeAssistant, config: Mapping[str, Any], component: EntityComponent, automation_id: str):
    """Process config and add a single automation."""
    # ... rest of the function

@websocket_api.websocket_command({'type': 'automation/config', 'entity_id': str})
def websocket_config(hass: Home