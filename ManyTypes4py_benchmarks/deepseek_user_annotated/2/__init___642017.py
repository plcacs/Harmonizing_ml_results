"""Allow to set up simple automation rules via the config file."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any, Protocol, cast, TypedDict, Optional, Union, List, Dict, Set, FrozenSet

from propcache.api import cached_property
import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_MODE,
    ATTR_NAME,
    CONF_ALIAS,
    CONF_CONDITIONS,
    CONF_DEVICE_ID,
    CONF_ENTITY_ID,
    CONF_EVENT_DATA,
    CONF_ID,
    CONF_MODE,
    CONF_PATH,
    CONF_PLATFORM,
    CONF_VARIABLES,
    CONF_ZONE,
    EVENT_HOMEASSISTANT_STARTED,
    SERVICE_RELOAD,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Context,
    CoreState,
    Event,
    HomeAssistant,
    ServiceCall,
    callback,
    split_entity_id,
    valid_entity_id,
)
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, TemplateError
from homeassistant.helpers import condition, config_validation as cv
from homeassistant.helpers.entity import ToggleEntity
from homeassistant.helpers.entity_component import EntityComponent
from homeassistant.helpers.issue_registry import (
    IssueSeverity,
    async_create_issue,
    async_delete_issue,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.script import (
    ATTR_CUR,
    ATTR_MAX,
    CONF_MAX,
    CONF_MAX_EXCEEDED,
    Script,
    ScriptRunResult,
    script_stack_cv,
)
from homeassistant.helpers.script_variables import ScriptVariables
from homeassistant.helpers.service import (
    ReloadServiceHelper,
    async_register_admin_service,
)
from homeassistant.helpers.trace import (
    TraceElement,
    script_execution_set,
    trace_append_element,
    trace_get,
    trace_path,
)
from homeassistant.helpers.trigger import async_initialize_triggers
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass
from homeassistant.util.dt import parse_datetime
from homeassistant.util.hass_dict import HassKey

from .config import AutomationConfig, ValidationStatus
from .const import (
    CONF_ACTIONS,
    CONF_INITIAL_STATE,
    CONF_TRACE,
    CONF_TRIGGER_VARIABLES,
    CONF_TRIGGERS,
    DEFAULT_INITIAL_STATE,
    DOMAIN,
    LOGGER,
)
from .helpers import async_get_blueprints
from .trace import trace_automation

DATA_COMPONENT: HassKey[EntityComponent[BaseAutomationEntity]] = HassKey(DOMAIN)
ENTITY_ID_FORMAT = DOMAIN + ".{}"


CONF_SKIP_CONDITION = "skip_condition"
CONF_STOP_ACTIONS = "stop_actions"
DEFAULT_STOP_ACTIONS = True

EVENT_AUTOMATION_RELOADED = "automation_reloaded"
EVENT_AUTOMATION_TRIGGERED = "automation_triggered"

ATTR_LAST_TRIGGERED = "last_triggered"
ATTR_SOURCE = "source"
ATTR_VARIABLES = "variables"
SERVICE_TRIGGER = "trigger"


class IfAction(Protocol):
    """Define the format of if_action."""

    config: List[ConfigType]

    def __call__(self, variables: Optional[Mapping[str, Any]] = None) -> bool:
        """AND all conditions."""


class AutomationTriggeredEventData(TypedDict):
    """Dictionary representing automation triggered event data."""

    name: str
    entity_id: str
    source: Optional[str]


@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    """Return true if specified automation entity_id is on."""
    return hass.states.is_state(entity_id, STATE_ON)


def _automations_with_x(
    hass: HomeAssistant, referenced_id: str, property_name: str
) -> List[str]:
    """Return all automations that reference the x."""
    if DATA_COMPONENT not in hass.data:
        return []

    return [
        automation_entity.entity_id
        for automation_entity in hass.data[DATA_COMPONENT].entities
        if referenced_id in getattr(automation_entity, property_name)
    ]


def _x_in_automation(
    hass: HomeAssistant, entity_id: str, property_name: str
) -> List[str]:
    """Return all x in an automation."""
    if DATA_COMPONENT not in hass.data:
        return []

    if (automation_entity := hass.data[DATA_COMPONENT].get_entity(entity_id)) is None:
        return []

    return list(getattr(automation_entity, property_name))


@callback
def automations_with_entity(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all automations that reference the entity."""
    return _automations_with_x(hass, entity_id, "referenced_entities")


@callback
def entities_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all entities in an automation."""
    return _x_in_automation(hass, entity_id, "referenced_entities")


@callback
def automations_with_device(hass: HomeAssistant, device_id: str) -> List[str]:
    """Return all automations that reference the device."""
    return _automations_with_x(hass, device_id, "referenced_devices")


@callback
def devices_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all devices in an automation."""
    return _x_in_automation(hass, entity_id, "referenced_devices")


@callback
def automations_with_area(hass: HomeAssistant, area_id: str) -> List[str]:
    """Return all automations that reference the area."""
    return _automations_with_x(hass, area_id, "referenced_areas")


@callback
def areas_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all areas in an automation."""
    return _x_in_automation(hass, entity_id, "referenced_areas")


@callback
def automations_with_floor(hass: HomeAssistant, floor_id: str) -> List[str]:
    """Return all automations that reference the floor."""
    return _automations_with_x(hass, floor_id, "referenced_floors")


@callback
def floors_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all floors in an automation."""
    return _x_in_automation(hass, entity_id, "referenced_floors")


@callback
def automations_with_label(hass: HomeAssistant, label_id: str) -> List[str]:
    """Return all automations that reference the label."""
    return _automations_with_x(hass, label_id, "referenced_labels")


@callback
def labels_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all labels in an automation."""
    return _x_in_automation(hass, entity_id, "referenced_labels")


@callback
def automations_with_blueprint(hass: HomeAssistant, blueprint_path: str) -> List[str]:
    """Return all automations that reference the blueprint."""
    if DOMAIN not in hass.data:
        return []

    return [
        automation_entity.entity_id
        for automation_entity in hass.data[DATA_COMPONENT].entities
        if automation_entity.referenced_blueprint == blueprint_path
    ]


@callback
def blueprint_in_automation(hass: HomeAssistant, entity_id: str) -> Optional[str]:
    """Return the blueprint the automation is based on or None."""
    if DATA_COMPONENT not in hass.data:
        return None

    if (automation_entity := hass.data[DATA_COMPONENT].get_entity(entity_id)) is None:
        return None

    return automation_entity.referenced_blueprint


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up all automations."""
    hass.data[DATA_COMPONENT] = component = EntityComponent[BaseAutomationEntity](
        LOGGER, DOMAIN, hass
    )

    # Register automation as valid domain for Blueprint
    async_get_blueprints(hass)

    await _async_process_config(hass, config, component)

    # Add some default blueprints to blueprints/automation
    hass.async_create_task(
        async_get_blueprints(hass).async_populate(), eager_start=True
    )

    async def trigger_service_handler(
        entity: BaseAutomationEntity, service_call: ServiceCall
    ) -> None:
        """Handle forced automation trigger."""
        await entity.async_trigger(
            {**service_call.data[ATTR_VARIABLES], "trigger": {"platform": None}},
            skip_condition=service_call.data[CONF_SKIP_CONDITION],
            context=service_call.context,
        )

    component.async_register_entity_service(
        SERVICE_TRIGGER,
        {
            vol.Optional(ATTR_VARIABLES, default={}): dict,
            vol.Optional(CONF_SKIP_CONDITION, default=True): bool,
        },
        trigger_service_handler,
    )
    component.async_register_entity_service(SERVICE_TOGGLE, None, "async_toggle")
    component.async_register_entity_service(SERVICE_TURN_ON, None, "async_turn_on")
    component.async_register_entity_service(
        SERVICE_TURN_OFF,
        {vol.Optional(CONF_STOP_ACTIONS, default=DEFAULT_STOP_ACTIONS): cv.boolean},
        "async_turn_off",
    )

    async def reload_service_handler(service_call: ServiceCall) -> None:
        """Remove all automations and load new ones from config."""
        await async_get_blueprints(hass).async_reset_cache()
        if (conf := await component.async_prepare_reload(skip_reset=True)) is None:
            return
        if automation_id := service_call.data.get(CONF_ID):
            await _async_process_single_config(hass, conf, component, automation_id)
        else:
            await _async_process_config(hass, conf, component)
        hass.bus.async_fire(EVENT_AUTOMATION_RELOADED, context=service_call.context)

    def reload_targets(service_call: ServiceCall) -> Set[Optional[str]]:
        if automation_id := service_call.data.get(CONF_ID):
            return {automation_id}
        return {automation.unique_id for automation in component.entities}

    reload_helper = ReloadServiceHelper(reload_service_handler, reload_targets)

    async_register_admin_service(
        hass,
        DOMAIN,
        SERVICE_RELOAD,
        reload_helper.execute_service,
        schema=vol.Schema({vol.Optional(CONF_ID): str}),
    )

    websocket_api.async_register_command(hass, websocket_config)

    return True


class BaseAutomationEntity(ToggleEntity, ABC):
    """Base class for automation entities."""

    _entity_component_unrecorded_attributes = frozenset(
        {ATTR_LAST_TRIGGERED, ATTR_MODE, ATTR_CUR, ATTR_MAX, CONF_ID}
    )
    raw_config: Optional[ConfigType]

    @property
    def capability_attributes(self) -> Optional[Dict[str, Any]]:
        """Return capability attributes."""
        if self.unique_id is not None:
            return {CONF_ID: self.unique_id}
        return None

    @cached_property
    @abstractmethod
    def referenced_labels(self) -> Set[str]:
        """Return a set of referenced labels."""

    @cached_property
    @abstractmethod
    def referenced_floors(self) -> Set[str]:
        """Return a set of referenced floors."""

    @cached_property
    @abstractmethod
    def referenced_areas(self) -> Set[str]:
        """Return a set of referenced areas."""

    @property
    @abstractmethod
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""

    @cached_property
    @abstractmethod
    def referenced_devices(self) -> Set[str]:
        """Return a set of referenced devices."""

    @cached_property
    @abstractmethod
    def referenced_entities(self) -> Set[str]:
        """Return a set of referenced entities."""

    @abstractmethod
    async def async_trigger(
        self,
        run_variables: Dict[str, Any],
        context: Optional[Context] = None,
        skip_condition: bool = False,
    ) -> Optional[ScriptRunResult]:
        """Trigger automation."""


class UnavailableAutomationEntity(BaseAutomationEntity):
    """A non-functional automation entity with its state set to unavailable."""

    _attr_should_poll = False
    _attr_available = False

    def __init__(
        self,
        automation_id: Optional[str],
        name: str,
        raw_config: Optional[ConfigType],
        validation_error: str,
        validation_status: ValidationStatus,
    ) -> None:
        """Initialize an automation entity."""
        self._attr_name = name
        self._attr_unique_id = automation_id
        self.raw_config = raw_config
        self._validation_error = validation_error
        self._validation_status = validation_status

    @cached_property
    def referenced_labels(self) -> Set[str]:
        """Return a set of referenced labels."""
        return set()

    @cached_property
    def referenced_floors(self) -> Set[str]:
        """Return a set of referenced floors."""
        return set()

    @cached_property
    def referenced_areas(self) -> Set[str]:
        """Return a set of referenced areas."""
        return set()

    @property
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""
        return None

    @cached_property
    def referenced_devices(self) -> Set[str]:
        """Return a set of referenced devices."""
        return set()

    @cached_property
    def referenced_entities(self) -> Set[str]:
        """Return a set of referenced entities."""
        return set()

    async def async_added_to_hass(self) -> None:
        """Create a repair issue to notify the user the automation has errors."""
        await super().async_added_to_hass()
        async_create_issue(
            self.hass,
            DOMAIN,
            f"{self.entity_id}_validation_{self._validation_status}",
            is_fixable=False,
            severity=IssueSeverity.ERROR,
            translation_key=f"validation_{self._validation_status}",
            translation_placeholders={
                "edit": f"/config/automation/edit/{self.unique_id}",
                "entity_id": self.entity_id,
                "error": self._validation_error,
                "name": self._attr_name or self.entity_id,
            },
        )

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        async_delete_issue(
            self.hass, DOMAIN, f"{self.entity_id}_validation_{self._validation_status}"
        )

    async def async_trigger(
        self,
        run_variables: Dict[str, Any],
        context: Optional[Context] = None,
        skip_condition: bool = False,
    ) -> None:
        """Trigger automation."""


class AutomationEntity(BaseAutomationEntity, RestoreEntity):
    """Entity to show status of entity."""

    _attr_should_poll = False

    def __init__(
        self,
        automation_id: Optional[str],
        name: str,
        trigger_config: List[ConfigType],
        cond_func: Optional[IfAction],
        action_script: Script,
        initial_state: Optional[bool],
        variables: Optional[ScriptVariables],
        trigger_variables: Optional[ScriptVariables],
        raw_config: Optional[ConfigType],
        blueprint_inputs: Optional[ConfigType],
        trace_config: ConfigType,
    ) -> None:
        """Initialize an automation entity."""
        self._attr_name = name
        self._trigger_config = trigger_config
        self._async_detach_triggers: Optional[CALLBACK_TYPE] = None
        self._cond_func = cond_func
        self.action_script = action_script
        self.action_script.change_listener = self.async_write_ha_state
        self._initial_state = initial_state
        self._is_enabled = False
        self._logger = LOGGER
        self._variables = variables
        self._trigger_variables = trigger_variables
        self.raw_config = raw_config
        self._blueprint_inputs = blueprint_inputs
        self._trace_config = trace_config
        self._attr_unique_id = automation_id

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the entity state attributes."""
        attrs = {
            ATTR_LAST_TRIGGERED: self.action_script.last_triggered,
            ATTR_MODE: self.action_script.script_mode,
            ATTR_CUR: self.action_script.runs,
        }
        if self.action_script.supports_max:
            attrs[ATTR_MAX] = self.action_script.max_runs
        return attrs

    @property
    def is_on(self) -> bool:
        """Return True if entity is on."""
        return self._async_detach_triggers is not None or self._is_enabled

    @property
    def referenced_labels(self) -> Set[str]:
        """Return a set of referenced labels."""
        return self.action_script.referenced_labels

    @property
    def referenced_floors(self) -> Set[str]:
        """Return a set of referenced floors."""
        return self.action_script.referenced_floors

    @cached_property
    def referenced_areas(self) -> Set[str]:
        """Return a set of referenced areas."""
        return self.action_script.referenced_areas

    @property
    def referenced