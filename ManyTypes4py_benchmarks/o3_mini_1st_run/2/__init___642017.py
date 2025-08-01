"""Allow to set up simple automation rules via the config file."""
from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import logging
from typing import Any, Optional, Protocol, cast, Union, List, Set, Dict, Tuple
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
from homeassistant.helpers.issue_registry import IssueSeverity, async_create_issue, async_delete_issue
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
from homeassistant.helpers.service import ReloadServiceHelper, async_register_admin_service
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

DATA_COMPONENT: HassKey = HassKey(DOMAIN)
ENTITY_ID_FORMAT: str = DOMAIN + '.{}'
CONF_SKIP_CONDITION: str = 'skip_condition'
CONF_STOP_ACTIONS: str = 'stop_actions'
DEFAULT_STOP_ACTIONS: bool = True
EVENT_AUTOMATION_RELOADED: str = 'automation_reloaded'
EVENT_AUTOMATION_TRIGGERED: str = 'automation_triggered'
ATTR_LAST_TRIGGERED: str = 'last_triggered'
ATTR_SOURCE: str = 'source'
ATTR_VARIABLES: str = 'variables'
SERVICE_TRIGGER: str = 'trigger'


class IfAction(Protocol):
    """Define the format of if_action."""

    config: Any

    def __call__(self, variables: Optional[Dict[str, Any]] = None) -> bool:
        ...


@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    """Return true if specified automation entity_id is on.

    Async friendly.
    """
    return hass.states.is_state(entity_id, STATE_ON)


def _automations_with_x(hass: HomeAssistant, referenced_id: str, property_name: str) -> List[str]:
    """Return all automations that reference the x."""
    if DATA_COMPONENT not in hass.data:
        return []
    return [
        automation_entity.entity_id
        for automation_entity in hass.data[DATA_COMPONENT].entities
        if referenced_id in getattr(automation_entity, property_name)
    ]


def _x_in_automation(hass: HomeAssistant, entity_id: str, property_name: str) -> List[str]:
    """Return all x in an automation."""
    if DATA_COMPONENT not in hass.data:
        return []
    if (automation_entity := hass.data[DATA_COMPONENT].get_entity(entity_id)) is None:
        return []
    return list(getattr(automation_entity, property_name))


@callback
def automations_with_entity(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all automations that reference the entity."""
    return _automations_with_x(hass, entity_id, 'referenced_entities')


@callback
def entities_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all entities in an automation."""
    return _x_in_automation(hass, entity_id, 'referenced_entities')


@callback
def automations_with_device(hass: HomeAssistant, device_id: str) -> List[str]:
    """Return all automations that reference the device."""
    return _automations_with_x(hass, device_id, 'referenced_devices')


@callback
def devices_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all devices in an automation."""
    return _x_in_automation(hass, entity_id, 'referenced_devices')


@callback
def automations_with_area(hass: HomeAssistant, area_id: str) -> List[str]:
    """Return all automations that reference the area."""
    return _automations_with_x(hass, area_id, 'referenced_areas')


@callback
def areas_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all areas in an automation."""
    return _x_in_automation(hass, entity_id, 'referenced_areas')


@callback
def automations_with_floor(hass: HomeAssistant, floor_id: str) -> List[str]:
    """Return all automations that reference the floor."""
    return _automations_with_x(hass, floor_id, 'referenced_floors')


@callback
def floors_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all floors in an automation."""
    return _x_in_automation(hass, entity_id, 'referenced_floors')


@callback
def automations_with_label(hass: HomeAssistant, label_id: str) -> List[str]:
    """Return all automations that reference the label."""
    return _automations_with_x(hass, label_id, 'referenced_labels')


@callback
def labels_in_automation(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all labels in an automation."""
    return _x_in_automation(hass, entity_id, 'referenced_labels')


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
    hass.data[DATA_COMPONENT] = component = EntityComponent[BaseAutomationEntity](LOGGER, DOMAIN, hass)
    async_get_blueprints(hass)
    await _async_process_config(hass, config, component)
    hass.async_create_task(async_get_blueprints(hass).async_populate(), eager_start=True)

    async def trigger_service_handler(entity: BaseAutomationEntity, service_call: ServiceCall) -> None:
        """Handle forced automation trigger, e.g. from frontend."""
        await entity.async_trigger(
            {**service_call.data[ATTR_VARIABLES], 'trigger': {'platform': None}},
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
    component.async_register_entity_service(SERVICE_TOGGLE, None, 'async_toggle')
    component.async_register_entity_service(SERVICE_TURN_ON, None, 'async_turn_on')
    component.async_register_entity_service(
        SERVICE_TURN_OFF,
        {vol.Optional(CONF_STOP_ACTIONS, default=DEFAULT_STOP_ACTIONS): cv.boolean},
        'async_turn_off',
    )

    async def reload_service_handler(service_call: ServiceCall) -> None:
        """Remove all automations and load new ones from config."""
        await async_get_blueprints(hass).async_reset_cache()
        if (conf := (await component.async_prepare_reload(skip_reset=True))) is None:
            return
        if (automation_id := service_call.data.get(CONF_ID)) is not None:
            await _async_process_single_config(hass, conf, component, automation_id)
        else:
            await _async_process_config(hass, conf, component)
        hass.bus.async_fire(EVENT_AUTOMATION_RELOADED, context=service_call.context)

    def reload_targets(service_call: ServiceCall) -> Set[str]:
        if (automation_id := service_call.data.get(CONF_ID)) is not None:
            return {automation_id}
        return {automation.unique_id for automation in component.entities if automation.unique_id is not None}

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
    _entity_component_unrecorded_attributes = frozenset((ATTR_LAST_TRIGGERED, ATTR_MODE, ATTR_CUR, ATTR_MAX, CONF_ID))

    @property
    def capability_attributes(self) -> Optional[Mapping[str, Any]]:
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
        self, run_variables: Dict[str, Any], context: Optional[Context] = None, skip_condition: bool = False
    ) -> Any:
        """Trigger automation."""


class UnavailableAutomationEntity(BaseAutomationEntity):
    """A non-functional automation entity with its state set to unavailable.

    This class is instantiated when an automation fails to validate.
    """
    _attr_should_poll = False
    _attr_available = False

    def __init__(
        self,
        automation_id: str,
        name: str,
        raw_config: Any,
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
            f'{self.entity_id}_validation_{self._validation_status}',
            is_fixable=False,
            severity=IssueSeverity.ERROR,
            translation_key=f'validation_{self._validation_status}',
            translation_placeholders={
                'edit': f'/config/automation/edit/{self.unique_id}',
                'entity_id': self.entity_id,
                'error': self._validation_error,
                'name': self._attr_name or self.entity_id,
            },
        )

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        async_delete_issue(self.hass, DOMAIN, f'{self.entity_id}_validation_{self._validation_status}')

    async def async_trigger(
        self, run_variables: Dict[str, Any], context: Optional[Context] = None, skip_condition: bool = False
    ) -> None:
        """Trigger automation."""
        # Unavailable automation does nothing.
        return None


class AutomationEntity(BaseAutomationEntity, RestoreEntity):
    """Entity to show status of entity."""
    _attr_should_poll = False

    def __init__(
        self,
        automation_id: str,
        name: str,
        trigger_config: Any,
        cond_func: Optional[IfAction],
        action_script: Script,
        initial_state: Optional[bool],
        variables: Any,
        trigger_variables: Any,
        raw_config: Any,
        blueprint_inputs: Any,
        trace_config: Any,
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
        attrs: Dict[str, Any] = {
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
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""
        if self._blueprint_inputs is None:
            return None
        return cast(str, self._blueprint_inputs[CONF_USE_BLUEPRINT][CONF_PATH])

    @cached_property
    def referenced_devices(self) -> Set[str]:
        """Return a set of referenced devices."""
        referenced: Set[str] = self.action_script.referenced_devices.copy()
        if self._cond_func is not None:
            for conf in self._cond_func.config:
                referenced |= condition.async_extract_devices(conf)
        for conf in self._trigger_config:
            referenced |= set(_trigger_extract_devices(conf))
        return referenced

    @cached_property
    def referenced_entities(self) -> Set[str]:
        """Return a set of referenced entities."""
        referenced: Set[str] = self.action_script.referenced_entities.copy()
        if self._cond_func is not None:
            for conf in self._cond_func.config:
                referenced |= condition.async_extract_entities(conf)
        for conf in self._trigger_config:
            for entity_id in _trigger_extract_entities(conf):
                referenced.add(entity_id)
        return referenced

    async def async_added_to_hass(self) -> None:
        """Startup with initial state or previous state."""
        await super().async_added_to_hass()
        self._logger = logging.getLogger(f'{__name__}.{split_entity_id(self.entity_id)[1]}')
        self.action_script.update_logger(self._logger)
        if (state := (await self.async_get_last_state())):
            enable_automation: bool = state.state == STATE_ON
            last_triggered = state.attributes.get('last_triggered')
            if last_triggered is not None:
                self.action_script.last_triggered = parse_datetime(last_triggered)
            self._logger.debug('Loaded automation %s with state %s from state storage last state %s', self.entity_id, enable_automation, state)
        else:
            enable_automation = DEFAULT_INITIAL_STATE
            self._logger.debug('Automation %s not in state storage, state %s from default is used', self.entity_id, enable_automation)
        if self._initial_state is not None:
            enable_automation = self._initial_state
            self._logger.debug('Automation %s initial state %s overridden from config initial_state', self.entity_id, enable_automation)
        if enable_automation:
            await self._async_enable()

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the entity on and update the state."""
        await self._async_enable()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the entity off."""
        if CONF_STOP_ACTIONS in kwargs:
            await self._async_disable(kwargs[CONF_STOP_ACTIONS])
        else:
            await self._async_disable()
        self.async_write_ha_state()

    async def async_trigger(
        self, run_variables: Dict[str, Any], context: Optional[Context] = None, skip_condition: bool = False
    ) -> Any:
        """Trigger automation.

        This method is a coroutine.
        """
        reason = ''
        alias = ''
        if 'trigger' in run_variables:
            if 'description' in run_variables['trigger']:
                reason = f' by {run_variables["trigger"]["description"]}'
            if 'alias' in run_variables['trigger']:
                alias = f" trigger '{run_variables['trigger']['alias']}'"
        self._logger.debug('Automation%s triggered%s', alias, reason)
        parent_id: Optional[str] = None if context is None else context.id
        trigger_context = Context(parent_id=parent_id)
        with trace_automation(
            self.hass,
            self.unique_id,
            self.raw_config,
            self._blueprint_inputs,
            trigger_context,
            self._trace_config,
        ) as automation_trace:
            this: Optional[Dict[str, Any]] = None
            if (state := self.hass.states.get(self.entity_id)):
                this = state.as_dict()
            variables: Dict[str, Any] = {'this': this, **(run_variables or {})}
            if self._variables:
                try:
                    variables = self._variables.async_render(self.hass, variables)
                except TemplateError as err:
                    self._logger.error('Error rendering variables: %s', err)
                    automation_trace.set_error(err)
                    return None
            automation_trace.set_trace(trace_get())
            trigger_description = variables.get('trigger', {}).get('description')
            automation_trace.set_trigger_description(trigger_description)
            if 'trigger' in variables and 'idx' in variables['trigger']:
                trigger_path = f'trigger/{variables["trigger"]["idx"]}'
            else:
                trigger_path = 'trigger'
            trace_element = TraceElement(variables, trigger_path)
            trace_append_element(trace_element)
            if not skip_condition and self._cond_func is not None and (not self._cond_func(variables)):
                self._logger.debug('Conditions not met, aborting automation. Condition summary: %s', trace_get(clear=False))
                script_execution_set('failed_conditions')
                return None
            self.async_set_context(trigger_context)
            event_data: Dict[str, Any] = {ATTR_NAME: self.name, ATTR_ENTITY_ID: self.entity_id}
            if 'trigger' in variables and 'description' in variables['trigger']:
                event_data[ATTR_SOURCE] = variables['trigger']['description']

            @callback
            def started_action() -> None:
                self.hass.bus.async_fire_internal(EVENT_AUTOMATION_TRIGGERED, event_data, context=trigger_context)

            script_stack_cv.set([])
            try:
                with trace_path('action'):
                    return await self.action_script.async_run(variables, trigger_context, started_action)
            except ServiceNotFound as err:
                async_create_issue(
                    self.hass,
                    DOMAIN,
                    f'{self.entity_id}_service_not_found_{err.domain}.{err.service}',
                    is_fixable=True,
                    is_persistent=True,
                    severity=IssueSeverity.ERROR,
                    translation_key='service_not_found',
                    translation_placeholders={
                        'service': f'{err.domain}.{err.service}',
                        'entity_id': self.entity_id,
                        'name': self._attr_name or self.entity_id,
                        'edit': f'/config/automation/edit/{self.unique_id}',
                    },
                )
                automation_trace.set_error(err)
            except (vol.Invalid, HomeAssistantError) as err:
                self._logger.error('Error while executing automation %s: %s', self.entity_id, err)
                automation_trace.set_error(err)
            except Exception as err:
                self._logger.exception('While executing automation %s', self.entity_id)
                automation_trace.set_error(err)
            return None

    async def async_will_remove_from_hass(self) -> None:
        """Remove listeners when removing automation from Home Assistant."""
        await super().async_will_remove_from_hass()
        await self._async_disable()

    async def _async_enable_automation(self, event: Event) -> None:
        """Start automation on startup."""
        if not self._is_enabled or self._async_detach_triggers is not None:
            return
        self._async_detach_triggers = await self._async_attach_triggers(True)
        self.async_write_ha_state()

    async def _async_enable(self) -> None:
        """Enable this automation entity.

        This method is not expected to write state to the
        state machine.
        """
        if self._is_enabled:
            return
        self._is_enabled = True
        if self.hass.state is not CoreState.not_running:
            self._async_detach_triggers = await self._async_attach_triggers(False)
            return
        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, self._async_enable_automation)

    async def _async_disable(self, stop_actions: bool = DEFAULT_STOP_ACTIONS) -> None:
        """Disable the automation entity.

        This method is not expected to write state to the
        state machine.
        """
        if not self._is_enabled and (not self.action_script.runs):
            return
        self._is_enabled = False
        if self._async_detach_triggers is not None:
            self._async_detach_triggers()
            self._async_detach_triggers = None
        if stop_actions:
            await self.action_script.async_stop()

    def _log_callback(self, level: int, msg: str, **kwargs: Any) -> None:
        """Log helper callback."""
        self._logger.log(level, '%s %s', msg, self.name, **kwargs)

    async def _async_trigger_if_enabled(
        self, run_variables: Dict[str, Any], context: Optional[Context] = None, skip_condition: bool = False
    ) -> Any:
        """Trigger automation if enabled.

        If the trigger starts but has a delay, the automation will be triggered
        when the delay has passed so we need to make sure its still enabled before
        executing the action.
        """
        if not self._is_enabled:
            return None
        return await self.async_trigger(run_variables, context, skip_condition)

    async def _async_attach_triggers(self, home_assistant_start: bool) -> Optional[CALLBACK_TYPE]:
        """Set up the triggers."""
        this: Optional[Dict[str, Any]] = None
        if (state := self.hass.states.get(self.entity_id)):
            this = state.as_dict()
        variables: Dict[str, Any] = {'this': this}
        if self._trigger_variables:
            try:
                variables = self._trigger_variables.async_render(self.hass, variables, limited=True)
            except TemplateError as err:
                self._logger.error('Error rendering trigger variables: %s', err)
                return None
        return await async_initialize_triggers(
            self.hass,
            self._trigger_config,
            self._async_trigger_if_enabled,
            DOMAIN,
            str(self.name),
            self._log_callback,
            home_assistant_start,
            variables,
        )


@dataclass(slots=True)
class AutomationEntityConfig:
    """Container for prepared automation entity configuration."""
    config_block: Mapping[str, Any]
    list_no: int
    raw_blueprint_inputs: Any
    raw_config: Any
    validation_error: Any
    validation_status: ValidationStatus


async def _prepare_automation_config(
    hass: HomeAssistant, config: ConfigType, wanted_automation_id: Optional[str]
) -> List[AutomationEntityConfig]:
    """Parse configuration and prepare automation entity configuration."""
    automation_configs: List[AutomationEntityConfig] = []
    conf: Any = config[DOMAIN]
    for list_no, config_block in enumerate(conf):
        automation_id: Optional[str] = config_block.get(CONF_ID)
        if wanted_automation_id is not None and automation_id != wanted_automation_id:
            continue
        raw_config: Any = cast(AutomationConfig, config_block).raw_config
        raw_blueprint_inputs: Any = cast(AutomationConfig, config_block).raw_blueprint_inputs
        validation_error: Any = cast(AutomationConfig, config_block).validation_error
        validation_status: ValidationStatus = cast(AutomationConfig, config_block).validation_status
        automation_configs.append(
            AutomationEntityConfig(config_block, list_no, raw_blueprint_inputs, raw_config, validation_error, validation_status)
        )
    return automation_configs


def _automation_name(automation_config: AutomationEntityConfig) -> str:
    """Return the configured name of an automation."""
    config_block: Mapping[str, Any] = automation_config.config_block
    list_no: int = automation_config.list_no
    return config_block.get(CONF_ALIAS) or f'{DOMAIN} {list_no}'


async def _create_automation_entities(
    hass: HomeAssistant, automation_configs: List[AutomationEntityConfig]
) -> List[BaseAutomationEntity]:
    """Create automation entities from prepared configuration."""
    entities: List[BaseAutomationEntity] = []
    for automation_config in automation_configs:
        config_block: Mapping[str, Any] = automation_config.config_block
        automation_id: Optional[str] = config_block.get(CONF_ID)
        name: str = _automation_name(automation_config)
        if automation_config.validation_status != ValidationStatus.OK:
            entities.append(
                UnavailableAutomationEntity(
                    automation_id, name, automation_config.raw_config, cast(str, automation_config.validation_error), automation_config.validation_status
                )
            )
            continue
        initial_state: Optional[bool] = config_block.get(CONF_INITIAL_STATE)
        action_script: Script = Script(
            hass,
            config_block[CONF_ACTIONS],
            name,
            DOMAIN,
            running_description='automation actions',
            script_mode=config_block[CONF_MODE],
            max_runs=config_block[CONF_MAX],
            max_exceeded=config_block[CONF_MAX_EXCEEDED],
            logger=LOGGER,
        )
        if CONF_CONDITIONS in config_block:
            cond_func = await _async_process_if(hass, name, config_block)
            if cond_func is None:
                continue
        else:
            cond_func = None
        variables: Any = None
        if CONF_TRIGGER_VARIABLES in config_block and CONF_VARIABLES in config_block:
            variables = ScriptVariables(dict(config_block[CONF_TRIGGER_VARIABLES].as_dict()))
            variables.variables.update(config_block[CONF_VARIABLES].as_dict())
        elif CONF_TRIGGER_VARIABLES in config_block:
            variables = config_block[CONF_TRIGGER_VARIABLES]
        elif CONF_VARIABLES in config_block:
            variables = config_block[CONF_VARIABLES]
        entity = AutomationEntity(
            automation_id,
            name,
            config_block[CONF_TRIGGERS],
            cond_func,
            action_script,
            initial_state,
            variables,
            config_block.get(CONF_TRIGGER_VARIABLES),
            automation_config.raw_config,
            automation_config.raw_blueprint_inputs,
            config_block[CONF_TRACE],
        )
        entities.append(entity)
    return entities


async def _async_process_config(
    hass: HomeAssistant, config: ConfigType, component: EntityComponent[BaseAutomationEntity]
) -> None:
    """Process config and add automations."""

    def automation_matches_config(automation: BaseAutomationEntity, config_item: AutomationEntityConfig) -> bool:
        name: str = _automation_name(config_item)
        return automation.name == name and automation.raw_config == config_item.raw_config

    def find_matches(
        automations: List[BaseAutomationEntity], automation_configs: List[AutomationEntityConfig]
    ) -> Tuple[Set[int], Set[int]]:
        """Find matches between a list of automation entities and a list of configurations.

        An automation or configuration is only allowed to match at most once to handle
        the case of multiple automations with identical configuration.

        Returns a tuple of sets of indices: ({automation_matches}, {config_matches})
        """
        automation_matches: Set[int] = set()
        config_matches: Set[int] = set()
        automation_configs_with_id: Dict[str, Tuple[int, AutomationEntityConfig]] = {}
        automation_configs_without_id: List[Tuple[int, AutomationEntityConfig]] = []
        for config_idx, automation_config in enumerate(automation_configs):
            if (automation_id := automation_config.config_block.get(CONF_ID)) is not None:
                automation_configs_with_id[automation_id] = (config_idx, automation_config)
                continue
            automation_configs_without_id.append((config_idx, automation_config))
        for automation_idx, automation in enumerate(automations):
            if automation.unique_id:
                if automation.unique_id not in automation_configs_with_id:
                    continue
                config_idx, automation_config = automation_configs_with_id.pop(automation.unique_id)
                if automation_matches_config(automation, automation_config):
                    automation_matches.add(automation_idx)
                    config_matches.add(config_idx)
                continue
            for config_idx, automation_config in automation_configs_without_id:
                if config_idx in config_matches:
                    continue
                if automation_matches_config(automation, automation_config):
                    automation_matches.add(automation_idx)
                    config_matches.add(config_idx)
                    break
        return automation_matches, config_matches

    automation_configs: List[AutomationEntityConfig] = await _prepare_automation_config(hass, config, None)
    automations: List[BaseAutomationEntity] = list(component.entities)
    automation_matches, config_matches = find_matches(automations, automation_configs)
    tasks = [automation.async_remove() for idx, automation in enumerate(automations) if idx not in automation_matches]
    await asyncio.gather(*tasks)
    updated_automation_configs = [config_item for idx, config_item in enumerate(automation_configs) if idx not in config_matches]
    entities: List[BaseAutomationEntity] = await _create_automation_entities(hass, updated_automation_configs)
    await component.async_add_entities(entities)


def _automation_matches_config(
    automation: Optional[BaseAutomationEntity], config_item: Optional[AutomationEntityConfig]
) -> bool:
    """Return False if an automation's config has been changed."""
    if not automation:
        return False
    if not config_item:
        return False
    name: str = _automation_name(config_item)
    return automation.name == name and automation.raw_config == config_item.raw_config


async def _async_process_single_config(
    hass: HomeAssistant, config: ConfigType, component: EntityComponent[BaseAutomationEntity], automation_id: str
) -> None:
    """Process config and add a single automation."""
    automation_configs: List[AutomationEntityConfig] = await _prepare_automation_config(hass, config, automation_id)
    automation: Optional[BaseAutomationEntity] = next((x for x in component.entities if x.unique_id == automation_id), None)
    automation_config: Optional[AutomationEntityConfig] = automation_configs[0] if automation_configs else None
    if _automation_matches_config(automation, automation_config):
        return
    if automation:
        await automation.async_remove()
    entities: List[BaseAutomationEntity] = await _create_automation_entities(hass, automation_configs)
    await component.async_add_entities(entities)


async def _async_process_if(hass: HomeAssistant, name: str, config: Mapping[str, Any]) -> Optional[IfAction]:
    """Process if checks."""
    if_configs: Any = config[CONF_CONDITIONS]
    try:
        if_action: IfAction = await condition.async_conditions_from_config(hass, if_configs, LOGGER, name)
    except HomeAssistantError as ex:
        LOGGER.warning('Invalid condition: %s', ex)
        return None
    result: IfAction = if_action
    result.config = if_configs
    return result


@callback
def _trigger_extract_devices(trigger_conf: Mapping[str, Any]) -> List[str]:
    """Extract devices from a trigger config."""
    if trigger_conf[CONF_PLATFORM] == 'device':
        return [trigger_conf[CONF_DEVICE_ID]]
    if trigger_conf[CONF_PLATFORM] == 'event' and CONF_EVENT_DATA in trigger_conf and (CONF_DEVICE_ID in trigger_conf[CONF_EVENT_DATA]) and isinstance(trigger_conf[CONF_EVENT_DATA][CONF_DEVICE_ID], str):
        return [trigger_conf[CONF_EVENT_DATA][CONF_DEVICE_ID]]
    if trigger_conf[CONF_PLATFORM] == 'tag' and CONF_DEVICE_ID in trigger_conf:
        return trigger_conf[CONF_DEVICE_ID]
    return []


@callback
def _trigger_extract_entities(trigger_conf: Mapping[str, Any]) -> List[str]:
    """Extract entities from a trigger config."""
    if trigger_conf[CONF_PLATFORM] in ('state', 'numeric_state'):
        return trigger_conf[CONF_ENTITY_ID]
    if trigger_conf[CONF_PLATFORM] == 'calendar':
        return [trigger_conf[CONF_ENTITY_ID]]
    if trigger_conf[CONF_PLATFORM] == 'zone':
        return trigger_conf[CONF_ENTITY_ID] + [trigger_conf[CONF_ZONE]]
    if trigger_conf[CONF_PLATFORM] == 'geo_location':
        return [trigger_conf[CONF_ZONE]]
    if trigger_conf[CONF_PLATFORM] == 'sun':
        return ['sun.sun']
    if (
        trigger_conf[CONF_PLATFORM] == 'event'
        and CONF_EVENT_DATA in trigger_conf
        and (CONF_ENTITY_ID in trigger_conf[CONF_EVENT_DATA])
        and isinstance(trigger_conf[CONF_EVENT_DATA][CONF_ENTITY_ID], str)
        and valid_entity_id(trigger_conf[CONF_EVENT_DATA][CONF_ENTITY_ID])
    ):
        return [trigger_conf[CONF_EVENT_DATA][CONF_ENTITY_ID]]
    return []


@websocket_api.websocket_command({'type': 'automation/config', 'entity_id': str})
def websocket_config(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Mapping[str, Any]) -> None:
    """Get automation config."""
    automation: Optional[BaseAutomationEntity] = hass.data[DATA_COMPONENT].get_entity(msg['entity_id'])
    if automation is None:
        connection.send_error(msg['id'], websocket_api.ERR_NOT_FOUND, 'Entity not found')
        return
    connection.send_result(msg['id'], {'config': automation.raw_config})
