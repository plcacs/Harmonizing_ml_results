#!/usr/bin/env python3
"""Support for scripts."""
from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import logging
from typing import Any, cast, Optional, List, Dict, Coroutine
from propcache.api import cached_property
import voluptuous as vol
from homeassistant.components import websocket_api
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_MODE,
    ATTR_NAME,
    CONF_ALIAS,
    CONF_DESCRIPTION,
    CONF_ICON,
    CONF_MODE,
    CONF_NAME,
    CONF_PATH,
    CONF_SEQUENCE,
    CONF_VARIABLES,
    SERVICE_RELOAD,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
)
from homeassistant.core import Context, HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.config_validation import make_entity_service_schema
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
from homeassistant.helpers.service import async_set_service_schema
from homeassistant.helpers.trace import trace_get, trace_path
from homeassistant.helpers.typing import ConfigType
from homeassistant.loader import bind_hass
from homeassistant.util.async_ import create_eager_task
from homeassistant.util.dt import parse_datetime
from .config import ScriptConfig, ValidationStatus
from .const import (
    ATTR_LAST_ACTION,
    ATTR_LAST_TRIGGERED,
    ATTR_VARIABLES,
    CONF_FIELDS,
    CONF_TRACE,
    DOMAIN,
    ENTITY_ID_FORMAT,
    EVENT_SCRIPT_STARTED,
    LOGGER,
)
from .helpers import async_get_blueprints
from .trace import trace_script

SCRIPT_SERVICE_SCHEMA: vol.Schema = vol.Schema(dict)
SCRIPT_TURN_ONOFF_SCHEMA: vol.Schema = make_entity_service_schema({vol.Optional(ATTR_VARIABLES): {str: cv.match_all}})
RELOAD_SERVICE_SCHEMA: vol.Schema = vol.Schema({})


@bind_hass
def is_on(hass: HomeAssistant, entity_id: str) -> bool:
    """Return if the script is on based on the statemachine."""
    return hass.states.is_state(entity_id, STATE_ON)


def _scripts_with_x(hass: HomeAssistant, referenced_id: str, property_name: str) -> List[str]:
    """Return all scripts that reference the x."""
    if DOMAIN not in hass.data:
        return []
    component: EntityComponent[BaseScriptEntity] = hass.data[DOMAIN]
    return [script_entity.entity_id for script_entity in component.entities if referenced_id in getattr(script_entity, property_name)]


def _x_in_script(hass: HomeAssistant, entity_id: str, property_name: str) -> List[str]:
    """Return all x in a script."""
    if DOMAIN not in hass.data:
        return []
    component: EntityComponent[BaseScriptEntity] = hass.data[DOMAIN]
    script_entity = component.get_entity(entity_id)
    if script_entity is None:
        return []
    return list(getattr(script_entity, property_name))


@callback
def scripts_with_entity(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all scripts that reference the entity."""
    return _scripts_with_x(hass, entity_id, 'referenced_entities')


@callback
def entities_in_script(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all entities in script."""
    return _x_in_script(hass, entity_id, 'referenced_entities')


@callback
def scripts_with_device(hass: HomeAssistant, device_id: str) -> List[str]:
    """Return all scripts that reference the device."""
    return _scripts_with_x(hass, device_id, 'referenced_devices')


@callback
def devices_in_script(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all devices in script."""
    return _x_in_script(hass, entity_id, 'referenced_devices')


@callback
def scripts_with_area(hass: HomeAssistant, area_id: str) -> List[str]:
    """Return all scripts that reference the area."""
    return _scripts_with_x(hass, area_id, 'referenced_areas')


@callback
def areas_in_script(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all areas in a script."""
    return _x_in_script(hass, entity_id, 'referenced_areas')


@callback
def scripts_with_floor(hass: HomeAssistant, floor_id: str) -> List[str]:
    """Return all scripts that reference the floor."""
    return _scripts_with_x(hass, floor_id, 'referenced_floors')


@callback
def floors_in_script(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all floors in a script."""
    return _x_in_script(hass, entity_id, 'referenced_floors')


@callback
def scripts_with_label(hass: HomeAssistant, label_id: str) -> List[str]:
    """Return all scripts that reference the label."""
    return _scripts_with_x(hass, label_id, 'referenced_labels')


@callback
def labels_in_script(hass: HomeAssistant, entity_id: str) -> List[str]:
    """Return all labels in a script."""
    return _x_in_script(hass, entity_id, 'referenced_labels')


@callback
def scripts_with_blueprint(hass: HomeAssistant, blueprint_path: str) -> List[str]:
    """Return all scripts that reference the blueprint."""
    if DOMAIN not in hass.data:
        return []
    component: EntityComponent[BaseScriptEntity] = hass.data[DOMAIN]
    return [script_entity.entity_id for script_entity in component.entities if script_entity.referenced_blueprint == blueprint_path]


@callback
def blueprint_in_script(hass: HomeAssistant, entity_id: str) -> Optional[str]:
    """Return the blueprint the script is based on or None."""
    if DOMAIN not in hass.data:
        return None
    component: EntityComponent[BaseScriptEntity] = hass.data[DOMAIN]
    script_entity = component.get_entity(entity_id)
    if script_entity is None:
        return None
    return script_entity.referenced_blueprint


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Load the scripts from the configuration."""
    hass.data[DOMAIN] = component = EntityComponent[BaseScriptEntity](LOGGER, DOMAIN, hass)
    async_get_blueprints(hass)
    await _async_process_config(hass, config, component)
    hass.async_create_task(async_get_blueprints(hass).async_populate(), eager_start=True)

    async def reload_service(service: ServiceCall) -> None:
        """Call a service to reload scripts."""
        await async_get_blueprints(hass).async_reset_cache()
        conf = await component.async_prepare_reload(skip_reset=True)
        if conf is None:
            return
        await _async_process_config(hass, conf, component)

    async def turn_on_service(service: ServiceCall) -> None:
        """Call a service to turn script on."""
        variables: Optional[Dict[str, Any]] = service.data.get(ATTR_VARIABLES)
        script_entities = await component.async_extract_from_service(service)
        for script_entity in script_entities:
            await script_entity.async_turn_on(variables=variables, context=service.context, wait=False)

    async def turn_off_service(service: ServiceCall) -> None:
        """Cancel a script."""
        script_entities = await component.async_extract_from_service(service)
        if not script_entities:
            return
        await asyncio.wait([create_eager_task(script_entity.async_turn_off()) for script_entity in script_entities])

    async def toggle_service(service: ServiceCall) -> None:
        """Toggle a script."""
        script_entities = await component.async_extract_from_service(service)
        for script_entity in script_entities:
            await script_entity.async_toggle(context=service.context, wait=False)

    hass.services.async_register(DOMAIN, SERVICE_RELOAD, reload_service, schema=RELOAD_SERVICE_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_TURN_ON, turn_on_service, schema=SCRIPT_TURN_ONOFF_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_TURN_OFF, turn_off_service, schema=SCRIPT_TURN_ONOFF_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_TOGGLE, toggle_service, schema=SCRIPT_TURN_ONOFF_SCHEMA)
    websocket_api.async_register_command(hass, websocket_config)
    return True


@dataclass(slots=True)
class ScriptEntityConfig:
    """Container for prepared script entity configuration."""
    config_block: ScriptConfig
    key: str
    raw_blueprint_inputs: Optional[Dict[str, Any]]
    raw_config: Dict[str, Any]
    validation_error: Optional[str]
    validation_status: ValidationStatus


async def _prepare_script_config(hass: HomeAssistant, config: ConfigType) -> List[ScriptEntityConfig]:
    """Parse configuration and prepare script entity configuration."""
    script_configs: List[ScriptEntityConfig] = []
    conf: Dict[str, Any] = config[DOMAIN]
    for key, config_block in conf.items():
        raw_config: Dict[str, Any] = cast(ScriptConfig, config_block).raw_config
        raw_blueprint_inputs: Optional[Dict[str, Any]] = cast(ScriptConfig, config_block).raw_blueprint_inputs
        validation_error: Optional[str] = cast(ScriptConfig, config_block).validation_error
        validation_status: ValidationStatus = cast(ScriptConfig, config_block).validation_status
        script_configs.append(
            ScriptEntityConfig(config_block, key, raw_blueprint_inputs, raw_config, validation_error, validation_status)
        )
    return script_configs


async def _create_script_entities(hass: HomeAssistant, script_configs: List[ScriptEntityConfig]) -> List[BaseScriptEntity]:
    """Create script entities from prepared configuration."""
    entities: List[BaseScriptEntity] = []
    for script_config in script_configs:
        if script_config.validation_status != ValidationStatus.OK:
            entities.append(
                UnavailableScriptEntity(
                    script_config.key,
                    script_config.raw_config,
                    cast(str, script_config.validation_error),
                    script_config.validation_status,
                )
            )
            continue
        entity = ScriptEntity(hass, script_config.key, script_config.config_block, script_config.raw_config, script_config.raw_blueprint_inputs)
        entities.append(entity)
    return entities


async def _async_process_config(hass: HomeAssistant, config: ConfigType, component: EntityComponent[BaseScriptEntity]) -> None:
    """Process script configuration."""
    def script_matches_config(script: BaseScriptEntity, config: ScriptEntityConfig) -> bool:
        return script.unique_id == config.key and getattr(script, "raw_config", None) == config.raw_config

    def find_matches(
        scripts: List[BaseScriptEntity], script_configs: List[ScriptEntityConfig]
    ) -> tuple[set[int], set[int]]:
        """Find matches between a list of script entities and a list of configurations.

        A script or configuration is only allowed to match at most once to handle
        the case of multiple scripts with identical configuration.

        Returns a tuple of sets of indices: ({script_matches}, {config_matches})
        """
        script_matches: set[int] = set()
        config_matches: set[int] = set()
        for script_idx, script in enumerate(scripts):
            for config_idx, script_config in enumerate(script_configs):
                if config_idx in config_matches:
                    continue
                if script_matches_config(script, script_config):
                    script_matches.add(script_idx)
                    config_matches.add(config_idx)
                    break
        return (script_matches, config_matches)

    script_configs = await _prepare_script_config(hass, config)
    scripts: List[BaseScriptEntity] = list(component.entities)
    script_matches, config_matches = find_matches(scripts, script_configs)
    tasks = [script.async_remove() for idx, script in enumerate(scripts) if idx not in script_matches]
    await asyncio.gather(*tasks)
    updated_script_configs = [config for idx, config in enumerate(script_configs) if idx not in config_matches]
    entities = await _create_script_entities(hass, updated_script_configs)
    await component.async_add_entities(entities)


class BaseScriptEntity(ToggleEntity, ABC):
    """Base class for script entities."""
    _entity_component_unrecorded_attributes = frozenset({ATTR_LAST_TRIGGERED, ATTR_MODE, ATTR_CUR, ATTR_MAX, ATTR_LAST_ACTION})

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
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""

    @cached_property
    @abstractmethod
    def referenced_devices(self) -> set[str]:
        """Return a set of referenced devices."""

    @cached_property
    @abstractmethod
    def referenced_entities(self) -> set[str]:
        """Return a set of referenced entities."""


class UnavailableScriptEntity(BaseScriptEntity):
    """A non-functional script entity with its state set to unavailable.

    This class is instantiated when an script fails to validate.
    """
    _attr_should_poll = False
    _attr_available = False

    def __init__(self, key: str, raw_config: Dict[str, Any], validation_error: str, validation_status: ValidationStatus) -> None:
        """Initialize a script entity."""
        self._attr_name = raw_config.get(CONF_ALIAS, key) if raw_config else key
        self._attr_unique_id = key
        self.raw_config = raw_config
        self._validation_error = validation_error
        self._validation_status = validation_status

    @cached_property
    def referenced_labels(self) -> set[str]:
        """Return a set of referenced labels."""
        return set()

    @cached_property
    def referenced_floors(self) -> set[str]:
        """Return a set of referenced floors."""
        return set()

    @cached_property
    def referenced_areas(self) -> set[str]:
        """Return a set of referenced areas."""
        return set()

    @property
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""
        return None

    @cached_property
    def referenced_devices(self) -> set[str]:
        """Return a set of referenced devices."""
        return set()

    @cached_property
    def referenced_entities(self) -> set[str]:
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
                'edit': f'/config/script/edit/{self.unique_id}',
                'entity_id': self.entity_id,
                'error': self._validation_error,
                'name': self._attr_name or self.entity_id,
            },
        )

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity will be removed from hass."""
        await super().async_will_remove_from_hass()
        async_delete_issue(self.hass, DOMAIN, f'{self.entity_id}_validation_{self._validation_status}')


class ScriptEntity(BaseScriptEntity, RestoreEntity):
    """Representation of a script entity."""
    icon: Optional[str] = None
    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        key: str,
        cfg: Dict[str, Any],
        raw_config: Dict[str, Any],
        blueprint_inputs: Optional[Dict[str, Any]],
    ) -> None:
        """Initialize the script."""
        self.icon = cfg.get(CONF_ICON)
        self.description: str = cfg[CONF_DESCRIPTION]
        self.fields: Any = cfg[CONF_FIELDS]
        self._attr_unique_id = key
        self.entity_id = ENTITY_ID_FORMAT.format(key)
        self.script: Script = Script(
            hass,
            cfg[CONF_SEQUENCE],
            cfg.get(CONF_ALIAS, key),
            DOMAIN,
            running_description='script sequence',
            change_listener=self.async_change_listener,
            script_mode=cfg[CONF_MODE],
            max_runs=cfg[CONF_MAX],
            max_exceeded=cfg[CONF_MAX_EXCEEDED],
            logger=logging.getLogger(f'{__name__}.{key}'),
            variables=cfg.get(CONF_VARIABLES),
        )
        self._changed: asyncio.Event = asyncio.Event()
        self.raw_config = raw_config
        self._trace_config = cfg[CONF_TRACE]
        self._blueprint_inputs = blueprint_inputs
        self._attr_name = self.script.name

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        script = self.script
        attrs: Dict[str, Any] = {ATTR_LAST_TRIGGERED: script.last_triggered, ATTR_MODE: script.script_mode, ATTR_CUR: script.runs}
        if script.supports_max:
            attrs[ATTR_MAX] = script.max_runs
        if script.last_action:
            attrs[ATTR_LAST_ACTION] = script.last_action
        return attrs

    @property
    def is_on(self) -> bool:
        """Return true if script is on."""
        return self.script.is_running

    @cached_property
    def referenced_labels(self) -> set[str]:
        """Return a set of referenced labels."""
        return self.script.referenced_labels

    @cached_property
    def referenced_floors(self) -> set[str]:
        """Return a set of referenced floors."""
        return self.script.referenced_floors

    @cached_property
    def referenced_areas(self) -> set[str]:
        """Return a set of referenced areas."""
        return self.script.referenced_areas

    @property
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""
        if self._blueprint_inputs is None:
            return None
        path: str = self._blueprint_inputs[CONF_USE_BLUEPRINT][CONF_PATH]
        return path

    @cached_property
    def referenced_devices(self) -> set[str]:
        """Return a set of referenced devices."""
        return self.script.referenced_devices

    @cached_property
    def referenced_entities(self) -> set[str]:
        """Return a set of referenced entities."""
        return self.script.referenced_entities

    @callback
    def async_change_listener(self) -> None:
        """Update state."""
        self.async_write_ha_state()
        self._changed.set()

    async def async_turn_on(self, **kwargs: Any) -> Optional[ServiceResponse]:
        """Run the script.

        Depending on the script's run mode, this may do nothing, restart the script or
        fire an additional parallel run.
        """
        variables: Optional[Dict[str, Any]] = kwargs.get('variables')
        context: Context = kwargs['context']
        wait: bool = kwargs.get('wait', True)
        return await self._async_start_run(variables, context, wait)

    async def _async_start_run(
        self, variables: Optional[Dict[str, Any]], context: Context, wait: bool
    ) -> Optional[ServiceResponse]:
        """Start the run of a script."""
        self.async_set_context(context)
        self.hass.bus.async_fire(EVENT_SCRIPT_STARTED, {ATTR_NAME: self.script.name, ATTR_ENTITY_ID: self.entity_id}, context=context)
        coro: Coroutine[Any, Any, ScriptRunResult] = self._async_run(variables, context)
        if wait:
            if (script_stack := script_stack_cv.get()):
                script_stack_cv.set(script_stack.copy())
            script_result = await coro
            return script_result.service_response if script_result else None
        script_stack_cv.set([])
        self._changed.clear()
        self.hass.async_create_task(coro, eager_start=True)
        await self._changed.wait()
        return None

    async def _async_run(self, variables: Optional[Dict[str, Any]], context: Context) -> ScriptRunResult:
        with trace_script(self.hass, self._attr_unique_id, self.raw_config, self._blueprint_inputs, context, self._trace_config) as script_trace:
            script_trace.set_trace(trace_get())
            with trace_path('sequence'):
                this: Optional[Dict[str, Any]] = None
                if (state := self.hass.states.get(self.entity_id)):
                    this = state.as_dict()
                script_vars: Dict[str, Any] = {'this': this, **(variables or {})}
                return await self.script.async_run(script_vars, context)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Stop running the script.

        If multiple runs are in progress, all will be stopped.
        """
        await self.script.async_stop()

    async def _service_handler(self, service: ServiceCall) -> Optional[Dict[str, Any]]:
        """Execute a service call to script.<script name>."""
        response: Optional[ServiceResponse] = await self._async_start_run(variables=service.data, context=service.context, wait=True)
        if service.return_response:
            return response or {}
        return None

    async def async_added_to_hass(self) -> None:
        """Restore last triggered on startup and register service."""
        if TYPE_CHECKING:
            assert self.unique_id is not None
            assert self.registry_entry is not None
        unique_id: str = self.unique_id
        hass: HomeAssistant = self.hass
        hass.services.async_register(DOMAIN, unique_id, self._service_handler, schema=SCRIPT_SERVICE_SCHEMA, supports_response=SupportsResponse.OPTIONAL)
        service_desc: Dict[str, Any] = {CONF_NAME: self.registry_entry.name or self.name, CONF_DESCRIPTION: self.description, CONF_FIELDS: self.fields}
        async_set_service_schema(hass, DOMAIN, unique_id, service_desc)
        if (state := (await self.async_get_last_state())) and (last_triggered := state.attributes.get('last_triggered')):
            self.script.last_triggered = parse_datetime(last_triggered)

    async def async_will_remove_from_hass(self) -> None:
        """Stop script and remove service when it will be removed from HA."""
        await self.script.async_stop()
        self.hass.services.async_remove(DOMAIN, self._attr_unique_id)


@websocket_api.websocket_command({'type': 'script/config', 'entity_id': str})
def websocket_config(hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: Dict[str, Any]) -> None:
    """Get script config."""
    component: EntityComponent[BaseScriptEntity] = hass.data[DOMAIN]
    script = component.get_entity(msg['entity_id'])
    if script is None:
        connection.send_error(msg['id'], websocket_api.ERR_NOT_FOUND, 'Entity not found')
        return
    connection.send_result(msg['id'], {'config': script.raw_config})