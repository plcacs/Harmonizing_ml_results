from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Container, Coroutine, Mapping
import copy
from dataclasses import dataclass
import types
from typing import Any, cast
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import HomeAssistant, callback, split_entity_id
from homeassistant.data_entry_flow import UnknownHandler
from . import entity_registry as er, selector
from .typing import UNDEFINED, UndefinedType

class SchemaFlowError(Exception):
    """Validation failed."""

@dataclass
class SchemaFlowStep:
    """Define a config or options flow step."""

@dataclass(slots=True)
class SchemaFlowFormStep(SchemaFlowStep):
    """Define a config or options flow form step."""
    schema: vol.Schema = None
    validate_user_input: Callable[[SchemaCommonFlowHandler, Any], Any] = None
    next_step: Callable[[Any], Any] = None
    suggested_values: Any = UNDEFINED
    preview: Any = None

@dataclass(slots=True)
class SchemaFlowMenuStep(SchemaFlowStep):
    """Define a config or options flow menu step."""

class SchemaCommonFlowHandler:
    """Handle a schema based config or options flow."""

    def __init__(self, handler, flow, options):
        """Initialize a common handler."""
        self._flow = flow
        self._handler = handler
        self._options = options if options is not None else {}
        self._flow_state = {}

    @property
    def parent_handler(self) -> Any:
        """Return parent handler."""
        return self._handler

    @property
    def options(self) -> Any:
        """Return the options linked to the current flow handler."""
        return self._options

    @property
    def flow_state(self) -> Any:
        """Return the flow state, used to store temporary data."""
        return self._flow_state

    async def async_step(self, step_id: str, user_input: Any = None) -> Any:
        """Handle a step."""
        if isinstance(self._flow[step_id], SchemaFlowFormStep):
            return await self._async_form_step(step_id, user_input)
        return await self._async_menu_step(step_id, user_input)

    async def _get_schema(self, form_step: SchemaFlowFormStep) -> Any:
        if form_step.schema is None:
            return None
        if isinstance(form_step.schema, vol.Schema):
            return form_step.schema
        return await form_step.schema(self)

    async def _async_form_step(self, step_id: str, user_input: Any = None) -> Any:
        """Handle a form step."""
        form_step = cast(SchemaFlowFormStep, self._flow[step_id])
        if user_input is not None and (data_schema := (await self._get_schema(form_step))) and data_schema.schema and (not self._handler.show_advanced_options):
            for key in data_schema.schema:
                if isinstance(key, (vol.Optional, vol.Required)):
                    if key.description and key.description.get('advanced') and (key.default is not vol.UNDEFINED) and (key not in self._options):
                        user_input[str(key.schema)] = cast(Callable[[], Any], key.default)()
        if user_input is not None and form_step.validate_user_input is not None:
            try:
                user_input = await form_step.validate_user_input(self, user_input)
            except SchemaFlowError as exc:
                return await self._show_next_step(step_id, exc, user_input)
        if user_input is not None:
            self._update_and_remove_omitted_optional_keys(self._options, user_input, data_schema)
        if user_input is not None or form_step.schema is None:
            return await self._show_next_step_or_create_entry(form_step)
        return await self._show_next_step(step_id)

    def _update_and_remove_omitted_optional_keys(self, values: Any, user_input: Any, data_schema: Any) -> None:
        values.update(user_input)
        if data_schema and data_schema.schema:
            for key in data_schema.schema:
                if isinstance(key, vol.Optional) and key not in user_input and (not (key.description and key.description.get('advanced') and (not self._handler.show_advanced_options))):
                    values.pop(key.schema, None)

    async def _show_next_step_or_create_entry(self, form_step: SchemaFlowFormStep) -> Any:
        if callable(form_step.next_step):
            next_step_id_or_end_flow = await form_step.next_step(self._options)
        else:
            next_step_id_or_end_flow = form_step.next_step
        if next_step_id_or_end_flow is None:
            return self._handler.async_create_entry(data=self._options)
        return await self._show_next_step(next_step_id_or_end_flow)

    async def _show_next_step(self, next_step_id: str, error: SchemaFlowError = None, user_input: Any = None) -> Any:
        """Show form for next step."""
        if isinstance(self._flow[next_step_id], SchemaFlowMenuStep):
            menu_step = cast(SchemaFlowMenuStep, self._flow[next_step_id])
            return self._handler.async_show_menu(step_id=next_step_id, menu_options=menu_step.options)
        form_step = cast(SchemaFlowFormStep, self._flow[next_step_id])
        if (data_schema := (await self._get_schema(form_step))) is None:
            return await self._show_next_step_or_create_entry(form_step)
        suggested_values = {}
        if form_step.suggested_values is UNDEFINED:
            suggested_values = self._options
        elif form_step.suggested_values:
            suggested_values = await form_step.suggested_values(self)
        if user_input:
            suggested_values = copy.deepcopy(suggested_values)
            self._update_and_remove_omitted_optional_keys(suggested_values, user_input, await self._get_schema(form_step))
        if data_schema.schema:
            data_schema = self._handler.add_suggested_values_to_schema(data_schema, suggested_values)
        errors = {'base': str(error)} if error else None
        last_step = None
        if not callable(form_step.next_step):
            last_step = form_step.next_step is None
        return self._handler.async_show_form(step_id=next_step_id, data_schema=data_schema, errors=errors, last_step=last_step, preview=form_step.preview)

    async def _async_menu_step(self, step_id: str, user_input: Any = None) -> Any:
        """Handle a menu step."""
        menu_step = cast(SchemaFlowMenuStep, self._flow[step_id])
        return self._handler.async_show_menu(step_id=step_id, menu_options=menu_step.options)

class SchemaConfigFlowHandler(ConfigFlow, ABC):
    """Handle a schema based config flow."""
    options_flow: Any = None
    VERSION: int = 1

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize a subclass."""
        super().__init_subclass__(**kwargs)

        @callback
        def _async_get_options_flow(config_entry: ConfigEntry) -> Any:
            """Get the options flow for this handler."""
            if cls.options_flow is None:
                raise UnknownHandler
            return SchemaOptionsFlowHandler(config_entry, cls.options_flow, cls.async_options_flow_finished, cls.async_setup_preview)
        cls.async_get_options_flow = _async_get_options_flow
        for step in cls.config_flow:
            setattr(cls, f'async_step_{step}', cls._async_step(step))

    def __init__(self) -> None:
        """Initialize config flow."""
        self._common_handler = SchemaCommonFlowHandler(self, self.config_flow, None)

    @staticmethod
    async def async_setup_preview(hass: HomeAssistant) -> None:
        """Set up preview."""

    @classmethod
    @callback
    def async_supports_options_flow(cls, config_entry: ConfigEntry) -> Any:
        """Return options flow support for this handler."""
        return cls.options_flow is not None

    @staticmethod
    def _async_step(step_id: str) -> Callable:
        """Generate a step handler."""

        async def _async_step(self, user_input: Any = None) -> Any:
            """Handle a config flow step."""
            return await self._common_handler.async_step(step_id, user_input)
        return _async_step

    @abstractmethod
    @callback
    def async_config_entry_title(self, options: Any) -> Any:
        """Return config entry title."""

    @callback
    def async_config_flow_finished(self, options: Any) -> None:
        """Take necessary actions after the config flow is finished, if needed."""

    @callback
    @staticmethod
    def async_options_flow_finished(hass: HomeAssistant, options: Any) -> None:
        """Take necessary actions after the options flow is finished, if needed."""

    @callback
    def async_create_entry(self, data: Any, **kwargs: Any) -> Any:
        """Finish config flow and create a config entry."""
        self.async_config_flow_finished(data)
        return super().async_create_entry(data={}, options=data, title=self.async_config_entry_title(data), **kwargs)

class SchemaOptionsFlowHandler(OptionsFlow):
    """Handle a schema based options flow."""

    def __init__(self, config_entry: ConfigEntry, options_flow: Any, async_options_flow_finished: Callable = None, async_setup_preview: Callable = None) -> None:
        """Initialize options flow."""
        self._options = copy.deepcopy(dict(config_entry.options))
        self._common_handler = SchemaCommonFlowHandler(self, options_flow, self.options)
        self._async_options_flow_finished = async_options_flow_finished
        for step in options_flow:
            setattr(self, f'async_step_{step}', types.MethodType(self._async_step(step), self))
        if async_setup_preview:
            setattr(self, 'async_setup_preview', async_setup_preview)

    @property
    def options(self) -> Any:
        """Return a mutable copy of the config entry options."""
        return self._options

    @staticmethod
    def _async_step(step_id: str) -> Callable:
        """Generate a step handler."""

        async def _async_step(self, user_input: Any = None) -> Any:
            """Handle an options flow step."""
            return await self._common_handler.async_step(step_id, user_input)
        return _async_step

    @callback
    def async_create_entry(self, data: Any, **kwargs: Any) -> Any:
        """Finish config flow and create a config entry."""
        if self._async_options_flow_finished:
            self._async_options_flow_finished(self.hass, data)
        return super().async_create_entry(data=data, **kwargs)

@callback
def wrapped_entity_config_entry_title(hass: HomeAssistant, entity_id_or_uuid: str) -> str:
    """Generate title for a config entry wrapping a single entity."""
    registry = er.async_get(hass)
    entity_id = er.async_validate_entity_id(registry, entity_id_or_uuid)
    object_id = split_entity_id(entity_id)[1]
    entry = registry.async_get(entity_id)
    if entry:
        return entry.name or entry.original_name or object_id
    state = hass.states.get(entity_id)
    if state:
        return state.name or object_id
    return object_id

@callback
def entity_selector_without_own_entities(handler: SchemaCommonFlowHandler, entity_selector_config: Any) -> selector.EntitySelector:
    """Return an entity selector which excludes own entities."""
    entity_registry = er.async_get(handler.hass)
    entities = er.async_entries_for_config_entry(entity_registry, handler.config_entry.entry_id)
    entity_ids = [ent.entity_id for ent in entities]
    final_selector_config = entity_selector_config.copy()
    final_selector_config['exclude_entities'] = entity_ids
    return selector.EntitySelector(final_selector_config)
