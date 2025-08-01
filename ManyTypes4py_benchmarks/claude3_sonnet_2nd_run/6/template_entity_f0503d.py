"""TemplateEntity utility class."""
from __future__ import annotations
from collections.abc import Callable, Mapping
import contextlib
import itertools
import logging
from typing import Any, cast, Optional, Dict, List, Union, TypeVar, Set
from propcache.api import under_cached_property
import voluptuous as vol
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import CONF_ENTITY_PICTURE_TEMPLATE, CONF_FRIENDLY_NAME, CONF_ICON, CONF_ICON_TEMPLATE, CONF_NAME, CONF_PATH, CONF_VARIABLES, STATE_UNKNOWN
from homeassistant.core import CALLBACK_TYPE, Context, Event, EventStateChangedData, HomeAssistant, State, callback, validate_state
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, TrackTemplateResultInfo, async_track_template_result
from homeassistant.helpers.script import Script, _VarsType
from homeassistant.helpers.start import async_at_start
from homeassistant.helpers.template import Template, TemplateStateFromEntityId, result_as_boolean
from homeassistant.helpers.trigger_template_entity import TEMPLATE_ENTITY_BASE_SCHEMA, make_template_entity_base_schema
from homeassistant.helpers.typing import ConfigType
from .const import CONF_ATTRIBUTE_TEMPLATES, CONF_ATTRIBUTES, CONF_AVAILABILITY, CONF_AVAILABILITY_TEMPLATE, CONF_PICTURE

_LOGGER = logging.getLogger(__name__)

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA = vol.Schema({vol.Optional(CONF_AVAILABILITY): cv.template})
TEMPLATE_ENTITY_ICON_SCHEMA = vol.Schema({vol.Optional(CONF_ICON): cv.template})
TEMPLATE_ENTITY_COMMON_SCHEMA = vol.Schema({
    vol.Optional(CONF_ATTRIBUTES): vol.Schema({cv.string: cv.template}),
    vol.Optional(CONF_AVAILABILITY): cv.template,
    vol.Optional(CONF_VARIABLES): cv.SCRIPT_VARIABLES_SCHEMA
}).extend(TEMPLATE_ENTITY_BASE_SCHEMA.schema)

def make_template_entity_common_schema(default_name: str) -> vol.Schema:
    """Return a schema with default name."""
    return vol.Schema({
        vol.Optional(CONF_ATTRIBUTES): vol.Schema({cv.string: cv.template}),
        vol.Optional(CONF_AVAILABILITY): cv.template
    }).extend(make_template_entity_base_schema(default_name).schema)

TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY = vol.Schema({
    vol.Optional(CONF_ATTRIBUTE_TEMPLATES, default={}): vol.Schema({cv.string: cv.template})
})
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY = vol.Schema({
    vol.Optional(CONF_AVAILABILITY_TEMPLATE): cv.template
})
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY = vol.Schema({
    vol.Optional(CONF_ENTITY_PICTURE_TEMPLATE): cv.template,
    vol.Optional(CONF_ICON_TEMPLATE): cv.template
}).extend(TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY.schema)

LEGACY_FIELDS: Dict[str, str] = {
    CONF_ICON_TEMPLATE: CONF_ICON,
    CONF_ENTITY_PICTURE_TEMPLATE: CONF_PICTURE,
    CONF_AVAILABILITY_TEMPLATE: CONF_AVAILABILITY,
    CONF_ATTRIBUTE_TEMPLATES: CONF_ATTRIBUTES,
    CONF_FRIENDLY_NAME: CONF_NAME
}

def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant, 
    entity_cfg: Dict[str, Any], 
    extra_legacy_fields: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Rewrite legacy config."""
    entity_cfg = {**entity_cfg}
    if extra_legacy_fields is None:
        extra_legacy_fields = {}
    for from_key, to_key in itertools.chain(LEGACY_FIELDS.items(), extra_legacy_fields.items()):
        if from_key not in entity_cfg or to_key in entity_cfg:
            continue
        val = entity_cfg.pop(from_key)
        if isinstance(val, str):
            val = Template(val, hass)
        entity_cfg[to_key] = val
    if CONF_NAME in entity_cfg and isinstance(entity_cfg[CONF_NAME], str):
        entity_cfg[CONF_NAME] = Template(entity_cfg[CONF_NAME], hass)
    return entity_cfg

class _TemplateAttribute:
    """Attribute value linked to template result."""

    def __init__(
        self, 
        entity: TemplateEntity, 
        attribute: str, 
        template: Template, 
        validator: Optional[Callable[[Any], Any]] = None, 
        on_update: Optional[Callable[[Any], None]] = None, 
        none_on_template_error: bool = False
    ) -> None:
        """Template attribute."""
        self._entity = entity
        self._attribute = attribute
        self.template = template
        self.validator = validator
        self.on_update = on_update
        self.async_update: Optional[Callable[[Any], None]] = None
        self.none_on_template_error = none_on_template_error

    @callback
    def async_setup(self) -> None:
        """Config update path for the attribute."""
        if self.on_update:
            return
        if not hasattr(self._entity, self._attribute):
            raise AttributeError(f"Attribute '{self._attribute}' does not exist.")
        self.on_update = self._default_update

    @callback
    def _default_update(self, result: Any) -> None:
        attr_result = None if isinstance(result, TemplateError) else result
        setattr(self._entity, self._attribute, attr_result)

    @callback
    def handle_result(
        self, 
        event: Optional[Event], 
        template: Template, 
        last_result: Any, 
        result: Any
    ) -> None:
        """Handle a template result event callback."""
        if isinstance(result, TemplateError):
            _LOGGER.error("TemplateError('%s') while processing template '%s' for attribute '%s' in entity '%s'", 
                         result, self.template, self._attribute, self._entity.entity_id)
            if self.none_on_template_error:
                self._default_update(result)
            else:
                assert self.on_update
                self.on_update(result)
            return
        if not self.validator:
            assert self.on_update
            self.on_update(result)
            return
        try:
            validated = self.validator(result)
        except vol.Invalid as ex:
            _LOGGER.error("Error validating template result '%s' from template '%s' for attribute '%s' in entity %s validation message '%s'", 
                         result, self.template, self._attribute, self._entity.entity_id, ex.msg)
            assert self.on_update
            self.on_update(None)
            return
        assert self.on_update
        self.on_update(validated)
        return

class TemplateEntity(Entity):
    """Entity that uses templates to calculate attributes."""
    _attr_available: bool = True
    _attr_entity_picture: Optional[str] = None
    _attr_icon: Optional[str] = None

    def __init__(
        self, 
        hass: HomeAssistant, 
        *, 
        availability_template: Optional[Template] = None, 
        icon_template: Optional[Template] = None, 
        entity_picture_template: Optional[Template] = None, 
        attribute_templates: Optional[Dict[str, Template]] = None, 
        config: Optional[Dict[str, Any]] = None, 
        fallback_name: Optional[str] = None, 
        unique_id: Optional[str] = None
    ) -> None:
        """Template Entity."""
        self._template_attrs: Dict[Template, List[_TemplateAttribute]] = {}
        self._template_result_info: Optional[TrackTemplateResultInfo] = None
        self._attr_extra_state_attributes: Dict[str, Any] = {}
        self._self_ref_update_count: int = 0
        self._attr_unique_id = unique_id
        self._preview_callback: Optional[Callable[[Optional[str], Optional[Dict[str, Any]], Optional[Set[str]], Optional[str]], None]] = None
        
        if config is None:
            self._attribute_templates = attribute_templates
            self._availability_template = availability_template
            self._icon_template = icon_template
            self._entity_picture_template = entity_picture_template
            self._friendly_name_template: Optional[Template] = None
            self._run_variables: Union[Dict[str, Any], Template] = {}
            self._blueprint_inputs: Optional[Dict[str, Any]] = None
        else:
            self._attribute_templates = config.get(CONF_ATTRIBUTES)
            self._availability_template = config.get(CONF_AVAILABILITY)
            self._icon_template = config.get(CONF_ICON)
            self._entity_picture_template = config.get(CONF_PICTURE)
            self._friendly_name_template = config.get(CONF_NAME)
            self._run_variables = config.get(CONF_VARIABLES, {})
            self._blueprint_inputs = config.get('raw_blueprint_inputs')

        class DummyState(State):
            """None-state for template entities not yet added to the state machine."""

            def __init__(self) -> None:
                """Initialize a new state."""
                super().__init__('unknown.unknown', STATE_UNKNOWN)
                self.entity_id = None

            @under_cached_property
            def name(self) -> str:
                """Name of this state."""
                return '<None>'
                
        variables: Dict[str, Any] = {'this': DummyState()}
        self._attr_name = fallback_name
        
        if self._friendly_name_template:
            with contextlib.suppress(TemplateError):
                self._attr_name = self._friendly_name_template.async_render(variables=variables, parse_result=False)
        
        if self._entity_picture_template:
            with contextlib.suppress(TemplateError):
                self._attr_entity_picture = self._entity_picture_template.async_render(variables=variables, parse_result=False)
        
        if self._icon_template:
            with contextlib.suppress(TemplateError):
                self._attr_icon = self._icon_template.async_render(variables=variables, parse_result=False)

    @callback
    def _render_variables(self) -> Dict[str, Any]:
        if isinstance(self._run_variables, dict):
            return self._run_variables
        return self._run_variables.async_render(self.hass, {'this': TemplateStateFromEntityId(self.hass, self.entity_id)})

    @callback
    def _update_available(self, result: Any) -> None:
        if isinstance(result, TemplateError):
            self._attr_available = True
            return
        self._attr_available = result_as_boolean(result)

    @callback
    def _update_state(self, result: Any) -> None:
        if self._availability_template:
            return
        self._attr_available = not isinstance(result, TemplateError)

    @callback
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None:
        """Create a template tracker for the attribute."""

        def _update_attribute(result: Any) -> None:
            attr_result = None if isinstance(result, TemplateError) else result
            self._attr_extra_state_attributes[attribute_key] = attr_result
            
        self.add_template_attribute(attribute_key, attribute_template, None, _update_attribute)

    @property
    def referenced_blueprint(self) -> Optional[str]:
        """Return referenced blueprint or None."""
        if self._blueprint_inputs is None:
            return None
        return cast(str, self._blueprint_inputs[CONF_USE_BLUEPRINT][CONF_PATH])

    def add_template_attribute(
        self, 
        attribute: str, 
        template: Template, 
        validator: Optional[Callable[[Any], Any]] = None, 
        on_update: Optional[Callable[[Any], None]] = None, 
        none_on_template_error: bool = False
    ) -> None:
        """Call in the constructor to add a template linked to a attribute.

        Parameters
        ----------
        attribute
            The name of the attribute to link to. This attribute must exist
            unless a custom on_update method is supplied.
        template
            The template to calculate.
        validator
            Validator function to parse the result and ensure it's valid.
        on_update
            Called to store the template result rather than storing it
            the supplied attribute. Passed the result of the validator, or None
            if the template or validator resulted in an error.
        none_on_template_error
            If True, the attribute will be set to None if the template errors.

        """
        if self.hass is None:
            raise ValueError('hass cannot be None')
        if template.hass is None:
            raise ValueError('template.hass cannot be None')
        template_attribute = _TemplateAttribute(self, attribute, template, validator, on_update, none_on_template_error)
        self._template_attrs.setdefault(template, [])
        self._template_attrs[template].append(template_attribute)

    @callback
    def _handle_results(self, event: Optional[Event], updates: List[TrackTemplateResult]) -> None:
        """Call back the results to the attributes."""
        if event:
            self.async_set_context(event.context)
        entity_id = event and event.data.get('entity_id')
        if entity_id and entity_id == self.entity_id:
            self._self_ref_update_count += 1
        else:
            self._self_ref_update_count = 0
        if self._self_ref_update_count > len(self._template_attrs):
            for update in updates:
                _LOGGER.warning('Template loop detected while processing event: %s, skipping template render for Template[%s]', event, update.template.template)
            return
        for update in updates:
            for template_attr in self._template_attrs[update.template]:
                template_attr.handle_result(event, update.template, update.last_result, update.result)
        if not self._preview_callback:
            self.async_write_ha_state()
            return
        try:
            calculated_state = self._async_calculate_state()
            validate_state(calculated_state.state)
        except Exception as err:
            self._preview_callback(None, None, None, str(err))
        else:
            assert self._template_result_info
            self._preview_callback(calculated_state.state, calculated_state.attributes, self._template_result_info.listeners, None)

    @callback
    def _async_template_startup(self, _hass: Optional[HomeAssistant], log_fn: Optional[Callable[[int, str], None]] = None) -> None:
        template_var_tups: List[TrackTemplate] = []
        has_availability_template = False
        variables: Dict[str, Any] = {'this': TemplateStateFromEntityId(self.hass, self.entity_id), **self._render_variables()}
        
        for template, attributes in self._template_attrs.items():
            template_var_tup = TrackTemplate(template, variables)
            is_availability_template = False
            for attribute in attributes:
                if attribute._attribute == '_attr_available':
                    has_availability_template = True
                    is_availability_template = True
                attribute.async_setup()
            if is_availability_template:
                template_var_tups.insert(0, template_var_tup)
            else:
                template_var_tups.append(template_var_tup)
                
        result_info = async_track_template_result(
            self.hass, 
            template_var_tups, 
            self._handle_results, 
            log_fn=log_fn, 
            has_super_template=has_availability_template
        )
        self.async_on_remove(result_info.async_remove)
        self._template_result_info = result_info
        result_info.async_refresh()

    @callback
    def _async_setup_templates(self) -> None:
        """Set up templates."""
        if self._availability_template is not None:
            self.add_template_attribute('_attr_available', self._availability_template, None, self._update_available)
        if self._attribute_templates is not None:
            for key, value in self._attribute_templates.items():
                self._add_attribute_template(key, value)
        if self._icon_template is not None:
            self.add_template_attribute('_attr_icon', self._icon_template, vol.Or(cv.whitespace, cv.icon))
        if self._entity_picture_template is not None:
            self.add_template_attribute('_attr_entity_picture', self._entity_picture_template, cv.string)
        if self._friendly_name_template is not None and (not self._friendly_name_template.is_static):
            self.add_template_attribute('_attr_name', self._friendly_name_template, cv.string)

    @callback
    def async_start_preview(
        self, 
        preview_callback: Callable[[Optional[str], Optional[Dict[str, Any]], Optional[Set[str]], Optional[str]], None]
    ) -> CALLBACK_TYPE:
        """Render a preview."""

        def log_template_error(level: int, msg: str) -> None:
            preview_callback(None, None, None, msg)
            
        self._preview_callback = preview_callback
        self._async_setup_templates()
        try:
            self._async_template_startup(None, log_template_error)
        except Exception as err:
            preview_callback(None, None, None, str(err))
        return self._call_on_remove_callbacks

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added to hass."""
        self._async_setup_templates()
        async_at_start(self.hass, self._async_template_startup)

    async def async_update(self) -> None:
        """Call for forced update."""
        assert self._template_result_info
        self._template_result_info.async_refresh()

    async def async_run_script(
        self, 
        script: Script, 
        *, 
        run_variables: Optional[_VarsType] = None, 
        context: Optional[Context] = None
    ) -> None:
        """Run an action script."""
        if run_variables is None:
            run_variables = {}
        await script.async_run(
            run_variables={'this': TemplateStateFromEntityId(self.hass, self.entity_id), **self._render_variables(), **run_variables}, 
            context=context
        )
