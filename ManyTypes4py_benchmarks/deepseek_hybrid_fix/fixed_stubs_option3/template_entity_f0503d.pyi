from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional
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

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema

def make_template_entity_common_schema(default_name: str) -> vol.Schema: ...

TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema
LEGACY_FIELDS: dict[str, str]

def rewrite_common_legacy_to_modern_conf(hass: HomeAssistant, entity_cfg: dict[str, Any], extra_legacy_fields: Optional[dict[str, str]] = ...) -> dict[str, Any]: ...

class _TemplateAttribute:
    _entity: Any
    _attribute: str
    template: Template
    validator: Optional[Callable[[Any], Any]]
    on_update: Optional[Callable[[Any], None]]
    async_update: None
    none_on_template_error: bool
    def __init__(self, entity: Any, attribute: str, template: Template, validator: Optional[Callable[[Any], Any]] = ..., on_update: Optional[Callable[[Any], None]] = ..., none_on_template_error: bool = ...) -> None: ...
    @callback
    def async_setup(self) -> None: ...
    @callback
    def _default_update(self, result: Any) -> None: ...
    @callback
    def handle_result(self, event: Optional[Event], template: Template, last_result: Any, result: Any) -> None: ...

class TemplateEntity(Entity):
    _attr_available: bool
    _attr_entity_picture: Any
    _attr_icon: Any
    _template_attrs: Dict[Template, List[_TemplateAttribute]]
    _template_result_info: Optional[TrackTemplateResultInfo]
    _attr_extra_state_attributes: Dict[str, Any]
    _self_ref_update_count: int
    _attribute_templates: Optional[Dict[str, Template]]
    _availability_template: Optional[Template]
    _icon_template: Optional[Template]
    _entity_picture_template: Optional[Template]
    _friendly_name_template: Optional[Template]
    _run_variables: Dict[str, Any] | _VarsType
    _blueprint_inputs: Optional[Dict[str, Any]]
    _preview_callback: Optional[Callable[[Optional[str], Optional[Dict[str, Any]], Optional[List[Any]], Optional[str]], None]]
    def __init__(self, hass: HomeAssistant, *, availability_template: Optional[Template] = ..., icon_template: Optional[Template] = ..., entity_picture_template: Optional[Template] = ..., attribute_templates: Optional[Dict[str, Template]] = ..., config: Optional[ConfigType] = ..., fallback_name: Optional[str] = ..., unique_id: Optional[str] = ...) -> None: ...
    @callback
    def _render_variables(self) -> Dict[str, Any]: ...
    @callback
    def _update_available(self, result: Any) -> None: ...
    @callback
    def _update_state(self, result: Any) -> None: ...
    @callback
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None: ...
    @property
    def referenced_blueprint(self) -> Optional[str]: ...
    def add_template_attribute(self, attribute: str, template: Template, validator: Optional[Callable[[Any], Any]] = ..., on_update: Optional[Callable[[Any], None]] = ..., none_on_template_error: bool = ...) -> None: ...
    @callback
    def _handle_results(self, event: Optional[Event], updates: List[TrackTemplateResult]) -> None: ...
    @callback
    def _async_template_startup(self, _hass: Optional[HomeAssistant], log_fn: Optional[Callable[[int, str], None]] = ...) -> None: ...
    @callback
    def _async_setup_templates(self) -> None: ...
    @callback
    def async_start_preview(self, preview_callback: Callable[[Optional[str], Optional[Dict[str, Any]], Optional[List[Any]], Optional[str]], None]) -> List[Callable[[], None]]: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_update(self) -> None: ...
    async def async_run_script(self, script: Script, *, run_variables: Optional[Dict[str, Any]] = ..., context: Optional[Context] = ...) -> None: ...