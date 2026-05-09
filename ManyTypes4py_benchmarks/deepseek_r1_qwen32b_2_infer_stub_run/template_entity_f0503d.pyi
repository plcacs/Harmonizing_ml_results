"""TemplateEntity utility class."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any, Callable, Dict, List, Optional, Union

import voluptuous as vol
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import CONF_ENTITY_PICTURE_TEMPLATE, CONF_FRIENDLY_NAME, CONF_ICON, CONF_ICON_TEMPLATE, CONF_NAME, CONF_PATH, CONF_VARIABLES, STATE_UNKNOWN
from homeassistant.core import CALLBACK_TYPE, Context, Event, EventStateChangedData, HomeAssistant, State, callback, validate_state
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, TrackTemplateResultInfo
from homeassistant.helpers.script import Script
from homeassistant.helpers.start import async_at_start
from homeassistant.helpers.template import Template
from homeassistant.helpers.trigger_template_entity import TEMPLATE_ENTITY_BASE_SCHEMA
from homeassistant.helpers.typing import ConfigType

_LOGGER: logging.Logger = ...

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema = ...

def make_template_entity_common_schema(default_name: str) -> vol.Schema:
    ...

TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema = ...
LEGACY_FIELDS: Dict[str, str] = ...

def rewrite_common_legacy_to_modern_conf(hass: HomeAssistant, entity_cfg: ConfigType, extra_legacy_fields: Optional[Dict[str, str]] = None) -> ConfigType:
    ...

class _TemplateAttribute:
    def __init__(self, entity: TemplateEntity, attribute: str, template: Template, validator: Optional[Callable] = None, on_update: Optional[Callable] = None, none_on_template_error: bool = False) -> None:
        ...

    @callback
    def async_setup(self) -> None:
        ...

    @callback
    def _default_update(self, result: Any) -> None:
        ...

    @callback
    def handle_result(self, event: Event, template: Template, last_result: Any, result: Any) -> None:
        ...

class TemplateEntity(Entity):
    _attr_available: bool
    _attr_entity_picture: Optional[str]
    _attr_icon: Optional[str]

    def __init__(self, hass: HomeAssistant, *, availability_template: Optional[Template] = None, icon_template: Optional[Template] = None, entity_picture_template: Optional[Template] = None, attribute_templates: Optional[Dict[str, Template]] = None, config: Optional[ConfigType] = None, fallback_name: Optional[str] = None, unique_id: Optional[str] = None) -> None:
        ...

    @callback
    def _render_variables(self) -> Dict[str, Any]:
        ...

    @callback
    def _update_available(self, result: Any) -> None:
        ...

    @callback
    def _update_state(self, result: Any) -> None:
        ...

    @callback
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None:
        ...

    @property
    def referenced_blueprint(self) -> Optional[str]:
        ...

    def add_template_attribute(self, attribute: str, template: Template, validator: Optional[Callable] = None, on_update: Optional[Callable] = None, none_on_template_error: bool = False) -> None:
        ...

    @callback
    def _handle_results(self, event: Optional[Event], updates: List[TrackTemplateResult]) -> None:
        ...

    @callback
    def _async_template_startup(self, _hass: HomeAssistant, log_fn: Optional[Callable] = None) -> None:
        ...

    @callback
    def _async_setup_templates(self) -> None:
        ...

    @callback
    def async_start_preview(self, preview_callback: Callable) -> Callable:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_update(self) -> None:
        ...

    async def async_run_script(self, script: Script, *, run_variables: Optional[Dict[str, Any]] = None, context: Optional[Context] = None) -> None:
        ...