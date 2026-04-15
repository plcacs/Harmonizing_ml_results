"""TemplateEntity utility class."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional, Union
import voluptuous as vol
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import (
    CONF_ENTITY_PICTURE_TEMPLATE,
    CONF_FRIENDLY_NAME,
    CONF_ICON,
    CONF_ICON_TEMPLATE,
    CONF_NAME,
    CONF_PATH,
    CONF_VARIABLES,
)
from homeassistant.core import CALLBACK_TYPE, Context, Event, HomeAssistant, callback
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, TrackTemplateResultInfo
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template, TemplateStateFromEntityId
from homeassistant.helpers.trigger_template_entity import TEMPLATE_ENTITY_BASE_SCHEMA
from homeassistant.helpers.typing import ConfigType

_LOGGER: logging.Logger = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema = ...
LEGACY_FIELDS: dict[str, str] = ...

def make_template_entity_common_schema(
    default_name: str
) -> vol.Schema: ...

def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: dict[str, Any],
    extra_legacy_fields: Optional[dict[str, str]] = None
) -> dict[str, Any]: ...

class _TemplateAttribute:
    def __init__(
        self,
        entity: Any,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], None]] = None,
        none_on_template_error: bool = False
    ) -> None: ...
    
    @callback
    def async_setup(self) -> None: ...
    
    @callback
    def _default_update(self, result: Any) -> None: ...
    
    @callback
    def handle_result(
        self,
        event: Optional[Event],
        template: Template,
        last_result: Any,
        result: Any
    ) -> None: ...

class TemplateEntity(Entity):
    _attr_available: bool = ...
    _attr_entity_picture: Optional[str] = ...
    _attr_icon: Optional[str] = ...
    
    def __init__(
        self,
        hass: HomeAssistant,
        *,
        availability_template: Optional[Template] = None,
        icon_template: Optional[Template] = None,
        entity_picture_template: Optional[Template] = None,
        attribute_templates: Optional[dict[str, Template]] = None,
        config: Optional[ConfigType] = None,
        fallback_name: Optional[str] = None,
        unique_id: Optional[str] = None
    ) -> None: ...
    
    @callback
    def _render_variables(self) -> dict[str, Any]: ...
    
    @callback
    def _update_available(self, result: Any) -> None: ...
    
    @callback
    def _update_state(self, result: Any) -> None: ...
    
    @callback
    def _add_attribute_template(
        self,
        attribute_key: str,
        attribute_template: Template
    ) -> None: ...
    
    @property
    def referenced_blueprint(self) -> Optional[str]: ...
    
    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], None]] = None,
        none_on_template_error: bool = False
    ) -> None: ...
    
    @callback
    def _handle_results(
        self,
        event: Optional[Event],
        updates: list[TrackTemplateResult]
    ) -> None: ...
    
    @callback
    def _async_template_startup(
        self,
        _hass: Optional[HomeAssistant],
        log_fn: Optional[Callable[[int, str], None]] = None
    ) -> None: ...
    
    @callback
    def _async_setup_templates(self) -> None: ...
    
    @callback
    def async_start_preview(
        self,
        preview_callback: Callable[[Optional[str], Optional[dict[str, Any]], Optional[list[Any]], Optional[str]], None]
    ) -> list[Callable[[], None]]: ...
    
    async def async_added_to_hass(self) -> None: ...
    
    async def async_update(self) -> None: ...
    
    async def async_run_script(
        self,
        script: Script,
        *,
        run_variables: Optional[dict[str, Any]] = None,
        context: Optional[Context] = None
    ) -> None: ...