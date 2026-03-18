```pyi
"""TemplateEntity utility class."""
from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any
import voluptuous as vol
from homeassistant.core import CALLBACK_TYPE, Context, Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplateResultInfo
from homeassistant.helpers.script import Script, _VarsType
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType

_LOGGER: Any

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema
TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema
LEGACY_FIELDS: dict[str, str]

def make_template_entity_common_schema(default_name: Any) -> vol.Schema: ...
def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: dict[str, Any],
    extra_legacy_fields: dict[str, str] | None = None,
) -> dict[str, Any]: ...

class _TemplateAttribute:
    _entity: Entity
    _attribute: str
    template: Template
    validator: Callable[[Any], Any] | None
    on_update: Callable[[Any], None] | None
    async_update: Any
    none_on_template_error: bool
    def __init__(
        self,
        entity: Entity,
        attribute: str,
        template: Template,
        validator: Callable[[Any], Any] | None = None,
        on_update: Callable[[Any], None] | None = None,
        none_on_template_error: bool = False,
    ) -> None: ...
    @callback
    def async_setup(self) -> None: ...
    @callback
    def _default_update(self, result: Any) -> None: ...
    @callback
    def handle_result(
        self,
        event: Event[Any] | None,
        template: Template,
        last_result: Any,
        result: Any,
    ) -> None: ...

class TemplateEntity(Entity):
    _attr_available: bool
    _attr_entity_picture: str | None
    _attr_icon: str | None
    _template_attrs: dict[Template, list[_TemplateAttribute]]
    _template_result_info: TrackTemplateResultInfo | None
    _attr_extra_state_attributes: dict[str, Any]
    _self_ref_update_count: int
    _attr_unique_id: str | None
    _preview_callback: Callable[[str | None, dict[str, Any] | None, Any, str | None], None] | None
    _attribute_templates: dict[str, Template] | None
    _availability_template: Template | None
    _icon_template: Template | None
    _entity_picture_template: Template | None
    _friendly_name_template: Template | None
    _run_variables: dict[str, Any] | Any
    _blueprint_inputs: dict[str, Any] | None
    def __init__(
        self,
        hass: HomeAssistant,
        *,
        availability_template: Template | None = None,
        icon_template: Template | None = None,
        entity_picture_template: Template | None = None,
        attribute_templates: dict[str, Template] | None = None,
        config: ConfigType | None = None,
        fallback_name: str | None = None,
        unique_id: str | None = None,
    ) -> None: ...
    @callback
    def _render_variables(self) -> dict[str, Any]: ...
    @callback
    def _update_available(self, result: Any) -> None: ...
    @callback
    def _update_state(self, result: Any) -> None: ...
    @callback
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None: ...
    @property
    def referenced_blueprint(self) -> str | None: ...
    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Callable[[Any], Any] | None = None,
        on_update: Callable[[Any], None] | None = None,
        none_on_template_error: bool = False,
    ) -> None: ...
    @callback
    def _handle_results(self, event: Event[Any] | None, updates: Any) -> None: ...
    @callback
    def _async_template_startup(self, _hass: HomeAssistant, log_fn: Callable[[int, str], None] | None = None) -> None: ...
    @callback
    def _async_setup_templates(self) -> None: ...
    @callback
    def async_start_preview(self, preview_callback: Callable[[str | None, dict[str, Any] | None, Any, str | None], None]) -> CALLBACK_TYPE: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_update(self) -> None: ...
    async def async_run_script(
        self,
        script: Script,
        *,
        run_variables: dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> None: ...
```