"""TemplateEntity utility class."""

from collections.abc import Callable, Mapping
from typing import Any, Optional, Union, cast
import voluptuous as vol
from homeassistant.core import Context, Event, HomeAssistant, State
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplateResult
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema[Any] = ...
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema[Any] = ...
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema[Any] = ...

def make_template_entity_common_schema(default_name: str) -> vol.Schema[Any]: ...

TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema[Any] = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema[Any] = ...
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema[Any] = ...
LEGACY_FIELDS: dict[str, str] = ...

def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: dict[str, Any],
    extra_legacy_fields: Optional[dict[str, str]] = None,
) -> dict[str, Any]: ...

class _TemplateAttribute:
    """Attribute value linked to template result."""

    def __init__(
        self,
        entity: Entity,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], Any]] = None,
        none_on_template_error: bool = False,
    ) -> None: ...

    def async_setup(self) -> None: ...
    def _default_update(self, result: Any) -> None: ...
    def handle_result(
        self,
        event: Event,
        template: Template,
        last_result: Any,
        result: Any,
    ) -> None: ...

class TemplateEntity(Entity):
    """Entity that uses templates to calculate attributes."""
    _attr_available: bool
    _attr_entity_picture: Optional[str]
    _attr_icon: Optional[str]

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        availability_template: Optional[Template] = None,
        icon_template: Optional[Template] = None,
        entity_picture_template: Optional[Template] = None,
        attribute_templates: Optional[dict[str, Template]] = None,
        config: Optional[dict[str, Any]] = None,
        fallback_name: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> None: ...

    def _render_variables(self) -> dict[str, Any]: ...
    def _update_available(self, result: Any) -> None: ...
    def _update_state(self, result: Any) -> None: ...
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None: ...

    @property
    def referenced_blueprint(self) -> Optional[str]: ...

    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], Any]] = None,
        none_on_template_error: bool = False,
    ) -> None: ...

    def _handle_results(self, event: Optional[Event], updates: list[TrackTemplateResult]) -> None: ...
    def _async_template_startup(self, _hass: Any, log_fn: Optional[Callable[[int, str], None]] = None) -> None: ...
    def _async_setup_templates(self) -> None: ...
    def async_start_preview(
        self,
        preview_callback: Callable[[Optional[str], Optional[dict[str, Any]], Optional[list[Any]], Optional[str]], None],
    ) -> Callable[[], None]: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_update(self) -> None: ...
    async def async_run_script(
        self,
        script: Script,
        *,
        run_variables: Optional[dict[str, Any]] = None,
        context: Optional[Context] = None,
    ) -> None: ...