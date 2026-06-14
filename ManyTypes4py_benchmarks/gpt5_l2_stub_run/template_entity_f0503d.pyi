from typing import Any, Callable, Iterable, Mapping
import logging
import voluptuous as vol
from homeassistant.core import CALLBACK_TYPE, Context, Event, HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import TrackTemplateResult
from homeassistant.helpers.script import Script
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType

_LOGGER: logging.Logger = ...

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema = ...
LEGACY_FIELDS: dict[str, str] = ...

def make_template_entity_common_schema(default_name: str) -> vol.Schema: ...
def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: Mapping[str, Any],
    extra_legacy_fields: Mapping[str, str] | None = ...,
) -> dict[str, Any]: ...

class _TemplateAttribute:
    def __init__(
        self,
        entity: "TemplateEntity",
        attribute: str,
        template: Template,
        validator: Callable[[Any], Any] | None = ...,
        on_update: Callable[[Any | TemplateError | None], None] | None = ...,
        none_on_template_error: bool = ...,
    ) -> None: ...
    def async_setup(self) -> None: ...
    def _default_update(self, result: Any | TemplateError) -> None: ...
    def handle_result(
        self,
        event: Event | None,
        template: Template,
        last_result: Any,
        result: Any | TemplateError,
    ) -> None: ...

class TemplateEntity(Entity):
    _attr_available: bool
    _attr_entity_picture: str | None
    _attr_icon: str | None

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        availability_template: Template | None = ...,
        icon_template: Template | None = ...,
        entity_picture_template: Template | None = ...,
        attribute_templates: Mapping[str, Template] | None = ...,
        config: ConfigType | None = ...,
        fallback_name: str | None = ...,
        unique_id: str | None = ...,
    ) -> None: ...
    def _render_variables(self) -> dict[str, Any]: ...
    def _update_available(self, result: Any | TemplateError) -> None: ...
    def _update_state(self, result: Any | TemplateError) -> None: ...
    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None: ...
    @property
    def referenced_blueprint(self) -> str | None: ...
    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Callable[[Any], Any] | None = ...,
        on_update: Callable[[Any | TemplateError | None], None] | None = ...,
        none_on_template_error: bool = ...,
    ) -> None: ...
    def _handle_results(self, event: Event | None, updates: Iterable[TrackTemplateResult]) -> None: ...
    def _async_template_startup(self, _hass: HomeAssistant | None, log_fn: Callable[[int, str], None] | None = ...) -> None: ...
    def _async_setup_templates(self) -> None: ...
    def async_start_preview(
        self,
        preview_callback: Callable[[str | None, Mapping[str, Any] | None, Any, str | None], None],
    ) -> CALLBACK_TYPE: ...
    async def async_added_to_hass(self) -> None: ...
    async def async_update(self) -> None: ...
    async def async_run_script(
        self,
        script: Script,
        *,
        run_variables: Mapping[str, Any] | None = ...,
        context: Context | None = ...,
    ) -> None: ...