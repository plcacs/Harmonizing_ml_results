from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import voluptuous as vol
from homeassistant.core import CALLBACK_TYPE, Context, HomeAssistant
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.script import Script, _VarsType
from homeassistant.helpers.template import Template
from homeassistant.helpers.typing import ConfigType

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema = ...

def make_template_entity_common_schema(default_name: str) -> vol.Schema: ...

TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema = ...
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema = ...
LEGACY_FIELDS: dict[str, str] = ...

def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: Mapping[str, Any],
    extra_legacy_fields: Mapping[str, str] | None = ...,
) -> dict[str, Any]: ...

class TemplateEntity(Entity):
    _attr_available: bool = ...
    _attr_entity_picture: str | None = ...
    _attr_icon: str | None = ...

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

    @property
    def referenced_blueprint(self) -> str | None: ...

    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Callable[[Any], Any] | None = ...,
        on_update: Callable[[Any], None] | None = ...,
        none_on_template_error: bool = ...,
    ) -> None: ...

    def async_start_preview(
        self,
        preview_callback: Callable[
            [str | None, Mapping[str, Any] | None, Mapping[str, set[str]] | None, str | None],
            None,
        ],
    ) -> CALLBACK_TYPE: ...

    async def async_added_to_hass(self) -> None: ...
    async def async_update(self) -> None: ...
    async def async_run_script(
        self,
        script: Script,
        *,
        run_variables: _VarsType | None = ...,
        context: Context | None = ...,
    ) -> None: ...