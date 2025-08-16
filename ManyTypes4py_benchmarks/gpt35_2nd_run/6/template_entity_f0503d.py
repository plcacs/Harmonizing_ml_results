from __future__ import annotations
from collections.abc import Callable, Mapping
import contextlib
import itertools
import logging
from typing import Any, cast
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

_LOGGER: logging.Logger

TEMPLATE_ENTITY_AVAILABILITY_SCHEMA: vol.Schema
TEMPLATE_ENTITY_ICON_SCHEMA: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA: vol.Schema
TEMPLATE_ENTITY_ATTRIBUTES_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_AVAILABILITY_SCHEMA_LEGACY: vol.Schema
TEMPLATE_ENTITY_COMMON_SCHEMA_LEGACY: vol.Schema
LEGACY_FIELDS: dict[str, str]

def make_template_entity_common_schema(default_name: str) -> vol.Schema

def rewrite_common_legacy_to_modern_conf(hass: HomeAssistant, entity_cfg: dict, extra_legacy_fields: dict = None) -> dict

class _TemplateAttribute:
    def __init__(self, entity: TemplateEntity, attribute: str, template: Template, validator: Callable = None, on_update: Callable = None, none_on_template_error: bool = False) -> None

    def async_setup(self) -> None

    def _default_update(self, result: Any) -> None

    def handle_result(self, event: Event, template: Template, last_result: Any, result: Any) -> None

class TemplateEntity(Entity):
    def __init__(self, hass: HomeAssistant, availability_template: Template = None, icon_template: Template = None, entity_picture_template: Template = None, attribute_templates: dict = None, config: dict = None, fallback_name: str = None, unique_id: str = None) -> None

    def _render_variables(self) -> dict

    def _update_available(self, result: Any) -> None

    def _update_state(self, result: Any) -> None

    def _add_attribute_template(self, attribute_key: str, attribute_template: Template) -> None

    @property
    def referenced_blueprint(self) -> str

    def add_template_attribute(self, attribute: str, template: Template, validator: Callable = None, on_update: Callable = None, none_on_template_error: bool = False) -> None

    def _handle_results(self, event: Event, updates: list[TrackTemplateResult]) -> None

    def _async_template_startup(self, _hass: HomeAssistant, log_fn: Callable = None) -> None

    def _async_setup_templates(self) -> None

    def async_start_preview(self, preview_callback: Callable) -> Callable

    async def async_added_to_hass(self) -> None

    async def async_update(self) -> None

    async def async_run_script(self, script: Script, run_variables: dict = None, context: Context = None) -> None
