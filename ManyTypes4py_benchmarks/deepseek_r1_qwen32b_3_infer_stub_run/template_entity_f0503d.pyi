"""Stub file for 'template_entity_f0503d' module."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from contextlib import suppress as contextlib_suppress
from itertools import chain as itertools_chain
from logging import Logger
from homeassistant.components.blueprint import CONF_USE_BLUEPRINT
from homeassistant.const import (
    CONF_ENTITY_PICTURE_TEMPLATE,
    CONF_FRIENDLY_NAME,
    CONF_ICON,
    CONF_ICON_TEMPLATE,
    CONF_NAME,
    CONF_PATH,
    CONF_VARIABLES,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Context,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
    validate_state,
)
from homeassistant.exceptions import TemplateError
from homeassistant.helpers.config_validation import ConfigType
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.event import (
    TrackTemplate,
    TrackTemplateResult,
    TrackTemplateResultInfo,
)
from homeassistant.helpers.script import Script, _VarsType
from homeassistant.helpers.start import async_at_start
from homeassistant.helpers.template import (
    Template,
    TemplateStateFromEntityId,
    result_as_boolean,
)
from homeassistant.helpers.trigger_template_entity import (
    TEMPLATE_ENTITY_BASE_SCHEMA,
    make_template_entity_base_schema,
)
from voluptuous import Schema as vol_Schema
from .const import (
    CONF_ATTRIBUTE_TEMPLATES,
    CONF_ATTRIBUTES,
    CONF_AVAILABILITY,
    CONF_AVAILABILITY_TEMPLATE,
    CONF_PICTURE,
)

_LOGGER: Logger = logging.getLogger(__name__)

def rewrite_common_legacy_to_modern_conf(
    hass: HomeAssistant,
    entity_cfg: Dict[str, Any],
    extra_legacy_fields: Optional[Dict[str, str]] = None,
) -> ConfigType:
    ...

class _TemplateAttribute:
    _entity: Entity
    _attribute: str
    template: Template
    validator: Optional[Callable[[Any], Any]]
    on_update: Optional[Callable[[Any], None]]
    async_update: Optional[Callable[..., None]]
    none_on_template_error: bool

    def __init__(
        self,
        entity: Entity,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], None]] = None,
        none_on_template_error: bool = False,
    ) -> None:
        ...

    @callback
    def async_setup(self) -> None:
        ...

    @callback
    def _default_update(self, result: Any) -> None:
        ...

    @callback
    def handle_result(
        self,
        event: Event,
        template: Template,
        last_result: Any,
        result: Union[TemplateError, Any],
    ) -> None:
        ...

class TemplateEntity(Entity):
    _attr_available: bool
    _attr_entity_picture: Optional[str]
    _attr_icon: Optional[str]
    _attr_extra_state_attributes: Dict[str, Any]
    _self_ref_update_count: int
    _attr_unique_id: Optional[str]
    _preview_callback: Optional[Callable[..., None]]
    _template_attrs: Dict[Template, List[_TemplateAttribute]]
    _template_result_info: Optional[TrackTemplateResultInfo]

    def __init__(
        self,
        hass: HomeAssistant,
        availability_template: Optional[Template] = None,
        icon_template: Optional[Template] = None,
        entity_picture_template: Optional[Template] = None,
        attribute_templates: Optional[Dict[str, Template]] = None,
        config: Optional[ConfigType] = None,
        fallback_name: Optional[str] = None,
        unique_id: Optional[str] = None,
    ) -> None:
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
    def _add_attribute_template(
        self,
        attribute_key: str,
        attribute_template: Template,
    ) -> None:
        ...

    @property
    def referenced_blueprint(self) -> Optional[str]:
        ...

    def add_template_attribute(
        self,
        attribute: str,
        template: Template,
        validator: Optional[Callable[[Any], Any]] = None,
        on_update: Optional[Callable[[Any], None]] = None,
        none_on_template_error: bool = False,
    ) -> None:
        ...

    @callback
    def _handle_results(
        self,
        event: Optional[Event],
        updates: List[TrackTemplateResult],
    ) -> None:
        ...

    @callback
    def _async_template_startup(
        self,
        _hass: Optional[HomeAssistant] = None,
        log_fn: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        ...

    @callback
    def _async_setup_templates(self) -> None:
        ...

    @callback
    def async_start_preview(
        self,
        preview_callback: Callable[..., None],
    ) -> Callable[..., None]:
        ...

    async def async_added_to_hass(self) -> None:
        ...

    async def async_update(self) -> None:
        ...

    async def async_run_script(
        self,
        script: Script,
        run_variables: Optional[Dict[str, Any]] = None,
        context: Optional[Context] = None,
    ) -> None:
        ...