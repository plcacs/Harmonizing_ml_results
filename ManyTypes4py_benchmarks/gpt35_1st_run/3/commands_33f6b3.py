from __future__ import annotations
from collections.abc import Callable
from functools import lru_cache, partial
import json
import logging
from typing import Any, cast
import voluptuous as vol
from homeassistant.auth.models import User
from homeassistant.auth.permissions.const import POLICY_READ
from homeassistant.auth.permissions.events import SUBSCRIBE_ALLOWLIST
from homeassistant.const import EVENT_STATE_CHANGED, MATCH_ALL, SIGNAL_BOOTSTRAP_INTEGRATIONS
from homeassistant.core import Context, Event, EventStateChangedData, HomeAssistant, ServiceResponse, State, callback
from homeassistant.exceptions import HomeAssistantError, ServiceNotFound, ServiceValidationError, TemplateError, Unauthorized
from homeassistant.helpers import config_validation as cv, entity, template
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entityfilter import INCLUDE_EXCLUDE_BASE_FILTER_SCHEMA, convert_include_exclude_filter
from homeassistant.helpers.event import TrackTemplate, TrackTemplateResult, async_track_template_result
from homeassistant.helpers.json import JSON_DUMP, ExtendedJSONEncoder, find_paths_unserializable_data, json_bytes, json_fragment
from homeassistant.helpers.service import async_get_all_descriptions
from homeassistant.loader import IntegrationNotFound, async_get_integration, async_get_integration_descriptions, async_get_integrations
from homeassistant.setup import async_get_loaded_integrations, async_get_setup_timings
from homeassistant.util.json import format_unserializable_data
from . import const, decorators, messages
from .connection import ActiveConnection
from .messages import construct_result_message
ALL_SERVICE_DESCRIPTIONS_JSON_CACHE: str = 'websocket_api_all_service_descriptions_json'
_LOGGER: logging.Logger = logging.getLogger(__name__)

@callback
def async_register_commands(hass: HomeAssistant, async_reg: Callable[[HomeAssistant, Callable], None]) -> None:
    """Register commands."""
    async_reg(hass, handle_call_service)
    async_reg(hass, handle_entity_source)
    async_reg(hass, handle_execute_script)
    async_reg(hass, handle_fire_event)
    async_reg(hass, handle_get_config)
    async_reg(hass, handle_get_services)
    async_reg(hass, handle_get_states)
    async_reg(hass, handle_manifest_get)
    async_reg(hass, handle_integration_setup_info)
    async_reg(hass, handle_manifest_list)
    async_reg(hass, handle_ping)
    async_reg(hass, handle_render_template)
    async_reg(hass, handle_subscribe_bootstrap_integrations)
    async_reg(hass, handle_subscribe_events)
    async_reg(hass, handle_subscribe_trigger)
    async_reg(hass, handle_test_condition)
    async_reg(hass, handle_unsubscribe_events)
    async_reg(hass, handle_validate_config)
    async_reg(hass, handle_subscribe_entities)
    async_reg(hass, handle_supported_features)
    async_reg(hass, handle_integration_descriptions)
