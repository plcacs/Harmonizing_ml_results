"""Provide pre-made queries on top of the recorder component."""
from __future__ import annotations
from datetime import datetime
from typing import Any
from sqlalchemy.orm.session import Session
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.recorder import get_instance
from ..filters import Filters
from .const import NEED_ATTRIBUTE_DOMAINS, SIGNIFICANT_DOMAINS
from .modern import get_full_significant_states_with_session as _modern_get_full_significant_states_with_session, get_last_state_changes as _modern_get_last_state_changes, get_significant_states as _modern_get_significant_states, get_significant_states_with_session as _modern_get_significant_states_with_session, state_changes_during_period as _modern_state_changes_during_period
__all__ = ['NEED_ATTRIBUTE_DOMAINS', 'SIGNIFICANT_DOMAINS', 'get_full_significant_states_with_session', 'get_last_state_changes', 'get_significant_states', 'get_significant_states_with_session', 'state_changes_during_period']

def get_full_significant_states_with_session(hass, session, start_time, end_time=None, entity_ids=None, filters=None, include_start_time_state=True, significant_changes_only=True, no_attributes=False):
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_full_significant_states_with_session as _legacy_get_full_significant_states_with_session
        _target = _legacy_get_full_significant_states_with_session
    else:
        _target = _modern_get_full_significant_states_with_session
    return _target(hass, session, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, no_attributes)

def get_last_state_changes(hass, number_of_states, entity_id):
    """Return the last number_of_states."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_last_state_changes as _legacy_get_last_state_changes
        _target = _legacy_get_last_state_changes
    else:
        _target = _modern_get_last_state_changes
    return _target(hass, number_of_states, entity_id)

def get_significant_states(hass, start_time, end_time=None, entity_ids=None, filters=None, include_start_time_state=True, significant_changes_only=True, minimal_response=False, no_attributes=False, compressed_state_format=False):
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_significant_states as _legacy_get_significant_states
        _target = _legacy_get_significant_states
    else:
        _target = _modern_get_significant_states
    return _target(hass, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, minimal_response, no_attributes, compressed_state_format)

def get_significant_states_with_session(hass, session, start_time, end_time=None, entity_ids=None, filters=None, include_start_time_state=True, significant_changes_only=True, minimal_response=False, no_attributes=False, compressed_state_format=False):
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_significant_states_with_session as _legacy_get_significant_states_with_session
        _target = _legacy_get_significant_states_with_session
    else:
        _target = _modern_get_significant_states_with_session
    return _target(hass, session, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, minimal_response, no_attributes, compressed_state_format)

def state_changes_during_period(hass, start_time, end_time=None, entity_id=None, no_attributes=False, descending=False, limit=None, include_start_time_state=True):
    """Return a list of states that changed during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import state_changes_during_period as _legacy_state_changes_during_period
        _target = _legacy_state_changes_during_period
    else:
        _target = _modern_state_changes_during_period
    return _target(hass, start_time, end_time, entity_id, no_attributes, descending, limit, include_start_time_state)