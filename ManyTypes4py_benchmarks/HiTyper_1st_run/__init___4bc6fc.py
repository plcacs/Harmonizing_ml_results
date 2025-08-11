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

def get_full_significant_states_with_session(hass: Union[str, homeassistancore.HomeAssistant, None], session: Union[datetime.timedelta, bool, float], start_time: Union[datetime.timedelta, bool, float], end_time: Union[None, datetime.timedelta, bool, float]=None, entity_ids: Union[None, datetime.timedelta, bool, float]=None, filters: Union[None, datetime.timedelta, bool, float]=None, include_start_time_state: bool=True, significant_changes_only: bool=True, no_attributes: bool=False) -> Union[bool, str, None]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_full_significant_states_with_session as _legacy_get_full_significant_states_with_session
        _target = _legacy_get_full_significant_states_with_session
    else:
        _target = _modern_get_full_significant_states_with_session
    return _target(hass, session, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, no_attributes)

def get_last_state_changes(hass: Union[homeassistancore.HomeAssistant, None, str, typing.Iterable[str], int], number_of_states: Union[str, int, typing.Mapping], entity_id: Union[str, int, typing.Mapping]) -> Union[bool, tuple[typing.Union[str,bool]], str]:
    """Return the last number_of_states."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_last_state_changes as _legacy_get_last_state_changes
        _target = _legacy_get_last_state_changes
    else:
        _target = _modern_get_last_state_changes
    return _target(hass, number_of_states, entity_id)

def get_significant_states(hass: Union[bool, str, float], start_time: Union[datetime.datetime, bool, float], end_time: Union[None, datetime.datetime, bool, float]=None, entity_ids: Union[None, datetime.datetime, bool, float]=None, filters: Union[None, datetime.datetime, bool, float]=None, include_start_time_state: bool=True, significant_changes_only: bool=True, minimal_response: bool=False, no_attributes: bool=False, compressed_state_format: bool=False) -> Union[float, datetime.datetime, GregorianDateTime]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_significant_states as _legacy_get_significant_states
        _target = _legacy_get_significant_states
    else:
        _target = _modern_get_significant_states
    return _target(hass, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, minimal_response, no_attributes, compressed_state_format)

def get_significant_states_with_session(hass: Union[homeassistancore.HomeAssistant, str, None], session: Union[datetime.datetime.datetime, str, None, grouper.models.base.session.Session], start_time: Union[datetime.datetime.datetime, str, None, grouper.models.base.session.Session], end_time: Union[None, datetime.datetime.datetime, str, grouper.models.base.session.Session]=None, entity_ids: Union[None, datetime.datetime.datetime, str, grouper.models.base.session.Session]=None, filters: Union[None, datetime.datetime.datetime, str, grouper.models.base.session.Session]=None, include_start_time_state: bool=True, significant_changes_only: bool=True, minimal_response: bool=False, no_attributes: bool=False, compressed_state_format: bool=False) -> Union[datetime.datetime, str, int]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import get_significant_states_with_session as _legacy_get_significant_states_with_session
        _target = _legacy_get_significant_states_with_session
    else:
        _target = _modern_get_significant_states_with_session
    return _target(hass, session, start_time, end_time, entity_ids, filters, include_start_time_state, significant_changes_only, minimal_response, no_attributes, compressed_state_format)

def state_changes_during_period(hass: Union[str, datetime.datetime, int], start_time: Union[str, datetime.datetime, int], end_time: Union[None, str, datetime.datetime, int]=None, entity_id: Union[None, str, datetime.datetime, int]=None, no_attributes: bool=False, descending: bool=False, limit: Union[None, str, datetime.datetime, int]=None, include_start_time_state: bool=True) -> Union[datetime.datetime.datetime, datetime.timedelta, str]:
    """Return a list of states that changed during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import state_changes_during_period as _legacy_state_changes_during_period
        _target = _legacy_state_changes_during_period
    else:
        _target = _modern_state_changes_during_period
    return _target(hass, start_time, end_time, entity_id, no_attributes, descending, limit, include_start_time_state)