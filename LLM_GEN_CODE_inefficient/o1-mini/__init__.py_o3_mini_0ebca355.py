"""Provide pre-made queries on top of the recorder component."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.orm.session import Session

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.recorder import get_instance

from ..filters import Filters
from .const import NEED_ATTRIBUTE_DOMAINS, SIGNIFICANT_DOMAINS
from .modern import (
    get_full_significant_states_with_session as _modern_get_full_significant_states_with_session,
    get_last_state_changes as _modern_get_last_state_changes,
    get_significant_states as _modern_get_significant_states,
    get_significant_states_with_session as _modern_get_significant_states_with_session,
    state_changes_during_period as _modern_state_changes_during_period,
)

__all__: List[str] = [
    "NEED_ATTRIBUTE_DOMAINS",
    "SIGNIFICANT_DOMAINS",
    "get_full_significant_states_with_session",
    "get_last_state_changes",
    "get_significant_states",
    "get_significant_states_with_session",
    "state_changes_during_period",
]


def get_full_significant_states_with_session(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[List[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    no_attributes: bool = False,
) -> Dict[str, List[State]]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import (  # pylint: disable=import-outside-toplevel
            get_full_significant_states_with_session as _legacy_get_full_significant_states_with_session,
        )

        _target: Callable[..., Dict[str, List[State]]] = _legacy_get_full_significant_states_with_session
    else:
        _target = _modern_get_full_significant_states_with_session
    return _target(
        hass,
        session,
        start_time,
        end_time,
        entity_ids,
        filters,
        include_start_time_state,
        significant_changes_only,
        no_attributes,
    )


def get_last_state_changes(
    hass: HomeAssistant, number_of_states: int, entity_id: str
) -> Dict[str, List[State]]:
    """Return the last number_of_states."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import (  # pylint: disable=import-outside-toplevel
            get_last_state_changes as _legacy_get_last_state_changes,
        )

        _target: Callable[..., Dict[str, List[State]]] = _legacy_get_last_state_changes
    else:
        _target = _modern_get_last_state_changes
    return _target(hass, number_of_states, entity_id)


def get_significant_states(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[List[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False,
) -> Dict[str, List[Union[State, Dict[str, Any]]]]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import (  # pylint: disable=import-outside-toplevel
            get_significant_states as _legacy_get_significant_states,
        )

        _target: Callable[..., Dict[str, List[Union[State, Dict[str, Any]]]]] = _legacy_get_significant_states
    else:
        _target = _modern_get_significant_states
    return _target(
        hass,
        start_time,
        end_time,
        entity_ids,
        filters,
        include_start_time_state,
        significant_changes_only,
        minimal_response,
        no_attributes,
        compressed_state_format,
    )


def get_significant_states_with_session(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_ids: Optional[List[str]] = None,
    filters: Optional[Filters] = None,
    include_start_time_state: bool = True,
    significant_changes_only: bool = True,
    minimal_response: bool = False,
    no_attributes: bool = False,
    compressed_state_format: bool = False,
) -> Dict[str, List[Union[State, Dict[str, Any]]]]:
    """Return a dict of significant states during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import (  # pylint: disable=import-outside-toplevel
            get_significant_states_with_session as _legacy_get_significant_states_with_session,
        )

        _target: Callable[..., Dict[str, List[Union[State, Dict[str, Any]]]]] = _legacy_get_significant_states_with_session
    else:
        _target = _modern_get_significant_states_with_session
    return _target(
        hass,
        session,
        start_time,
        end_time,
        entity_ids,
        filters,
        include_start_time_state,
        significant_changes_only,
        minimal_response,
        no_attributes,
        compressed_state_format,
    )


def state_changes_during_period(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    entity_id: Optional[str] = None,
    no_attributes: bool = False,
    descending: bool = False,
    limit: Optional[int] = None,
    include_start_time_state: bool = True,
) -> Dict[str, List[State]]:
    """Return a list of states that changed during a time period."""
    if not get_instance(hass).states_meta_manager.active:
        from .legacy import (  # pylint: disable=import-outside-toplevel
            state_changes_during_period as _legacy_state_changes_during_period,
        )

        _target: Callable[..., Dict[str, List[State]]] = _legacy_state_changes_during_period
    else:
        _target = _modern_state_changes_during_period
    return _target(
        hass,
        start_time,
        end_time,
        entity_id,
        no_attributes,
        descending,
        limit,
        include_start_time_state,
    )
