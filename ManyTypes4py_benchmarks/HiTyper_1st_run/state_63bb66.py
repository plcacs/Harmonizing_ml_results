"""Models states in for Recorder."""
from __future__ import annotations
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any
from propcache.api import cached_property
from sqlalchemy.engine.row import Row
from homeassistant.const import COMPRESSED_STATE_ATTRIBUTES, COMPRESSED_STATE_LAST_CHANGED, COMPRESSED_STATE_LAST_UPDATED, COMPRESSED_STATE_STATE
from homeassistant.core import Context, State
from homeassistant.util import dt as dt_util
from .state_attributes import decode_attributes_from_source
_LOGGER = logging.getLogger(__name__)
EMPTY_CONTEXT = Context(id=None)

def extract_metadata_ids(entity_id_to_metadata_id: Any) -> list:
    """Extract metadata ids from entity_id_to_metadata_id."""
    return [metadata_id for metadata_id in entity_id_to_metadata_id.values() if metadata_id is not None]

class LazyState(State):
    """A lazy version of core State after schema 31."""

    def __init__(self, row: Union[int, str, None, list[tuple[str]]], attr_cache: Union[dict, datetime.datetime.datetime, None], start_time_ts: Union[datetime.datetime.date.time, datetime.datetime.timedelta, None, datetime.date], entity_id: Union[bool, None, str, list["DeliveryItem"]], state: Union[str, dict, None], last_updated_ts: Union[datetime.datetime.date.time, datetime.datetime.timedelta, None, datetime.date], no_attributes: Union[int, None, dict]) -> None:
        """Init the lazy state."""
        self._row = row
        self.entity_id = entity_id
        self.state = state or ''
        self._attributes = None
        self._last_updated_ts = last_updated_ts or start_time_ts
        self.attr_cache = attr_cache
        self.context = EMPTY_CONTEXT

    @cached_property
    def attributes(self):
        """State attributes."""
        return decode_attributes_from_source(getattr(self._row, 'attributes', None), self.attr_cache)

    @cached_property
    def _last_changed_ts(self):
        """Last changed timestamp."""
        return getattr(self._row, 'last_changed_ts', None)

    @cached_property
    def last_changed(self):
        """Last changed datetime."""
        return dt_util.utc_from_timestamp(self._last_changed_ts or self._last_updated_ts)

    @cached_property
    def _last_reported_ts(self):
        """Last reported timestamp."""
        return getattr(self._row, 'last_reported_ts', None)

    @cached_property
    def last_reported(self):
        """Last reported datetime."""
        return dt_util.utc_from_timestamp(self._last_reported_ts or self._last_updated_ts)

    @cached_property
    def last_updated(self):
        """Last updated datetime."""
        if TYPE_CHECKING:
            assert self._last_updated_ts is not None
        return dt_util.utc_from_timestamp(self._last_updated_ts)

    @cached_property
    def last_updated_timestamp(self):
        """Last updated timestamp."""
        if TYPE_CHECKING:
            assert self._last_updated_ts is not None
        return self._last_updated_ts

    @cached_property
    def last_changed_timestamp(self):
        """Last changed timestamp."""
        ts = self._last_changed_ts or self._last_updated_ts
        if TYPE_CHECKING:
            assert ts is not None
        return ts

    @cached_property
    def last_reported_timestamp(self):
        """Last reported timestamp."""
        ts = self._last_reported_ts or self._last_updated_ts
        if TYPE_CHECKING:
            assert ts is not None
        return ts

    def as_dict(self) -> dict[typing.Text, typing.Union[datetime.date,datetime.datetime.datetime,None]]:
        """Return a dict representation of the LazyState.

        Async friendly.

        To be used for JSON serialization.
        """
        last_updated_isoformat = self.last_updated.isoformat()
        if self._last_changed_ts == self._last_updated_ts:
            last_changed_isoformat = last_updated_isoformat
        else:
            last_changed_isoformat = self.last_changed.isoformat()
        return {'entity_id': self.entity_id, 'state': self.state, 'attributes': self._attributes or self.attributes, 'last_changed': last_changed_isoformat, 'last_updated': last_updated_isoformat}

def row_to_compressed_state(row: Union[str, float, int], attr_cache: Union[float, int, list[str], None], start_time_ts: Union[int, None, datetime.date], entity_id: Union[int, None, dict, list, tuple], state: Union[int, datetime.datetime.datetime, dict, None], last_updated_ts: Union[int, None, datetime.date], no_attributes: Union[int, None, dict, list, tuple]):
    """Convert a database row to a compressed state schema 41 and later."""
    comp_state = {COMPRESSED_STATE_STATE: state}
    if not no_attributes:
        comp_state[COMPRESSED_STATE_ATTRIBUTES] = decode_attributes_from_source(getattr(row, 'attributes', None), attr_cache)
    row_last_updated_ts = last_updated_ts or start_time_ts
    comp_state[COMPRESSED_STATE_LAST_UPDATED] = row_last_updated_ts
    if (row_last_changed_ts := getattr(row, 'last_changed_ts', None)) and row_last_changed_ts and (row_last_updated_ts != row_last_changed_ts):
        comp_state[COMPRESSED_STATE_LAST_CHANGED] = row_last_changed_ts
    return comp_state