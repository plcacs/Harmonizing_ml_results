from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, TYPE_CHECKING, Dict, List, Optional

from propcache.api import cached_property
from sqlalchemy.engine.row import Row

from homeassistant.const import (
    COMPRESSED_STATE_ATTRIBUTES,
    COMPRESSED_STATE_LAST_CHANGED,
    COMPRESSED_STATE_LAST_UPDATED,
    COMPRESSED_STATE_STATE,
)
from homeassistant.core import Context, State
from homeassistant.util import dt as dt_util

from .state_attributes import decode_attributes_from_source

_LOGGER: logging.Logger = logging.getLogger(__name__)

EMPTY_CONTEXT: Context = Context(id=None)


def extract_metadata_ids(
    entity_id_to_metadata_id: Dict[str, Optional[int]],
) -> List[int]:
    """Extract metadata ids from entity_id_to_metadata_id."""
    return [
        metadata_id
        for metadata_id in entity_id_to_metadata_id.values()
        if metadata_id is not None
    ]


class LazyState(State):
    """A lazy version of core State after schema 31."""

    def __init__(
        self,
        row: Row,
        attr_cache: Dict[str, Dict[str, Any]],
        start_time_ts: Optional[float],
        entity_id: str,
        state: str,
        last_updated_ts: Optional[float],
        no_attributes: bool,
    ) -> None:
        """Init the lazy state."""
        self._row: Row = row
        self.entity_id: str = entity_id
        self.state: str = state or ""
        self._attributes: Optional[Dict[str, Any]] = None
        self._last_updated_ts: Optional[float] = last_updated_ts or start_time_ts
        self.attr_cache: Dict[str, Dict[str, Any]] = attr_cache
        self.context: Context = EMPTY_CONTEXT

    @cached_property
    def attributes(self) -> Dict[str, Any]:  # type: ignore[override]
        """State attributes."""
        return decode_attributes_from_source(
            getattr(self._row, "attributes", None), self.attr_cache
        )

    @cached_property
    def _last_changed_ts(self) -> Optional[float]:
        """Last changed timestamp."""
        return getattr(self._row, "last_changed_ts", None)

    @cached_property
    def last_changed(self) -> datetime:  # type: ignore[override]
        """Last changed datetime."""
        return dt_util.utc_from_timestamp(
            self._last_changed_ts or self._last_updated_ts  # type: ignore[arg-type]
        )

    @cached_property
    def _last_reported_ts(self) -> Optional[float]:
        """Last reported timestamp."""
        return getattr(self._row, "last_reported_ts", None)

    @cached_property
    def last_reported(self) -> datetime:  # type: ignore[override]
        """Last reported datetime."""
        return dt_util.utc_from_timestamp(
            self._last_reported_ts or self._last_updated_ts  # type: ignore[arg-type]
        )

    @cached_property
    def last_updated(self) -> datetime:  # type: ignore[override]
        """Last updated datetime."""
        if TYPE_CHECKING:
            assert self._last_updated_ts is not None
        return dt_util.utc_from_timestamp(self._last_updated_ts)  # type: ignore[arg-type]

    @cached_property
    def last_updated_timestamp(self) -> float:  # type: ignore[override]
        """Last updated timestamp."""
        if TYPE_CHECKING:
            assert self._last_updated_ts is not None
        return self._last_updated_ts  # type: ignore

    @cached_property
    def last_changed_timestamp(self) -> float:  # type: ignore[override]
        """Last changed timestamp."""
        ts: Optional[float] = self._last_changed_ts or self._last_updated_ts
        if TYPE_CHECKING:
            assert ts is not None
        return ts  # type: ignore

    @cached_property
    def last_reported_timestamp(self) -> float:  # type: ignore[override]
        """Last reported timestamp."""
        ts: Optional[float] = self._last_reported_ts or self._last_updated_ts
        if TYPE_CHECKING:
            assert ts is not None
        return ts  # type: ignore

    def as_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        """Return a dict representation of the LazyState.

        Async friendly.

        To be used for JSON serialization.
        """
        last_updated_isoformat: str = self.last_updated.isoformat()
        if self._last_changed_ts == self._last_updated_ts:
            last_changed_isoformat: str = last_updated_isoformat
        else:
            last_changed_isoformat = self.last_changed.isoformat()
        return {
            "entity_id": self.entity_id,
            "state": self.state,
            "attributes": self._attributes or self.attributes,
            "last_changed": last_changed_isoformat,
            "last_updated": last_updated_isoformat,
        }


def row_to_compressed_state(
    row: Row,
    attr_cache: Dict[str, Dict[str, Any]],
    start_time_ts: Optional[float],
    entity_id: str,
    state: str,
    last_updated_ts: Optional[float],
    no_attributes: bool,
) -> Dict[str, Any]:
    """Convert a database row to a compressed state schema 41 and later."""
    comp_state: Dict[str, Any] = {COMPRESSED_STATE_STATE: state}
    if not no_attributes:
        comp_state[COMPRESSED_STATE_ATTRIBUTES] = decode_attributes_from_source(
            getattr(row, "attributes", None), attr_cache
        )
    row_last_updated_ts: float = last_updated_ts or start_time_ts  # type: ignore[assignment]
    comp_state[COMPRESSED_STATE_LAST_UPDATED] = row_last_updated_ts
    row_last_changed_ts: Optional[float] = getattr(row, "last_changed_ts", None)
    if row_last_changed_ts and row_last_updated_ts != row_last_changed_ts:
        comp_state[COMPRESSED_STATE_LAST_CHANGED] = row_last_changed_ts
    return comp_state