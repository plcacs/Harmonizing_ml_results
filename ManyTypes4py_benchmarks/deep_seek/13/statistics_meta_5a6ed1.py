"""Support managing StatesMeta."""
from __future__ import annotations
import logging
import threading
from typing import TYPE_CHECKING, Final, Literal, Optional, Dict, Set, Tuple, Any, cast
from lru import LRU
from sqlalchemy import lambda_stmt, select
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.expression import true
from sqlalchemy.sql.lambdas import StatementLambdaElement
from ..db_schema import StatisticsMeta
from ..models import StatisticMetaData
from ..util import execute_stmt_lambda_element

if TYPE_CHECKING:
    from ..core import Recorder

CACHE_SIZE: Final[int] = 8192
_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
QUERY_STATISTIC_META: Final[Tuple] = (
    StatisticsMeta.id, 
    StatisticsMeta.statistic_id, 
    StatisticsMeta.source, 
    StatisticsMeta.unit_of_measurement, 
    StatisticsMeta.has_mean, 
    StatisticsMeta.has_sum, 
    StatisticsMeta.name
)
INDEX_ID: Final[int] = 0
INDEX_STATISTIC_ID: Final[int] = 1
INDEX_SOURCE: Final[int] = 2
INDEX_UNIT_OF_MEASUREMENT: Final[int] = 3
INDEX_HAS_MEAN: Final[int] = 4
INDEX_HAS_SUM: Final[int] = 5
INDEX_NAME: Final[int] = 6

def _generate_get_metadata_stmt(
    statistic_ids: Optional[Set[str]] = None, 
    statistic_type: Optional[Literal['mean', 'sum']] = None, 
    statistic_source: Optional[str] = None
) -> StatementLambdaElement:
    """Generate a statement to fetch metadata."""
    stmt = lambda_stmt(lambda: select(*QUERY_STATISTIC_META))
    if statistic_ids:
        stmt += lambda q: q.where(StatisticsMeta.statistic_id.in_(statistic_ids))
    if statistic_source is not None:
        stmt += lambda q: q.where(StatisticsMeta.source == statistic_source))
    if statistic_type == 'mean':
        stmt += lambda q: q.where(StatisticsMeta.has_mean == true())
    elif statistic_type == 'sum':
        stmt += lambda q: q.where(StatisticsMeta.has_sum == true())
    return stmt

class StatisticsMetaManager:
    """Manage the StatisticsMeta table."""

    def __init__(self, recorder: Recorder) -> None:
        """Initialize the statistics meta manager."""
        self.recorder: Recorder = recorder
        self._stat_id_to_id_meta: LRU[str, Tuple[int, Dict[str, Any]]] = LRU(CACHE_SIZE)

    def _clear_cache(self, statistic_ids: Set[str]) -> None:
        """Clear the cache."""
        for statistic_id in statistic_ids:
            self._stat_id_to_id_meta.pop(statistic_id, None)

    def _get_from_database(
        self, 
        session: Session, 
        statistic_ids: Optional[Set[str]] = None, 
        statistic_type: Optional[Literal['mean', 'sum']] = None, 
        statistic_source: Optional[str] = None
    ) -> Dict[str, Tuple[int, Dict[str, Any]]]:
        """Fetch meta data and process it into results and/or cache."""
        update_cache: bool = not session.new and (not session.dirty) and (self.recorder.thread_id == threading.get_ident())
        results: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        with session.no_autoflush:
            stat_id_to_id_meta = self._stat_id_to_id_meta
            for row in execute_stmt_lambda_element(session, _generate_get_metadata_stmt(statistic_ids, statistic_type, statistic_source), orm_rows=False):
                statistic_id: str = row[INDEX_STATISTIC_ID]
                row_id: int = row[INDEX_ID]
                meta: Dict[str, Any] = {
                    'has_mean': row[INDEX_HAS_MEAN], 
                    'has_sum': row[INDEX_HAS_SUM], 
                    'name': row[INDEX_NAME], 
                    'source': row[INDEX_SOURCE], 
                    'statistic_id': statistic_id, 
                    'unit_of_measurement': row[INDEX_UNIT_OF_MEASUREMENT]
                }
                id_meta: Tuple[int, Dict[str, Any]] = (row_id, meta)
                results[statistic_id] = id_meta
                if update_cache:
                    stat_id_to_id_meta[statistic_id] = id_meta
        return results

    def _assert_in_recorder_thread(self) -> None:
        """Assert that we are in the recorder thread."""
        if self.recorder.thread_id != threading.get_ident():
            raise RuntimeError('Detected unsafe call not in recorder thread')

    def _add_metadata(self, session: Session, statistic_id: str, new_metadata: Dict[str, Any]) -> int:
        """Add metadata to the database."""
        self._assert_in_recorder_thread()
        meta = StatisticsMeta.from_meta(new_metadata)
        session.add(meta)
        session.flush()
        _LOGGER.debug('Added new statistics metadata for %s, new_metadata: %s', statistic_id, new_metadata)
        return meta.id

    def _update_metadata(
        self, 
        session: Session, 
        statistic_id: str, 
        new_metadata: Dict[str, Any], 
        old_metadata_dict: Dict[str, Tuple[int, Dict[str, Any]]]
    ) -> Tuple[Optional[str], int]:
        """Update metadata in the database."""
        metadata_id, old_metadata = old_metadata_dict[statistic_id]
        if not (old_metadata['has_mean'] != new_metadata['has_mean'] or 
                old_metadata['has_sum'] != new_metadata['has_sum'] or 
                old_metadata['name'] != new_metadata['name'] or 
                (old_metadata['unit_of_measurement'] != new_metadata['unit_of_measurement'])):
            return (None, metadata_id)
        self._assert_in_recorder_thread()
        session.query(StatisticsMeta).filter_by(statistic_id=statistic_id).update({
            StatisticsMeta.has_mean: new_metadata['has_mean'], 
            StatisticsMeta.has_sum: new_metadata['has_sum'], 
            StatisticsMeta.name: new_metadata['name'], 
            StatisticsMeta.unit_of_measurement: new_metadata['unit_of_measurement']
        }, synchronize_session=False)
        self._clear_cache({statistic_id})
        _LOGGER.debug('Updated statistics metadata for %s, old_metadata: %s, new_metadata: %s', statistic_id, old_metadata, new_metadata)
        return (statistic_id, metadata_id)

    def load(self, session: Session) -> None:
        """Load the statistic_id to metadata_id mapping into memory."""
        self.get_many(session)

    def get(self, session: Session, statistic_id: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Resolve statistic_id to the metadata_id."""
        return self.get_many(session, {statistic_id}).get(statistic_id)

    def get_many(
        self, 
        session: Session, 
        statistic_ids: Optional[Set[str]] = None, 
        statistic_type: Optional[Literal['mean', 'sum']] = None, 
        statistic_source: Optional[str] = None
    ) -> Dict[str, Tuple[int, Dict[str, Any]]]:
        """Fetch meta data."""
        if statistic_ids is None:
            return self._get_from_database(session, statistic_type=statistic_type, statistic_source=statistic_source)
        if statistic_type is not None or statistic_source is not None:
            raise ValueError('Providing statistic_type and statistic_source is mutually exclusive of statistic_ids')
        results: Dict[str, Tuple[int, Dict[str, Any]] = self.get_from_cache_threadsafe(statistic_ids)
        if not (missing_statistic_id := statistic_ids.difference(results)):
            return results
        return results | self._get_from_database(session, statistic_ids=missing_statistic_id)

    def get_from_cache_threadsafe(self, statistic_ids: Set[str]) -> Dict[str, Tuple[int, Dict[str, Any]]]:
        """Get metadata from cache."""
        return {statistic_id: id_meta for statistic_id in statistic_ids if (id_meta := self._stat_id_to_id_meta.get(statistic_id))}

    def update_or_add(
        self, 
        session: Session, 
        new_metadata: Dict[str, Any], 
        old_metadata_dict: Dict[str, Tuple[int, Dict[str, Any]]]
    ) -> Tuple[Optional[str], int]:
        """Get metadata_id for a statistic_id."""
        statistic_id: str = new_metadata['statistic_id']
        if statistic_id not in old_metadata_dict:
            return (statistic_id, self._add_metadata(session, statistic_id, new_metadata))
        return self._update_metadata(session, statistic_id, new_metadata, old_metadata_dict)

    def update_unit_of_measurement(self, session: Session, statistic_id: str, new_unit: Optional[str]) -> None:
        """Update the unit of measurement for a statistic_id."""
        self._assert_in_recorder_thread()
        session.query(StatisticsMeta).filter(StatisticsMeta.statistic_id == statistic_id).update({StatisticsMeta.unit_of_measurement: new_unit})
        self._clear_cache({statistic_id})

    def update_statistic_id(self, session: Session, source: str, old_statistic_id: str, new_statistic_id: str) -> None:
        """Update the statistic_id for a statistic_id."""
        self._assert_in_recorder_thread()
        if self.get(session, new_statistic_id):
            _LOGGER.error('Cannot rename statistic_id `%s` to `%s` because the new statistic_id is already in use', old_statistic_id, new_statistic_id)
            return
        session.query(StatisticsMeta).filter((StatisticsMeta.statistic_id == old_statistic_id) & (StatisticsMeta.source == source)).update({StatisticsMeta.statistic_id: new_statistic_id})
        self._clear_cache({old_statistic_id})

    def delete(self, session: Session, statistic_ids: Set[str]) -> None:
        """Clear statistics for a list of statistic_ids."""
        self._assert_in_recorder_thread()
        session.query(StatisticsMeta).filter(StatisticsMeta.statistic_id.in_(statistic_ids)).delete(synchronize_session=False)
        self._clear_cache(statistic_ids)

    def reset(self) -> None:
        """Reset the cache."""
        self._stat_id_to_id_meta.clear()

    def adjust_lru_size(self, new_size: int) -> None:
        """Adjust the LRU cache size."""
        lru = self._stat_id_to_id_meta
        if new_size > lru.get_size():
            lru.set_size(new_size)
