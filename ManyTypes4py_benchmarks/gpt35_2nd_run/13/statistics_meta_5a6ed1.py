from __future__ import annotations
import logging
import threading
from typing import TYPE_CHECKING, Final, Literal, Dict, Optional, Set, Tuple
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
_LOGGER: logging.Logger = logging.getLogger(__name__)
QUERY_STATISTIC_META: Final[Tuple] = (StatisticsMeta.id, StatisticsMeta.statistic_id, StatisticsMeta.source, StatisticsMeta.unit_of_measurement, StatisticsMeta.has_mean, StatisticsMeta.has_sum, StatisticsMeta.name)
INDEX_ID: Final[int] = 0
INDEX_STATISTIC_ID: Final[int] = 1
INDEX_SOURCE: Final[int] = 2
INDEX_UNIT_OF_MEASUREMENT: Final[int] = 3
INDEX_HAS_MEAN: Final[int] = 4
INDEX_HAS_SUM: Final[int] = 5
INDEX_NAME: Final[int] = 6

def _generate_get_metadata_stmt(statistic_ids: Optional[Set[int]] = None, statistic_type: Optional[str] = None, statistic_source: Optional[str] = None) -> StatementLambdaElement:
    ...

class StatisticsMetaManager:
    def __init__(self, recorder: Recorder):
        ...

    def _clear_cache(self, statistic_ids: Set[int]):
        ...

    def _get_from_database(self, session: Session, statistic_ids: Optional[Set[int]] = None, statistic_type: Optional[str] = None, statistic_source: Optional[str] = None) -> Dict[int, Tuple[int, Dict[str, any]]]:
        ...

    def _assert_in_recorder_thread(self):
        ...

    def _add_metadata(self, session: Session, statistic_id: int, new_metadata: Dict[str, any]) -> int:
        ...

    def _update_metadata(self, session: Session, statistic_id: int, new_metadata: Dict[str, any], old_metadata_dict: Dict[int, Tuple[int, Dict[str, any]]]) -> Tuple[Optional[int], int]:
        ...

    def load(self, session: Session):
        ...

    def get(self, session: Session, statistic_id: int) -> Optional[int]:
        ...

    def get_many(self, session: Session, statistic_ids: Optional[Set[int]] = None, statistic_type: Optional[str] = None, statistic_source: Optional[str] = None) -> Dict[int, Tuple[int, Dict[str, any]]]:
        ...

    def get_from_cache_threadsafe(self, statistic_ids: Set[int]) -> Dict[int, Tuple[int, Dict[str, any]]]:
        ...

    def update_or_add(self, session: Session, new_metadata: Dict[str, any], old_metadata_dict: Dict[int, Tuple[int, Dict[str, any]]]) -> Tuple[Optional[int], int]:
        ...

    def update_unit_of_measurement(self, session: Session, statistic_id: int, new_unit: str):
        ...

    def update_statistic_id(self, session: Session, source: str, old_statistic_id: int, new_statistic_id: int):
        ...

    def delete(self, session: Session, statistic_ids: Set[int]):
        ...

    def reset(self):
        ...

    def adjust_lru_size(self, new_size: int):
        ...
