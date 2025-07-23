"""Statistics helper."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence, Mapping
import dataclasses
from datetime import datetime, timedelta
from functools import lru_cache, partial
from itertools import chain, groupby
import logging
from operator import itemgetter
import re
from time import time as time_time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast, Optional, Union, Dict, List, Set, Tuple, DefaultDict
from sqlalchemy import Select, and_, bindparam, func, lambda_stmt, select, text
from sqlalchemy.engine.row import Row
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.session import Session
from sqlalchemy.sql.lambdas import StatementLambdaElement
import voluptuous as vol
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT
from homeassistant.core import HomeAssistant, callback, valid_entity_id
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.singleton import singleton
from homeassistant.helpers.typing import UNDEFINED, UndefinedType
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    AreaConverter, BaseUnitConverter, BloodGlucoseConcentrationConverter,
    ConductivityConverter, DataRateConverter, DistanceConverter,
    DurationConverter, ElectricCurrentConverter, ElectricPotentialConverter,
    EnergyConverter, EnergyDistanceConverter, InformationConverter,
    MassConverter, PowerConverter, PressureConverter, SpeedConverter,
    TemperatureConverter, UnitlessRatioConverter, VolumeConverter,
    VolumeFlowRateConverter
)
from .const import (
    DOMAIN, EVENT_RECORDER_5MIN_STATISTICS_GENERATED,
    EVENT_RECORDER_HOURLY_STATISTICS_GENERATED,
    INTEGRATION_PLATFORM_COMPILE_STATISTICS,
    INTEGRATION_PLATFORM_LIST_STATISTIC_IDS,
    INTEGRATION_PLATFORM_UPDATE_STATISTICS_ISSUES,
    INTEGRATION_PLATFORM_VALIDATE_STATISTICS, SupportedDialect
)
from .db_schema import (
    STATISTICS_TABLES, Statistics, StatisticsBase, StatisticsMeta,
    StatisticsRuns, StatisticsShortTerm
)
from .models import (
    StatisticData, StatisticDataTimestamp, StatisticMetaData,
    StatisticResult, datetime_to_timestamp_or_none, process_timestamp
)
from .util import (
    execute, execute_stmt_lambda_element,
    filter_unique_constraint_integrity_error, get_instance,
    retryable_database_job, session_scope
)

if TYPE_CHECKING:
    from . import Recorder

QUERY_STATISTICS = (
    Statistics.metadata_id, Statistics.start_ts, Statistics.mean,
    Statistics.min, Statistics.max, Statistics.last_reset_ts,
    Statistics.state, Statistics.sum
)
QUERY_STATISTICS_SHORT_TERM = (
    StatisticsShortTerm.metadata_id, StatisticsShortTerm.start_ts,
    StatisticsShortTerm.mean, StatisticsShortTerm.min,
    StatisticsShortTerm.max, StatisticsShortTerm.last_reset_ts,
    StatisticsShortTerm.state, StatisticsShortTerm.sum
)
QUERY_STATISTICS_SUMMARY_MEAN = (
    StatisticsShortTerm.metadata_id, func.avg(StatisticsShortTerm.mean),
    func.min(StatisticsShortTerm.min), func.max(StatisticsShortTerm.max)
)
QUERY_STATISTICS_SUMMARY_SUM = (
    StatisticsShortTerm.metadata_id, StatisticsShortTerm.start_ts,
    StatisticsShortTerm.last_reset_ts, StatisticsShortTerm.state,
    StatisticsShortTerm.sum,
    func.row_number().over(
        partition_by=StatisticsShortTerm.metadata_id,
        order_by=StatisticsShortTerm.start_ts.desc()
    ).label('rownum')
)

STATISTIC_UNIT_TO_UNIT_CONVERTER = {
    **{unit: AreaConverter for unit in AreaConverter.VALID_UNITS},
    **{unit: BloodGlucoseConcentrationConverter for unit in BloodGlucoseConcentrationConverter.VALID_UNITS},
    **{unit: ConductivityConverter for unit in ConductivityConverter.VALID_UNITS},
    **{unit: DataRateConverter for unit in DataRateConverter.VALID_UNITS},
    **{unit: DistanceConverter for unit in DistanceConverter.VALID_UNITS},
    **{unit: DurationConverter for unit in DurationConverter.VALID_UNITS},
    **{unit: ElectricCurrentConverter for unit in ElectricCurrentConverter.VALID_UNITS},
    **{unit: ElectricPotentialConverter for unit in ElectricPotentialConverter.VALID_UNITS},
    **{unit: EnergyConverter for unit in EnergyConverter.VALID_UNITS},
    **{unit: EnergyDistanceConverter for unit in EnergyDistanceConverter.VALID_UNITS},
    **{unit: InformationConverter for unit in InformationConverter.VALID_UNITS},
    **{unit: MassConverter for unit in MassConverter.VALID_UNITS},
    **{unit: PowerConverter for unit in PowerConverter.VALID_UNITS},
    **{unit: PressureConverter for unit in PressureConverter.VALID_UNITS},
    **{unit: SpeedConverter for unit in SpeedConverter.VALID_UNITS},
    **{unit: TemperatureConverter for unit in TemperatureConverter.VALID_UNITS},
    **{unit: UnitlessRatioConverter for unit in UnitlessRatioConverter.VALID_UNITS},
    **{unit: VolumeConverter for unit in VolumeConverter.VALID_UNITS},
    **{unit: VolumeFlowRateConverter for unit in VolumeFlowRateConverter.VALID_UNITS}
}

UNIT_CLASSES = {
    unit: converter.UNIT_CLASS
    for unit, converter in STATISTIC_UNIT_TO_UNIT_CONVERTER.items()
}

DATA_SHORT_TERM_STATISTICS_RUN_CACHE = 'recorder_short_term_statistics_run_cache'

def mean(values: List[float]) -> float:
    """Return the mean of the values."""
    return sum(values) / len(values)

_LOGGER = logging.getLogger(__name__)

@dataclasses.dataclass(slots=True)
class ShortTermStatisticsRunCache:
    """Cache for short term statistics runs."""
    _latest_id_by_metadata_id: Dict[int, int] = dataclasses.field(default_factory=dict)

    def get_latest_ids(self, metadata_ids: Set[int]) -> Dict[int, int]:
        """Return the latest short term statistics ids for the metadata_ids."""
        return {
            metadata_id: id_
            for metadata_id, id_ in self._latest_id_by_metadata_id.items()
            if metadata_id in metadata_ids
        }

    def set_latest_id_for_metadata_id(self, metadata_id: int, id_: int) -> None:
        """Cache the latest id for the metadata_id."""
        self._latest_id_by_metadata_id[metadata_id] = id_

    def set_latest_ids_for_metadata_ids(self, metadata_id_to_id: Dict[int, int]) -> None:
        """Cache the latest id for the each metadata_id."""
        self._latest_id_by_metadata_id.update(metadata_id_to_id)

class BaseStatisticsRow(TypedDict, total=False):
    """A processed row of statistic data."""

class StatisticsRow(BaseStatisticsRow, total=False):
    """A processed row of statistic data."""

def get_display_unit(hass: HomeAssistant, statistic_id: str, statistic_unit: str) -> str:
    """Return the unit which the statistic will be displayed in."""
    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)) is None:
        return statistic_unit
    state_unit = statistic_unit
    if (state := hass.states.get(statistic_id)):
        state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if state_unit == statistic_unit or state_unit not in converter.VALID_UNITS:
        return statistic_unit
    return state_unit

def _get_statistic_to_display_unit_converter(
    statistic_unit: str,
    state_unit: str,
    requested_units: Optional[Dict[str, str]],
    allow_none: bool = True
) -> Optional[Callable[[float], float]]:
    """Prepare a converter from the statistics unit to display unit."""
    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)) is None:
        return None
    unit_class = converter.UNIT_CLASS
    if requested_units and unit_class in requested_units:
        display_unit = requested_units[unit_class]
    else:
        display_unit = state_unit
    if display_unit not in converter.VALID_UNITS:
        return None
    if display_unit == statistic_unit:
        return None
    if allow_none:
        return converter.converter_factory_allow_none(
            from_unit=statistic_unit, to_unit=display_unit
        )
    return converter.converter_factory(from_unit=statistic_unit, to_unit=display_unit)

def _get_display_to_statistic_unit_converter(
    display_unit: str,
    statistic_unit: str
) -> Optional[Callable[[float], float]]:
    """Prepare a converter from the display unit to the statistics unit."""
    if display_unit == statistic_unit or (
        converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)
    ) is None:
        return None
    return converter.converter_factory(from_unit=display_unit, to_unit=statistic_unit)

def _get_unit_converter(
    from_unit: str,
    to_unit: str
) -> Optional[Callable[[Optional[float]], Optional[float]]]:
    """Prepare a converter from a unit to another unit."""
    for conv in STATISTIC_UNIT_TO_UNIT_CONVERTER.values():
        if from_unit in conv.VALID_UNITS and to_unit in conv.VALID_UNITS:
            if from_unit == to_unit:
                return None
            return conv.converter_factory_allow_none(
                from_unit=from_unit, to_unit=to_unit
            )
    raise HomeAssistantError

def can_convert_units(from_unit: str, to_unit: str) -> bool:
    """Return True if it's possible to convert from from_unit to to_unit."""
    for converter in STATISTIC_UNIT_TO_UNIT_CONVERTER.values():
        if from_unit in converter.VALID_UNITS and to_unit in converter.VALID_UNITS:
            return True
    return False

@dataclasses.dataclass
class PlatformCompiledStatistics:
    """Compiled Statistics from a platform."""
    platform_stats: List[Dict[str, Any]]
    current_metadata: Dict[str, Dict[str, Any]]

def split_statistic_id(entity_id: str) -> Tuple[str, str]:
    """Split a state entity ID into domain and object ID."""
    return entity_id.split(':', 1)

VALID_STATISTIC_ID = re.compile('^(?!.+__)(?!_)[\\da-z_]+(?<!_):(?!_)[\\da-z_]+(?<!_)$')

def valid_statistic_id(statistic_id: str) -> bool:
    """Test if a statistic ID is a valid format."""
    return VALID_STATISTIC_ID.match(statistic_id) is not None

def validate_statistic_id(value: str) -> str:
    """Validate statistic ID."""
    if valid_statistic_id(value):
        return value
    raise vol.Invalid(f'Statistics ID {value} is an invalid statistic ID')

@dataclasses.dataclass
class ValidationIssue:
    """Error or warning message."""
    data: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return dictionary version."""
        return dataclasses.asdict(self)

def get_start_time() -> datetime:
    """Return start time."""
    now = dt_util.utcnow()
    current_period_minutes = now.minute - now.minute % 5
    current_period = now.replace(minute=current_period_minutes, second=0, microsecond=0)
    return current_period - timedelta(minutes=5)

def _compile_hourly_statistics_summary_mean_stmt(
    start_time_ts: float,
    end_time_ts: float
) -> StatementLambdaElement:
    """Generate the summary mean statement for hourly statistics."""
    return lambda_stmt(
        lambda: select(*QUERY_STATISTICS_SUMMARY_MEAN)
        .filter(StatisticsShortTerm.start_ts >= start_time_ts)
        .filter(StatisticsShortTerm.start_ts < end_time_ts)
        .group_by(StatisticsShortTerm.metadata_id)
        .order_by(StatisticsShortTerm.metadata_id)
    )

def _compile_hourly_statistics_last_sum_stmt(
    start_time_ts: float,
    end_time_ts: float
) -> StatementLambdaElement:
    """Generate the summary mean statement for hourly statistics."""
    return lambda_stmt(
        lambda: select(
            (subquery := select(*QUERY_STATISTICS_SUMMARY_SUM)
             .filter(StatisticsShortTerm.start_ts >= start_time_ts)
             .filter(StatisticsShortTerm.start_ts < end_time_ts)
             .subquery())
        )
        .filter(subquery.c.rownum == 1)
        .order_by(subquery.c.metadata_id)
    )

def _compile_hourly_statistics(session: Session, start: datetime) -> None:
    """Compile hourly statistics."""
    start_time = start.replace(minute=0)
    start_time_ts = start_time.timestamp()
    end_time = start_time + Statistics.duration
    end_time_ts = end_time.timestamp()
    summary: Dict[int, Dict[str, Any]] = {}
    stmt = _compile_hourly_statistics_summary_mean_stmt(start_time_ts, end_time_ts)
    stats = execute_stmt_lambda_element(session, stmt)
    if stats:
        for stat in stats:
            metadata_id, _mean, _min, _max = stat
            summary[metadata_id] = {
                'start_ts': start_time_ts,
                'mean': _mean,
                'min': _min,
                'max': _max
            }
    stmt = _compile_hourly_statistics_last_sum_stmt(start_time_ts, end_time_ts)
    stats = execute_stmt_lambda_element(session, stmt)
    if stats:
        for stat in stats:
            metadata_id, start, last_reset_ts, state, _sum, _ = stat
            if metadata_id in summary:
                summary[metadata_id].update({
                    'last_reset_ts': last_reset_ts,
                    'state': state,
                    'sum': _sum
                })
            else:
                summary[metadata_id] = {
                    'start_ts': start_time_ts,
                    'last_reset_ts': last_reset_ts,
                    'state': state,
                    'sum': _sum
                }
    now_timestamp = time_time()
    session.add_all(
        Statistics.from_stats_ts(metadata_id, summary_item, now_timestamp)
        for metadata_id, summary_item in summary.items()
    )

@retryable_database_job('compile missing statistics')
def compile_missing_statistics(instance: 'Recorder') -> bool:
    """Compile missing statistics."""
    now = dt_util.utcnow()
    period_size = 5
    last_period_minutes = now.minute - now.minute % period_size
    last_period = now.replace(minute=last_period_minutes, second=0, microsecond=0)
    start = now - timedelta(days=instance.keep_days)
    start = start.replace(minute=0, second=0, microsecond=0)
    commit_interval = 60 / period_size * 12
    with session_scope(
        session=instance.get_session(),
        exception_filter=filter_unique_constraint_integrity_error(instance, 'statistic')
    ) as session:
        if (last_run := session.query(func.max(StatisticsRuns.start)).scalar()):
            start = max(start, process_timestamp(last_run) + StatisticsShortTerm.duration)
        periods_without_commit = 0
        while start < last_period:
            periods_without_commit += 1
            end = start + timedelta(minutes=period_size)
            _LOGGER.debug('Compiling missing statistics for %s-%s', start, end)
            modified_statistic_ids = _compile_statistics(instance, session, start, end >= last_period)
            if periods_without_commit == commit_interval or modified_statistic_ids:
                session.commit()
                session.expunge_all()
                periods_without_commit = 0
            start = end
    return True

@retryable_database_job('compile statistics')
def compile_statistics(
    instance: 'Recorder',
    start: datetime,
    fire_events: bool
) -> bool:
    """Compile 5-minute statistics for all integrations with a recorder platform."""
    modified_statistic_ids = None
    with session_scope(
        session=instance.get_session(),
        exception_filter=filter_unique_constraint_integrity_error(instance, 'statistic')
    ) as session:
        modified_statistic_ids = _compile_statistics(instance, session, start, fire_events)
    if modified_statistic_ids:
        with session_scope(session=instance.get_session(), read_only=True) as session:
            instance.statistics_meta_manager.get_many(session, modified_statistic_ids)
    return True

def _get_first_id_stmt(start: datetime) -> StatementLambdaElement:
    """Return a statement that returns the first run_id at start."""
    return lambda_stmt(
        lambda: select(StatisticsRuns.run_id).filter_by(start=start)
    )

def _compile_statistics(
    instance: 'Recorder',
    session: Session,
    start: datetime,
    fire_events: bool
) -> Set[str]:
    """Compile 5-minute statistics for all integrations with a recorder platform."""
    assert start.tzinfo == dt_util.UTC, 'start must be in UTC'
    end = start + StatisticsShortTerm.duration
    statistics_meta_manager = instance.statistics_meta_manager
    modified_statistic_ids: Set[str] = set()
    if execute_stmt_lambda_element(session, _get_first_id_stmt(start)):
        _LOGGER.debug('Statistics already compiled for %s-%s', start, end)
        return modified_statistic_ids
    _LOGGER.debug('Compiling statistics for %s-%s', start, end)
    platform_stats: List[Dict[str, Any]] = []
    current_metadata: Dict[str