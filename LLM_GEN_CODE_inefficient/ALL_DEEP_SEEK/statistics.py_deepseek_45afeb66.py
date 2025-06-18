"""Statistics helper."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
import dataclasses
from datetime import datetime, timedelta
from functools import lru_cache, partial
from itertools import chain, groupby
import logging
from operator import itemgetter
import re
from time import time as time_time
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

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
    AreaConverter,
    BaseUnitConverter,
    BloodGlucoseConcentrationConverter,
    ConductivityConverter,
    DataRateConverter,
    DistanceConverter,
    DurationConverter,
    ElectricCurrentConverter,
    ElectricPotentialConverter,
    EnergyConverter,
    EnergyDistanceConverter,
    InformationConverter,
    MassConverter,
    PowerConverter,
    PressureConverter,
    SpeedConverter,
    TemperatureConverter,
    UnitlessRatioConverter,
    VolumeConverter,
    VolumeFlowRateConverter,
)

from .const import (
    DOMAIN,
    EVENT_RECORDER_5MIN_STATISTICS_GENERATED,
    EVENT_RECORDER_HOURLY_STATISTICS_GENERATED,
    INTEGRATION_PLATFORM_COMPILE_STATISTICS,
    INTEGRATION_PLATFORM_LIST_STATISTIC_IDS,
    INTEGRATION_PLATFORM_UPDATE_STATISTICS_ISSUES,
    INTEGRATION_PLATFORM_VALIDATE_STATISTICS,
    SupportedDialect,
)
from .db_schema import (
    STATISTICS_TABLES,
    Statistics,
    StatisticsBase,
    StatisticsMeta,
    StatisticsRuns,
    StatisticsShortTerm,
)
from .models import (
    StatisticData,
    StatisticDataTimestamp,
    StatisticMetaData,
    StatisticResult,
    datetime_to_timestamp_or_none,
    process_timestamp,
)
from .util import (
    execute,
    execute_stmt_lambda_element,
    filter_unique_constraint_integrity_error,
    get_instance,
    retryable_database_job,
    session_scope,
)

if TYPE_CHECKING:
    from . import Recorder

QUERY_STATISTICS = (
    Statistics.metadata_id,
    Statistics.start_ts,
    Statistics.mean,
    Statistics.min,
    Statistics.max,
    Statistics.last_reset_ts,
    Statistics.state,
    Statistics.sum,
)

QUERY_STATISTICS_SHORT_TERM = (
    StatisticsShortTerm.metadata_id,
    StatisticsShortTerm.start_ts,
    StatisticsShortTerm.mean,
    StatisticsShortTerm.min,
    StatisticsShortTerm.max,
    StatisticsShortTerm.last_reset_ts,
    StatisticsShortTerm.state,
    StatisticsShortTerm.sum,
)

QUERY_STATISTICS_SUMMARY_MEAN = (
    StatisticsShortTerm.metadata_id,
    func.avg(StatisticsShortTerm.mean),
    func.min(StatisticsShortTerm.min),
    func.max(StatisticsShortTerm.max),
)

QUERY_STATISTICS_SUMMARY_SUM = (
    StatisticsShortTerm.metadata_id,
    StatisticsShortTerm.start_ts,
    StatisticsShortTerm.last_reset_ts,
    StatisticsShortTerm.state,
    StatisticsShortTerm.sum,
    func.row_number()
    .over(
        partition_by=StatisticsShortTerm.metadata_id,
        order_by=StatisticsShortTerm.start_ts.desc(),
    )
    .label("rownum"),
)


STATISTIC_UNIT_TO_UNIT_CONVERTER: dict[str | None, type[BaseUnitConverter]] = {
    **{unit: AreaConverter for unit in AreaConverter.VALID_UNITS},
    **{
        unit: BloodGlucoseConcentrationConverter
        for unit in BloodGlucoseConcentrationConverter.VALID_UNITS
    },
    **{unit: ConductivityConverter for unit in ConductivityConverter.VALID_UNITS},
    **{unit: DataRateConverter for unit in DataRateConverter.VALID_UNITS},
    **{unit: DistanceConverter for unit in DistanceConverter.VALID_UNITS},
    **{unit: DurationConverter for unit in DurationConverter.VALID_UNITS},
    **{unit: ElectricCurrentConverter for unit in ElectricCurrentConverter.VALID_UNITS},
    **{
        unit: ElectricPotentialConverter
        for unit in ElectricPotentialConverter.VALID_UNITS
    },
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
    **{unit: VolumeFlowRateConverter for unit in VolumeFlowRateConverter.VALID_UNITS},
}


UNIT_CLASSES = {
    unit: converter.UNIT_CLASS
    for unit, converter in STATISTIC_UNIT_TO_UNIT_CONVERTER.items()
}

DATA_SHORT_TERM_STATISTICS_RUN_CACHE = "recorder_short_term_statistics_run_cache"


def mean(values: list[float]) -> float | None:
    """Return the mean of the values.

    This is a very simple version that only works
    with a non-empty list of floats. The built-in
    statistics.mean is more robust but is almost
    an order of magnitude slower.
    """
    return sum(values) / len(values)


_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(slots=True)
class ShortTermStatisticsRunCache:
    """Cache for short term statistics runs."""

    # This is a mapping of metadata_id:id of the last short term
    # statistics run for each metadata_id
    _latest_id_by_metadata_id: dict[int, int] = dataclasses.field(default_factory=dict)

    def get_latest_ids(self, metadata_ids: set[int]) -> dict[int, int]:
        """Return the latest short term statistics ids for the metadata_ids."""
        return {
            metadata_id: id_
            for metadata_id, id_ in self._latest_id_by_metadata_id.items()
            if metadata_id in metadata_ids
        }

    def set_latest_id_for_metadata_id(self, metadata_id: int, id_: int) -> None:
        """Cache the latest id for the metadata_id."""
        self._latest_id_by_metadata_id[metadata_id] = id_

    def set_latest_ids_for_metadata_ids(
        self, metadata_id_to_id: dict[int, int]
    ) -> None:
        """Cache the latest id for the each metadata_id."""
        self._latest_id_by_metadata_id.update(metadata_id_to_id)


class BaseStatisticsRow(TypedDict, total=False):
    """A processed row of statistic data."""

    start: float


class StatisticsRow(BaseStatisticsRow, total=False):
    """A processed row of statistic data."""

    end: float
    last_reset: float | None
    state: float | None
    sum: float | None
    min: float | None
    max: float | None
    mean: float | None
    change: float | None


def get_display_unit(
    hass: HomeAssistant,
    statistic_id: str,
    statistic_unit: str | None,
) -> str | None:
    """Return the unit which the statistic will be displayed in."""

    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)) is None:
        return statistic_unit

    state_unit: str | None = statistic_unit
    if state := hass.states.get(statistic_id):
        state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)

    if state_unit == statistic_unit or state_unit not in converter.VALID_UNITS:
        # Guard against invalid state unit in the DB
        return statistic_unit

    return state_unit


def _get_statistic_to_display_unit_converter(
    statistic_unit: str | None,
    state_unit: str | None,
    requested_units: dict[str, str] | None,
    allow_none: bool = True,
) -> Callable[[float | None], float | None] | Callable[[float], float] | None:
    """Prepare a converter from the statistics unit to display unit."""
    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)) is None:
        return None

    display_unit: str | None
    unit_class = converter.UNIT_CLASS
    if requested_units and unit_class in requested_units:
        display_unit = requested_units[unit_class]
    else:
        display_unit = state_unit

    if display_unit not in converter.VALID_UNITS:
        # Guard against invalid state unit in the DB
        return None

    if display_unit == statistic_unit:
        return None

    if allow_none:
        return converter.converter_factory_allow_none(
            from_unit=statistic_unit, to_unit=display_unit
        )
    return converter.converter_factory(from_unit=statistic_unit, to_unit=display_unit)


def _get_display_to_statistic_unit_converter(
    display_unit: str | None,
    statistic_unit: str | None,
) -> Callable[[float], float] | None:
    """Prepare a converter from the display unit to the statistics unit."""
    if (
        display_unit == statistic_unit
        or (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit)) is None
    ):
        return None
    return converter.converter_factory(from_unit=display_unit, to_unit=statistic_unit)


def _get_unit_converter(
    from_unit: str, to_unit: str
) -> Callable[[float | None], float | None] | None:
    """Prepare a converter from a unit to another unit."""
    for conv in STATISTIC_UNIT_TO_UNIT_CONVERTER.values():
        if from_unit in conv.VALID_UNITS and to_unit in conv.VALID_UNITS:
            if from_unit == to_unit:
                return None
            return conv.converter_factory_allow_none(
                from_unit=from_unit, to_unit=to_unit
            )
    raise HomeAssistantError


def can_convert_units(from_unit: str | None, to_unit: str | None) -> bool:
    """Return True if it's possible to convert from from_unit to to_unit."""
    for converter in STATISTIC_UNIT_TO_UNIT_CONVERTER.values():
        if from_unit in converter.VALID_UNITS and to_unit in converter.VALID_UNITS:
            return True
    return False


@dataclasses.dataclass
class PlatformCompiledStatistics:
    """Compiled Statistics from a platform."""

    platform_stats: list[StatisticResult]
    current_metadata: dict[str, tuple[int, StatisticMetaData]]


def split_statistic_id(entity_id: str) -> list[str]:
    """Split a state entity ID into domain and object ID."""
    return entity_id.split(":", 1)


VALID_STATISTIC_ID = re.compile(r"^(?!.+__)(?!_)[\da-z_]+(?<!_):(?!_)[\da-z_]+(?<!_)$")


def valid_statistic_id(statistic_id: str) -> bool:
    """Test if a statistic ID is a valid format.

    Format: <domain>:<statistic> where both are slugs.
    """
    return VALID_STATISTIC_ID.match(statistic_id) is not None


def validate_statistic_id(value: str) -> str:
    """Validate statistic ID."""
    if valid_statistic_id(value):
        return value

    raise vol.Invalid(f"Statistics ID {value} is an invalid statistic ID")


@dataclasses.dataclass
class ValidationIssue:
    """Error or warning message."""

    type: str
    data: dict[str, str | None] | None = None

    def as_dict(self) -> dict:
        """Return dictionary version."""
        return dataclasses.asdict(self)


def get_start_time() -> datetime:
    """Return start time."""
    now = dt_util.utcnow()
    current_period_minutes = now.minute - now.minute % 5
    current_period = now.replace(minute=current_period_minutes, second=0, microsecond=0)
    return current_period - timedelta(minutes=5)


def _compile_hourly_statistics_summary_mean_stmt(
    start_time_ts: float, end_time_ts: float
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
    start_time_ts: float, end_time_ts: float
) -> StatementLambdaElement:
    """Generate the summary mean statement for hourly statistics."""
    return lambda_stmt(
        lambda: select(
            subquery := (
                select(*QUERY_STATISTICS_SUMMARY_SUM)
                .filter(StatisticsShortTerm.start_ts >= start_time_ts)
                .filter(StatisticsShortTerm.start_ts < end_time_ts)
                .subquery()
            )
        )
        .filter(subquery.c.rownum == 1)
        .order_by(subquery.c.metadata_id)
    )


def _compile_hourly_statistics(session: Session, start: datetime) -> None:
    """Compile hourly statistics.

    This will summarize 5-minute statistics for one hour:
    - average, min max is computed by a database query
    - sum is taken from the last 5-minute entry during the hour
    """
    start_time = start.replace(minute=0)
    start_time_ts = start_time.timestamp()
    end_time = start_time + Statistics.duration
    end_time_ts = end_time.timestamp()

    # Compute last hour's average, min, max
    summary: dict[int, StatisticDataTimestamp] = {}
    stmt = _compile_hourly_statistics_summary_mean_stmt(start_time_ts, end_time_ts)
    stats = execute_stmt_lambda_element(session, stmt)

    if stats:
        for stat in stats:
            metadata_id, _mean, _min, _max = stat
            summary[metadata_id] = {
                "start_ts": start_time_ts,
                "mean": _mean,
                "min": _min,
                "max": _max,
            }

    stmt = _compile_hourly_statistics_last_sum_stmt(start_time_ts, end_time_ts)
    # Get last hour's last sum
    stats = execute_stmt_lambda_element(session, stmt)

    if stats:
        for stat in stats:
            metadata_id, start, last_reset_ts, state, _sum, _ = stat
            if metadata_id in summary:
                summary[metadata_id].update(
                    {
                        "last_reset_ts": last_reset_ts,
                        "state": state,
                        "sum": _sum,
                    }
                )
            else:
                summary[metadata_id] = {
                    "start_ts": start_time_ts,
                    "last_reset_ts": last_reset_ts,
                    "state": state,
                    "sum": _sum,
                }

    # Insert compiled hourly statistics in the database
    now_timestamp = time_time()
    session.add_all(
        Statistics.from_stats_ts(metadata_id, summary_item, now_timestamp)
        for metadata_id, summary_item in summary.items()
    )


@retryable_database_job("compile missing statistics")
def compile_missing_statistics(instance: Recorder) -> bool:
    """Compile missing statistics."""
    now = dt_util.utcnow()
    period_size = 5
    last_period_minutes = now.minute - now.minute % period_size
    last_period = now.replace(minute=last_period_minutes, second=0, microsecond=0)
    start = now - timedelta(days=instance.keep_days)
    start = start.replace(minute=0, second=0, microsecond=0)
    # Commit every 12 hours of data
    commit_interval = 60 / period_size * 12

    with session_scope(
        session=instance.get_session(),
        exception_filter=filter_unique_constraint_integrity_error(
            instance, "statistic"
        ),
    ) as session:
        # Find the newest statistics run, if any
        if last_run := session.query(func.max(StatisticsRuns.start)).scalar():
            start = max(
                start, process_timestamp(last_run) + StatisticsShortTerm.duration
            )

        periods_without_commit = 0
        while start < last_period:
            periods_without_commit += 1
            end = start + timedelta(minutes=period_size)
            _LOGGER.debug("Compiling missing statistics for %s-%s", start, end)
            modified_statistic_ids = _compile_statistics(
                instance, session, start, end >= last_period
            )
            if periods_without_commit == commit_interval or modified_statistic_ids:
                session.commit()
                session.expunge_all()
                periods_without_commit = 0
            start = end

    return True


@retryable_database_job("compile statistics")
def compile_statistics(instance: Recorder, start: datetime, fire_events: bool) -> bool:
    """Compile 5-minute statistics for all integrations with a recorder platform.

    The actual calculation is delegated to the platforms.
    """
    # Define modified_statistic_ids outside of the "with" statement as
    # _compile_statistics may raise and be trapped by
    # filter_unique_constraint_integrity_error which would make
    # modified_statistic_ids unbound.
    modified_statistic_ids: set[str] | None = None

    # Return if we already have 5-minute statistics for the requested period
    with session_scope(
        session=instance.get_session(),
        exception_filter=filter_unique_constraint_integrity_error(
            instance, "statistic"
        ),
    ) as session:
        modified_statistic_ids = _compile_statistics(
            instance, session, start, fire_events
        )

    if modified_statistic_ids:
        # In the rare case that we have modified statistic_ids, we reload the modified
        # statistics meta data into the cache in a fresh session to ensure that the
        # cache is up to date and future calls to get statistics meta data will
        # not have to hit the database again.
        with session_scope(session=instance.get_session(), read_only=True) as session:
            instance.statistics_meta_manager.get_many(session, modified_statistic_ids)

    return True


def _get_first_id_stmt(start: datetime) -> StatementLambdaElement:
    """Return a statement that returns the first run_id at start."""
    return lambda_stmt(lambda: select(StatisticsRuns.run_id).filter_by(start=start))


def _compile_statistics(
    instance: Recorder, session: Session, start: datetime, fire_events: bool
) -> set[str]:
    """Compile 5-minute statistics for all integrations with a recorder platform.

    This is a helper function for compile_statistics and compile_missing_statistics
    that does not retry on database errors since both callers already retry.

    returns a set of modified statistic_ids if any were modified.
    """
    assert start.tzinfo == dt_util.UTC, "start must be in UTC"
    end = start + StatisticsShortTerm.duration
    statistics_meta_manager = instance.statistics_meta_manager
    modified_statistic_ids: set[str] = set()

    # Return if we already have 5-minute statistics for the requested period
    if execute_stmt_lambda_element(session, _get_first_id_stmt(start)):
        _LOGGER.debug("Statistics already compiled for %s-%s", start, end)
        return modified_statistic_ids

    _LOGGER.debug("Compiling statistics for %s-%s", start, end)
    platform_stats: list[StatisticResult] = []
    current_metadata: dict[str, tuple[int, StatisticMetaData]] = {}
    # Collect statistics from all platforms implementing support
    for domain, platform in instance.hass.data[DOMAIN].recorder_platforms.items():
        if not (
            platform_compile_statistics := getattr(
                platform, INTEGRATION_PLATFORM_COMPILE_STATISTICS, None
            )
        ):
            continue
        compiled: PlatformCompiledStatistics = platform_compile_statistics(
            instance.hass, session, start, end
        )
        _LOGGER.debug(
            "Statistics for %s during %s-%s: %s",
            domain,
            start,
            end,
            compiled.platform_stats,
        )
        platform_stats.extend(compiled.platform_stats)
        current_metadata.update(compiled.current_metadata)

    new_short_term_stats: list[StatisticsBase] = []
    updated_metadata_ids: set[int] = set()
    now_timestamp = time_time()
    # Insert collected statistics in the database
    for stats in platform_stats:
        modified_statistic_id, metadata_id = statistics_meta_manager.update_or_add(
            session, stats["meta"], current_metadata
        )
        if modified_statistic_id is not None:
            modified_statistic_ids.add(modified_statistic_id)
        updated_metadata_ids.add(metadata_id)
        if new_stat := _insert_statistics(
            session, StatisticsShortTerm, metadata_id, stats["stat"], now_timestamp
        ):
            new_short_term_stats.append(new_stat)

    if start.minute == 50:
        # Once every hour, update issues
        for platform in instance.hass.data[DOMAIN].recorder_platforms.values():
            if not (
                platform_update_issues := getattr(
                    platform, INTEGRATION_PLATFORM_UPDATE_STATISTICS_ISSUES, None
                )
            ):
                continue
            platform_update_issues(instance.hass, session)

    if start.minute == 55:
        # A full hour is ready, summarize it
        _compile_hourly_statistics(session, start)

    session.add(StatisticsRuns(start=start))

    if fire_events:
        instance.hass.bus.fire(EVENT_RECORDER_5MIN_STATISTICS_GENERATED)
        if start.minute == 55:
            instance.hass.bus.fire(EVENT_RECORDER_HOURLY_STATISTICS_GENERATED)

    if updated_metadata_ids:
        # These are always the newest statistics, so we can update
        # the run cache without having to check the start_ts.
        session.flush()  # populate the ids of the new StatisticsShortTerm rows
        run_cache = get_short_term_statistics_run_cache(instance.hass)
        # metadata_id is typed to allow None, but we know it's not None here
        # so we can safely cast it to int.
        run_cache.set_latest_ids_for_metadata_ids(
            cast(
                dict[int, int],
                {
                    new_stat.metadata_id: new_stat.id
                    for new_stat in new_short_term_stats
                },
            )
        )

    return modified_statistic_ids


def _adjust_sum_statistics(
    session: Session,
    table: type[StatisticsBase],
    metadata_id: int,
    start_time: datetime,
    adj: float,
) -> None:
    """Adjust statistics in the database."""
    start_time_ts = start_time.timestamp()
    try:
        session.query(table).filter_by(metadata_id=metadata_id).filter(
            table.start_ts >= start_time_ts
        ).update(
            {
                table.sum: table.sum + adj,
            },
            synchronize_session=False,
        )
    except SQLAlchemyError:
        _LOGGER.exception(
            "Unexpected exception when updating statistics %s",
            id,
        )


def _insert_statistics(
    session: Session,
    table: type[StatisticsBase],
    metadata_id: int,
    statistic: StatisticData,
    now_timestamp: float,
) -> StatisticsBase | None:
    """Insert statistics in the database."""
    try:
        stat = table.from_stats(metadata_id, statistic, now_timestamp)
        session.add(stat)
    except SQLAlchemyError:
        _LOGGER.exception(
            "Unexpected exception when inserting statistics %s:%s ",
            metadata_id,
            statistic,
        )
        return None
    return stat


def _update_statistics(
    session: Session,
    table: type[StatisticsBase],
    stat_id: int,
    statistic: StatisticData,
) -> None:
    """Insert statistics in the database."""
    try:
        session.query(table).filter_by(id=stat_id).update(
            {
                table.mean: statistic.get("mean"),
                table.min: statistic.get("min"),
                table.max: statistic.get("max"),
                table.last_reset_ts: datetime_to_timestamp_or_none(
                    statistic.get("last_reset")
                ),
                table.state: statistic.get("state"),
                table.sum: statistic.get("sum"),
            },
            synchronize_session=False,
        )
    except SQLAlchemyError:
        _LOGGER.exception(
            "Unexpected exception when updating statistics %s:%s ",
            stat_id,
            statistic,
        )


def get_metadata_with_session(
    instance: Recorder,
    session: Session,
    *,
    statistic_ids: set[str] | None = None,
    statistic_type: Literal["mean", "sum"] | None = None,
    statistic_source: str | None = None,
) -> dict[str, tuple[int, StatisticMetaData]]:
    """Fetch meta data.

    Returns a dict of (metadata_id, StatisticMetaData) tuples indexed by statistic_id.
    If statistic_ids is given, fetch metadata only for the listed statistics_ids.
    If statistic_type is given, fetch metadata only for statistic_ids supporting it.
    """
    return instance.statistics_meta_manager.get_many(
        session,
        statistic_ids=statistic_ids,
        statistic_type=statistic_type,
        statistic_source=statistic_source,
    )


def get_metadata(
    hass: HomeAssistant,
    *,
    statistic_ids: set[str] | None = None,
    statistic_type: Literal["mean", "sum"] | None = None,
    statistic_source: str | None = None,
) -> dict[str, tuple[int, StatisticMetaData]]:
    """Return metadata for statistic_ids."""
    with session_scope(hass=hass, read_only=True) as session:
        return get_metadata_with_session(
            get_instance(hass),
            session,
            statistic_ids=statistic_ids,
            statistic_type=statistic_type,
            statistic_source=statistic_source,
        )


def clear_statistics(instance: Recorder, statistic_ids: list[str]) -> None:
    """Clear statistics for a list of statistic_ids."""
    with session_scope(session=instance.get_session()) as session:
        instance.statistics_meta_manager.delete(session, statistic_ids)


def update_statistics_metadata(
    instance: Recorder,
    statistic_id: str,
    new_statistic_id: str | None | UndefinedType,
    new_unit_of_measurement: str | None | UndefinedType,
) -> None:
    """Update statistics metadata for a statistic_id."""
    statistics_meta_manager = instance.statistics_meta_manager
    if new_unit_of_measurement is not UNDEFINED:
        with session_scope(session=instance.get_session()) as session:
            statistics_meta_manager.update_unit_of_measurement(
                session, statistic_id, new_unit_of_measurement
            )
    if new_statistic_id is not UNDEFINED and new_statistic_id is not None:
        with session_scope(
            session=instance.get_session(),
            exception_filter=filter_unique_constraint_integrity_error(
                instance, "statistic"
            ),
        ) as session:
            statistics_meta_manager.update_statistic_id(
                session, DOMAIN, statistic_id, new_statistic_id
            )


async def async_list_statistic_ids(
    hass: HomeAssistant,
    statistic_ids: set[str] | None = None,
    statistic_type: Literal["mean", "sum"] | None = None,
) -> list[dict]:
    """Return all statistic_ids (or filtered one) and unit of measurement.

    Queries the database for existing statistic_ids, as well as integrations with
    a recorder platform for statistic_ids which will be added in the next statistics
    period.
    """
    instance = get_instance(hass)

    if statistic_ids is not None:
        # Try to get the results from the cache since there is nearly
        # always a cache hit.
        statistics_meta_manager = instance.statistics_meta_manager
        metadata = statistics_meta_manager.get_from_cache_threadsafe(statistic_ids)
        if not statistic_ids.difference(metadata):
            result = _statistic_by_id_from_metadata(hass, metadata)
            return _flatten_list_statistic_ids_metadata_result(result)

    return await instance.async_add_executor_job(
        list_statistic_ids,
        hass,
        statistic_ids,
        statistic_type,
    )


def _statistic_by_id_from_metadata(
    hass: HomeAssistant,
    metadata: dict[str, tuple[int, StatisticMetaData]],
) -> dict[str, dict[str, Any]]:
    """Return a list of results for a given metadata dict."""
    return {
        meta["statistic_id"]: {
            "display_unit_of_measurement": get_display_unit(
                hass, meta["statistic_id"], meta["unit_of_measurement"]
            ),
            "has_mean": meta["has_mean"],
            "has_sum": meta["has_sum"],
            "name": meta["name"],
            "source": meta["source"],
            "unit_class": UNIT_CLASSES.get(meta["unit_of_measurement"]),
            "unit_of_measurement": meta["unit_of_measurement"],
        }
        for _, meta in metadata.values()
    }


def _flatten_list_statistic_ids_metadata_result(
    result: dict[str, dict[str, Any]],
) -> list[dict]:
    """Return a flat dict of metadata."""
    return [
        {
            "statistic_id": _id,
            "display_unit_of_measurement": info["display_unit_of_measurement"],
            "has_mean": info["has_mean"],
            "has_sum": info["has_sum"],
            "name": info.get("name"),
            "source": info["source"],
            "statistics_unit_of_measurement": info["unit_of_measurement"],
            "unit_class": info["unit_class"],
        }
        for _id, info in result.items()
    ]


def list_statistic_ids(
    hass: HomeAssistant,
    statistic_ids: set[str] | None = None,
    statistic_type: Literal["mean", "sum"] | None = None,
) -> list[dict]:
    """Return all statistic_ids (or filtered one) and unit of measurement.

    Queries the database for existing statistic_ids, as well as integrations with
    a recorder platform for statistic_ids which will be added in the next statistics
    period.
    """
    result = {}
    instance = get_instance(hass)
    statistics_meta_manager = instance.statistics_meta_manager

    # Query the database
    with session_scope(hass=hass, read_only=True) as session:
        metadata = statistics_meta_manager.get_many(
            session, statistic_type=statistic_type, statistic_ids=statistic_ids
        )
        result = _statistic_by_id_from_metadata(hass, metadata)

    if not statistic_ids or statistic_ids.difference(result):
        # If we want all statistic_ids, or some are missing, we need to query
        # the integrations for the missing ones.
        #
        # Query all integrations with a registered recorder platform
        for platform in hass.data[DOMAIN].recorder_platforms.values():
            if not (
                platform_list_statistic_ids := getattr(
                    platform, INTEGRATION_PLATFORM_LIST_STATISTIC_IDS, None
                )
            ):
                continue
            platform_statistic_ids = platform_list_statistic_ids(
                hass, statistic_ids=statistic_ids, statistic_type=statistic_type
            )

            for key, meta in platform_statistic_ids.items():
                if key in result:
                    # The database has a higher priority than the integration
                    continue
                result[key] = {
                    "display_unit_of_measurement": meta["unit_of_measurement"],
                    "has_mean": meta["has_mean"],
                    "has_sum": meta["has_sum"],
                    "name": meta["name"],
                    "source": meta["source"],
                    "unit_class": UNIT_CLASSES.get(meta["unit_of_measurement"]),
                    "unit_of_measurement": meta["unit_of_measurement"],
                }

    # Return a list of statistic_id + metadata
    return _flatten_list_statistic_ids_metadata_result(result)


def _reduce_statistics(
    stats: dict[str, list[StatisticsRow]],
    same_period: Callable[[float, float], bool],
    period_start_end: Callable[[float], tuple[float, float]],
    period: timedelta,
    types: set[Literal["last_reset", "max", "mean", "min", "state", "sum"]],
) -> dict[str, list[StatisticsRow]]:
    """Reduce hourly statistics to daily or monthly statistics."""
    result: dict[str, list[StatisticsRow]] = defaultdict(list)
    period_seconds = period.total_seconds()
    _want_mean = "mean" in types
    _want_min = "min" in types
    _want_max = "max" in types
    _want_last_reset = "last_reset" in types
    _want_state = "state" in types
    _want_sum = "sum" in types
    for statistic_id, stat_list in stats.items():
        max_values: list[float] = []
        mean_values: list[float] = []
        min_values: list[float] = []
        prev_stat: StatisticsRow = stat_list[0]
        fake_entry: StatisticsRow = {"start": stat_list[-1]["start"] + period_seconds}

        # Loop over the hourly statistics + a fake entry to end the period
        for statistic in chain(stat_list, (fake_entry,)):
            if not same_period(prev_stat["start"], statistic["start"]):
                start, end = period_start_end(prev_stat["start"])
                # The previous statistic was the last entry of the period
                row: StatisticsRow = {
                    "start": start,
                    "end": end,
                }
                if _want_mean:
                    row["mean"] = mean(mean_values) if mean_values else None
                    mean_values.clear()
                if _want_min:
                    row["min"] = min(min_values) if min_values else None
                    min_values.clear()
                if _want_max:
                    row["max"] = max(max_values) if max_values else None
                    max_values.clear()
                if _want_last_reset:
                    row["last_reset"] = prev_stat.get("last_reset")
                if _want_state:
                    row["state"] = prev_stat.get("state")
                if _want_sum:
                    row["sum"] = prev_stat["sum"]
                result[statistic_id].append(row)
            if _want_max and (_max := statistic.get("