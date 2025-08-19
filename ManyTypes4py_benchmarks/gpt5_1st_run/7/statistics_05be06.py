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
from typing import TYPE_CHECKING, Any, Final, Mapping, Optional, Pattern, Container, TypedDict, cast

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

QUERY_STATISTICS: tuple[Any, ...] = (
    Statistics.metadata_id,
    Statistics.start_ts,
    Statistics.mean,
    Statistics.min,
    Statistics.max,
    Statistics.last_reset_ts,
    Statistics.state,
    Statistics.sum,
)
QUERY_STATISTICS_SHORT_TERM: tuple[Any, ...] = (
    StatisticsShortTerm.metadata_id,
    StatisticsShortTerm.start_ts,
    StatisticsShortTerm.mean,
    StatisticsShortTerm.min,
    StatisticsShortTerm.max,
    StatisticsShortTerm.last_reset_ts,
    StatisticsShortTerm.state,
    StatisticsShortTerm.sum,
)
QUERY_STATISTICS_SUMMARY_MEAN: tuple[Any, ...] = (
    StatisticsShortTerm.metadata_id,
    func.avg(StatisticsShortTerm.mean),
    func.min(StatisticsShortTerm.min),
    func.max(StatisticsShortTerm.max),
)
QUERY_STATISTICS_SUMMARY_SUM: tuple[Any, ...] = (
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
STATISTIC_UNIT_TO_UNIT_CONVERTER: dict[str, type[BaseUnitConverter]] = {
    **{unit: AreaConverter for unit in AreaConverter.VALID_UNITS},
    **{
        unit: BloodGlucoseConcentrationConverter
        for unit in BloodGlucoseConcentrationConverter.VALID_UNITS
    },
    **{unit: ConductivityConverter for unit in ConductivityConverter.VALID_UNITS},
    **{unit: DataRateConverter for unit in DataRateConverter.VALID_UNITS},
    **{unit: DistanceConverter for unit in DistanceConverter.VALID_UNITS},
    **{unit: DurationConverter for unit in DurationConverter.VALID_UNITS},
    **{
        unit: ElectricCurrentConverter
        for unit in ElectricCurrentConverter.VALID_UNITS
    },
    **{
        unit: ElectricPotentialConverter
        for unit in ElectricPotentialConverter.VALID_UNITS
    },
    **{unit: EnergyConverter for unit in EnergyConverter.VALID_UNITS},
    **{
        unit: EnergyDistanceConverter
        for unit in EnergyDistanceConverter.VALID_UNITS
    },
    **{unit: InformationConverter for unit in InformationConverter.VALID_UNITS},
    **{unit: MassConverter for unit in MassConverter.VALID_UNITS},
    **{unit: PowerConverter for unit in PowerConverter.VALID_UNITS},
    **{unit: PressureConverter for unit in PressureConverter.VALID_UNITS},
    **{unit: SpeedConverter for unit in SpeedConverter.VALID_UNITS},
    **{unit: TemperatureConverter for unit in TemperatureConverter.VALID_UNITS},
    **{
        unit: UnitlessRatioConverter
        for unit in UnitlessRatioConverter.VALID_UNITS
    },
    **{unit: VolumeConverter for unit in VolumeConverter.VALID_UNITS},
    **{
        unit: VolumeFlowRateConverter
        for unit in VolumeFlowRateConverter.VALID_UNITS
    },
}
UNIT_CLASSES: dict[str, str] = {
    unit: converter.UNIT_CLASS for unit, converter in STATISTIC_UNIT_TO_UNIT_CONVERTER.items()
}
DATA_SHORT_TERM_STATISTICS_RUN_CACHE: Final[str] = "recorder_short_term_statistics_run_cache"


def mean(values: Sequence[float]) -> float:
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

    _latest_id_by_metadata_id: dict[int, int] = dataclasses.field(default_factory=dict)

    def get_latest_ids(self, metadata_ids: Container[int]) -> dict[int, int]:
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
        self, metadata_id_to_id: Mapping[int, int]
    ) -> None:
        """Cache the latest id for the each metadata_id."""
        self._latest_id_by_metadata_id.update(metadata_id_to_id)


class BaseStatisticsRow(TypedDict, total=False):
    """A processed row of statistic data."""


class StatisticsRow(BaseStatisticsRow, total=False):
    """A processed row of statistic data."""


def get_display_unit(
    hass: HomeAssistant, statistic_id: str, statistic_unit: Optional[str]
) -> Optional[str]:
    """Return the unit which the statistic will be displayed in."""
    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit or "")) is None:
        return statistic_unit
    state_unit: Optional[str] = statistic_unit
    if state := hass.states.get(statistic_id):
        state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if state_unit == statistic_unit or state_unit not in converter.VALID_UNITS:
        return statistic_unit
    return state_unit


def _get_statistic_to_display_unit_converter(
    statistic_unit: Optional[str],
    state_unit: Optional[str],
    requested_units: Optional[Mapping[str, str]],
    allow_none: bool = True,
) -> Optional[Callable[[Any], Any]]:
    """Prepare a converter from the statistics unit to display unit."""
    if (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit or "")) is None:
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
    display_unit: Optional[str], statistic_unit: Optional[str]
) -> Optional[Callable[[float], float]]:
    """Prepare a converter from the display unit to the statistics unit."""
    if (
        display_unit == statistic_unit
        or (converter := STATISTIC_UNIT_TO_UNIT_CONVERTER.get(statistic_unit or ""))
        is None
    ):
        return None
    return converter.converter_factory(from_unit=display_unit, to_unit=statistic_unit)


def _get_unit_converter(
    from_unit: str, to_unit: str
) -> Optional[Callable[[Optional[float]], Optional[float]]]:
    """Prepare a converter from a unit to another unit."""
    for conv in STATISTIC_UNIT_TO_UNIT_CONVERTER.values():
        if from_unit in conv.VALID_UNITS and to_unit in conv.VALID_UNITS:
            if from_unit == to_unit:
                return None
            return conv.converter_factory_allow_none(from_unit=from_unit, to_unit=to_unit)
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


def split_statistic_id(entity_id: str) -> tuple[str, str]:
    """Split a state entity ID into domain and object ID."""
    parts = entity_id.split(":", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


VALID_STATISTIC_ID: Pattern[str] = re.compile(
    r"^(?!.+__)(?!_)[\da-z_]+(?<!_):(?!_)[\da-z_]+(?<!_)$"
)


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

    data: Any = None

    def as_dict(self) -> dict[str, Any]:
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
            (
                subquery := select(*QUERY_STATISTICS_SUMMARY_SUM)
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
    summary: dict[int, dict[str, Any]] = {}
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
    stats = execute_stmt_lambda_element(session, stmt)
    if stats:
        for stat in stats:
            metadata_id, start_ts, last_reset_ts, state, _sum, _ = stat
            if metadata_id in summary:
                summary[metadata_id].update(
                    {"last_reset_ts": last_reset_ts, "state": state, "sum": _sum}
                )
            else:
                summary[metadata_id] = {
                    "start_ts": start_time_ts,
                    "last_reset_ts": last_reset_ts,
                    "state": state,
                    "sum": _sum,
                }
    now_timestamp = time_time()
    session.add_all(
        (
            Statistics.from_stats_ts(metadata_id, summary_item, now_timestamp)
            for metadata_id, summary_item in summary.items()
        )
    )


@retryable_database_job("compile missing statistics")
def compile_missing_statistics(instance: "Recorder") -> bool:
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
        exception_filter=filter_unique_constraint_integrity_error(
            instance, "statistic"
        ),
    ) as session:
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
def compile_statistics(
    instance: "Recorder", start: datetime, fire_events: bool
) -> bool:
    """Compile 5-minute statistics for all integrations with a recorder platform.

    The actual calculation is delegated to the platforms.
    """
    modified_statistic_ids: Optional[set[str]] = None
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
        with session_scope(session=instance.get_session(), read_only=True) as session:
            instance.statistics_meta_manager.get_many(session, modified_statistic_ids)
    return True


def _get_first_id_stmt(start: datetime) -> StatementLambdaElement:
    """Return a statement that returns the first run_id at start."""
    return lambda_stmt(lambda: select(StatisticsRuns.run_id).filter_by(start=start))


def _compile_statistics(
    instance: "Recorder", session: Session, start: datetime, fire_events: bool
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
    if execute_stmt_lambda_element(session, _get_first_id_stmt(start)):
        _LOGGER.debug("Statistics already compiled for %s-%s", start, end)
        return modified_statistic_ids
    _LOGGER.debug("Compiling statistics for %s-%s", start, end)
    platform_stats: list[dict[str, Any]] = []
    current_metadata: dict[str, Any] = {}
    for domain, platform in instance.hass.data[DOMAIN].recorder_platforms.items():
        if not (
            platform_compile_statistics := getattr(
                platform, INTEGRATION_PLATFORM_COMPILE_STATISTICS, None
            )
        ):
            continue
        compiled = platform_compile_statistics(instance.hass, session, start, end)
        _LOGGER.debug(
            "Statistics for %s during %s-%s: %s",
            domain,
            start,
            end,
            compiled.platform_stats,
        )
        platform_stats.extend(compiled.platform_stats)
        current_metadata.update(compiled.current_metadata)
    new_short_term_stats: list[StatisticsShortTerm] = []
    updated_metadata_ids: set[int] = set()
    now_timestamp = time_time()
    for stats in platform_stats:
        modified_statistic_id, metadata_id = statistics_meta_manager.update_or_add(
            session, stats["meta"], current_metadata
        )
        if modified_statistic_id is not None:
            modified_statistic_ids.add(modified_statistic_id)
        updated_metadata_ids.add(metadata_id)
        if new_stat := _insert_statistics(
            session,
            StatisticsShortTerm,
            metadata_id,
            stats["stat"],
            now_timestamp,
        ):
            new_short_term_stats.append(cast(StatisticsShortTerm, new_stat))
    if start.minute == 50:
        for platform in instance.hass.data[DOMAIN].recorder_platforms.values():
            if not (
                platform_update_issues := getattr(
                    platform, INTEGRATION_PLATFORM_UPDATE_STATISTICS_ISSUES, None
                )
            ):
                continue
            platform_update_issues(instance.hass, session)
    if start.minute == 55:
        _compile_hourly_statistics(session, start)
    session.add(StatisticsRuns(start=start))
    if fire_events:
        instance.hass.bus.fire(EVENT_RECORDER_5MIN_STATISTICS_GENERATED)
        if start.minute == 55:
            instance.hass.bus.fire(EVENT_RECORDER_HOURLY_STATISTICS_GENERATED)
    if updated_metadata_ids:
        session.flush()
        run_cache = get_short_term_statistics_run_cache(instance.hass)
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
        ).update({table.sum: table.sum + adj}, synchronize_session=False)
    except SQLAlchemyError:
        _LOGGER.exception("Unexpected exception when updating statistics %s", id)


def _insert_statistics(
    session: Session,
    table: type[StatisticsBase],
    metadata_id: int,
    statistic: Mapping[str, Any],
    now_timestamp: float,
) -> Optional[StatisticsBase]:
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
    session: Session, table: type[StatisticsBase], stat_id: int, statistic: Mapping[str, Any]
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
            "Unexpected exception when updating statistics %s:%s ", stat_id, statistic
        )


def get_metadata_with_session(
    instance: "Recorder",
    session: Session,
    *,
    statistic_ids: Optional[set[str]] = None,
    statistic_type: Optional[str] = None,
    statistic_source: Optional[str] = None,
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
    statistic_ids: Optional[set[str]] = None,
    statistic_type: Optional[str] = None,
    statistic_source: Optional[str] = None,
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


def clear_statistics(instance: "Recorder", statistic_ids: set[str]) -> None:
    """Clear statistics for a list of statistic_ids."""
    with session_scope(session=instance.get_session()) as session:
        instance.statistics_meta_manager.delete(session, statistic_ids)


def update_statistics_metadata(
    instance: "Recorder",
    statistic_id: str,
    new_statistic_id: str | None | UndefinedType,
    new_unit_of_measurement: str | UndefinedType,
) -> None:
    """Update statistics metadata for a statistic_id."""
    statistics_meta_manager = instance.statistics_meta_manager
    if new_unit_of_measurement is not UNDEFINED:
        with session_scope(session=instance.get_session()) as session:
            statistics_meta_manager.update_unit_of_measurement(
                session, statistic_id, cast(str, new_unit_of_measurement)
            )
    if new_statistic_id is not UNDEFINED and new_statistic_id is not None:
        with session_scope(
            session=instance.get_session(),
            exception_filter=filter_unique_constraint_integrity_error(
                instance, "statistic"
            ),
        ) as session:
            statistics_meta_manager.update_statistic_id(
                session, DOMAIN, statistic_id, cast(str, new_statistic_id)
            )


async def async_list_statistic_ids(
    hass: HomeAssistant,
    statistic_ids: Optional[set[str]] = None,
    statistic_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return all statistic_ids (or filtered one) and unit of measurement.

    Queries the database for existing statistic_ids, as well as integrations with
    a recorder platform for statistic_ids which will be added in the next statistics
    period.
    """
    instance = get_instance(hass)
    if statistic_ids is not None:
        statistics_meta_manager = instance.statistics_meta_manager
        metadata = statistics_meta_manager.get_from_cache_threadsafe(statistic_ids)
        if not statistic_ids.difference(metadata):
            result = _statistic_by_id_from_metadata(hass, metadata)
            return _flatten_list_statistic_ids_metadata_result(result)
    return await instance.async_add_executor_job(
        list_statistic_ids, hass, statistic_ids, statistic_type
    )


def _statistic_by_id_from_metadata(
    hass: HomeAssistant, metadata: dict[str, tuple[int, StatisticMetaData]]
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
    result: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
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
    statistic_ids: Optional[set[str]] = None,
    statistic_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return all statistic_ids (or filtered one) and unit of measurement.

    Queries the database for existing statistic_ids, as well as integrations with
    a recorder platform for statistic_ids which will be added in the next statistics
    period.
    """
    result: dict[str, dict[str, Any]] = {}
    instance = get_instance(hass)
    statistics_meta_manager = instance.statistics_meta_manager
    with session_scope(hass=hass, read_only=True) as session:
        metadata = statistics_meta_manager.get_many(
            session, statistic_type=statistic_type, statistic_ids=statistic_ids
        )
        result = _statistic_by_id_from_metadata(hass, metadata)
    if not statistic_ids or statistic_ids.difference(result):
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
    return _flatten_list_statistic_ids_metadata_result(result)


def _reduce_statistics(
    stats: Mapping[str, list[dict[str, Any]]],
    same_period: Callable[[float, float], bool],
    period_start_end: Callable[[float], tuple[float, float]],
    period: timedelta,
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Reduce hourly statistics to daily or monthly statistics."""
    result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
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
        prev_stat = stat_list[0]
        fake_entry: dict[str, Any] = {
            "start": cast(float, prev_stat["start"]) + (stat_list[-1]["start"] - prev_stat["start"])  # type: ignore[index]
        }
        fake_entry["start"] = stat_list[-1]["start"] + period_seconds
        for statistic in chain(stat_list, (fake_entry,)):
            if not same_period(prev_stat["start"], statistic["start"]):  # type: ignore[index]
                start, end = period_start_end(prev_stat["start"])  # type: ignore[index]
                row: dict[str, Any] = {"start": start, "end": end}
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
                    row["sum"] = prev_stat["sum"]  # type: ignore[index]
                result[statistic_id].append(row)
            if _want_max and (_max := statistic.get("max")) is not None:
                max_values.append(_max)
            if _want_mean and (_mean := statistic.get("mean")) is not None:
                mean_values.append(_mean)
            if _want_min and (_min := statistic.get("min")) is not None:
                min_values.append(_min)
            prev_stat = statistic
    return result


def reduce_day_ts_factory() -> tuple[
    Callable[[float, float], bool], Callable[[float], tuple[float, float]]
]:
    """Return functions to match same day and day start end."""
    _lower_bound = 0.0
    _upper_bound = 0.0
    _local_from_timestamp = partial(
        datetime.fromtimestamp, tz=dt_util.get_default_time_zone()
    )

    def _same_day_ts(time1: float, time2: float) -> bool:
        """Return True if time1 and time2 are in the same date."""
        nonlocal _lower_bound, _upper_bound
        if not _lower_bound <= time1 < _upper_bound:
            _lower_bound, _upper_bound = _day_start_end_ts_cached(time1)
        return _lower_bound <= time2 < _upper_bound

    def _day_start_end_ts(time: float) -> tuple[float, float]:
        """Return the start and end of the period (day) time is within."""
        start_local = _local_from_timestamp(time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (
            start_local.timestamp(),
            (start_local + timedelta(days=1)).timestamp(),
        )

    _day_start_end_ts_cached = lru_cache(maxsize=6)(_day_start_end_ts)
    return (_same_day_ts, _day_start_end_ts_cached)


def _reduce_statistics_per_day(
    stats: Mapping[str, list[dict[str, Any]]], types: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Reduce hourly statistics to daily statistics."""
    _same_day_ts, _day_start_end_ts = reduce_day_ts_factory()
    return _reduce_statistics(stats, _same_day_ts, _day_start_end_ts, timedelta(days=1), types)


def reduce_week_ts_factory() -> tuple[
    Callable[[float, float], bool], Callable[[float], tuple[float, float]]
]:
    """Return functions to match same week and week start end."""
    _lower_bound = 0.0
    _upper_bound = 0.0
    _local_from_timestamp = partial(
        datetime.fromtimestamp, tz=dt_util.get_default_time_zone()
    )

    def _same_week_ts(time1: float, time2: float) -> bool:
        """Return True if time1 and time2 are in the same year and week."""
        nonlocal _lower_bound, _upper_bound
        if not _lower_bound <= time1 < _upper_bound:
            _lower_bound, _upper_bound = _week_start_end_ts_cached(time1)
        return _lower_bound <= time2 < _upper_bound

    def _week_start_end_ts(time: float) -> tuple[float, float]:
        """Return the start and end of the period (week) time is within."""
        time_local = _local_from_timestamp(time)
        start_local = time_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=time_local.weekday())
        return (
            start_local.timestamp(),
            (start_local + timedelta(days=7)).timestamp(),
        )

    _week_start_end_ts_cached = lru_cache(maxsize=6)(_week_start_end_ts)
    return (_same_week_ts, _week_start_end_ts_cached)


def _reduce_statistics_per_week(
    stats: Mapping[str, list[dict[str, Any]]], types: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Reduce hourly statistics to weekly statistics."""
    _same_week_ts, _week_start_end_ts = reduce_week_ts_factory()
    return _reduce_statistics(
        stats, _same_week_ts, _week_start_end_ts, timedelta(days=7), types
    )


def _find_month_end_time(timestamp: datetime) -> datetime:
    """Return the end of the month (midnight at the first day of the next month)."""
    return (timestamp.replace(day=28) + timedelta(days=4)).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


def reduce_month_ts_factory() -> tuple[
    Callable[[float, float], bool], Callable[[float], tuple[float, float]]
]:
    """Return functions to match same month and month start end."""
    _lower_bound = 0.0
    _upper_bound = 0.0
    _local_from_timestamp = partial(
        datetime.fromtimestamp, tz=dt_util.get_default_time_zone()
    )

    def _same_month_ts(time1: float, time2: float) -> bool:
        """Return True if time1 and time2 are in the same year and month."""
        nonlocal _lower_bound, _upper_bound
        if not _lower_bound <= time1 < _upper_bound:
            _lower_bound, _upper_bound = _month_start_end_ts_cached(time1)
        return _lower_bound <= time2 < _upper_bound

    def _month_start_end_ts(time: float) -> tuple[float, float]:
        """Return the start and end of the period (month) time is within."""
        start_local = _local_from_timestamp(time).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end_local = _find_month_end_time(start_local)
        return (start_local.timestamp(), end_local.timestamp())

    _month_start_end_ts_cached = lru_cache(maxsize=6)(_month_start_end_ts)
    return (_same_month_ts, _month_start_end_ts_cached)


def _reduce_statistics_per_month(
    stats: Mapping[str, list[dict[str, Any]]], types: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Reduce hourly statistics to monthly statistics."""
    _same_month_ts, _month_start_end_ts = reduce_month_ts_factory()
    return _reduce_statistics(
        stats, _same_month_ts, _month_start_end_ts, timedelta(days=31), types
    )


def _generate_statistics_during_period_stmt(
    start_time: datetime,
    end_time: Optional[datetime],
    metadata_ids: Optional[Iterable[int]],
    table: type[StatisticsBase],
    types: set[str],
) -> StatementLambdaElement:
    """Prepare a database query for statistics during a given period.

    This prepares a lambda_stmt query, so we don't insert the parameters yet.
    """
    start_time_ts = start_time.timestamp()
    stmt = _generate_select_columns_for_types_stmt(table, types)
    stmt += lambda q: q.filter(table.start_ts >= start_time_ts)
    if end_time is not None:
        end_time_ts = end_time.timestamp()
        stmt += lambda q: q.filter(table.start_ts < end_time_ts)
    if metadata_ids:
        stmt += lambda q: q.filter(table.metadata_id.in_(metadata_ids))
    stmt += lambda q: q.order_by(table.metadata_id, table.start_ts)
    return stmt


def _generate_max_mean_min_statistic_in_sub_period_stmt(
    columns: Select,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    table: type[StatisticsBase],
    metadata_id: int,
) -> StatementLambdaElement:
    stmt = lambda_stmt(lambda: columns.filter(table.metadata_id == metadata_id))
    if start_time is not None:
        start_time_ts = start_time.timestamp()
        stmt += lambda q: q.filter(table.start_ts >= start_time_ts)
    if end_time is not None:
        end_time_ts = end_time.timestamp()
        stmt += lambda q: q.filter(table.start_ts < end_time_ts)
    return stmt


def _get_max_mean_min_statistic_in_sub_period(
    session: Session,
    result: dict[str, Any],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    table: type[StatisticsBase],
    types: set[str],
    metadata_id: int,
) -> None:
    """Return max, mean and min during the period."""
    columns: Select[Any] = select()
    if "max" in types:
        columns = columns.add_columns(func.max(table.max))
    if "mean" in types:
        columns = columns.add_columns(func.avg(table.mean))
        columns = columns.add_columns(func.count(table.mean))
    if "min" in types:
        columns = columns.add_columns(func.min(table.min))
    stmt = _generate_max_mean_min_statistic_in_sub_period_stmt(
        columns, start_time, end_time, table, metadata_id
    )
    stats = cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt))
    if not stats:
        return
    if "max" in types and (new_max := stats[0].max) is not None:
        old_max = result.get("max")
        result["max"] = max(new_max, old_max) if old_max is not None else new_max
    if "mean" in types and stats[0].avg is not None:
        duration = stats[0].count * table.duration.total_seconds()
        result["duration"] = result.get("duration", 0.0) + duration
        result["mean_acc"] = result.get("mean_acc", 0.0) + stats[0].avg * duration
    if "min" in types and (new_min := stats[0].min) is not None:
        old_min = result.get("min")
        result["min"] = min(new_min, old_min) if old_min is not None else new_min


def _get_max_mean_min_statistic(
    session: Session,
    head_start_time: Optional[datetime],
    head_end_time: Optional[datetime],
    main_start_time: Optional[datetime],
    main_end_time: Optional[datetime],
    tail_start_time: Optional[datetime],
    tail_end_time: Optional[datetime],
    tail_only: bool,
    metadata_id: int,
    types: set[str],
) -> dict[str, Optional[float]]:
    """Return max, mean and min during the period.

    The mean is a time weighted average, combining hourly and 5-minute statistics if
    necessary.
    """
    max_mean_min: dict[str, Any] = {}
    result: dict[str, Optional[float]] = {}
    if tail_start_time is not None:
        _get_max_mean_min_statistic_in_sub_period(
            session,
            max_mean_min,
            tail_start_time,
            tail_end_time,
            StatisticsShortTerm,
            types,
            metadata_id,
        )
    if not tail_only:
        _get_max_mean_min_statistic_in_sub_period(
            session,
            max_mean_min,
            main_start_time,
            main_end_time,
            Statistics,
            types,
            metadata_id,
        )
    if head_start_time is not None:
        _get_max_mean_min_statistic_in_sub_period(
            session,
            max_mean_min,
            head_start_time,
            head_end_time,
            StatisticsShortTerm,
            types,
            metadata_id,
        )
    if "max" in types:
        result["max"] = max_mean_min.get("max")
    if "mean" in types:
        if "mean_acc" not in max_mean_min:
            result["mean"] = None
        else:
            result["mean"] = cast(float, max_mean_min["mean_acc"]) / cast(
                float, max_mean_min["duration"]
            )
    if "min" in types:
        result["min"] = max_mean_min.get("min")
    return result


def _first_statistic(
    session: Session, table: type[StatisticsBase], metadata_id: int
) -> Optional[datetime]:
    """Return the date of the oldest statistic row for a given metadata id."""
    stmt = lambda_stmt(
        lambda: select(table.start_ts)
        .filter(table.metadata_id == metadata_id)
        .order_by(table.start_ts.asc())
        .limit(1)
    )
    if stats := cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt)):
        return dt_util.utc_from_timestamp(stats[0].start_ts)
    return None


def _last_statistic(
    session: Session, table: type[StatisticsBase], metadata_id: int
) -> Optional[datetime]:
    """Return the date of the newest statistic row for a given metadata id."""
    stmt = lambda_stmt(
        lambda: select(table.start_ts)
        .filter(table.metadata_id == metadata_id)
        .order_by(table.start_ts.desc())
        .limit(1)
    )
    if stats := cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt)):
        return dt_util.utc_from_timestamp(stats[0].start_ts)
    return None


def _get_oldest_sum_statistic(
    session: Session,
    head_start_time: Optional[datetime],
    main_start_time: Optional[datetime],
    tail_start_time: Optional[datetime],
    oldest_stat: Optional[datetime],
    oldest_5_min_stat: Optional[datetime],
    tail_only: bool,
    metadata_id: int,
) -> float:
    """Return the oldest non-NULL sum during the period."""

    def _get_oldest_sum_statistic_in_sub_period(
        session: Session, start_time: Optional[datetime], table: type[StatisticsBase], metadata_id: int
    ) -> Optional[float]:
        """Return the oldest non-NULL sum during the period."""
        stmt = lambda_stmt(
            lambda: select(table.sum)
            .filter(table.metadata_id == metadata_id)
            .filter(table.sum.is_not(None))
            .order_by(table.start_ts.asc())
            .limit(1)
        )
        if start_time is not None:
            start_time_local = start_time + table.duration - timedelta.resolution
            if table == StatisticsShortTerm:
                minutes = start_time_local.minute - start_time_local.minute % 5
                period = start_time_local.replace(
                    minute=minutes, second=0, microsecond=0
                )
            else:
                period = start_time_local.replace(minute=0, second=0, microsecond=0)
            prev_period = period - table.duration
            prev_period_ts = prev_period.timestamp()
            stmt += lambda q: q.filter(table.start_ts >= prev_period_ts)
        stats = cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt))
        return stats[0].sum if stats else None

    oldest_sum: Optional[float] = None
    if not tail_only and main_start_time is not None and (oldest_stat is not None):
        period = main_start_time.replace(minute=0, second=0, microsecond=0)
        prev_period = period - Statistics.duration
        if prev_period < oldest_stat:
            return 0.0
    if (
        head_start_time is not None
        and oldest_5_min_stat is not None
        and (oldest_stat is None or oldest_5_min_stat < oldest_stat or oldest_5_min_stat <= head_start_time)
        and (
            (oldest_sum := _get_oldest_sum_statistic_in_sub_period(
                session, head_start_time, StatisticsShortTerm, metadata_id
            ))
            is not None
        )
    ):
        return oldest_sum
    if not tail_only:
        oldest_sum = _get_oldest_sum_statistic_in_sub_period(
            session, main_start_time, Statistics, metadata_id
        )
        if oldest_sum is not None:
            return oldest_sum
        return 0.0
    if (
        tail_start_time is not None
        and (
            oldest_sum := _get_oldest_sum_statistic_in_sub_period(
                session, tail_start_time, StatisticsShortTerm, metadata_id
            )
        )
        is not None
    ):
        return oldest_sum
    return 0.0


def _get_newest_sum_statistic(
    session: Session,
    head_start_time: Optional[datetime],
    head_end_time: Optional[datetime],
    main_start_time: Optional[datetime],
    main_end_time: Optional[datetime],
    tail_start_time: Optional[datetime],
    tail_end_time: Optional[datetime],
    tail_only: bool,
    metadata_id: int,
) -> Optional[float]:
    """Return the newest non-NULL sum during the period."""

    def _get_newest_sum_statistic_in_sub_period(
        session: Session,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        table: type[StatisticsBase],
        metadata_id: int,
    ) -> Optional[float]:
        """Return the newest non-NULL sum during the period."""
        stmt = lambda_stmt(
            lambda: select(table.sum)
            .filter(table.metadata_id == metadata_id)
            .filter(table.sum.is_not(None))
            .order_by(table.start_ts.desc())
            .limit(1)
        )
        if start_time is not None:
            start_time_ts = start_time.timestamp()
            stmt += lambda q: q.filter(table.start_ts >= start_time_ts)
        if end_time is not None:
            end_time_ts = end_time.timestamp()
            stmt += lambda q: q.filter(table.start_ts < end_time_ts)
        stats = cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt))
        return stats[0].sum if stats else None

    newest_sum: Optional[float] = None
    if tail_start_time is not None:
        newest_sum = _get_newest_sum_statistic_in_sub_period(
            session, tail_start_time, tail_end_time, StatisticsShortTerm, metadata_id
        )
        if newest_sum is not None:
            return newest_sum
    if not tail_only:
        newest_sum = _get_newest_sum_statistic_in_sub_period(
            session, main_start_time, main_end_time, Statistics, metadata_id
        )
        if newest_sum is not None:
            return newest_sum
    if head_start_time is not None:
        newest_sum = _get_newest_sum_statistic_in_sub_period(
            session, head_start_time, head_end_time, StatisticsShortTerm, metadata_id
        )
    return newest_sum


def statistic_during_period(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime],
    statistic_id: str,
    types: set[str],
    units: Optional[Mapping[str, str]],
) -> dict[str, Optional[float]]:
    """Return a statistic data point for the UTC period start_time - end_time."""
    metadata: Optional[tuple[int, StatisticMetaData]] = None
    if not types:
        types = {"max", "mean", "min", "change"}
    result: dict[str, Optional[float]] = {}
    with session_scope(hass=hass, read_only=True) as session:
        if not (metadata := get_instance(hass).statistics_meta_manager.get(session, statistic_id)):
            return result
        metadata_id = metadata[0]
        oldest_stat = _first_statistic(session, Statistics, metadata_id)
        oldest_5_min_stat: Optional[datetime] = None
        if not valid_statistic_id(statistic_id):
            oldest_5_min_stat = _first_statistic(session, StatisticsShortTerm, metadata_id)
        now = dt_util.utcnow()
        if end_time is not None and end_time > now:
            end_time = now
        tail_only = (
            start_time is not None
            and end_time is not None
            and (end_time - start_time < Statistics.duration)
        )
        head_start_time: Optional[datetime] = None
        head_end_time: Optional[datetime] = None
        if (
            not tail_only
            and oldest_stat is not None
            and (oldest_5_min_stat is not None)
            and (oldest_5_min_stat - oldest_stat < Statistics.duration)
            and (start_time is None or start_time < oldest_5_min_stat)
        ):
            head_start_time = oldest_5_min_stat
            head_end_time = oldest_5_min_stat.replace(
                minute=0, second=0, microsecond=0
            ) + Statistics.duration
        elif not tail_only and start_time is not None and start_time.minute:
            head_start_time = start_time
            head_end_time = start_time.replace(minute=0, second=0, microsecond=0) + Statistics.duration
        tail_start_time: Optional[datetime] = None
        tail_end_time: Optional[datetime] = None
        if end_time is None:
            tail_start_time = _last_statistic(session, Statistics, metadata_id)
            if tail_start_time:
                tail_start_time += Statistics.duration
            else:
                tail_start_time = now.replace(minute=0, second=0, microsecond=0)
        elif tail_only:
            tail_start_time = start_time
            tail_end_time = end_time
        elif end_time.minute:
            tail_start_time = end_time.replace(minute=0, second=0, microsecond=0)
            tail_end_time = end_time
        main_start_time: Optional[datetime] = None
        main_end_time: Optional[datetime] = None
        if not tail_only:
            main_start_time = start_time if head_end_time is None else head_end_time
            main_end_time = end_time if tail_start_time is None else tail_start_time
        if not types.isdisjoint({"max", "mean", "min"}):
            result = _get_max_mean_min_statistic(
                session,
                head_start_time,
                head_end_time,
                main_start_time,
                main_end_time,
                tail_start_time,
                tail_end_time,
                tail_only,
                metadata_id,
                types,
            )
        if "change" in types:
            if start_time is None:
                oldest_sum = 0.0
            else:
                oldest_sum = _get_oldest_sum_statistic(
                    session,
                    head_start_time,
                    main_start_time,
                    tail_start_time,
                    oldest_stat,
                    oldest_5_min_stat,
                    tail_only,
                    metadata_id,
                )
            newest_sum = _get_newest_sum_statistic(
                session,
                head_start_time,
                head_end_time,
                main_start_time,
                main_end_time,
                tail_start_time,
                tail_end_time,
                tail_only,
                metadata_id,
            )
            if oldest_sum is not None and newest_sum is not None:
                result["change"] = newest_sum - oldest_sum
            else:
                result["change"] = None
    state_unit = unit = metadata[1]["unit_of_measurement"]
    if state := hass.states.get(statistic_id):
        state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    convert = _get_statistic_to_display_unit_converter(unit, state_unit, units)
    if not convert:
        return result
    return {key: cast(Optional[float], convert(value)) for key, value in result.items()}


_type_column_mapping: dict[str, str] = {
    "last_reset": "last_reset_ts",
    "max": "max",
    "mean": "mean",
    "min": "min",
    "state": "state",
    "sum": "sum",
}


def _generate_select_columns_for_types_stmt(
    table: type[StatisticsBase], types: set[str]
) -> StatementLambdaElement:
    columns: Select[Any] = select(table.metadata_id, table.start_ts)
    track_on: list[Optional[str]] = [table.__tablename__]
    for key, column in _type_column_mapping.items():
        if key in types:
            columns = columns.add_columns(getattr(table, column))
            track_on.append(column)
        else:
            track_on.append(None)
    return lambda_stmt(lambda: columns, track_on=track_on)


def _extract_metadata_and_discard_impossible_columns(
    metadata: dict[str, tuple[int, StatisticMetaData]], types: set[str]
) -> list[int]:
    """Extract metadata ids from metadata and discard impossible columns."""
    metadata_ids: list[int] = []
    has_mean = False
    has_sum = False
    for metadata_id, stats_metadata in metadata.values():
        metadata_ids.append(metadata_id)
        has_mean |= stats_metadata["has_mean"]
        has_sum |= stats_metadata["has_sum"]
    if not has_mean:
        types.discard("mean")
        types.discard("min")
        types.discard("max")
    if not has_sum:
        types.discard("sum")
        types.discard("state")
    return metadata_ids


def _augment_result_with_change(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    units: Optional[Mapping[str, str]],
    _types: set[str],
    table: type[StatisticsBase],
    metadata: dict[str, tuple[int, StatisticMetaData]],
    result: dict[str, list[dict[str, Any]]],
) -> None:
    """Add change to the result."""
    drop_sum = "sum" not in _types
    prev_sums: dict[str, float] = {}
    if tmp := _statistics_at_time(
        session,
        {metadata[statistic_id][0] for statistic_id in result},
        table,
        start_time,
        {"sum"},
    ):
        _metadata = dict(metadata.values())
        for row in tmp:
            metadata_by_id = _metadata[row.metadata_id]
            statistic_id = metadata_by_id["statistic_id"]
            state_unit = unit = metadata_by_id["unit_of_measurement"]
            if state := hass.states.get(statistic_id):
                state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
            convert = _get_statistic_to_display_unit_converter(unit, state_unit, units)
            if convert is not None:
                prev_sums[statistic_id] = cast(float, convert(row.sum))
            else:
                prev_sums[statistic_id] = row.sum
    for statistic_id, rows in result.items():
        prev_sum = prev_sums.get(statistic_id) or 0
        for statistics_row in rows:
            if "sum" not in statistics_row:
                continue
            if drop_sum:
                _sum = statistics_row.pop("sum")
            else:
                _sum = statistics_row["sum"]
            if _sum is None:
                statistics_row["change"] = None
                continue
            statistics_row["change"] = _sum - prev_sum
            prev_sum = _sum


def _statistics_during_period_with_session(
    hass: HomeAssistant,
    session: Session,
    start_time: datetime,
    end_time: Optional[datetime],
    statistic_ids: Optional[set[str]],
    period: str,
    units: Optional[Mapping[str, str]],
    _types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return statistic data points during UTC period start_time - end_time.

    If end_time is omitted, returns statistics newer than or equal to start_time.
    If statistic_ids is omitted, returns statistics for all statistics ids.
    """
    if statistic_ids is not None and (not isinstance(statistic_ids, set)):
        statistic_ids = set(statistic_ids)
    metadata = get_instance(hass).statistics_meta_manager.get_many(
        session, statistic_ids=statistic_ids
    )
    if not metadata:
        return {}
    types: set[str] = set()
    for stat_type in _types:
        if stat_type == "change":
            types.add("sum")
            continue
        types.add(stat_type)
    metadata_ids: Optional[list[int]] = None
    if statistic_ids is not None:
        metadata_ids = _extract_metadata_and_discard_impossible_columns(metadata, types)
    if period == "day":
        start_time = dt_util.as_local(start_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_time = start_time.replace()
        if end_time is not None:
            end_local = dt_util.as_local(end_time)
            end_time = end_local.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
    elif period == "week":
        start_local = dt_util.as_local(start_time)
        start_time = start_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=start_local.weekday())
        if end_time is not None:
            end_local = dt_util.as_local(end_time)
            end_time = end_local.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=end_local.weekday()) + timedelta(days=7)
    elif period == "month":
        start_time = dt_util.as_local(start_time).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        if end_time is not None:
            end_time = _find_month_end_time(dt_util.as_local(end_time))
    table: type[StatisticsBase] = Statistics if period != "5minute" else StatisticsShortTerm
    stmt = _generate_statistics_during_period_stmt(
        start_time, end_time, metadata_ids, table, types
    )
    stats = cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt, orm_rows=False))
    if not stats:
        return {}
    result = _sorted_statistics_to_dict(
        hass, stats, statistic_ids, metadata, True, table, units, types
    )
    if period == "day":
        result = _reduce_statistics_per_day(result, types)
    if period == "week":
        result = _reduce_statistics_per_week(result, types)
    if period == "month":
        result = _reduce_statistics_per_month(result, types)
    if "change" in _types:
        _augment_result_with_change(
            hass, session, start_time, units, _types, table, metadata, result
        )
    return result


def statistics_during_period(
    hass: HomeAssistant,
    start_time: datetime,
    end_time: Optional[datetime],
    statistic_ids: Optional[set[str]],
    period: str,
    units: Optional[Mapping[str, str]],
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return statistic data points during UTC period start_time - end_time.

    If end_time is omitted, returns statistics newer than or equal to start_time.
    If statistic_ids is omitted, returns statistics for all statistics ids.
    """
    with session_scope(hass=hass, read_only=True) as session:
        return _statistics_during_period_with_session(
            hass, session, start_time, end_time, statistic_ids, period, units, types
        )


def _get_last_statistics_stmt(
    metadata_id: int, number_of_stats: int
) -> StatementLambdaElement:
    """Generate a statement for number_of_stats statistics for a given statistic_id."""
    return lambda_stmt(
        lambda: select(*QUERY_STATISTICS)
        .filter_by(metadata_id=metadata_id)
        .order_by(Statistics.metadata_id, Statistics.start_ts.desc())
        .limit(number_of_stats)
    )


def _get_last_statistics_short_term_stmt(
    metadata_id: int, number_of_stats: int
) -> StatementLambdaElement:
    """Generate a statement for number_of_stats short term statistics.

    For a given statistic_id.
    """
    return lambda_stmt(
        lambda: select(*QUERY_STATISTICS_SHORT_TERM)
        .filter_by(metadata_id=metadata_id)
        .order_by(StatisticsShortTerm.metadata_id, StatisticsShortTerm.start_ts.desc())
        .limit(number_of_stats)
    )


def _get_last_statistics(
    hass: HomeAssistant,
    number_of_stats: int,
    statistic_id: str,
    convert_units: bool,
    table: type[StatisticsBase],
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return the last number_of_stats statistics for a given statistic_id."""
    statistic_ids: set[str] = {statistic_id}
    with session_scope(hass=hass, read_only=True) as session:
        metadata = get_instance(hass).statistics_meta_manager.get_many(
            session, statistic_ids=statistic_ids
        )
        if not metadata:
            return {}
        metadata_ids = _extract_metadata_and_discard_impossible_columns(metadata, types)
        metadata_id = metadata_ids[0]
        if table == Statistics:
            stmt = _get_last_statistics_stmt(metadata_id, number_of_stats)
        else:
            stmt = _get_last_statistics_short_term_stmt(metadata_id, number_of_stats)
        stats = cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt, orm_rows=False))
        if not stats:
            return {}
        return _sorted_statistics_to_dict(
            hass, stats, statistic_ids, metadata, convert_units, table, None, types
        )


def get_last_statistics(
    hass: HomeAssistant,
    number_of_stats: int,
    statistic_id: str,
    convert_units: bool,
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return the last number_of_stats statistics for a statistic_id."""
    return _get_last_statistics(
        hass, number_of_stats, statistic_id, convert_units, Statistics, types
    )


def get_last_short_term_statistics(
    hass: HomeAssistant,
    number_of_stats: int,
    statistic_id: str,
    convert_units: bool,
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return the last number_of_stats short term statistics for a statistic_id."""
    return _get_last_statistics(
        hass, number_of_stats, statistic_id, convert_units, StatisticsShortTerm, types
    )


def get_latest_short_term_statistics_by_ids(
    session: Session, ids: Iterable[int]
) -> list[Row[Any]]:
    """Return the latest short term statistics for a list of ids."""
    stmt = _latest_short_term_statistics_by_ids_stmt(ids)
    return list(cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt)))


def _latest_short_term_statistics_by_ids_stmt(
    ids: Iterable[int],
) -> StatementLambdaElement:
    """Create the statement for finding the latest short term stat rows by id."""
    return lambda_stmt(
        lambda: select(*QUERY_STATISTICS_SHORT_TERM).filter(
            StatisticsShortTerm.id.in_(ids)
        )
    )


def get_latest_short_term_statistics_with_session(
    hass: HomeAssistant,
    session: Session,
    statistic_ids: set[str],
    types: set[str],
    metadata: Optional[dict[str, tuple[int, StatisticMetaData]]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return the latest short term statistics for a list of statistic_ids with a session."""
    if not metadata:
        metadata = get_instance(hass).statistics_meta_manager.get_many(
            session, statistic_ids=statistic_ids
        )
    if not metadata:
        return {}
    metadata_ids = set(_extract_metadata_and_discard_impossible_columns(metadata, types))
    run_cache = get_short_term_statistics_run_cache(hass)
    stats: list[Row[Any]] = []
    if metadata_id_to_id := run_cache.get_latest_ids(metadata_ids):
        stats = get_latest_short_term_statistics_by_ids(
            session, metadata_id_to_id.values()
        )
    if (missing_metadata_ids := (metadata_ids - set(metadata_id_to_id))) and (
        found_latest_ids := {
            latest_id
            for metadata_id in missing_metadata_ids
            if (latest_id := cache_latest_short_term_statistic_id_for_metadata_id(run_cache, session, metadata_id))
            is not None
        }
    ):
        stats.extend(get_latest_short_term_statistics_by_ids(session, found_latest_ids))
    if not stats:
        return {}
    return _sorted_statistics_to_dict(
        hass, stats, statistic_ids, metadata, False, StatisticsShortTerm, None, types
    )


def _generate_statistics_at_time_stmt(
    table: type[StatisticsBase],
    metadata_ids: Iterable[int],
    start_time_ts: float,
    types: set[str],
) -> StatementLambdaElement:
    """Create the statement for finding the statistics for a given time."""
    stmt = _generate_select_columns_for_types_stmt(table, types)
    stmt += lambda q: q.select_from(StatisticsMeta).join(
        table,
        and_(
            table.start_ts
            == select(table.start_ts)
            .where(
                (StatisticsMeta.id == table.metadata_id)
                & (table.start_ts < start_time_ts)
            )
            .order_by(table.start_ts.desc())
            .limit(1)
            .scalar_subquery()
            .correlate(StatisticsMeta),
            table.metadata_id == StatisticsMeta.id,
        ),
    ).where(table.metadata_id.in_(metadata_ids))
    return stmt


def _statistics_at_time(
    session: Session,
    metadata_ids: Iterable[int],
    table: type[StatisticsBase],
    start_time: datetime,
    types: set[str],
) -> Sequence[Row[Any]]:
    """Return last known statistics, earlier than start_time, for the metadata_ids."""
    start_time_ts = start_time.timestamp()
    stmt = _generate_statistics_at_time_stmt(table, metadata_ids, start_time_ts, types)
    return cast(Sequence[Row[Any]], execute_stmt_lambda_element(session, stmt))


def _build_sum_converted_stats(
    db_rows: Sequence[Row[Any]],
    table_duration_seconds: float,
    start_ts_idx: int,
    sum_idx: int,
    convert: Callable[[float], float],
) -> list[dict[str, Any]]:
    """Build a list of sum statistics."""
    return [
        {
            "start": (start_ts := db_row[start_ts_idx]),
            "end": start_ts + table_duration_seconds,
            "sum": None if (v := db_row[sum_idx]) is None else convert(v),
        }
        for db_row in db_rows
    ]


def _build_sum_stats(
    db_rows: Sequence[Row[Any]],
    table_duration_seconds: float,
    start_ts_idx: int,
    sum_idx: int,
) -> list[dict[str, Any]]:
    """Build a list of sum statistics."""
    return [
        {
            "start": (start_ts := db_row[start_ts_idx]),
            "end": start_ts + table_duration_seconds,
            "sum": db_row[sum_idx],
        }
        for db_row in db_rows
    ]


def _build_stats(
    db_rows: Sequence[Row[Any]],
    table_duration_seconds: float,
    start_ts_idx: int,
    row_mapping: tuple[tuple[str, int], ...],
) -> list[dict[str, Any]]:
    """Build a list of statistics without unit conversion."""
    return [
        {
            "start": (start_ts := db_row[start_ts_idx]),
            "end": start_ts + table_duration_seconds,
            **{key: db_row[idx] for key, idx in row_mapping},
        }
        for db_row in db_rows
    ]


def _build_converted_stats(
    db_rows: Sequence[Row[Any]],
    table_duration_seconds: float,
    start_ts_idx: int,
    row_mapping: tuple[tuple[str, int], ...],
    convert: Callable[[float], float],
) -> list[dict[str, Any]]:
    """Build a list of statistics with unit conversion."""
    return [
        {
            "start": (start_ts := db_row[start_ts_idx]),
            "end": start_ts + table_duration_seconds,
            **{
                key: None if (v := db_row[idx]) is None else convert(v)
                for key, idx in row_mapping
            },
        }
        for db_row in db_rows
    ]


def _sorted_statistics_to_dict(
    hass: HomeAssistant,
    stats: Sequence[Row[Any]],
    statistic_ids: Optional[set[str]],
    _metadata: dict[str, tuple[int, StatisticMetaData]],
    convert_units: bool,
    table: type[StatisticsBase],
    units: Optional[Mapping[str, str]],
    types: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Convert SQL results into JSON friendly data structure."""
    assert stats, "stats must not be empty"
    result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[int, StatisticMetaData] = dict(_metadata.values())
    field_map = {idx_key: idx for idx, idx_key in enumerate(stats[0]._fields)}  # type: ignore[attr-defined]
    metadata_id_idx = field_map["metadata_id"]
    start_ts_idx = field_map["start_ts"]
    stats_by_meta_id: dict[int, list[Row[Any]]] = {}
    seen_statistic_ids: set[str] = set()
    key_func = itemgetter(metadata_id_idx)
    for meta_id, group in groupby(stats, key_func):
        stats_by_meta_id[meta_id] = list(group)
        seen_statistic_ids.add(metadata[meta_id]["statistic_id"])
    if statistic_ids is not None:
        for stat_id in statistic_ids:
            if stat_id in seen_statistic_ids:
                result[stat_id] = []
    if "last_reset_ts" in field_map:
        field_map["last_reset"] = field_map.pop("last_reset_ts")
    sum_idx = field_map["sum"] if "sum" in types else None
    sum_only = len(types) == 1 and sum_idx is not None
    row_mapping: tuple[tuple[str, int], ...] = tuple(
        ((key, field_map[key]) for key in types if key in field_map)
    )
    table_duration_seconds = table.duration.total_seconds()
    for meta_id, db_rows in stats_by_meta_id.items():
        metadata_by_id = metadata[meta_id]
        statistic_id = metadata_by_id["statistic_id"]
        if convert_units:
            state_unit = unit = metadata_by_id["unit_of_measurement"]
            if state := hass.states.get(statistic_id):
                state_unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
            convert = _get_statistic_to_display_unit_converter(
                unit, state_unit, units, allow_none=False
            )
        else:
            convert = None
        build_args = (db_rows, table_duration_seconds, start_ts_idx)
        if sum_only:
            assert sum_idx is not None
            if convert:
                _stats = _build_sum_converted_stats(
                    *build_args, sum_idx, cast(Callable[[float], float], convert)
                )
            else:
                _stats = _build_sum_stats(*build_args, sum_idx)
        elif convert:
            _stats = _build_converted_stats(
                *build_args, row_mapping, cast(Callable[[float], float], convert)
            )
        else:
            _stats = _build_stats(*build_args, row_mapping)
        result[statistic_id] = _stats
    return result


def validate_statistics(hass: HomeAssistant) -> dict[str, Any]:
    """Validate statistics."""
    platform_validation: dict[str, Any] = {}
    for platform in hass.data[DOMAIN].recorder_platforms.values():
        if platform_validate_statistics := getattr(
            platform, INTEGRATION_PLATFORM_VALIDATE_STATISTICS, None
        ):
            platform_validation.update(platform_validate_statistics(hass))
    return platform_validation


def update_statistics_issues(hass: HomeAssistant) -> None:
    """Update statistics issues."""
    with session_scope(hass=hass, read_only=True) as session:
        for platform in hass.data[DOMAIN].recorder_platforms.values():
            if platform_update_statistics_issues := getattr(
                platform, INTEGRATION_PLATFORM_UPDATE_STATISTICS_ISSUES, None
            ):
                platform_update_statistics_issues(hass, session)


def _statistics_exists(
    session: Session, table: type[StatisticsBase], metadata_id: int, start: datetime
) -> Optional[int]:
    """Return id if a statistics entry already exists."""
    start_ts = start.timestamp()
    result = (
        session.query(table.id)
        .filter((table.metadata_id == metadata_id) & (table.start_ts == start_ts))
        .first()
    )
    return result.id if result else None


@callback
def _async_import_statistics(
    hass: HomeAssistant, metadata: StatisticMetaData, statistics: list[StatisticData]
) -> None:
    """Validate timestamps and insert an import_statistics job in the queue."""
    for statistic in statistics:
        start = statistic["start"]
        if start.tzinfo is None or start.tzinfo.utcoffset(start) is None:
            raise HomeAssistantError(
                "Naive timestamp: no or invalid timezone info provided"
            )
        if start.minute != 0 or start.second != 0 or start.microsecond != 0:
            raise HomeAssistantError(
                "Invalid timestamp: timestamps must be from the top of the hour (minutes and seconds = 0)"
            )
        statistic["start"] = dt_util.as_utc(start)
        if "last_reset" in statistic and statistic["last_reset"] is not None:
            last_reset = statistic["last_reset"]
            if last_reset.tzinfo is None or last_reset.tzinfo.utcoffset(last_reset) is None:
                raise HomeAssistantError("Naive timestamp")
            statistic["last_reset"] = dt_util.as_utc(last_reset)
    get_instance(hass).async_import_statistics(metadata, statistics, Statistics)


@callback
def async_import_statistics(
    hass: HomeAssistant, metadata: StatisticMetaData, statistics: list[StatisticData]
) -> None:
    """Import hourly statistics from an internal source.

    This inserts an import_statistics job in the recorder's queue.
    """
    if not valid_entity_id(metadata["statistic_id"]):
        raise HomeAssistantError("Invalid statistic_id")
    if not metadata["source"] or metadata["source"] != DOMAIN:
        raise HomeAssistantError("Invalid source")
    _async_import_statistics(hass, metadata, statistics)


@callback
def async_add_external_statistics(
    hass: HomeAssistant, metadata: StatisticMetaData, statistics: list[StatisticData]
) -> None:
    """Add hourly statistics from an external source.

    This inserts an import_statistics job in the recorder's queue.
    """
    if not valid_statistic_id(metadata["statistic_id"]):
        raise HomeAssistantError("Invalid statistic_id")
    domain, _object_id = split_statistic_id(metadata["statistic_id"])
    if not metadata["source"] or metadata["source"] != domain:
        raise HomeAssistantError("Invalid source")
    _async_import_statistics(hass, metadata, statistics)


def _import_statistics_with_session(
    instance: "Recorder",
    session: Session,
    metadata: StatisticMetaData,
    statistics: list[StatisticData],
    table: type[StatisticsBase],
) -> bool:
    """Import statistics to the database."""
    statistics_meta_manager = instance.statistics_meta_manager
    old_metadata_dict = statistics_meta_manager.get_many(
        session, statistic_ids={metadata["statistic_id"]}
    )
    _, metadata_id = statistics_meta_manager.update_or_add(
        session, metadata, old_metadata_dict
    )
    now_timestamp = time_time()
    for stat in statistics:
        if stat_id := _statistics_exists(session, table, metadata_id, stat["start"]):
            _update_statistics(session, table, stat_id, stat)
        else:
            _insert_statistics(session, table, metadata_id, stat, now_timestamp)
    if table != StatisticsShortTerm:
        return True
    run_cache = get_short_term_statistics_run_cache(instance.hass)
    cache_latest_short_term_statistic_id_for_metadata_id(
        run_cache, session, metadata_id
    )
    return True


@singleton(DATA_SHORT_TERM_STATISTICS_RUN_CACHE)
def get_short_term_statistics_run_cache(
    hass: HomeAssistant,
) -> ShortTermStatisticsRunCache:
    """Get the short term statistics run cache."""
    return ShortTermStatisticsRunCache()


def cache_latest_short_term_statistic_id_for_metadata_id(
    run_cache: ShortTermStatisticsRunCache, session: Session, metadata_id: int
) -> Optional[int]:
    """Cache the latest short term statistic for a given metadata_id.

    Returns the id of the latest short term statistic for the metadata_id
    that was added to the cache, or None if no latest short term statistic
    was found for the metadata_id.
    """
    if latest := cast(
        Sequence[Row[Any]],
        execute_stmt_lambda_element(
            session, _find_latest_short_term_statistic_for_metadata_id_stmt(metadata_id)
        ),
    ):
        id_ = latest[0].id
        run_cache.set_latest_id_for_metadata_id(metadata_id, id_)
        return id_
    return None


def _find_latest_short_term_statistic_for_metadata_id_stmt(
    metadata_id: int,
) -> StatementLambdaElement:
    """Create a statement to find the latest short term statistics for a metadata_id."""
    return lambda_stmt(
        lambda: select(StatisticsShortTerm.id)
        .where(StatisticsShortTerm.metadata_id == metadata_id)
        .order_by(StatisticsShortTerm.start_ts.desc())
        .limit(1)
    )


@retryable_database_job("statistics")
def import_statistics(
    instance: "Recorder",
    metadata: StatisticMetaData,
    statistics: list[StatisticData],
    table: type[StatisticsBase],
) -> bool:
    """Process an import_statistics job."""
    with session_scope(
        session=instance.get_session(),
        exception_filter=filter_unique_constraint_integrity_error(
            instance, "statistic"
        ),
    ) as session:
        return _import_statistics_with_session(
            instance, session, metadata, statistics, table
        )


@retryable_database_job("adjust_statistics")
def adjust_statistics(
    instance: "Recorder",
    statistic_id: str,
    start_time: datetime,
    sum_adjustment: float,
    adjustment_unit: str,
) -> bool:
    """Process an add_statistics job."""
    with session_scope(session=instance.get_session()) as session:
        metadata = instance.statistics_meta_manager.get_many(
            session, statistic_ids={statistic_id}
        )
        if statistic_id not in metadata:
            return True
        statistic_unit = metadata[statistic_id][1]["unit_of_measurement"]
        if convert := _get_display_to_statistic_unit_converter(
            adjustment_unit, statistic_unit
        ):
            sum_adjustment = convert(sum_adjustment)
        _adjust_sum_statistics(
            session, StatisticsShortTerm, metadata[statistic_id][0], start_time, sum_adjustment
        )
        _adjust_sum_statistics(
            session,
            Statistics,
            metadata[statistic_id][0],
            start_time.replace(minute=0),
            sum_adjustment,
        )
    return True


def _change_statistics_unit_for_table(
    session: Session,
    table: type[StatisticsBase],
    metadata_id: int,
    convert: Callable[[Optional[float]], Optional[float]],
) -> None:
    """Insert statistics in the database."""
    columns = (table.id, table.mean, table.min, table.max, table.state, table.sum)
    query = session.query(*columns).filter_by(metadata_id=bindparam("metadata_id"))
    rows = execute(query.params(metadata_id=metadata_id))
    for row in rows:
        session.query(table).filter(table.id == row.id).update(
            {
                table.mean: convert(row.mean),
                table.min: convert(row.min),
                table.max: convert(row.max),
                table.state: convert(row.state),
                table.sum: convert(row.sum),
            },
            synchronize_session=False,
        )


def change_statistics_unit(
    instance: "Recorder", statistic_id: str, new_unit: str, old_unit: str
) -> None:
    """Change statistics unit for a statistic_id."""
    statistics_meta_manager = instance.statistics_meta_manager
    with session_scope(session=instance.get_session()) as session:
        metadata = statistics_meta_manager.get(session, statistic_id)
        if (
            metadata is None
            or metadata[1]["source"] != DOMAIN
            or metadata[1]["unit_of_measurement"] != old_unit
        ):
            _LOGGER.warning("Could not change statistics unit for %s", statistic_id)
            return
        metadata_id = metadata[0]
        if not (convert := _get_unit_converter(old_unit, new_unit)):
            _LOGGER.warning(
                "Statistics unit of measurement for %s is already %s",
                statistic_id,
                new_unit,
            )
            return
        tables: tuple[type[StatisticsBase], ...] = (Statistics, StatisticsShortTerm)
        for table in tables:
            _change_statistics_unit_for_table(session, table, metadata_id, convert)
        statistics_meta_manager.update_unit_of_measurement(
            session, statistic_id, new_unit
        )


@callback
def async_change_statistics_unit(
    hass: HomeAssistant,
    statistic_id: str,
    *,
    new_unit_of_measurement: str,
    old_unit_of_measurement: str,
) -> None:
    """Change statistics unit for a statistic_id."""
    if not can_convert_units(old_unit_of_measurement, new_unit_of_measurement):
        raise HomeAssistantError(
            f"Can't convert {old_unit_of_measurement} to {new_unit_of_measurement}"
        )
    get_instance(hass).async_change_statistics_unit(
        statistic_id,
        new_unit_of_measurement=new_unit_of_measurement,
        old_unit_of_measurement=old_unit_of_measurement,
    )


def cleanup_statistics_timestamp_migration(instance: "Recorder") -> bool:
    """Clean up the statistics migration from timestamp to datetime.

    Returns False if there are more rows to update.
    Returns True if all rows have been updated.
    """
    engine = instance.engine
    assert engine is not None
    if engine.dialect.name == SupportedDialect.SQLITE:
        for table in STATISTICS_TABLES:
            with session_scope(session=instance.get_session()) as session:
                session.connection().execute(
                    text(
                        f"update {table} set start = NULL, created = NULL, last_reset = NULL;"
                    )
                )
    elif engine.dialect.name == SupportedDialect.MYSQL:
        for table in STATISTICS_TABLES:
            with session_scope(session=instance.get_session()) as session:
                if (
                    session.connection()
                    .execute(
                        text(
                            f"UPDATE {table} set start=NULL, created=NULL, last_reset=NULL where start is not NULL LIMIT 100000;"
                        )
                    )
                    .rowcount
                ):
                    return False
    elif engine.dialect.name == SupportedDialect.POSTGRESQL:
        for table in STATISTICS_TABLES:
            with session_scope(session=instance.get_session()) as session:
                if (
                    session.connection()
                    .execute(
                        text(
                            f"UPDATE {table} set start=NULL, created=NULL, last_reset=NULL where id in (select id from {table} where start is not NULL LIMIT 100000)"
                        )
                    )
                    .rowcount
                ):
                    return False
    from .migration import _drop_index

    for table in STATISTICS_TABLES:
        _drop_index(instance.get_session, table, f"ix_{table}_start")
    return True