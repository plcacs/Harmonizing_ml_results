import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from django.conf import settings
from django.db import connection, models
from django.db.models import Model
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Composable, Identifier, Literal
from typing_extensions import Literal as TypingLiteral, override

from analytics.models import (
    BaseCount,
    FillState,
    InstallationCount,
    RealmCount,
    StreamCount,
    UserCount,
    installation_epoch,
)
from zerver.lib.timestamp import ceiling_to_day, ceiling_to_hour, floor_to_hour, verify_UTC
from zerver.models import Message, Realm, Stream, UserActivityInterval, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType

if settings.ZILENCER_ENABLED:
    from zilencer.models import (
        RemoteInstallationCount,
        RemoteRealm,
        RemoteRealmCount,
        RemoteZulipServer,
    )


logger = logging.getLogger("zulip.analytics")


TIMEDELTA_MAX = timedelta(days=365 * 1000)


class CountStat:
    HOUR: TypingLiteral["hour"] = "hour"
    DAY: TypingLiteral["day"] = "day"
    FREQUENCIES = frozenset([HOUR, DAY])

    @property
    def time_increment(self) -> timedelta:
        if self.frequency == CountStat.HOUR:
            return timedelta(hours=1)
        return timedelta(days=1)

    def __init__(
        self,
        property: str,
        data_collector: "DataCollector",
        frequency: str,
        interval: Optional[timedelta] = None,
    ) -> None:
        self.property = property
        self.data_collector = data_collector
        if frequency not in self.FREQUENCIES:
            raise AssertionError(f"Unknown frequency: {frequency}")
        self.frequency = frequency
        if interval is not None:
            self.interval = interval
        else:
            self.interval = self.time_increment

    @override
    def __repr__(self) -> str:
        return f"<CountStat: {self.property}>"

    def last_successful_fill(self) -> Optional[datetime]:
        fillstate = FillState.objects.filter(property=self.property).first()
        if fillstate is None:
            return None
        if fillstate.state == FillState.DONE:
            return fillstate.end_time
        return fillstate.end_time - self.time_increment

    def current_month_accumulated_count_for_user(self, user: UserProfile) -> int:
        now = timezone_now()
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        if now.month == 12:
            start_of_next_month = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            start_of_next_month = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)

        assert self.data_collector.output_table == UserCount
        result = cast(Any, self.data_collector.output_table).objects.filter(
            user=user,
            property=self.property,
            end_time__gte=start_of_month,
            end_time__lt=start_of_next_month,
        ).aggregate(models.Sum("value"))

        total_value = result["value__sum"] or 0
        return total_value


class LoggingCountStat(CountStat):
    def __init__(self, property: str, output_table: Type[BaseCount], frequency: str) -> None:
        CountStat.__init__(self, property, DataCollector(output_table, None), frequency)


class DependentCountStat(CountStat):
    def __init__(
        self,
        property: str,
        data_collector: "DataCollector",
        frequency: str,
        interval: Optional[timedelta] = None,
        dependencies: Sequence[str] = [],
    ) -> None:
        CountStat.__init__(self, property, data_collector, frequency, interval=interval)
        self.dependencies = dependencies


class DataCollector:
    def __init__(
        self,
        output_table: Type[BaseCount],
        pull_function: Optional[Callable[[str, datetime, datetime, Optional[Realm]], int]],
    ) -> None:
        self.output_table = output_table
        self.pull_function = pull_function

    def depends_on_realm(self) -> bool:
        return self.output_table in (UserCount, StreamCount)


def process_count_stat(
    stat: CountStat, fill_to_time: datetime, realm: Optional[Realm] = None
) -> None:
    verify_UTC(fill_to_time)
    if floor_to_hour(fill_to_time) != fill_to_time:
        raise ValueError(f"fill_to_time must be on an hour boundary: {fill_to_time}")

    fill_state = FillState.objects.filter(property=stat.property).first()
    if fill_state is None:
        currently_filled = installation_epoch()
        fill_state = FillState.objects.create(
            property=stat.property, end_time=currently_filled, state=FillState.DONE
        )
        logger.info("INITIALIZED %s %s", stat.property, currently_filled)
    elif fill_state.state == FillState.STARTED:
        logger.info("UNDO START %s %s", stat.property, fill_state.end_time)
        do_delete_counts_at_hour(stat, fill_state.end_time)
        currently_filled = fill_state.end_time - stat.time_increment
        do_update_fill_state(fill_state, currently_filled, FillState.DONE)
        logger.info("UNDO DONE %s", stat.property)
    elif fill_state.state == FillState.DONE:
        currently_filled = fill_state.end_time
    else:
        raise AssertionError(f"Unknown value for FillState.state: {fill_state.state}.")

    if isinstance(stat, DependentCountStat):
        for dependency in stat.dependencies:
            dependency_fill_time = COUNT_STATS[dependency].last_successful_fill()
            if dependency_fill_time is None:
                logger.warning(
                    "DependentCountStat %s run before dependency %s.", stat.property, dependency
                )
                return
            fill_to_time = min(fill_to_time, dependency_fill_time)

    currently_filled += stat.time_increment
    while currently_filled <= fill_to_time:
        logger.info("START %s %s", stat.property, currently_filled)
        start = time.time()
        do_update_fill_state(fill_state, currently_filled, FillState.STARTED)
        do_fill_count_stat_at_hour(stat, currently_filled, realm)
        do_update_fill_state(fill_state, currently_filled, FillState.DONE)
        end = time.time()
        currently_filled += stat.time_increment
        logger.info("DONE %s (%dms)", stat.property, (end - start) * 1000)


def do_update_fill_state(fill_state: FillState, end_time: datetime, state: int) -> None:
    fill_state.end_time = end_time
    fill_state.state = state
    fill_state.save()


def do_fill_count_stat_at_hour(
    stat: CountStat, end_time: datetime, realm: Optional[Realm] = None
) -> None:
    start_time = end_time - stat.interval
    if not isinstance(stat, LoggingCountStat):
        timer = time.time()
        assert stat.data_collector.pull_function is not None
        rows_added = stat.data_collector.pull_function(stat.property, start_time, end_time, realm)
        logger.info(
            "%s run pull_function (%dms/%sr)",
            stat.property,
            (time.time() - timer) * 1000,
            rows_added,
        )
    do_aggregate_to_summary_table(stat, end_time, realm)


def do_delete_counts_at_hour(stat: CountStat, end_time: datetime) -> None:
    if isinstance(stat, LoggingCountStat):
        InstallationCount.objects.filter(property=stat.property, end_time=end_time).delete()
        if stat.data_collector.depends_on_realm():
            RealmCount.objects.filter(property=stat.property, end_time=end_time).delete()
    else:
        UserCount.objects.filter(property=stat.property, end_time=end_time).delete()
        StreamCount.objects.filter(property=stat.property, end_time=end_time).delete()
        RealmCount.objects.filter(property=stat.property, end_time=end_time).delete()
        InstallationCount.objects.filter(property=stat.property, end_time=end_time).delete()


def do_aggregate_to_summary_table(
    stat: CountStat, end_time: datetime, realm: Optional[Realm] = None
) -> None:
    cursor = connection.cursor()

    output_table = stat.data_collector.output_table
    if realm is not None:
        realm_clause: Composable = SQL("AND zerver_realm.id = {}").format(Literal(realm.id))
    else:
        realm_clause = SQL("")

    if stat.data_collector.depends_on_realm():
        realmcount_query = SQL(
            """
            INSERT INTO analytics_realmcount
                (realm_id, value, property, subgroup, end_time)
            SELECT
                zerver_realm.id, COALESCE(sum({output_table}.value), 0), %(property)s,
                {output_table}.subgroup, %(end_time)s
            FROM zerver_realm
            JOIN {output_table}
            ON
                zerver_realm.id = {output_table}.realm_id
            WHERE
                {output_table}.property = %(property)s AND
                {output_table}.end_time = %(end_time)s
                {realm_clause}
            GROUP BY zerver_realm.id, {output_table}.subgroup
        """
        ).format(
            output_table=Identifier(output_table._meta.db_table),
            realm_clause=realm_clause,
        )
        start = time.time()
        cursor.execute(
            realmcount_query,
            {
                "property": stat.property,
                "end_time": end_time,
            },
        )
        end = time.time()
        logger.info(
            "%s RealmCount aggregation (%dms/%sr)",
            stat.property,
            (end - start) * 1000,
            cursor.rowcount,
        )

    if realm is None:
        installationcount_query = SQL(
            """
            INSERT INTO analytics_installationcount
                (value, property, subgroup, end_time)
            SELECT
                sum(value), %(property)s, analytics_realmcount.subgroup, %(end_time)s
            FROM analytics_realmcount
            WHERE
                property = %(property)s AND
                end_time = %(end_time)s
            GROUP BY analytics_realmcount.subgroup
        """
        )
        start = time.time()
        cursor.execute(
            installationcount_query,
            {
                "property": stat.property,
                "end_time": end_time,
            },
        )
        end = time.time()
        logger.info(
            "%s InstallationCount aggregation (%dms/%sr)",
            stat.property,
            (end - start) * 1000,
            cursor.rowcount,
        )

    cursor.close()


def do_increment_logging_stat(
    model_object_for_bucket: Union[Realm, UserProfile, Stream, "RemoteRealm", "RemoteZulipServer"],
    stat: CountStat,
    subgroup: Union[str, int, bool, None],
    event_time: datetime,
    increment: int = 1,
) -> None:
    if not increment:
        return

    table = stat.data_collector.output_table
    id_args: Dict[str, Optional[int]] = {}
    conflict_args: List[str] = []
    if table == RealmCount:
        assert isinstance(model_object_for_bucket, Realm)
        id_args = {"realm_id": model_object_for_bucket.id}
        conflict_args = ["realm_id"]
    elif table == UserCount:
        assert isinstance(model_object_for_bucket, UserProfile)
        id_args = {
            "realm_id": model_object_for_bucket.realm_id,
            "user_id": model_object_for_bucket.id,
        }
        conflict_args = ["user_id"]
    elif table == StreamCount:
        assert isinstance(model_object_for_bucket, Stream)
        id_args = {
            "realm_id": model_object_for_bucket.realm_id,
            "stream_id": model_object_for_bucket.id,
        }
        conflict_args = ["stream_id"]
    elif table == RemoteInstallationCount:
        assert isinstance(model_object_for_bucket, RemoteZulipServer)
        id_args = {"server_id": model_object_for_bucket.id, "remote_id": None}
        conflict_args = ["server_id"]
    elif table == RemoteRealmCount:
        assert isinstance(model_object_for_bucket, RemoteRealm)
        id_args = {
            "server_id": model_object_for_bucket.server_id,
            "remote_realm_id": model_object_for_bucket.id,
            "remote_id": None,
            "realm_id": None,
        }
        conflict_args = [
            "remote_realm_id",
        ]
    else:
        raise AssertionError("Unsupported CountStat output_table")

    if stat.frequency == CountStat.DAY:
        end_time = ceiling_to_day(event_time)
    elif stat.frequency == CountStat.HOUR:
        end_time = ceiling_to_hour(event_time)
    else:
        raise AssertionError("Unsupported CountStat frequency")

    is_subgroup: SQL = SQL("NULL")
    if subgroup is not None:
        is_subgroup = SQL("NOT NULL")
        subgroup = str(subgroup)
        conflict_args.append("subgroup")

    id_column_names = SQL(", ").join(map(Identifier, id_args.keys()))
    id_values = SQL(", ").join(map(Literal, id_args.values()))
    conflict_columns = SQL(", ").join(map(Identifier, conflict_args))

    sql_query = SQL(
        """
        INSERT INTO {table_name}(property, subgroup, end_time, value, {id_column_names})
        VALUES (%s, %s, %s, %s, {id_values})
        ON CONFLICT (property, end_time, {conflict_columns})
        WHERE subgroup IS {is_subgroup}
        DO UPDATE SET
            value = {table_name}.value + EXCLUDED.value
        """
    ).format(
        table_name=Identifier(table._meta.db_table),
        id_column_names=id_column_names,
        id_values=id_values,
        conflict_columns=conflict_columns,
        is_subgroup=is_subgroup,
    )
    with connection.cursor() as cursor:
        cursor.execute(sql_query, [stat.property, subgroup, end_time, increment])


def do_drop_all_analytics_tables() -> None:
    UserCount.objects.all().delete()
    StreamCount.objects.all().delete()
    RealmCount.objects.all().delete()
    InstallationCount.objects.all().delete()
    FillState.objects.all().delete()


def do_drop_single_stat(property: str) -> None:
    UserCount.objects.filter(property=property).delete()
    StreamCount.objects.filter(property=property).delete()
    RealmCount.objects.filter(property=property).delete()
    InstallationCount.objects.filter(property=property).delete()
    FillState.objects.filter(property=property).delete()


QueryFn = Callable[[Dict[str, Composable]], Composable]


def do_pull_by_sql_query(
    property: str,
    start_time: datetime,
    end_time: datetime,
    query: QueryFn,
    group_by: Optional[Tuple[Type[Model], str]],
) -> int:
    if group_by is None:
        subgroup: Composable = SQL("NULL")
        group_by_clause: Composable = SQL("")
    else:
        subgroup = Identifier(group_by[0]._meta.db_table, group_by[1])
        group_by_clause = SQL(", {}").format(subgroup)

    query_ = query(
        {
            "subgroup": subgroup,
            "group_by_clause": group_by_clause,
        }
    )
    cursor = connection.cursor()
    cursor.execute(
        query_,
        {
            "property": property,
            "time_start": start_time,
            "time_end": end_time,
        },
    )
    rowcount = cursor.rowcount
    cursor.close()
    return rowcount


def sql_data_collector(
    output_table: Type[BaseCount],
    query: QueryFn,
    group_by: Optional[Tuple[Type[Model], str]],
) -> DataCollector:
    def pull_function(
        property: str, start_time: datetime, end_time: datetime, realm: Optional[Realm] = None
    ) -> int:
        return do_pull_by_sql_query(property, start_time, end_time, query, group_by)

    return DataCollector(output_table, pull_function)


def count_upload_space_used_by_realm_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause: Composable = SQL("")
    else:
        realm_clause = SQL("zerver_attachment.realm_id = {} AND").format(Literal(realm.id))

    return lambda kwargs: SQL(
        """
            INSERT INTO analytics_realmcount (realm_id, property, end_time, value)
            SELECT
                zerver_attachment.realm_id,
                %(property)s,
                %(time_end)s,
                COALESCE(SUM(zerver_attachment.size), 0)
            FROM
                zerver_attachment
            WHERE
                {realm_clause}
                zerver_attachment.create_time < %(time_end)s
            GROUP BY
                zerver_attachment.realm_id
        """
    ).format(**kwargs, realm_clause=realm_clause)


def do_pull_minutes_active(
    property: str, start_time: datetime, end_time: datetime, realm: Optional[Realm] = None
) -> int:
    user_activity_intervals = (
        UserActivityInterval.objects.filter(
            end__gt=start_time,
            start__lt=end_time,
        )
        .select_related(
            "user_profile",
        )
        .values_list("user_profile_id", "user_profile__realm_id", "start", "end")
    )

    seconds_active: Dict[Tuple[int, int], float] = defaultdict(float)
    for user_id, realm_id, interval_start, interval_end