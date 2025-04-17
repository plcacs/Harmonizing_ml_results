import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import TypeAlias, Union, Optional, Dict, List, Tuple, Any

from django.conf import settings
from django.db import connection, models
from django.utils.timezone import now as timezone_now
from psycopg2.sql import SQL, Composable, Identifier, Literal
from typing_extensions import override

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


logger: logging.Logger = logging.getLogger("zulip.analytics")


TIMEDELTA_MAX: timedelta = timedelta(days=365 * 1000)


class CountStat:
    HOUR: str = "hour"
    DAY: str = "day"
    FREQUENCIES: frozenset[str] = frozenset([HOUR, DAY])

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
        self.property: str = property
        self.data_collector: DataCollector = data_collector
        if frequency not in self.FREQUENCIES:
            raise AssertionError(f"Unknown frequency: {frequency}")
        self.frequency: str = frequency
        self.interval: timedelta = interval if interval is not None else self.time_increment

    @override
    def __repr__(self) -> str:
        return f"<CountStat: {self.property}>"

    def last_successful_fill(self) -> Optional[datetime]:
        fillstate: Optional[FillState] = FillState.objects.filter(property=self.property).first()
        if fillstate is None:
            return None
        if fillstate.state == FillState.DONE:
            return fillstate.end_time
        return fillstate.end_time - self.time_increment

    def current_month_accumulated_count_for_user(self, user: UserProfile) -> int:
        now: datetime = timezone_now()
        start_of_month: datetime = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        start_of_next_month: datetime = datetime(
            now.year + 1, 1, 1, tzinfo=timezone.utc
        ) if now.month == 12 else datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)

        assert self.data_collector.output_table == UserCount
        result: Dict[str, Any] = self.data_collector.output_table.objects.filter(
            user=user,
            property=self.property,
            end_time__gte=start_of_month,
            end_time__lt=start_of_next_month,
        ).aggregate(models.Sum("value"))

        total_value: int = result["value__sum"] or 0
        return total_value


class LoggingCountStat(CountStat):
    def __init__(self, property: str, output_table: type[BaseCount], frequency: str) -> None:
        super().__init__(property, DataCollector(output_table, None), frequency)


class DependentCountStat(CountStat):
    def __init__(
        self,
        property: str,
        data_collector: "DataCollector",
        frequency: str,
        interval: Optional[timedelta] = None,
        dependencies: Sequence[str] = [],
    ) -> None:
        super().__init__(property, data_collector, frequency, interval=interval)
        self.dependencies: Sequence[str] = dependencies


class DataCollector:
    def __init__(
        self,
        output_table: type[BaseCount],
        pull_function: Optional[Callable[[str, datetime, datetime, Optional[Realm]], int]],
    ) -> None:
        self.output_table: type[BaseCount] = output_table
        self.pull_function: Optional[Callable[[str, datetime, datetime, Optional[Realm]], int]] = pull_function

    def depends_on_realm(self) -> bool:
        return self.output_table in (UserCount, StreamCount)


def process_count_stat(stat: CountStat, fill_to_time: datetime, realm: Optional[Realm] = None) -> None:
    verify_UTC(fill_to_time)
    if floor_to_hour(fill_to_time) != fill_to_time:
        raise ValueError(f"fill_to_time must be on an hour boundary: {fill_to_time}")

    fill_state: Optional[FillState] = FillState.objects.filter(property=stat.property).first()
    if fill_state is None:
        currently_filled: datetime = installation_epoch()
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
            dependency_fill_time: Optional[datetime] = COUNT_STATS[dependency].last_successful_fill()
            if dependency_fill_time is None:
                logger.warning(
                    "DependentCountStat %s run before dependency %s.", stat.property, dependency
                )
                return
            fill_to_time = min(fill_to_time, dependency_fill_time)

    currently_filled += stat.time_increment
    while currently_filled <= fill_to_time:
        logger.info("START %s %s", stat.property, currently_filled)
        start: float = time.time()
        do_update_fill_state(fill_state, currently_filled, FillState.STARTED)
        do_fill_count_stat_at_hour(stat, currently_filled, realm)
        do_update_fill_state(fill_state, currently_filled, FillState.DONE)
        end: float = time.time()
        currently_filled += stat.time_increment
        logger.info("DONE %s (%dms)", stat.property, (end - start) * 1000)


def do_update_fill_state(fill_state: FillState, end_time: datetime, state: int) -> None:
    fill_state.end_time = end_time
    fill_state.state = state
    fill_state.save()


def do_fill_count_stat_at_hour(
    stat: CountStat, end_time: datetime, realm: Optional[Realm] = None
) -> None:
    start_time: datetime = end_time - stat.interval
    if not isinstance(stat, LoggingCountStat):
        timer: float = time.time()
        assert stat.data_collector.pull_function is not None
        rows_added: int = stat.data_collector.pull_function(stat.property, start_time, end_time, realm)
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

    output_table: type[BaseCount] = stat.data_collector.output_table
    realm_clause: Composable = SQL("AND zerver_realm.id = {}").format(Literal(realm.id)) if realm is not None else SQL("")

    if stat.data_collector.depends_on_realm():
        realmcount_query: Composable = SQL(
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
        start: float = time.time()
        cursor.execute(
            realmcount_query,
            {
                "property": stat.property,
                "end_time": end_time,
            },
        )
        end: float = time.time()
        logger.info(
            "%s RealmCount aggregation (%dms/%sr)",
            stat.property,
            (end - start) * 1000,
            cursor.rowcount,
        )

    if realm is None:
        installationcount_query: Composable = SQL(
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
        start: float = time.time()
        cursor.execute(
            installationcount_query,
            {
                "property": stat.property,
                "end_time": end_time,
            },
        )
        end: float = time.time()
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
    subgroup: Optional[Union[str, int, bool]],
    event_time: datetime,
    increment: int = 1,
) -> None:
    if not increment:
        return

    table: type[BaseCount] = stat.data_collector.output_table
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

    end_time: datetime = ceiling_to_day(event_time) if stat.frequency == CountStat.DAY else ceiling_to_hour(event_time)

    is_subgroup: SQL = SQL("NULL") if subgroup is None else SQL("NOT NULL")
    if subgroup is not None:
        subgroup = str(subgroup)
        conflict_args.append("subgroup")

    id_column_names: Composable = SQL(", ").join(map(Identifier, id_args.keys()))
    id_values: Composable = SQL(", ").join(map(Literal, id_args.values()))
    conflict_columns: Composable = SQL(", ").join(map(Identifier, conflict_args))

    sql_query: Composable = SQL(
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


QueryFn: TypeAlias = Callable[[Dict[str, Composable]], Composable]


def do_pull_by_sql_query(
    property: str,
    start_time: datetime,
    end_time: datetime,
    query: QueryFn,
    group_by: Optional[Tuple[type[models.Model], str]],
) -> int:
    subgroup: Composable = SQL("NULL") if group_by is None else Identifier(group_by[0]._meta.db_table, group_by[1])
    group_by_clause: Composable = SQL("") if group_by is None else SQL(", {}").format(subgroup)

    query_: Composable = query(
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
    rowcount: int = cursor.rowcount
    cursor.close()
    return rowcount


def sql_data_collector(
    output_table: type[BaseCount],
    query: QueryFn,
    group_by: Optional[Tuple[type[models.Model], str]],
) -> DataCollector:
    def pull_function(
        property: str, start_time: datetime, end_time: datetime, realm: Optional[Realm] = None
    ) -> int:
        return do_pull_by_sql_query(property, start_time, end_time, query, group_by)

    return DataCollector(output_table, pull_function)


def count_upload_space_used_by_realm_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL("zerver_attachment.realm_id = {} AND").format(Literal(realm.id))

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
    user_activity_intervals: List[Tuple[int, int, datetime, datetime]] = (
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
    for user_id, realm_id, interval_start, interval_end in user_activity_intervals:
        if realm is None or realm.id == realm_id:
            start: datetime = max(start_time, interval_start)
            end: datetime = min(end_time, interval_end)
            seconds_active[(user_id, realm_id)] += (end - start).total_seconds()

    rows: List[UserCount] = [
        UserCount(
            user_id=ids[0],
            realm_id=ids[1],
            property=property,
            end_time=end_time,
            value=int(seconds // 60),
        )
        for ids, seconds in seconds_active.items()
        if seconds >= 60
    ]
    UserCount.objects.bulk_create(rows)
    return len(rows)


def count_message_by_user_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL(
        "zerver_userprofile.realm_id = {} AND zerver_message.realm_id = {} AND"
    ).format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_usercount
        (user_id, realm_id, value, property, subgroup, end_time)
    SELECT
        zerver_userprofile.id, zerver_userprofile.realm_id, count(*),
        %(property)s, {subgroup}, %(time_end)s
    FROM zerver_userprofile
    JOIN zerver_message
    ON
        zerver_userprofile.id = zerver_message.sender_id
    WHERE
        zerver_userprofile.date_joined < %(time_end)s AND
        zerver_message.date_sent >= %(time_start)s AND
        {realm_clause}
        zerver_message.date_sent < %(time_end)s
    GROUP BY zerver_userprofile.id {group_by_clause}
"""
    ).format(**kwargs, realm_clause=realm_clause)


def count_message_type_by_user_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL(
        "zerver_userprofile.realm_id = {} AND zerver_message.realm_id = {} AND"
    ).format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_usercount
            (realm_id, user_id, value, property, subgroup, end_time)
    SELECT realm_id, id, SUM(count) AS value, %(property)s, message_type, %(time_end)s
    FROM
    (
        SELECT zerver_userprofile.realm_id, zerver_userprofile.id, count(*),
        CASE WHEN
                  zerver_recipient.type = 1 THEN 'private_message'
             WHEN
                  zerver_recipient.type = 3 THEN 'huddle_message'
             WHEN
                  zerver_stream.invite_only = TRUE THEN 'private_stream'
             ELSE 'public_stream'
        END
        message_type

        FROM zerver_userprofile
        JOIN zerver_message
        ON
            zerver_userprofile.id = zerver_message.sender_id AND
            zerver_message.date_sent >= %(time_start)s AND
            {realm_clause}
            zerver_message.date_sent < %(time_end)s
        JOIN zerver_recipient
        ON
            zerver_message.recipient_id = zerver_recipient.id
        LEFT JOIN zerver_stream
        ON
            zerver_recipient.type_id = zerver_stream.id
        GROUP BY
            zerver_userprofile.realm_id, zerver_userprofile.id,
            zerver_recipient.type, zerver_stream.invite_only
    ) AS subquery
    GROUP BY realm_id, id, message_type
"""
    ).format(**kwargs, realm_clause=realm_clause)


def count_message_by_stream_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL(
        "zerver_stream.realm_id = {} AND zerver_message.realm_id = {} AND"
    ).format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_streamcount
        (stream_id, realm_id, value, property, subgroup, end_time)
    SELECT
        zerver_stream.id, zerver_stream.realm_id, count(*), %(property)s, {subgroup}, %(time_end)s
    FROM zerver_stream
    JOIN zerver_recipient
    ON
        zerver_stream.id = zerver_recipient.type_id
    JOIN zerver_message
    ON
        zerver_recipient.id = zerver_message.recipient_id
    JOIN zerver_userprofile
    ON
        zerver_message.sender_id = zerver_userprofile.id
    WHERE
        zerver_stream.date_created < %(time_end)s AND
        zerver_recipient.type = 2 AND
        zerver_message.date_sent >= %(time_start)s AND
        {realm_clause}
        zerver_message.date_sent < %(time_end)s
    GROUP BY zerver_stream.id {group_by_clause}
"""
    ).format(**kwargs, realm_clause=realm_clause)


def check_realmauditlog_by_user_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL("realm_id = {} AND").format(Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_realmcount
        (realm_id, value, property, subgroup, end_time)
    SELECT
        zerver_userprofile.realm_id, count(*), %(property)s, {subgroup}, %(time_end)s
    FROM zerver_userprofile
    JOIN (
            SELECT DISTINCT ON (modified_user_id)
                    modified_user_id, event_type
            FROM
                    zerver_realmauditlog
            WHERE
                    event_type IN ({user_created}, {user_activated}, {user_deactivated}, {user_reactivated}) AND
                    {realm_clause}
                    event_time < %(time_end)s
            ORDER BY
                    modified_user_id,
                    event_time DESC
    ) last_user_event ON last_user_event.modified_user_id = zerver_userprofile.id
    WHERE
        last_user_event.event_type in ({user_created}, {user_activated}, {user_reactivated})
    GROUP BY zerver_userprofile.realm_id {group_by_clause}
    """
    ).format(
        **kwargs,
        user_created=Literal(AuditLogEventType.USER_CREATED),
        user_activated=Literal(AuditLogEventType.USER_ACTIVATED),
        user_deactivated=Literal(AuditLogEventType.USER_DEACTIVATED),
        user_reactivated=Literal(AuditLogEventType.USER_REACTIVATED),
        realm_clause=realm_clause,
    )


def check_useractivityinterval_by_user_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL("zerver_userprofile.realm_id = {} AND").format(Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_usercount
        (user_id, realm_id, value, property, subgroup, end_time)
    SELECT
        zerver_userprofile.id, zerver_userprofile.realm_id, 1, %(property)s, {subgroup}, %(time_end)s
    FROM zerver_userprofile
    JOIN zerver_useractivityinterval
    ON
        zerver_userprofile.id = zerver_useractivityinterval.user_profile_id
    WHERE
        zerver_useractivityinterval.end >= %(time_start)s AND
        {realm_clause}
        zerver_useractivityinterval.start < %(time_end)s
    GROUP BY zerver_userprofile.id {group_by_clause}
"""
    ).format(**kwargs, realm_clause=realm_clause)


def count_realm_active_humans_query(realm: Optional[Realm]) -> QueryFn:
    realm_clause: Composable = SQL("") if realm is None else SQL("realm_id = {} AND").format(Literal(realm.id))
    return lambda kwargs: SQL(
        """
    INSERT INTO analytics_realmcount
        (realm_id, value, property, subgroup, end_time)
    SELECT
            active_usercount.realm_id, count(*), %(property)s, NULL, %(time_end)s
    FROM (
            SELECT
                    realm_id,
                    user_id
            FROM
                    analytics_usercount
            WHERE
                    property = '15day_actives::day'
                    {realm_clause}
                    AND end_time = %(time_end)s
    ) active_usercount
    JOIN zerver_userprofile ON active_usercount.user_id = zerver_userprofile.id
     AND active_usercount.realm_id = zerver_userprofile.realm_id
    JOIN (
            SELECT DISTINCT ON (modified_user_id)
                    modified_user_id, event_type
            FROM
                    zerver_realmauditlog
            WHERE
                    event_type IN ({user_created}, {user_activated}, {user_deactivated}, {user_reactivated})
                    AND event_time < %(time_end)s
            ORDER BY
                    modified_user_id,
                    event_time DESC
    ) last_user_event ON last_user_event.modified_user_id = active_usercount.user_id
    WHERE
            NOT zerver_userprofile.is_bot
            AND event_type IN ({user_created}, {user_activated}, {user_reactivated})
    GROUP BY
            active_usercount.realm_id
"""
    ).format(
        **kwargs,
        user_created=Literal(AuditLogEventType.USER_CREATED),
        user_activated=Literal(AuditLogEventType.USER_ACTIVATED),
        user_deactivated=Literal(AuditLogEventType.USER_DEACTIVATED),
        user_reactivated=Literal(AuditLogEventType.USER_REACTIVATED),
        realm_clause=realm_clause,
    )


count_stream_by_realm_query = lambda kwargs: SQL(
    """
    INSERT INTO analytics_realmcount
        (realm_id, value, property, subgroup, end_time)
    SELECT
        zerver_realm.id, count(*), %(property)s, {subgroup}, %(time_end)s
    FROM zerver_realm
    JOIN zerver_stream
    ON
        zerver_realm.id = zerver_stream.realm_id AND
    WHERE
        zerver_realm.date_created < %(time_end)s AND
        zerver_stream.date_created >= %(time_start)s AND
        zerver_stream.date_created < %(time_end)s
    GROUP BY zerver_realm.id {group_by_clause}
"""
).format(**kwargs)


def get_count_stats(realm: Optional[Realm] = None) -> Dict[str, CountStat]:
    count_stats_: List[CountStat] = [
        CountStat(
            "messages_sent:is_bot:hour",
            sql_data_collector(
                UserCount, count_message_by_user_query(realm), (UserProfile, "is_bot")
            ),
            CountStat.HOUR,
        ),
        CountStat(
            "messages_sent:message_type:day",
            sql_data_collector(UserCount, count_message_type_by_user_query(realm), None),
            CountStat.DAY,
        ),
        CountStat(
            "messages_sent:client:day",
            sql_data_collector(
                UserCount, count_message_by_user_query(realm), (Message, "sending_client_id")
            ),
            CountStat.DAY,
        ),
        CountStat(
            "messages_in_stream:is_bot:day",
            sql_data_collector(
                StreamCount, count_message_by_stream_query(realm), (UserProfile, "is_bot")
            ),
            CountStat.DAY,
        ),
        LoggingCountStat("ai_credit_usage::day", UserCount, CountStat.DAY),
        CountStat(
            "active_users_audit:is_bot:day",
            sql_data_collector(
                RealmCount, check_realmauditlog_by_user_query(realm), (UserProfile, "is_bot")
            ),
            CountStat.DAY,
        ),
        CountStat(
            "upload_quota_used_bytes::day",
            sql_data_collector(RealmCount, count_upload_space_used_by_realm_query(realm), None),
            CountStat.DAY,
        ),
        LoggingCountStat("messages_read::hour", UserCount, CountStat.HOUR),
        LoggingCountStat("messages_read_interactions::hour", UserCount, CountStat.HOUR),
        CountStat(
            "1day_actives::day",
            sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None),
            CountStat.DAY,
            interval=timedelta(days=1) - UserActivityInterval.MIN_INTERVAL_LENGTH,
        ),
        CountStat(
            "7day_actives::day",
            sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None),
            CountStat.DAY,
            interval=timedelta(days=7) - UserActivityInterval.MIN_INTERVAL_LENGTH,
        ),
        CountStat(
            "15day_actives::day",
            sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None),
            CountStat.DAY,
            interval=timedelta(days=15) - UserActivityInterval.MIN_INTERVAL_LENGTH,
        ),
        CountStat(
            "minutes_active::day", DataCollector(UserCount, do_pull_minutes_active), CountStat.DAY
        ),
        LoggingCountStat(
            "mobile_pushes_sent::day",
            RealmCount,
            CountStat.DAY,
        ),
        LoggingCountStat("invites_sent::day", RealmCount, CountStat.DAY),
        DependentCountStat(
            "realm_active_humans::day",
            sql_data_collector(RealmCount, count_realm_active_humans_query(realm), None),
            CountStat.DAY,
            dependencies=["15day_actives::day"],
        ),
    ]

    if settings.ZILENCER_ENABLED:
        count_stats_.append(
            LoggingCountStat(
                "mobile_pushes_received::day",
                RemoteRealmCount,
                CountStat.DAY,
            )
        )
        count_stats_.append(
            LoggingCountStat(
                "mobile_pushes_forwarded::day",
                RemoteRealmCount,
                CountStat.DAY,
            )
        )

    return OrderedDict((stat.property, stat) for stat in count_stats_)


BOUNCER_ONLY_REMOTE_COUNT_STAT_PROPERTIES: List[str] = [
    "mobile_pushes_received::day",
    "mobile_pushes_forwarded::day",
]

LOGGING_COUNT_STAT_PROPERTIES_NOT_SENT_TO_BOUNCER: Set[str] = {
    "invites_sent::day",
    "mobile_pushes_sent::day",
    "active_users_log:is_bot:day",
    "active_users:is_bot:day",
}

COUNT_STATS: Dict[str, CountStat] = get_count_stats()

REMOTE_INSTALLATION_COUNT_STATS: Dict[str, CountStat] = OrderedDict()

if settings.ZILENCER_ENABLED:
    REMOTE_INSTALLATION_COUNT_STATS["mobile_pushes_received::day"] = LoggingCountStat(
        "mobile_pushes_received::day",
        RemoteInstallationCount,
        CountStat.DAY,
    )
    REMOTE_INSTALLATION_COUNT_STATS["mobile_pushes_forwarded::day"] = LoggingCountStat(
        "mobile_pushes_forwarded::day",
        RemoteInstallationCount,
        CountStat.DAY,
    )

ALL_COUNT_STATS: Dict[str, CountStat] = OrderedDict(
    list(COUNT_STATS.items()) + list(REMOTE_INSTALLATION_COUNT_STATS.items())
