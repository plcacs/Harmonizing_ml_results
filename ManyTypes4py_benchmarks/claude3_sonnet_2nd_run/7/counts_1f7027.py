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
from analytics.models import BaseCount, FillState, InstallationCount, RealmCount, StreamCount, UserCount, installation_epoch
from zerver.lib.timestamp import ceiling_to_day, ceiling_to_hour, floor_to_hour, verify_UTC
from zerver.models import Message, Realm, Stream, UserActivityInterval, UserProfile
from zerver.models.realm_audit_logs import AuditLogEventType
if settings.ZILENCER_ENABLED:
    from zilencer.models import RemoteInstallationCount, RemoteRealm, RemoteRealmCount, RemoteZulipServer
logger = logging.getLogger('zulip.analytics')
TIMEDELTA_MAX = timedelta(days=365 * 1000)

class CountStat:
    HOUR = 'hour'
    DAY = 'day'
    FREQUENCIES = frozenset([HOUR, DAY])

    @property
    def time_increment(self) -> timedelta:
        if self.frequency == CountStat.HOUR:
            return timedelta(hours=1)
        return timedelta(days=1)

    def __init__(self, property: str, data_collector: 'DataCollector', frequency: str, interval: Optional[timedelta] = None) -> None:
        self.property = property
        self.data_collector = data_collector
        if frequency not in self.FREQUENCIES:
            raise AssertionError(f'Unknown frequency: {frequency}')
        self.frequency = frequency
        if interval is not None:
            self.interval = interval
        else:
            self.interval = self.time_increment

    @override
    def __repr__(self) -> str:
        return f'<CountStat: {self.property}>'

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
        result = self.data_collector.output_table.objects.filter(user=user, property=self.property, end_time__gte=start_of_month, end_time__lt=start_of_next_month).aggregate(models.Sum('value'))
        total_value = result['value__sum'] or 0
        return total_value

class LoggingCountStat(CountStat):

    def __init__(self, property: str, output_table: type[BaseCount], frequency: str) -> None:
        CountStat.__init__(self, property, DataCollector(output_table, None), frequency)

class DependentCountStat(CountStat):

    def __init__(self, property: str, data_collector: 'DataCollector', frequency: str, interval: Optional[timedelta] = None, dependencies: List[str] = []) -> None:
        CountStat.__init__(self, property, data_collector, frequency, interval=interval)
        self.dependencies = dependencies

class DataCollector:

    def __init__(self, output_table: type[BaseCount], pull_function: Optional[Callable[..., int]]) -> None:
        self.output_table = output_table
        self.pull_function = pull_function

    def depends_on_realm(self) -> bool:
        return self.output_table in (UserCount, StreamCount)

def process_count_stat(stat: CountStat, fill_to_time: datetime, realm: Optional[Realm] = None) -> None:
    verify_UTC(fill_to_time)
    if floor_to_hour(fill_to_time) != fill_to_time:
        raise ValueError(f'fill_to_time must be on an hour boundary: {fill_to_time}')
    fill_state = FillState.objects.filter(property=stat.property).first()
    if fill_state is None:
        currently_filled = installation_epoch()
        fill_state = FillState.objects.create(property=stat.property, end_time=currently_filled, state=FillState.DONE)
        logger.info('INITIALIZED %s %s', stat.property, currently_filled)
    elif fill_state.state == FillState.STARTED:
        logger.info('UNDO START %s %s', stat.property, fill_state.end_time)
        do_delete_counts_at_hour(stat, fill_state.end_time)
        currently_filled = fill_state.end_time - stat.time_increment
        do_update_fill_state(fill_state, currently_filled, FillState.DONE)
        logger.info('UNDO DONE %s', stat.property)
    elif fill_state.state == FillState.DONE:
        currently_filled = fill_state.end_time
    else:
        raise AssertionError(f'Unknown value for FillState.state: {fill_state.state}.')
    if isinstance(stat, DependentCountStat):
        for dependency in stat.dependencies:
            dependency_fill_time = COUNT_STATS[dependency].last_successful_fill()
            if dependency_fill_time is None:
                logger.warning('DependentCountStat %s run before dependency %s.', stat.property, dependency)
                return
            fill_to_time = min(fill_to_time, dependency_fill_time)
    currently_filled += stat.time_increment
    while currently_filled <= fill_to_time:
        logger.info('START %s %s', stat.property, currently_filled)
        start = time.time()
        do_update_fill_state(fill_state, currently_filled, FillState.STARTED)
        do_fill_count_stat_at_hour(stat, currently_filled, realm)
        do_update_fill_state(fill_state, currently_filled, FillState.DONE)
        end = time.time()
        currently_filled += stat.time_increment
        logger.info('DONE %s (%dms)', stat.property, (end - start) * 1000)

def do_update_fill_state(fill_state: FillState, end_time: datetime, state: int) -> None:
    fill_state.end_time = end_time
    fill_state.state = state
    fill_state.save()

def do_fill_count_stat_at_hour(stat: CountStat, end_time: datetime, realm: Optional[Realm] = None) -> None:
    start_time = end_time - stat.interval
    if not isinstance(stat, LoggingCountStat):
        timer = time.time()
        assert stat.data_collector.pull_function is not None
        rows_added = stat.data_collector.pull_function(stat.property, start_time, end_time, realm)
        logger.info('%s run pull_function (%dms/%sr)', stat.property, (time.time() - timer) * 1000, rows_added)
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

def do_aggregate_to_summary_table(stat: CountStat, end_time: datetime, realm: Optional[Realm] = None) -> None:
    cursor = connection.cursor()
    output_table = stat.data_collector.output_table
    if realm is not None:
        realm_clause = SQL('AND zerver_realm.id = {}').format(Literal(realm.id))
    else:
        realm_clause = SQL('')
    if stat.data_collector.depends_on_realm():
        realmcount_query = SQL('\n            INSERT INTO analytics_realmcount\n                (realm_id, value, property, subgroup, end_time)\n            SELECT\n                zerver_realm.id, COALESCE(sum({output_table}.value), 0), %(property)s,\n                {output_table}.subgroup, %(end_time)s\n            FROM zerver_realm\n            JOIN {output_table}\n            ON\n                zerver_realm.id = {output_table}.realm_id\n            WHERE\n                {output_table}.property = %(property)s AND\n                {output_table}.end_time = %(end_time)s\n                {realm_clause}\n            GROUP BY zerver_realm.id, {output_table}.subgroup\n        ').format(output_table=Identifier(output_table._meta.db_table), realm_clause=realm_clause)
        start = time.time()
        cursor.execute(realmcount_query, {'property': stat.property, 'end_time': end_time})
        end = time.time()
        logger.info('%s RealmCount aggregation (%dms/%sr)', stat.property, (end - start) * 1000, cursor.rowcount)
    if realm is None:
        installationcount_query = SQL('\n            INSERT INTO analytics_installationcount\n                (value, property, subgroup, end_time)\n            SELECT\n                sum(value), %(property)s, analytics_realmcount.subgroup, %(end_time)s\n            FROM analytics_realmcount\n            WHERE\n                property = %(property)s AND\n                end_time = %(end_time)s\n            GROUP BY analytics_realmcount.subgroup\n        ')
        start = time.time()
        cursor.execute(installationcount_query, {'property': stat.property, 'end_time': end_time})
        end = time.time()
        logger.info('%s InstallationCount aggregation (%dms/%sr)', stat.property, (end - start) * 1000, cursor.rowcount)
    cursor.close()

def do_increment_logging_stat(model_object_for_bucket: Union[Realm, UserProfile, Stream, 'RemoteZulipServer', 'RemoteRealm'], 
                             stat: CountStat, 
                             subgroup: Optional[str], 
                             event_time: datetime, 
                             increment: int = 1) -> None:
    if not increment:
        return
    table = stat.data_collector.output_table
    id_args: Dict[str, Any] = {}
    conflict_args: List[str] = []
    if table == RealmCount:
        assert isinstance(model_object_for_bucket, Realm)
        id_args = {'realm_id': model_object_for_bucket.id}
        conflict_args = ['realm_id']
    elif table == UserCount:
        assert isinstance(model_object_for_bucket, UserProfile)
        id_args = {'realm_id': model_object_for_bucket.realm_id, 'user_id': model_object_for_bucket.id}
        conflict_args = ['user_id']
    elif table == StreamCount:
        assert isinstance(model_object_for_bucket, Stream)
        id_args = {'realm_id': model_object_for_bucket.realm_id, 'stream_id': model_object_for_bucket.id}
        conflict_args = ['stream_id']
    elif table == RemoteInstallationCount:
        assert isinstance(model_object_for_bucket, RemoteZulipServer)
        id_args = {'server_id': model_object_for_bucket.id, 'remote_id': None}
        conflict_args = ['server_id']
    elif table == RemoteRealmCount:
        assert isinstance(model_object_for_bucket, RemoteRealm)
        id_args = {'server_id': model_object_for_bucket.server_id, 'remote_realm_id': model_object_for_bucket.id, 'remote_id': None, 'realm_id': None}
        conflict_args = ['remote_realm_id']
    else:
        raise AssertionError('Unsupported CountStat output_table')
    if stat.frequency == CountStat.DAY:
        end_time = ceiling_to_day(event_time)
    elif stat.frequency == CountStat.HOUR:
        end_time = ceiling_to_hour(event_time)
    else:
        raise AssertionError('Unsupported CountStat frequency')
    is_subgroup = SQL('NULL')
    if subgroup is not None:
        is_subgroup = SQL('NOT NULL')
        subgroup = str(subgroup)
        conflict_args.append('subgroup')
    id_column_names = SQL(', ').join(map(Identifier, id_args.keys()))
    id_values = SQL(', ').join(map(Literal, id_args.values()))
    conflict_columns = SQL(', ').join(map(Identifier, conflict_args))
    sql_query = SQL('\n        INSERT INTO {table_name}(property, subgroup, end_time, value, {id_column_names})\n        VALUES (%s, %s, %s, %s, {id_values})\n        ON CONFLICT (property, end_time, {conflict_columns})\n        WHERE subgroup IS {is_subgroup}\n        DO UPDATE SET\n            value = {table_name}.value + EXCLUDED.value\n        ').format(table_name=Identifier(table._meta.db_table), id_column_names=id_column_names, id_values=id_values, conflict_columns=conflict_columns, is_subgroup=is_subgroup)
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
QueryFn: TypeAlias = Callable[[dict[str, Composable]], Composable]

def do_pull_by_sql_query(property: str, start_time: datetime, end_time: datetime, query: QueryFn, 
                         group_by: Optional[Tuple[Any, str]]) -> int:
    if group_by is None:
        subgroup = SQL('NULL')
        group_by_clause = SQL('')
    else:
        subgroup = Identifier(group_by[0]._meta.db_table, group_by[1])
        group_by_clause = SQL(', {}').format(subgroup)
    query_ = query({'subgroup': subgroup, 'group_by_clause': group_by_clause})
    cursor = connection.cursor()
    cursor.execute(query_, {'property': property, 'time_start': start_time, 'time_end': end_time})
    rowcount = cursor.rowcount
    cursor.close()
    return rowcount

def sql_data_collector(output_table: type[BaseCount], query: QueryFn, 
                      group_by: Optional[Tuple[Any, str]]) -> 'DataCollector':

    def pull_function(property: str, start_time: datetime, end_time: datetime, realm: Optional[Realm] = None) -> int:
        return do_pull_by_sql_query(property, start_time, end_time, query, group_by)
    return DataCollector(output_table, pull_function)

def count_upload_space_used_by_realm_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('zerver_attachment.realm_id = {} AND').format(Literal(realm.id))
    return lambda kwargs: SQL('\n            INSERT INTO analytics_realmcount (realm_id, property, end_time, value)\n            SELECT\n                zerver_attachment.realm_id,\n                %(property)s,\n                %(time_end)s,\n                COALESCE(SUM(zerver_attachment.size), 0)\n            FROM\n                zerver_attachment\n            WHERE\n                {realm_clause}\n                zerver_attachment.create_time < %(time_end)s\n            GROUP BY\n                zerver_attachment.realm_id\n        ').format(**kwargs, realm_clause=realm_clause)

def do_pull_minutes_active(property: str, start_time: datetime, end_time: datetime, 
                          realm: Optional[Realm] = None) -> int:
    user_activity_intervals = UserActivityInterval.objects.filter(end__gt=start_time, start__lt=end_time).select_related('user_profile').values_list('user_profile_id', 'user_profile__realm_id', 'start', 'end')
    seconds_active: Dict[Tuple[int, int], float] = defaultdict(float)
    for user_id, realm_id, interval_start, interval_end in user_activity_intervals:
        if realm is None or realm.id == realm_id:
            start = max(start_time, interval_start)
            end = min(end_time, interval_end)
            seconds_active[user_id, realm_id] += (end - start).total_seconds()
    rows = [UserCount(user_id=ids[0], realm_id=ids[1], property=property, end_time=end_time, value=int(seconds // 60)) for ids, seconds in seconds_active.items() if seconds >= 60]
    UserCount.objects.bulk_create(rows)
    return len(rows)

def count_message_by_user_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('zerver_userprofile.realm_id = {} AND zerver_message.realm_id = {} AND').format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL('\n    INSERT INTO analytics_usercount\n        (user_id, realm_id, value, property, subgroup, end_time)\n    SELECT\n        zerver_userprofile.id, zerver_userprofile.realm_id, count(*),\n        %(property)s, {subgroup}, %(time_end)s\n    FROM zerver_userprofile\n    JOIN zerver_message\n    ON\n        zerver_userprofile.id = zerver_message.sender_id\n    WHERE\n        zerver_userprofile.date_joined < %(time_end)s AND\n        zerver_message.date_sent >= %(time_start)s AND\n        {realm_clause}\n        zerver_message.date_sent < %(time_end)s\n    GROUP BY zerver_userprofile.id {group_by_clause}\n').format(**kwargs, realm_clause=realm_clause)

def count_message_type_by_user_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('zerver_userprofile.realm_id = {} AND zerver_message.realm_id = {} AND').format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL("\n    INSERT INTO analytics_usercount\n            (realm_id, user_id, value, property, subgroup, end_time)\n    SELECT realm_id, id, SUM(count) AS value, %(property)s, message_type, %(time_end)s\n    FROM\n    (\n        SELECT zerver_userprofile.realm_id, zerver_userprofile.id, count(*),\n        CASE WHEN\n                  zerver_recipient.type = 1 THEN 'private_message'\n             WHEN\n                  zerver_recipient.type = 3 THEN 'huddle_message'\n             WHEN\n                  zerver_stream.invite_only = TRUE THEN 'private_stream'\n             ELSE 'public_stream'\n        END\n        message_type\n\n        FROM zerver_userprofile\n        JOIN zerver_message\n        ON\n            zerver_userprofile.id = zerver_message.sender_id AND\n            zerver_message.date_sent >= %(time_start)s AND\n            {realm_clause}\n            zerver_message.date_sent < %(time_end)s\n        JOIN zerver_recipient\n        ON\n            zerver_message.recipient_id = zerver_recipient.id\n        LEFT JOIN zerver_stream\n        ON\n            zerver_recipient.type_id = zerver_stream.id\n        GROUP BY\n            zerver_userprofile.realm_id, zerver_userprofile.id,\n            zerver_recipient.type, zerver_stream.invite_only\n    ) AS subquery\n    GROUP BY realm_id, id, message_type\n").format(**kwargs, realm_clause=realm_clause)

def count_message_by_stream_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('zerver_stream.realm_id = {} AND zerver_message.realm_id = {} AND').format(Literal(realm.id), Literal(realm.id))
    return lambda kwargs: SQL('\n    INSERT INTO analytics_streamcount\n        (stream_id, realm_id, value, property, subgroup, end_time)\n    SELECT\n        zerver_stream.id, zerver_stream.realm_id, count(*), %(property)s, {subgroup}, %(time_end)s\n    FROM zerver_stream\n    JOIN zerver_recipient\n    ON\n        zerver_stream.id = zerver_recipient.type_id\n    JOIN zerver_message\n    ON\n        zerver_recipient.id = zerver_message.recipient_id\n    JOIN zerver_userprofile\n    ON\n        zerver_message.sender_id = zerver_userprofile.id\n    WHERE\n        zerver_stream.date_created < %(time_end)s AND\n        zerver_recipient.type = 2 AND\n        zerver_message.date_sent >= %(time_start)s AND\n        {realm_clause}\n        zerver_message.date_sent < %(time_end)s\n    GROUP BY zerver_stream.id {group_by_clause}\n').format(**kwargs, realm_clause=realm_clause)

def check_realmauditlog_by_user_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('realm_id = {} AND').format(Literal(realm.id))
    return lambda kwargs: SQL('\n    INSERT INTO analytics_realmcount\n        (realm_id, value, property, subgroup, end_time)\n    SELECT\n        zerver_userprofile.realm_id, count(*), %(property)s, {subgroup}, %(time_end)s\n    FROM zerver_userprofile\n    JOIN (\n            SELECT DISTINCT ON (modified_user_id)\n                    modified_user_id, event_type\n            FROM\n                    zerver_realmauditlog\n            WHERE\n                    event_type IN ({user_created}, {user_activated}, {user_deactivated}, {user_reactivated}) AND\n                    {realm_clause}\n                    event_time < %(time_end)s\n            ORDER BY\n                    modified_user_id,\n                    event_time DESC\n    ) last_user_event ON last_user_event.modified_user_id = zerver_userprofile.id\n    WHERE\n        last_user_event.event_type in ({user_created}, {user_activated}, {user_reactivated})\n    GROUP BY zerver_userprofile.realm_id {group_by_clause}\n    ').format(**kwargs, user_created=Literal(AuditLogEventType.USER_CREATED), user_activated=Literal(AuditLogEventType.USER_ACTIVATED), user_deactivated=Literal(AuditLogEventType.USER_DEACTIVATED), user_reactivated=Literal(AuditLogEventType.USER_REACTIVATED), realm_clause=realm_clause)

def check_useractivityinterval_by_user_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('zerver_userprofile.realm_id = {} AND').format(Literal(realm.id))
    return lambda kwargs: SQL('\n    INSERT INTO analytics_usercount\n        (user_id, realm_id, value, property, subgroup, end_time)\n    SELECT\n        zerver_userprofile.id, zerver_userprofile.realm_id, 1, %(property)s, {subgroup}, %(time_end)s\n    FROM zerver_userprofile\n    JOIN zerver_useractivityinterval\n    ON\n        zerver_userprofile.id = zerver_useractivityinterval.user_profile_id\n    WHERE\n        zerver_useractivityinterval.end >= %(time_start)s AND\n        {realm_clause}\n        zerver_useractivityinterval.start < %(time_end)s\n    GROUP BY zerver_userprofile.id {group_by_clause}\n').format(**kwargs, realm_clause=realm_clause)

def count_realm_active_humans_query(realm: Optional[Realm]) -> QueryFn:
    if realm is None:
        realm_clause = SQL('')
    else:
        realm_clause = SQL('realm_id = {} AND').format(Literal(realm.id))
    return lambda kwargs: SQL("\n    INSERT INTO analytics_realmcount\n        (realm_id, value, property, subgroup, end_time)\n    SELECT\n            active_usercount.realm_id, count(*), %(property)s, NULL, %(time_end)s\n    FROM (\n            SELECT\n                    realm_id,\n                    user_id\n            FROM\n                    analytics_usercount\n            WHERE\n                    property = '15day_actives::day'\n                    {realm_clause}\n                    AND end_time = %(time_end)s\n    ) active_usercount\n    JOIN zerver_userprofile ON active_usercount.user_id = zerver_userprofile.id\n     AND active_usercount.realm_id = zerver_userprofile.realm_id\n    JOIN (\n            SELECT DISTINCT ON (modified_user_id)\n                    modified_user_id, event_type\n            FROM\n                    zerver_realmauditlog\n            WHERE\n                    event_type IN ({user_created}, {user_activated}, {user_deactivated}, {user_reactivated})\n                    AND event_time < %(time_end)s\n            ORDER BY\n                    modified_user_id,\n                    event_time DESC\n    ) last_user_event ON last_user_event.modified_user_id = active_usercount.user_id\n    WHERE\n            NOT zerver_userprofile.is_bot\n            AND event_type IN ({user_created}, {user_activated}, {user_reactivated})\n    GROUP BY\n            active_usercount.realm_id\n").format(**kwargs, user_created=Literal(AuditLogEventType.USER_CREATED), user_activated=Literal(AuditLogEventType.USER_ACTIVATED), user_deactivated=Literal(AuditLogEventType.USER_DEACTIVATED), user_reactivated=Literal(AuditLogEventType.USER_REACTIVATED), realm_clause=realm_clause)
count_stream_by_realm_query = lambda kwargs: SQL('\n    INSERT INTO analytics_realmcount\n        (realm_id, value, property, subgroup, end_time)\n    SELECT\n        zerver_realm.id, count(*), %(property)s, {subgroup}, %(time_end)s\n    FROM zerver_realm\n    JOIN zerver_stream\n    ON\n        zerver_realm.id = zerver_stream.realm_id AND\n    WHERE\n        zerver_realm.date_created < %(time_end)s AND\n        zerver_stream.date_created >= %(time_start)s AND\n        zerver_stream.date_created < %(time_end)s\n    GROUP BY zerver_realm.id {group_by_clause}\n').format(**kwargs)

def get_count_stats(realm: Optional[Realm] = None) -> OrderedDict[str, CountStat]:
    count_stats_: List[CountStat] = [CountStat('messages_sent:is_bot:hour', sql_data_collector(UserCount, count_message_by_user_query(realm), (UserProfile, 'is_bot')), CountStat.HOUR), CountStat('messages_sent:message_type:day', sql_data_collector(UserCount, count_message_type_by_user_query(realm), None), CountStat.DAY), CountStat('messages_sent:client:day', sql_data_collector(UserCount, count_message_by_user_query(realm), (Message, 'sending_client_id')), CountStat.DAY), CountStat('messages_in_stream:is_bot:day', sql_data_collector(StreamCount, count_message_by_stream_query(realm), (UserProfile, 'is_bot')), CountStat.DAY), LoggingCountStat('ai_credit_usage::day', UserCount, CountStat.DAY), CountStat('active_users_audit:is_bot:day', sql_data_collector(RealmCount, check_realmauditlog_by_user_query(realm), (UserProfile, 'is_bot')), CountStat.DAY), CountStat('upload_quota_used_bytes::day', sql_data_collector(RealmCount, count_upload_space_used_by_realm_query(realm), None), CountStat.DAY), LoggingCountStat('messages_read::hour', UserCount, CountStat.HOUR), LoggingCountStat('messages_read_interactions::hour', UserCount, CountStat.HOUR), CountStat('1day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=1) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('7day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=7) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('15day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=15) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('minutes_active::day', DataCollector(UserCount, do_pull_minutes_active), CountStat.DAY), LoggingCountStat('mobile_pushes_sent::day', RealmCount, CountStat.DAY), LoggingCountStat('invites_sent::day', RealmCount, CountStat.DAY), DependentCountStat('realm_active_humans::day', sql_data_collector(RealmCount, count_realm_active_humans_query(realm), None), CountStat.DAY, dependencies=['15day_actives::day'])]
    if settings.ZILENCER_ENABLED:
        count_stats_.append(LoggingCountStat('mobile_pushes_received::day', RemoteRealmCount, CountStat.DAY))
        count_stats_.append(LoggingCountStat('mobile_pushes_forwarded::day', RemoteRealmCount, CountStat.DAY))
    return OrderedDict(((stat.property, stat) for stat in count_stats_))
BOUNCER_ONLY_REMOTE_COUNT_STAT_PROPERTIES: List[str] = ['mobile_pushes_received::day', 'mobile_pushes_forwarded::day']
LOGGING_COUNT_STAT_PROPERTIES_NOT_SENT_TO_BOUNCER: set[str] = {'invites_sent::day', 'mobile_pushes_sent::day', 'active_users_log:is_bot:day', 'active_users:is_bot:day'}
COUNT_STATS: OrderedDict[str, CountStat] = get_count_stats()
REMOTE_INSTALLATION_COUNT_STATS: OrderedDict[str, CountStat] = OrderedDict()
if settings.ZILENCER_ENABLED:
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_received::day'] = LoggingCountStat('mobile_pushes_received::day', RemoteInstallationCount, CountStat.DAY)
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_forwarded::day'] = LoggingCountStat('mobile_pushes_forwarded::day', RemoteInstallationCount, CountStat.DAY)
ALL_COUNT_STATS: OrderedDict[str, CountStat] = OrderedDict(list(COUNT_STATS.items()) + list(REMOTE_INSTALLATION_COUNT_STATS.items()))
