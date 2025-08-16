from typing import Type, List, Dict

class CountStat:
    HOUR: str = 'hour'
    DAY: str = 'day'
    FREQUENCIES: frozenset = frozenset([HOUR, DAY])

    @property
    def time_increment(self) -> timedelta:
        if self.frequency == CountStat.HOUR:
            return timedelta(hours=1)
        return timedelta(days=1)

    def __init__(self, property: str, data_collector: DataCollector, frequency: str, interval=None) -> None:
        self.property: str = property
        self.data_collector: DataCollector = data_collector
        if frequency not in self.FREQUENCIES:
            raise AssertionError(f'Unknown frequency: {frequency}')
        self.frequency: str = frequency
        if interval is not None:
            self.interval = interval
        else:
            self.interval = self.time_increment

    def last_successful_fill(self) -> datetime:
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

    def __init__(self, property: str, output_table: Type[models.Model], frequency: str) -> None:
        CountStat.__init__(self, property, DataCollector(output_table, None), frequency)

class DependentCountStat(CountStat):

    def __init__(self, property: str, data_collector: DataCollector, frequency: str, interval=None, dependencies=[]) -> None:
        CountStat.__init__(self, property, data_collector, frequency, interval=interval)
        self.dependencies: List[str] = dependencies

class DataCollector:

    def __init__(self, output_table: Type[models.Model], pull_function: Callable) -> None:
        self.output_table: Type[models.Model] = output_table
        self.pull_function: Callable = pull_function

    def depends_on_realm(self) -> bool:
        return self.output_table in (UserCount, StreamCount)

def process_count_stat(stat: CountStat, fill_to_time: datetime, realm=None) -> None:
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

def do_update_fill_state(fill_state: FillState, end_time: datetime, state: str) -> None:
    fill_state.end_time = end_time
    fill_state.state = state
    fill_state.save()

def do_fill_count_stat_at_hour(stat: CountStat, end_time: datetime, realm=None) -> None:
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

def do_aggregate_to_summary_table(stat: CountStat, end_time: datetime, realm=None) -> None:
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

def do_increment_logging_stat(model_object_for_bucket, stat: CountStat, subgroup, event_time: datetime, increment=1) -> None:
    if not increment:
        return
    table = stat.data_collector.output_table
    id_args = {}
    conflict_args = []
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

QueryFn: TypeAlias = Callable[[Dict[str, Composable]], Composable]

def do_pull_by_sql_query(property: str, start_time: datetime, end_time: datetime, query: QueryFn, group_by) -> int:
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

def sql_data_collector(output_table: Type[models.Model], query: QueryFn, group_by) -> DataCollector:

    def pull_function(property: str, start_time: datetime, end_time: datetime, realm=None) -> int:
        return do_pull_by_sql_query(property, start_time, end_time, query, group_by)
    return DataCollector(output_table, pull_function)

def get_count_stats(realm=None) -> Dict[str, CountStat]:
    count_stats_: List[CountStat] = [CountStat('messages_sent:is_bot:hour', sql_data_collector(UserCount, count_message_by_user_query(realm), (UserProfile, 'is_bot')), CountStat.HOUR), CountStat('messages_sent:message_type:day', sql_data_collector(UserCount, count_message_type_by_user_query(realm), None), CountStat.DAY), CountStat('messages_sent:client:day', sql_data_collector(UserCount, count_message_by_user_query(realm), (Message, 'sending_client_id')), CountStat.DAY), CountStat('messages_in_stream:is_bot:day', sql_data_collector(StreamCount, count_message_by_stream_query(realm), (UserProfile, 'is_bot')), CountStat.DAY), LoggingCountStat('ai_credit_usage::day', UserCount, CountStat.DAY), CountStat('active_users_audit:is_bot:day', sql_data_collector(RealmCount, check_realmauditlog_by_user_query(realm), (UserProfile, 'is_bot')), CountStat.DAY), CountStat('upload_quota_used_bytes::day', sql_data_collector(RealmCount, count_upload_space_used_by_realm_query(realm), None), CountStat.DAY), LoggingCountStat('messages_read::hour', UserCount, CountStat.HOUR), LoggingCountStat('messages_read_interactions::hour', UserCount, CountStat.HOUR), CountStat('1day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=1) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('7day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=7) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('15day_actives::day', sql_data_collector(UserCount, check_useractivityinterval_by_user_query(realm), None), CountStat.DAY, interval=timedelta(days=15) - UserActivityInterval.MIN_INTERVAL_LENGTH), CountStat('minutes_active::day', DataCollector(UserCount, do_pull_minutes_active), CountStat.DAY), LoggingCountStat('mobile_pushes_sent::day', RealmCount, CountStat.DAY), LoggingCountStat('invites_sent::day', RealmCount, CountStat.DAY), DependentCountStat('realm_active_humans::day', sql_data_collector(RealmCount, count_realm_active_humans_query(realm), None), CountStat.DAY, dependencies=['15day_actives::day'])]
    if settings.ZILENCER_ENABLED:
        count_stats_.append(LoggingCountStat('mobile_pushes_received::day', RemoteRealmCount, CountStat.DAY))
        count_stats_.append(LoggingCountStat('mobile_pushes_forwarded::day', RemoteRealmCount, CountStat.DAY))
    return OrderedDict(((stat.property, stat) for stat in count_stats_))

BOUNCER_ONLY_REMOTE_COUNT_STAT_PROPERTIES: List[str] = ['mobile_pushes_received::day', 'mobile_pushes_forwarded::day']
LOGGING_COUNT_STAT_PROPERTIES_NOT_SENT_TO_BOUNCER: set = {'invites_sent::day', 'mobile_pushes_sent::day', 'active_users_log:is_bot:day', 'active_users:is_bot:day'}
COUNT_STATS: Dict[str, CountStat] = get_count_stats()
REMOTE_INSTALLATION_COUNT_STATS: OrderedDict = OrderedDict()
if settings.ZILENCER_ENABLED:
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_received::day'] = LoggingCountStat('mobile_pushes_received::day', RemoteInstallationCount, CountStat.DAY)
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_forwarded::day'] = LoggingCountStat('mobile_pushes_forwarded::day', RemoteInstallationCount, CountStat.DAY)
ALL_COUNT_STATS: OrderedDict = OrderedDict(list(COUNT_STATS.items()) + list(REMOTE_INSTALLATION_COUNT_STATS.items()))
