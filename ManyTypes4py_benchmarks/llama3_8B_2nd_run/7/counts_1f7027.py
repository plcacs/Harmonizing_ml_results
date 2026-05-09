from typing import TypeAlias, Union
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from psycopg2.sql import SQL, Composable, Identifier, Literal
from typing_extensions import override

CountStat: TypeAlias = 'CountStat'
DataCollector: TypeAlias = 'DataCollector'
QueryFn: TypeAlias = 'Callable[[dict[str, Composable]], Composable]'

def process_count_stat(stat: CountStat, fill_to_time: datetime, realm: Union[None, Realm] = None) -> None:
    # ...

def do_update_fill_state(fill_state: FillState, end_time: datetime, state: str) -> None:
    # ...

def do_fill_count_stat_at_hour(stat: CountStat, end_time: datetime, realm: Union[None, Realm] = None) -> None:
    # ...

def do_aggregate_to_summary_table(stat: CountStat, end_time: datetime, realm: Union[None, Realm] = None) -> None:
    # ...

def do_pull_by_sql_query(property: str, start_time: datetime, end_time: datetime, query: QueryFn, group_by: Union[None, tuple]) -> int:
    # ...

def sql_data_collector(output_table: Type[models.Model], query: QueryFn, group_by: Union[None, tuple]) -> DataCollector:
    # ...

def get_count_stats(realm: Union[None, Realm] = None) -> OrderedDict[str, CountStat]:
    # ...

def do_pull_minutes_active(property: str, start_time: datetime, end_time: datetime, realm: Union[None, Realm] = None) -> int:
    # ...

def count_upload_space_used_by_realm_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def count_message_by_user_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def count_message_type_by_user_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def count_message_by_stream_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def check_realmauditlog_by_user_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def check_useractivityinterval_by_user_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

def count_realm_active_humans_query(realm: Union[None, Realm]) -> QueryFn:
    # ...

COUNT_STATS: OrderedDict[str, CountStat] = get_count_stats()
REMOTE_INSTALLATION_COUNT_STATS: OrderedDict[str, CountStat] = OrderedDict()
if settings.ZILENCER_ENABLED:
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_received::day'] = LoggingCountStat('mobile_pushes_received::day', RemoteInstallationCount, CountStat.DAY)
    REMOTE_INSTALLATION_COUNT_STATS['mobile_pushes_forwarded::day'] = LoggingCountStat('mobile_pushes_forwarded::day', RemoteInstallationCount, CountStat.DAY)
ALL_COUNT_STATS: OrderedDict[str, CountStat] = OrderedDict(list(COUNT_STATS.items()) + list(REMOTE_INSTALLATION_COUNT_STATS.items()))
