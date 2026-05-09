from typing import TypeVar, Dict, List, Optional, Union
from django.db.models import QuerySet
from analytics.models import BaseCount, InstallationCount, RealmCount, StreamCount, UserCount, RemoteInstallationCount, RemoteRealmCount
from zerver.models import Realm, UserProfile, Stream, Client
from analytics.lib.counts import CountStat

CountT = TypeVar('CountT', bound=BaseCount)

def table_filtered_to_id(table: Type[CountT], key_id: Union[int, str]) -> QuerySet:
    if table == RealmCount:
        return table._default_manager.filter(realm_id=key_id)
    elif table == UserCount:
        return table._default_manager.filter(user_id=key_id)
    elif table == StreamCount:
        return table._default_manager.filter(stream_id=key_id)
    elif table == InstallationCount:
        return table._default_manager.all()
    elif table == RemoteInstallationCount:
        return table._default_manager.filter(server_id=key_id)
    elif table == RemoteRealmCount:
        return table._default_manager.filter(realm_id=key_id)
    else:
        raise AssertionError(f'Unknown table: {table}')

def client_label_map(name: str) -> str:
    if name == 'website':
        return 'Web app'
    if name.startswith('desktop app'):
        return 'Old desktop app'
    if name == 'ZulipElectron':
        return 'Desktop app'
    if name == 'ZulipTerminal':
        return 'Terminal app'
    if name == 'ZulipAndroid':
        return 'Old Android app'
    if name == 'ZulipiOS':
        return 'Old iOS app'
    if name == 'ZulipMobile':
        return 'Mobile app (React Native)'
    if name in ['ZulipFlutter', 'ZulipMobile/flutter']:
        return 'Mobile app beta (Flutter)'
    if name in ['ZulipPython', 'API: Python']:
        return 'Python API'
    if name.startswith('Zulip') and name.endswith('Webhook'):
        return name.removeprefix('Zulip').removesuffix('Webhook') + ' webhook'
    return name

def rewrite_client_arrays(value_arrays: Dict[str, List[int]]) -> Dict[str, List[int]]:
    mapped_arrays = {}
    for label, array in value_arrays.items():
        mapped_label = client_label_map(label)
        if mapped_label in mapped_arrays:
            for i in range(len(array)):
                mapped_arrays[mapped_label][i] += value_arrays[label][i]
        else:
            mapped_arrays[mapped_label] = [value_arrays[label][i] for i in range(len(array))]
    return mapped_arrays

def get_time_series_by_subgroup(stat: CountStat, table: Type[CountT], key_id: Union[int, str], end_times: List[datetime], subgroup_to_label: Dict[Optional[str], str], include_empty_subgroups: bool) -> Dict[str, List[int]]:
    queryset = table_filtered_to_id(table, key_id).filter(property=stat.property).values_list('subgroup', 'end_time', 'value')
    value_dicts = defaultdict(lambda: defaultdict(int))
    for subgroup, end_time, value in queryset:
        value_dicts[subgroup][end_time] = value
    value_arrays = {}
    for subgroup, label in subgroup_to_label.items():
        if subgroup in value_dicts or include_empty_subgroups:
            value_arrays[label] = [value_dicts[subgroup][end_time] for end_time in end_times]
    if stat == COUNT_STATS['messages_sent:client:day']:
        return rewrite_client_arrays(value_arrays)
    return value_arrays

def sort_by_totals(value_arrays: Dict[str, List[int]]) -> List[str]:
    totals = sorted(((sum(values), label) for label, values in value_arrays.items()), reverse=True)
    return [label for total, label in totals]

def sort_client_labels(data: Dict[str, Dict[str, List[int]]]) -> List[str]:
    realm_order = sort_by_totals(data['everyone'])
    user_order = sort_by_totals(data['user'])
    label_sort_values = {label: i for i, label in enumerate(realm_order)}
    for i, label in enumerate(user_order):
        label_sort_values[label] = min(i - 0.1, label_sort_values.get(label, i))
    return [label for label, sort_value in sorted(label_sort_values.items(), key=lambda x: x[1])]
