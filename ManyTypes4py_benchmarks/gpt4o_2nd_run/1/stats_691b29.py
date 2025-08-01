import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from django.conf import settings
from django.db.models import QuerySet
from django.http import HttpRequest, HttpResponse, HttpResponseNotFound
from django.shortcuts import render
from django.utils import translation
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _

from pydantic import BeforeValidator, Json, NonNegativeInt

from analytics.lib.counts import COUNT_STATS, CountStat
from analytics.lib.time_utils import time_range
from analytics.models import (
    BaseCount,
    InstallationCount,
    RealmCount,
    StreamCount,
    UserCount,
    installation_epoch,
)
from zerver.decorator import (
    require_non_guest_user,
    require_server_admin,
    require_server_admin_api,
    to_utc_datetime,
    zulip_login_required,
)
from zerver.lib.exceptions import JsonableError
from zerver.lib.i18n import get_and_set_request_language, get_language_translation_data
from zerver.lib.response import json_success
from zerver.lib.streams import access_stream_by_id
from zerver.lib.timestamp import convert_to_UTC
from zerver.lib.typed_endpoint import PathOnly, typed_endpoint
from zerver.models import Client, Realm, Stream, UserProfile
from zerver.models.realms import get_realm

if settings.ZILENCER_ENABLED:
    from zilencer.models import RemoteInstallationCount, RemoteRealmCount, RemoteZulipServer

MAX_TIME_FOR_FULL_ANALYTICS_GENERATION = timedelta(days=1, minutes=30)

def is_analytics_ready(realm: Realm) -> bool:
    return timezone_now() - realm.date_created > MAX_TIME_FOR_FULL_ANALYTICS_GENERATION

def render_stats(
    request: HttpRequest,
    data_url_suffix: str,
    realm: Optional[Realm],
    *,
    title: Optional[str] = None,
    analytics_ready: bool = True
) -> HttpResponse:
    assert request.user.is_authenticated
    if realm is not None:
        guest_users = UserProfile.objects.filter(
            realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST
        ).count()
        space_used = realm.currently_used_upload_space_bytes()
        if title:
            pass
        else:
            title = realm.name or realm.string_id
    else:
        assert title
        guest_users = None
        space_used = None
    request_language = get_and_set_request_language(
        request, request.user.default_language, translation.get_language_from_path(request.path_info)
    )
    page_params = dict(
        page_type='stats',
        data_url_suffix=data_url_suffix,
        upload_space_used=space_used,
        guest_users=guest_users,
        translation_data=get_language_translation_data(request_language),
    )
    return render(
        request,
        'analytics/stats.html',
        context=dict(target_name=title, page_params=page_params, analytics_ready=analytics_ready),
    )

@zulip_login_required
def stats(request: HttpRequest) -> HttpResponse:
    assert request.user.is_authenticated
    realm = request.user.realm
    if request.user.is_guest:
        raise JsonableError(_('Not allowed for guest users'))
    return render_stats(request, '', realm, analytics_ready=is_analytics_ready(realm))

@require_server_admin
@typed_endpoint
def stats_for_realm(request: HttpRequest, *, realm_str: str) -> HttpResponse:
    try:
        realm = get_realm(realm_str)
    except Realm.DoesNotExist:
        return HttpResponseNotFound()
    return render_stats(request, f'/realm/{realm_str}', realm, analytics_ready=is_analytics_ready(realm))

@require_server_admin
@typed_endpoint
def stats_for_remote_realm(
    request: HttpRequest, *, remote_server_id: int, remote_realm_id: int
) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server = RemoteZulipServer.objects.get(id=remote_server_id)
    return render_stats(
        request,
        f'/remote/{server.id}/realm/{remote_realm_id}',
        None,
        title=f'Realm {remote_realm_id} on server {server.hostname}',
    )

@require_server_admin_api
@typed_endpoint
def get_chart_data_for_realm(
    request: HttpRequest,
    user_profile: UserProfile,
    /,
    *,
    realm_str: str,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    try:
        realm = get_realm(realm_str)
    except Realm.DoesNotExist:
        raise JsonableError(_('Invalid organization'))
    return do_get_chart_data(
        request,
        user_profile,
        realm=realm,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_non_guest_user
@typed_endpoint
def get_chart_data_for_stream(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    stream_id: int,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    stream, ignored_sub = access_stream_by_id(
        user_profile, stream_id, require_active=True, require_content_access=False
    )
    return do_get_chart_data(
        request,
        user_profile,
        stream=stream,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_server_admin_api
@typed_endpoint
def get_chart_data_for_remote_realm(
    request: HttpRequest,
    user_profile: UserProfile,
    /,
    *,
    remote_server_id: int,
    remote_realm_id: int,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server = RemoteZulipServer.objects.get(id=remote_server_id)
    return do_get_chart_data(
        request,
        user_profile,
        server=server,
        remote=True,
        remote_realm_id=remote_realm_id,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_server_admin
def stats_for_installation(request: HttpRequest) -> HttpResponse:
    assert request.user.is_authenticated
    return render_stats(request, '/installation', None, title='installation')

@require_server_admin
def stats_for_remote_installation(request: HttpRequest, remote_server_id: int) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server = RemoteZulipServer.objects.get(id=remote_server_id)
    return render_stats(
        request, f'/remote/{server.id}/installation', None, title=f'remote installation {server.hostname}'
    )

@require_server_admin_api
@typed_endpoint
def get_chart_data_for_installation(
    request: HttpRequest,
    user_profile: UserProfile,
    /,
    *,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    return do_get_chart_data(
        request,
        user_profile,
        for_installation=True,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_server_admin_api
@typed_endpoint
def get_chart_data_for_remote_installation(
    request: HttpRequest,
    user_profile: UserProfile,
    /,
    *,
    remote_server_id: int,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server = RemoteZulipServer.objects.get(id=remote_server_id)
    return do_get_chart_data(
        request,
        user_profile,
        for_installation=True,
        remote=True,
        server=server,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_non_guest_user
@typed_endpoint
def get_chart_data(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> HttpResponse:
    return do_get_chart_data(
        request,
        user_profile,
        chart_name=chart_name,
        min_length=min_length,
        start=start,
        end=end,
    )

@require_non_guest_user
def do_get_chart_data(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    realm: Optional[Realm] = None,
    for_installation: bool = False,
    remote: bool = False,
    remote_realm_id: Optional[int] = None,
    server: Optional[RemoteZulipServer] = None,
    stream: Optional[Stream] = None,
) -> HttpResponse:
    TableType = Union[
        Type[RemoteInstallationCount],
        Type[InstallationCount],
        Type[RemoteRealmCount],
        Type[RealmCount],
    ]
    if for_installation:
        if remote:
            assert settings.ZILENCER_ENABLED
            aggregate_table: TableType = RemoteInstallationCount
            assert server is not None
        else:
            aggregate_table = InstallationCount
    elif remote:
        assert settings.ZILENCER_ENABLED
        aggregate_table = RemoteRealmCount
        assert server is not None
        assert remote_realm_id is not None
    else:
        aggregate_table = RealmCount

    if chart_name == 'number_of_humans':
        stats = [
            COUNT_STATS['1day_actives::day'],
            COUNT_STATS['realm_active_humans::day'],
            COUNT_STATS['active_users_audit:is_bot:day'],
        ]
        tables = (aggregate_table,)
        subgroup_to_label = {
            stats[0]: {None: '_1day'},
            stats[1]: {None: '_15day'},
            stats[2]: {'false': 'all_time'},
        }
        labels_sort_function = None
        include_empty_subgroups = True
    elif chart_name == 'messages_sent_over_time':
        stats = [COUNT_STATS['messages_sent:is_bot:hour']]
        tables = (aggregate_table, UserCount)
        subgroup_to_label = {stats[0]: {'false': 'human', 'true': 'bot'}}
        labels_sort_function = None
        include_empty_subgroups = True
    elif chart_name == 'messages_sent_by_message_type':
        stats = [COUNT_STATS['messages_sent:message_type:day']]
        tables = (aggregate_table, UserCount)
        subgroup_to_label = {
            stats[0]: {
                'public_stream': _('Public channels'),
                'private_stream': _('Private channels'),
                'private_message': _('Direct messages'),
                'huddle_message': _('Group direct messages'),
            }
        }
        labels_sort_function = lambda data: sort_by_totals(data['everyone'])
        include_empty_subgroups = True
    elif chart_name == 'messages_sent_by_client':
        stats = [COUNT_STATS['messages_sent:client:day']]
        tables = (aggregate_table, UserCount)
        subgroup_to_label = {
            stats[0]: {str(id): name for id, name in Client.objects.values_list('id', 'name')}
        }
        labels_sort_function = sort_client_labels
        include_empty_subgroups = False
    elif chart_name == 'messages_read_over_time':
        stats = [COUNT_STATS['messages_read::hour']]
        tables = (aggregate_table, UserCount)
        subgroup_to_label = {stats[0]: {None: 'read'}}
        labels_sort_function = None
        include_empty_subgroups = True
    elif chart_name == 'messages_sent_by_stream':
        if stream is None:
            raise JsonableError(
                _('Missing channel for chart: {chart_name}').format(chart_name=chart_name)
            )
        stats = [COUNT_STATS['messages_in_stream:is_bot:day']]
        tables = (aggregate_table, StreamCount)
        subgroup_to_label = {stats[0]: {'false': 'human', 'true': 'bot'}}
        labels_sort_function = None
        include_empty_subgroups = True
    else:
        raise JsonableError(_('Unknown chart name: {chart_name}').format(chart_name=chart_name))

    if start is not None:
        start = convert_to_UTC(start)
    if end is not None:
        end = convert_to_UTC(end)
    if start is not None and end is not None and (start > end):
        raise JsonableError(
            _('Start time is later than end time. Start: {start}, End: {end}').format(
                start=start, end=end
            )
        )
    if realm is None:
        realm = user_profile.realm
    if remote:
        assert server is not None
        assert aggregate_table is RemoteInstallationCount or aggregate_table is RemoteRealmCount
        aggregate_table_remote = cast(
            Union[Type[RemoteInstallationCount], Type[RemoteRealmCount]], aggregate_table
        )
        if not aggregate_table_remote.objects.filter(server=server).exists():
            raise JsonableError(
                _('No analytics data available. Please contact your server administrator.')
            )
        if start is None:
            first = aggregate_table_remote.objects.filter(server=server).order_by('remote_id').first()
            assert first is not None
            start = first.end_time
        if end is None:
            last = aggregate_table_remote.objects.filter(server=server).order_by('remote_id').last()
            assert last is not None
            end = last.end_time
    else:
        if start is None:
            if for_installation:
                start = installation_epoch()
            else:
                start = realm.date_created
        if end is None:
            end = max(
                (
                    stat.last_successful_fill() or datetime.min.replace(tzinfo=timezone.utc)
                    for stat in stats
                )
            )
        if start > end and timezone_now() - start > MAX_TIME_FOR_FULL_ANALYTICS_GENERATION:
            logging.warning(
                'User from realm %s attempted to access /stats, but the computed start time: %s (creation of realm or installation) is later than the computed end time: %s (last successful analytics update). Is the analytics cron job running?',
                realm.string_id,
                start,
                end,
            )
            raise JsonableError(
                _('No analytics data available. Please contact your server administrator.')
            )

    assert len({stat.frequency for stat in stats}) == 1
    end_times = time_range(start, end, stats[0].frequency, min_length)
    data: Dict[str, Any] = {
        'end_times': [int(end_time.timestamp()) for end_time in end_times],
        'frequency': stats[0].frequency,
    }
    aggregation_level: Dict[Type[BaseCount], str] = {
        InstallationCount: 'everyone',
        RealmCount: 'everyone',
        UserCount: 'user',
        StreamCount: 'everyone',
    }
    if settings.ZILENCER_ENABLED:
        aggregation_level[RemoteInstallationCount] = 'everyone'
        aggregation_level[RemoteRealmCount] = 'everyone'
    id_value: Dict[Type[BaseCount], int] = {
        InstallationCount: -1,
        RealmCount: realm.id,
        UserCount: user_profile.id,
    }
    if stream is not None:
        id_value[StreamCount] = stream.id
    if settings.ZILENCER_ENABLED:
        if server is not None:
            id_value[RemoteInstallationCount] = server.id
        if remote_realm_id is not None:
            id_value[RemoteRealmCount] = remote_realm_id

    for table in tables:
        data[aggregation_level[table]] = {}
        for stat in stats:
            data[aggregation_level[table]].update(
                get_time_series_by_subgroup(
                    stat,
                    table,
                    id_value[table],
                    end_times,
                    subgroup_to_label[stat],
                    include_empty_subgroups,
                )
            )
    if labels_sort_function is not None:
        data['display_order'] = labels_sort_function(data)
    else:
        data['display_order'] = None
    return json_success(request, data=data)

def sort_by_totals(value_arrays: Dict[str, List[int]]) -> List[str]:
    totals = sorted(
        ((sum(values), label) for label, values in value_arrays.items()), reverse=True
    )
    return [label for total, label in totals]

def sort_client_labels(data: Dict[str, Dict[str, List[int]]]) -> List[str]:
    realm_order = sort_by_totals(data['everyone'])
    user_order = sort_by_totals(data['user'])
    label_sort_values = {label: i for i, label in enumerate(realm_order)}
    for i, label in enumerate(user_order):
        label_sort_values[label] = min(i - 0.1, label_sort_values.get(label, i))
    return [label for label, sort_value in sorted(label_sort_values.items(), key=lambda x: x[1])]

CountT = TypeVar('CountT', bound=BaseCount)

def table_filtered_to_id(table: Type[CountT], key_id: int) -> QuerySet[CountT]:
    if table == RealmCount:
        return table._default_manager.filter(realm_id=key_id)
    elif table == UserCount:
        return table._default_manager.filter(user_id=key_id)
    elif table == StreamCount:
        return table._default_manager.filter(stream_id=key_id)
    elif table == InstallationCount:
        return table._default_manager.all()
    elif settings.ZILENCER_ENABLED and table == RemoteInstallationCount:
        return table._default_manager.filter(server_id=key_id)
    elif settings.ZILENCER_ENABLED and table == RemoteRealmCount:
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
    mapped_arrays: Dict[str, List[int]] = {}
    for label, array in value_arrays.items():
        mapped_label = client_label_map(label)
        if mapped_label in mapped_arrays:
            for i in range(len(array)):
                mapped_arrays[mapped_label][i] += value_arrays[label][i]
        else:
            mapped_arrays[mapped_label] = [value_arrays[label][i] for i in range(len(array))]
    return mapped_arrays

def get_time_series_by_subgroup(
    stat: CountStat,
    table: Type[BaseCount],
    key_id: int,
    end_times: List[datetime],
    subgroup_to_label: Dict[Optional[str], str],
    include_empty_subgroups: bool,
) -> Dict[str, List[int]]:
    queryset = table_filtered_to_id(table, key_id).filter(property=stat.property).values_list(
        'subgroup', 'end_time', 'value'
    )
    value_dicts: Dict[Optional[str], Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    for subgroup, end_time, value in queryset:
        value_dicts[subgroup][end_time] = value
    value_arrays: Dict[str, List[int]] = {}
    for subgroup, label in subgroup_to_label.items():
        if subgroup in value_dicts or include_empty_subgroups:
            value_arrays[label] = [value_dicts[subgroup][end_time] for end_time in end_times]
    if stat == COUNT_STATS['messages_sent:client:day']:
        return rewrite_client_arrays(value_arrays)
    return value_arrays
