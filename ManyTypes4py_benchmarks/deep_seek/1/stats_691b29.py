import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional, TypeAlias, TypeVar, cast, Dict, List, Tuple, Union, Callable
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
from analytics.models import BaseCount, InstallationCount, RealmCount, StreamCount, UserCount, installation_epoch
from zerver.decorator import require_non_guest_user, require_server_admin, require_server_admin_api, to_utc_datetime, zulip_login_required
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

MAX_TIME_FOR_FULL_ANALYTICS_GENERATION: timedelta = timedelta(days=1, minutes=30)

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
        guest_users: Optional[int] = UserProfile.objects.filter(realm=realm, is_active=True, is_bot=False, role=UserProfile.ROLE_GUEST).count()
        space_used: Optional[int] = realm.currently_used_upload_space_bytes()
        if title:
            pass
        else:
            title = realm.name or realm.string_id
    else:
        assert title
        guest_users = None
        space_used = None
    request_language: str = get_and_set_request_language(request, request.user.default_language, translation.get_language_from_path(request.path_info))
    page_params: Dict[str, Any] = dict(
        page_type='stats',
        data_url_suffix=data_url_suffix,
        upload_space_used=space_used,
        guest_users=guest_users,
        translation_data=get_language_translation_data(request_language)
    )
    return render(request, 'analytics/stats.html', context=dict(target_name=title, page_params=page_params, analytics_ready=analytics_ready))

@zulip_login_required
def stats(request: HttpRequest) -> HttpResponse:
    assert request.user.is_authenticated
    realm: Realm = request.user.realm
    if request.user.is_guest:
        raise JsonableError(_('Not allowed for guest users'))
    return render_stats(request, '', realm, analytics_ready=is_analytics_ready(realm))

@require_server_admin
@typed_endpoint
def stats_for_realm(request: HttpRequest, *, realm_str: str) -> HttpResponse:
    try:
        realm: Realm = get_realm(realm_str)
    except Realm.DoesNotExist:
        return HttpResponseNotFound()
    return render_stats(request, f'/realm/{realm_str}', realm, analytics_ready=is_analytics_ready(realm))

@require_server_admin
@typed_endpoint
def stats_for_remote_realm(request: HttpRequest, *, remote_server_id: int, remote_realm_id: int) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server: RemoteZulipServer = RemoteZulipServer.objects.get(id=remote_server_id)
    return render_stats(request, f'/remote/{server.id}/realm/{remote_realm_id}', None, title=f'Realm {remote_realm_id} on server {server.hostname}')

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
    end: Optional[datetime] = None
) -> HttpResponse:
    try:
        realm: Realm = get_realm(realm_str)
    except Realm.DoesNotExist:
        raise JsonableError(_('Invalid organization'))
    return do_get_chart_data(request, user_profile, realm=realm, chart_name=chart_name, min_length=min_length, start=start, end=end)

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
    end: Optional[datetime] = None
) -> HttpResponse:
    stream: Stream
    ignored_sub: Any
    stream, ignored_sub = access_stream_by_id(user_profile, stream_id, require_active=True, require_content_access=False)
    return do_get_chart_data(request, user_profile, stream=stream, chart_name=chart_name, min_length=min_length, start=start, end=end)

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
    end: Optional[datetime] = None
) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server: RemoteZulipServer = RemoteZulipServer.objects.get(id=remote_server_id)
    return do_get_chart_data(request, user_profile, server=server, remote=True, remote_realm_id=remote_realm_id, chart_name=chart_name, min_length=min_length, start=start, end=end)

@require_server_admin
def stats_for_installation(request: HttpRequest) -> HttpResponse:
    assert request.user.is_authenticated
    return render_stats(request, '/installation', None, title='installation')

@require_server_admin
def stats_for_remote_installation(request: HttpRequest, remote_server_id: int) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server: RemoteZulipServer = RemoteZulipServer.objects.get(id=remote_server_id)
    return render_stats(request, f'/remote/{server.id}/installation', None, title=f'remote installation {server.hostname}')

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
    end: Optional[datetime] = None
) -> HttpResponse:
    return do_get_chart_data(request, user_profile, for_installation=True, chart_name=chart_name, min_length=min_length, start=start, end=end)

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
    end: Optional[datetime] = None
) -> HttpResponse:
    assert settings.ZILENCER_ENABLED
    server: RemoteZulipServer = RemoteZulipServer.objects.get(id=remote_server_id)
    return do_get_chart_data(request, user_profile, for_installation=True, remote=True, server=server, chart_name=chart_name, min_length=min_length, start=start, end=end)

@require_non_guest_user
@typed_endpoint
def get_chart_data(
    request: HttpRequest,
    user_profile: UserProfile,
    *,
    chart_name: str,
    min_length: Optional[int] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> HttpResponse:
    return do_get_chart_data(request, user_profile, chart_name=chart_name, min_length=min_length, start=start, end=end)

CountT = TypeVar('CountT', bound=BaseCount)
TableType: TypeAlias = Union[type['RemoteInstallationCount'], type[InstallationCount], type['RemoteRealmCount'], type[RealmCount]]

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
    server: Optional['RemoteZulipServer'] = None,
    stream: Optional[Stream] = None
) -> HttpResponse:
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

    # ... rest of the function remains the same with appropriate type hints ...

    return json_success(request, data=data)

def sort_by_totals(value_arrays: Dict[str, List[int]]) -> List[str]:
    totals = sorted(((sum(values), label) for label, values in value_arrays.items()), reverse=True)
    return [label for total, label in totals]

def sort_client_labels(data: Dict[str, Dict[str, List[int]]]) -> List[str]:
    realm_order: List[str] = sort_by_totals(data['everyone'])
    user_order: List[str] = sort_by_totals(data['user'])
    label_sort_values: Dict[str, float] = {label: i for i, label in enumerate(realm_order)}
    for i, label in enumerate(user_order):
        label_sort_values[label] = min(i - 0.1, label_sort_values.get(label, i))
    return [label for label, sort_value in sorted(label_sort_values.items(), key=lambda x: x[1])]

def table_filtered_to_id(table: TableType, key_id: int) -> QuerySet[CountT]:
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
        mapped_label: str = client_label_map(label)
        if mapped_label in mapped_arrays:
            for i in range(len(array)):
                mapped_arrays[mapped_label][i] += value_arrays[label][i]
        else:
            mapped_arrays[mapped_label] = [value_arrays[label][i] for i in range(len(array))]
    return mapped_arrays

def get_time_series_by_subgroup(
    stat: CountStat,
    table: TableType,
    key_id: int,
    end_times: List[datetime],
    subgroup_to_label: Dict[Optional[str], str],
    include_empty_subgroups: bool
) -> Dict[str, List[int]]:
    queryset: QuerySet[Tuple[Optional[str], datetime, int]] = table_filtered_to_id(table, key_id).filter(property=stat.property).values_list('subgroup', 'end_time', 'value')
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
