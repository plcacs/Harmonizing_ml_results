import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from django.conf import settings
from django.utils.timezone import now as timezone_now
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.users import check_user_can_access_all_users, get_accessible_user_ids
from zerver.models import Realm, UserPresence, UserProfile


def func_0zr8n629(
    all_rows: List[Dict[str, Any]], 
    slim_presence: bool
) -> Dict[str, Dict[str, Any]]:
    if slim_presence:
        get_user_key: Callable[[Dict[str, Any]], str] = lambda row: str(row['user_profile_id'])
        get_user_presence_info: Callable[[datetime, datetime], Dict[str, Any]] = func_vryv3i17
    else:
        get_user_key = lambda row: row['user_profile__email']
        get_user_presence_info = func_tzp20bay
    user_statuses: Dict[str, Dict[str, Any]] = {}
    for presence_row in all_rows:
        user_key = get_user_key(presence_row)
        last_active_time = func_nvjy4bri(
            presence_row['last_active_time'], 
            presence_row['user_profile__date_joined']
        )
        last_connected_time = func_nvjy4bri(
            presence_row['last_connected_time'], 
            presence_row['user_profile__date_joined']
        )
        info = get_user_presence_info(last_active_time, last_connected_time)
        user_statuses[user_key] = info
    return user_statuses


def func_nvjy4bri(dt: Optional[datetime], date_joined: datetime) -> datetime:
    if dt is None:
        return date_joined
    return dt


def func_vryv3i17(
    last_active_time: datetime, 
    last_connected_time: datetime
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    result['active_timestamp'] = datetime_to_timestamp(last_active_time)
    result['idle_timestamp'] = datetime_to_timestamp(last_connected_time)
    return result


def func_tzp20bay(
    last_active_time: datetime, 
    last_connected_time: datetime
) -> Dict[str, Dict[str, Any]]:
    most_recent_info = func_sjovzerc(last_active_time, last_connected_time)
    result: Dict[str, Dict[str, Any]] = {}
    result['aggregated'] = dict(
        client=most_recent_info['client'], 
        status=most_recent_info['status'], 
        timestamp=most_recent_info['timestamp']
    )
    result['website'] = most_recent_info
    return result


def func_sjovzerc(
    last_active_time: datetime, 
    last_connected_time: datetime
) -> Dict[str, Union[str, bool, float]]:
    if last_active_time + timedelta(
        seconds=settings.PRESENCE_LEGACY_EVENT_OFFSET_FOR_ACTIVITY_SECONDS
    ) >= last_connected_time:
        status = UserPresence.LEGACY_STATUS_ACTIVE
        timestamp = datetime_to_timestamp(last_active_time)
    else:
        status = UserPresence.LEGACY_STATUS_IDLE
        timestamp = datetime_to_timestamp(last_connected_time)
    pushable = False
    return dict(
        client='website', 
        status=status, 
        timestamp=timestamp,
        pushable=pushable
    )


def func_bjxdrq8q(
    user_profile_id: int, 
    slim_presence: bool = False
) -> Dict[str, Dict[str, Any]]:
    query = UserPresence.objects.filter(user_profile_id=user_profile_id
        ).values('last_active_time', 'last_connected_time',
        'user_profile__email', 'user_profile_id',
        'user_profile__enable_offline_push_notifications',
        'user_profile__date_joined')
    presence_rows = list(query)
    return func_0zr8n629(presence_rows, slim_presence)


def func_okz0pz7c(
    realm: Realm, 
    slim_presence: bool = False,
    last_update_id_fetched_by_client: Optional[int] = None, 
    history_limit_days: Optional[int] = None,
    requesting_user_profile: Optional[UserProfile] = None
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    now = timezone_now()
    if history_limit_days is not None:
        fetch_since_datetime = now - timedelta(days=history_limit_days)
    else:
        fetch_since_datetime = now - timedelta(days=14)
    kwargs: Dict[str, Any] = dict()
    if last_update_id_fetched_by_client is not None:
        kwargs['last_update_id__gt'] = last_update_id_fetched_by_client
    if (last_update_id_fetched_by_client is None or 
        last_update_id_fetched_by_client <= 0):
        kwargs['last_connected_time__gte'] = fetch_since_datetime
    if history_limit_days != 0:
        query = UserPresence.objects.filter(realm_id=realm.id,
            user_profile__is_active=True, user_profile__is_bot=False, **kwargs)
    else:
        query = UserPresence.objects.none()
    if (settings.CAN_ACCESS_ALL_USERS_GROUP_LIMITS_PRESENCE and not
        check_user_can_access_all_users(requesting_user_profile)):
        assert requesting_user_profile is not None
        accessible_user_ids = get_accessible_user_ids(realm,
            requesting_user_profile)
        query = query.filter(user_profile_id__in=accessible_user_ids)
    presence_rows = list(query.values('last_active_time',
        'last_connected_time', 'user_profile__email', 'user_profile_id',
        'user_profile__enable_offline_push_notifications',
        'user_profile__date_joined', 'last_update_id'))
    if presence_rows:
        last_update_id_fetched_by_server = max(row['last_update_id'] for
            row in presence_rows)
    elif last_update_id_fetched_by_client is not None:
        last_update_id_fetched_by_server = last_update_id_fetched_by_client
    else:
        last_update_id_fetched_by_server = -1
    assert last_update_id_fetched_by_server is not None
    return (
        func_0zr8n629(presence_rows, slim_presence),
        last_update_id_fetched_by_server
    )


def func_on1sau0r(
    realm: Realm, 
    slim_presence: bool, 
    last_update_id_fetched_by_client: Optional[int],
    history_limit_days: Optional[int], 
    requesting_user_profile: Optional[UserProfile]
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    if realm.presence_disabled:
        return defaultdict(dict), -1
    return func_okz0pz7c(
        realm, 
        slim_presence,
        last_update_id_fetched_by_client, 
        history_limit_days,
        requesting_user_profile=requesting_user_profile
    )


def func_l4k66qkt(
    requesting_user_profile: UserProfile, 
    slim_presence: bool,
    last_update_id_fetched_by_client: Optional[int] = None, 
    history_limit_days: Optional[int] = None
) -> Dict[str, Any]:
    realm = requesting_user_profile.realm
    server_timestamp = time.time()
    presences, last_update_id_fetched_by_server = func_on1sau0r(
        realm,
        slim_presence, 
        last_update_id_fetched_by_client, 
        history_limit_days,
        requesting_user_profile=requesting_user_profile
    )
    response_dict: Dict[str, Any] = dict(
        presences=presences, 
        server_timestamp=server_timestamp, 
        presence_last_update_id=last_update_id_fetched_by_server
    )
    return response_dict
