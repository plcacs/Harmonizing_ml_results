import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.conf import settings
from django.utils.timezone import now as timezone_now
from zerver.lib.timestamp import datetime_to_timestamp
from zerver.lib.users import check_user_can_access_all_users, get_accessible_user_ids
from zerver.models import Realm, UserPresence, UserProfile


def get_presence_dicts_for_rows(all_rows: List[Dict[str, Any]], slim_presence: bool) -> Dict[str, Any]:
    if slim_presence:
        get_user_key: Callable[[Dict[str, Any]], str] = lambda row: str(row['user_profile_id'])
        get_user_presence_info: Callable[[datetime, datetime], Dict[str, Any]] = get_modern_user_presence_info
    else:
        get_user_key = lambda row: row['user_profile__email']  # type: Callable[[Dict[str, Any]], str]
        get_user_presence_info = get_legacy_user_presence_info  # type: Callable[[datetime, datetime], Dict[str, Any]]
    user_statuses: Dict[str, Any] = {}
    for presence_row in all_rows:
        user_key: str = get_user_key(presence_row)
        last_active_time: datetime = user_presence_datetime_with_date_joined_default(
            presence_row['last_active_time'], presence_row['user_profile__date_joined']
        )
        last_connected_time: datetime = user_presence_datetime_with_date_joined_default(
            presence_row['last_connected_time'], presence_row['user_profile__date_joined']
        )
        info: Dict[str, Any] = get_user_presence_info(last_active_time, last_connected_time)
        user_statuses[user_key] = info
    return user_statuses


def user_presence_datetime_with_date_joined_default(dt: Optional[datetime], date_joined: datetime) -> datetime:
    """
    Our data models support UserPresence objects not having None
    values for last_active_time/last_connected_time. The legacy API
    however has always sent timestamps, so for backward
    compatibility we cannot send such values through the API and need
    to default to a sane

    This helper functions expects to take a last_active_time or
    last_connected_time value and the date_joined of the user, which
    will serve as the default value if the first argument is None.
    """
    if dt is None:
        return date_joined
    return dt


def get_modern_user_presence_info(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    result['active_timestamp'] = datetime_to_timestamp(last_active_time)
    result['idle_timestamp'] = datetime_to_timestamp(last_connected_time)
    return result


def get_legacy_user_presence_info(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, Any]:
    """
    Reformats the modern UserPresence data structure so that legacy
    API clients can still access presence data.
    We expect this code to remain mostly unchanged until we can delete it.
    """
    most_recent_info: Dict[str, Any] = format_legacy_presence_dict(last_active_time, last_connected_time)
    result: Dict[str, Any] = {}
    result['aggregated'] = dict(
        client=most_recent_info['client'],
        status=most_recent_info['status'],
        timestamp=most_recent_info['timestamp']
    )
    result['website'] = most_recent_info
    return result


def format_legacy_presence_dict(last_active_time: datetime, last_connected_time: datetime) -> Dict[str, Any]:
    """
    This function assumes it's being called right after the presence object was updated,
    and is not meant to be used on old presence data.
    """
    if last_active_time + timedelta(seconds=settings.PRESENCE_LEGACY_EVENT_OFFSET_FOR_ACTIVITY_SECONDS) >= last_connected_time:
        status = UserPresence.LEGACY_STATUS_ACTIVE
        timestamp = datetime_to_timestamp(last_active_time)
    else:
        status = UserPresence.LEGACY_STATUS_IDLE
        timestamp = datetime_to_timestamp(last_connected_time)
    pushable: bool = False
    return dict(client='website', status=status, timestamp=timestamp, pushable=pushable)


def get_presence_for_user(user_profile_id: int, slim_presence: bool = False) -> Dict[str, Any]:
    query = UserPresence.objects.filter(user_profile_id=user_profile_id).values(
        'last_active_time',
        'last_connected_time',
        'user_profile__email',
        'user_profile_id',
        'user_profile__enable_offline_push_notifications',
        'user_profile__date_joined'
    )
    presence_rows: List[Dict[str, Any]] = list(query)
    return get_presence_dicts_for_rows(presence_rows, slim_presence)


def get_presence_dict_by_realm(
    realm: Realm,
    slim_presence: bool = False,
    last_update_id_fetched_by_client: Optional[int] = None,
    history_limit_days: Optional[int] = None,
    requesting_user_profile: Optional[UserProfile] = None
) -> Tuple[Dict[str, Any], int]:
    now: datetime = timezone_now()
    if history_limit_days is not None:
        fetch_since_datetime: datetime = now - timedelta(days=history_limit_days)
    else:
        fetch_since_datetime = now - timedelta(days=14)
    kwargs: Dict[str, Any] = dict()
    if last_update_id_fetched_by_client is not None:
        kwargs['last_update_id__gt'] = last_update_id_fetched_by_client
    if last_update_id_fetched_by_client is None or last_update_id_fetched_by_client <= 0:
        kwargs['last_connected_time__gte'] = fetch_since_datetime
    if history_limit_days != 0:
        query = UserPresence.objects.filter(
            realm_id=realm.id,
            user_profile__is_active=True,
            user_profile__is_bot=False,
            **kwargs
        )
    else:
        query = UserPresence.objects.none()
    if settings.CAN_ACCESS_ALL_USERS_GROUP_LIMITS_PRESENCE and (not check_user_can_access_all_users(requesting_user_profile)):
        assert requesting_user_profile is not None
        accessible_user_ids = get_accessible_user_ids(realm, requesting_user_profile)
        query = query.filter(user_profile_id__in=accessible_user_ids)
    presence_rows: List[Dict[str, Any]] = list(query.values(
        'last_active_time',
        'last_connected_time',
        'user_profile__email',
        'user_profile_id',
        'user_profile__enable_offline_push_notifications',
        'user_profile__date_joined',
        'last_update_id'
    ))
    if presence_rows:
        last_update_id_fetched_by_server: int = max((row['last_update_id'] for row in presence_rows))
    elif last_update_id_fetched_by_client is not None:
        last_update_id_fetched_by_server = last_update_id_fetched_by_client
    else:
        last_update_id_fetched_by_server = -1
    assert last_update_id_fetched_by_server is not None
    return (get_presence_dicts_for_rows(presence_rows, slim_presence), last_update_id_fetched_by_server)


def get_presences_for_realm(
    realm: Realm,
    slim_presence: bool,
    last_update_id_fetched_by_client: Optional[int],
    history_limit_days: Optional[int],
    requesting_user_profile: UserProfile
) -> Tuple[Dict[str, Any], int]:
    if realm.presence_disabled:
        return (defaultdict(dict), -1)
    return get_presence_dict_by_realm(
        realm,
        slim_presence,
        last_update_id_fetched_by_client,
        history_limit_days,
        requesting_user_profile=requesting_user_profile
    )


def get_presence_response(
    requesting_user_profile: UserProfile,
    slim_presence: bool,
    last_update_id_fetched_by_client: Optional[int] = None,
    history_limit_days: Optional[int] = None
) -> Dict[str, Any]:
    realm: Realm = requesting_user_profile.realm
    server_timestamp: float = time.time()
    presences, last_update_id_fetched_by_server = get_presences_for_realm(
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