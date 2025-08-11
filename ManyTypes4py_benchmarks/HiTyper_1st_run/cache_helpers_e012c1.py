import logging
from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any
from django.conf import settings
from django.contrib.sessions.models import Session
from django.db import connection
from django.db.models import QuerySet
from django.utils.timezone import now as timezone_now
from analytics.models import RealmCount
from zerver.lib.cache import cache_set_many, get_remote_cache_requests, get_remote_cache_time, user_profile_by_api_key_cache_key, user_profile_by_id_cache_key, user_profile_narrow_by_id_cache_key
from zerver.lib.safe_session_cached_db import SessionStore
from zerver.lib.sessions import session_engine
from zerver.models import Client, UserProfile
from zerver.models.clients import get_client_cache_key
from zerver.models.users import base_get_user_narrow_queryset, base_get_user_queryset

def get_users() -> Union[zerver.models.UserProfile, list[str], str]:
    return base_get_user_queryset().filter(long_term_idle=False, realm__in=get_active_realm_ids())

def user_cache_items(items_for_remote_cache: Union[models.User, None, zerver.models.UserProfile, app.models.user.UserCreate], user_profile: Union[models.User, None, zerver.models.UserProfile, app.models.user.UserCreate]) -> None:
    items_for_remote_cache[user_profile_by_api_key_cache_key(user_profile.api_key)] = (user_profile,)
    items_for_remote_cache[user_profile_by_id_cache_key(user_profile.id)] = (user_profile,)

def get_narrow_users() -> Union[models.characters.character_base.Character, str, zerver.models.Realm]:
    return base_get_user_narrow_queryset().filter(long_term_idle=False, realm__in=get_active_realm_ids())

def user_narrow_cache_items(items_for_remote_cache: Union[zerver.models.UserProfile, zerver.models.UserGroup, zerver.models.Realm], user_profile: Union[zerver.models.UserProfile, zerver.models.UserGroup, zerver.models.Realm]) -> None:
    items_for_remote_cache[user_profile_narrow_by_id_cache_key(user_profile.id)] = (user_profile,)

def client_cache_items(items_for_remote_cache: Union[zerver.models.Client, excars.models.user.User, str], client: Union[zerver.models.Client, excars.models.user.User, str]) -> None:
    items_for_remote_cache[get_client_cache_key(client.name)] = (client,)

def session_cache_items(items_for_remote_cache: Union[dict[str, str], sqlalchemy.orm.Session, bool], session: Union[str, rotkehlchen.db.dbhandler.DBHandler, zerver.models.Client]) -> None:
    if settings.SESSION_ENGINE != 'zerver.lib.safe_session_cached_db':
        return
    store = session_engine.SessionStore(session_key=session.session_key)
    assert isinstance(store, SessionStore)
    items_for_remote_cache[store.cache_key] = store.decode(session.session_data)

def get_active_realm_ids() -> Union[int, str]:
    """For installations like Zulip Cloud hosting a lot of realms, it only makes
    sense to do cache-filling work for realms that have any currently
    active users/clients.  Otherwise, we end up with every single-user
    trial organization that has ever been created costing us N streams
    worth of cache work (where N is the number of default streams for
    a new organization).
    """
    date = timezone_now() - timedelta(days=2)
    return RealmCount.objects.filter(end_time__gte=date, property='1day_actives::day', subgroup=None, value__gt=0).distinct('realm_id').values_list('realm_id', flat=True)
cache_fillers = {'user': (get_users, user_cache_items, 3600 * 24 * 7, 10000), 'user_narrow': (get_narrow_users, user_narrow_cache_items, 3600 * 24 * 7, 10000), 'client': (Client.objects.all, client_cache_items, 3600 * 24 * 7, 10000), 'session': (Session.objects.all, session_cache_items, 3600 * 24 * 7, 10000)}

class SQLQueryCounter:

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, execute: Union[dict[str, typing.Any], typing.Mapping], sql: Union[dict[str, typing.Any], typing.Mapping], params: Union[dict[str, typing.Any], typing.Mapping], many: Union[dict[str, typing.Any], typing.Mapping], context: Union[dict[str, typing.Any], typing.Mapping]) -> Union[bytes, typing.Deque, int]:
        self.count += 1
        return execute(sql, params, many, context)

def fill_remote_cache(cache: Union[str, dict[str, typing.Any], dict]) -> None:
    remote_cache_time_start = get_remote_cache_time()
    remote_cache_requests_start = get_remote_cache_requests()
    items_for_remote_cache = {}
    objects, items_filler, timeout, batch_size = cache_fillers[cache]
    count = 0
    db_query_counter = SQLQueryCounter()
    with connection.execute_wrapper(db_query_counter):
        for obj in objects():
            items_filler(items_for_remote_cache, obj)
            count += 1
            if count % batch_size == 0:
                cache_set_many(items_for_remote_cache, timeout=timeout)
                items_for_remote_cache = {}
        cache_set_many(items_for_remote_cache, timeout=timeout)
    logging.info('Successfully populated %s cache: %d items, %d DB queries, %d memcached sets, %.2f seconds', cache, count, db_query_counter.count, get_remote_cache_requests() - remote_cache_requests_start, get_remote_cache_time() - remote_cache_time_start)