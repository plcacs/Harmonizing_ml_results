import hashlib
import logging
import os
import re
import secrets
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from functools import _lru_cache_wrapper, lru_cache, wraps
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from django.conf import settings
from django.core.cache import caches
from django.core.cache.backends.base import BaseCache
from django.db.models import Q, QuerySet
from typing_extensions import ParamSpec
if TYPE_CHECKING:
    from zerver.models import Attachment, Message, MutedUser, Realm, Stream, SubMessage, UserProfile
MEMCACHED_MAX_KEY_LENGTH = 250
ParamT = ParamSpec('ParamT')
ReturnT = TypeVar('ReturnT')
logger = logging.getLogger()
remote_cache_time_start = 0.0
remote_cache_total_time = 0.0
remote_cache_total_requests = 0

def get_remote_cache_time() -> Union[int, float, typing.Final]:
    return remote_cache_total_time

def get_remote_cache_requests() -> Union[int, str]:
    return remote_cache_total_requests

def remote_cache_stats_start() -> None:
    global remote_cache_time_start
    remote_cache_time_start = time.time()

def remote_cache_stats_finish() -> None:
    global remote_cache_total_time, remote_cache_total_requests
    remote_cache_total_requests += 1
    remote_cache_total_time += time.time() - remote_cache_time_start

def get_or_create_key_prefix() -> typing.Text:
    if settings.PUPPETEER_TESTS:
        return 'puppeteer_tests:'
    elif settings.TEST_SUITE:
        return 'django_tests_unused:'
    os.makedirs(os.path.join(settings.DEPLOY_ROOT, 'var'), exist_ok=True)
    filename = os.path.join(settings.DEPLOY_ROOT, 'var', 'remote_cache_prefix')
    try:
        with open(filename, 'x') as f:
            prefix = secrets.token_hex(16) + ':'
            f.write(prefix + '\n')
    except FileExistsError:
        tries = 1
        while tries < 10:
            with open(filename) as f:
                prefix = f.readline().removesuffix('\n')
            if len(prefix) == 33:
                break
            tries += 1
            prefix = ''
            time.sleep(0.5)
    if not prefix:
        print('Could not read remote cache key prefix file')
        sys.exit(1)
    return prefix
KEY_PREFIX = get_or_create_key_prefix()

def bounce_key_prefix_for_testing(test_name: str) -> None:
    global KEY_PREFIX
    KEY_PREFIX = test_name + ':' + str(os.getpid()) + ':'
    KEY_PREFIX = hashlib.sha1(KEY_PREFIX.encode()).hexdigest() + ':'

def get_cache_backend(cache_name: Union[str, None, int]):
    if cache_name is None:
        cache_name = 'default'
    return caches[cache_name]

def cache_with_key(keyfunc: Union[int, str, None, float], cache_name: Union[None, int, str, float]=None, timeout: Union[None, int, str, float]=None):
    """Decorator which applies Django caching to a function.

    Decorator argument is a function which computes a cache key
    from the original function's arguments.  You are responsible
    for avoiding collisions with other uses of this decorator or
    other uses of caching."""

    def decorator(func: Any):

        @wraps(func)
        def func_with_caching(*args, **kwargs) -> QuerySet:
            key = keyfunc(*args, **kwargs)
            try:
                val = cache_get(key, cache_name=cache_name)
            except InvalidCacheKeyError:
                stack_trace = traceback.format_exc()
                log_invalid_cache_keys(stack_trace, [key])
                return func(*args, **kwargs)
            if val is not None:
                return val[0]
            val = func(*args, **kwargs)
            if isinstance(val, QuerySet):
                logging.error('cache_with_key attempted to store a full QuerySet object -- declining to cache', stack_info=True)
            else:
                cache_set(key, val, cache_name=cache_name, timeout=timeout)
            return val
        return func_with_caching
    return decorator

class InvalidCacheKeyError(Exception):
    pass

def log_invalid_cache_keys(stack_trace: Union[str, tuple[typing.Union[traceback.FrameSummary,str]], None], key: Union[str, tuple[typing.Union[traceback.FrameSummary,str]], None]) -> None:
    logger.warning('Invalid cache key used: %s\nStack trace: %s\n', key, stack_trace)

def validate_cache_key(key: str) -> None:
    if not key.startswith(KEY_PREFIX):
        key = KEY_PREFIX + key
    if not bool(re.fullmatch('([!-~])+', key)):
        raise InvalidCacheKeyError('Invalid characters in the cache key: ' + key)
    if len(key) > MEMCACHED_MAX_KEY_LENGTH:
        raise InvalidCacheKeyError(f'Cache key too long: {key} Length: {len(key)}')

def cache_set(key: str, val: Union[float, None, int, str], cache_name: Union[None, str, bool]=None, timeout: Union[None, float, int, str]=None) -> None:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)
    remote_cache_stats_start()
    cache_backend = get_cache_backend(cache_name)
    cache_backend.set(final_key, (val,), timeout=timeout)
    remote_cache_stats_finish()

def cache_get(key: str, cache_name: Union[None, str, typing.Iterable[str]]=None) -> Union[None, typing.Any, int, str]:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)
    remote_cache_stats_start()
    cache_backend = get_cache_backend(cache_name)
    ret = cache_backend.get(final_key)
    remote_cache_stats_finish()
    return ret

def cache_get_many(keys: Union[str, dict[str, typing.Any]], cache_name: Union[None, str]=None) -> dict:
    keys = [KEY_PREFIX + key for key in keys]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    ret = get_cache_backend(cache_name).get_many(keys)
    remote_cache_stats_finish()
    return {key.removeprefix(KEY_PREFIX): value for key, value in ret.items()}

def safe_cache_get_many(keys: Union[str, None, typing.Iterable[str]], cache_name: Union[None, str, int]=None) -> Union[str, dict[str, typing.Any], None, list[typing.Any]]:
    """Variant of cache_get_many that drops any keys that fail
    validation, rather than throwing an exception visible to the
    caller."""
    try:
        return cache_get_many(keys, cache_name)
    except InvalidCacheKeyError:
        stack_trace = traceback.format_exc()
        good_keys, bad_keys = filter_good_and_bad_keys(keys)
        log_invalid_cache_keys(stack_trace, bad_keys)
        return cache_get_many(good_keys, cache_name)

def cache_set_many(items: Union[dict, models.order_action.Parameters], cache_name: Union[None, str, int]=None, timeout: Union[None, str, int]=None) -> None:
    new_items = {}
    for key, item in items.items():
        new_key = KEY_PREFIX + key
        validate_cache_key(new_key)
        new_items[new_key] = item
    items = new_items
    remote_cache_stats_start()
    get_cache_backend(cache_name).set_many(items, timeout=timeout)
    remote_cache_stats_finish()

def safe_cache_set_many(items: dict, cache_name: Union[None, float, str]=None, timeout: Union[None, float, str]=None) -> Union[str, None, bool]:
    """Variant of cache_set_many that drops saving any keys that fail
    validation, rather than throwing an exception visible to the
    caller."""
    try:
        return cache_set_many(items, cache_name, timeout)
    except InvalidCacheKeyError:
        stack_trace = traceback.format_exc()
        good_keys, bad_keys = filter_good_and_bad_keys(list(items.keys()))
        log_invalid_cache_keys(stack_trace, bad_keys)
        good_items = {key: items[key] for key in good_keys}
        return cache_set_many(good_items, cache_name, timeout)

def cache_delete(key: str, cache_name: Union[None, str, typing.Iterable[str]]=None) -> None:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)
    remote_cache_stats_start()
    get_cache_backend(cache_name).delete(final_key)
    remote_cache_stats_finish()

def cache_delete_many(items: Union[str, dict[str, typing.Any]], cache_name: Union[None, str, list[str]]=None) -> None:
    keys = [KEY_PREFIX + item for item in items]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    get_cache_backend(cache_name).delete_many(keys)
    remote_cache_stats_finish()

def filter_good_and_bad_keys(keys: Union[str, bytes]) -> tuple[list[typing.Text]]:
    good_keys = []
    bad_keys = []
    for key in keys:
        try:
            validate_cache_key(key)
            good_keys.append(key)
        except InvalidCacheKeyError:
            bad_keys.append(key)
    return (good_keys, bad_keys)
ObjKT = TypeVar('ObjKT')
ItemT = TypeVar('ItemT')
CacheItemT = TypeVar('CacheItemT')
CompressedItemT = TypeVar('CompressedItemT')

def generic_bulk_cached_fetch(cache_key_function: Union[int, None, str], query_function: Union[int, None, str, tuple[typing.Union[int,str]]], object_ids: Union[list[int], str, list[str]], *, extractor: Union[dict[str, typing.Any], str, typing.Mapping], setter: Union[str, dict[str, typing.Any], list[str]], id_fetcher: Union[bytes, None, str, int], cache_transformer: Union[str, None, dict[str, str]]) -> Union[dict, dict[typing.Union[int,typing.Text], typing.Union[dict[str, str],dict,str]]]:
    if len(object_ids) == 0:
        return {}
    cache_keys = {}
    for object_id in object_ids:
        cache_keys[object_id] = cache_key_function(object_id)
    cached_objects_compressed = safe_cache_get_many([cache_keys[object_id] for object_id in object_ids])
    cached_objects = {key: extractor(val[0]) for key, val in cached_objects_compressed.items()}
    needed_ids = [object_id for object_id in object_ids if cache_keys[object_id] not in cached_objects]
    if len(needed_ids) > 0:
        db_objects = query_function(needed_ids)
    else:
        db_objects = []
    items_for_remote_cache = {}
    for obj in db_objects:
        key = cache_keys[id_fetcher(obj)]
        item = cache_transformer(obj)
        items_for_remote_cache[key] = (setter(item),)
        cached_objects[key] = item
    if len(items_for_remote_cache) > 0:
        safe_cache_set_many(items_for_remote_cache)
    return {object_id: cached_objects[cache_keys[object_id]] for object_id in object_ids if cache_keys[object_id] in cached_objects}

def bulk_cached_fetch(cache_key_function: Union[str, dict[str, typing.Any], bool], query_function: Union[str, dict[str, typing.Any], bool], object_ids: Union[str, dict[str, typing.Any], bool], *, id_fetcher: Union[str, dict[str, typing.Any], bool]) -> Union[bool, str, asgard.backends.accounts.AccountsBackend]:
    return generic_bulk_cached_fetch(cache_key_function, query_function, object_ids, id_fetcher=id_fetcher, extractor=lambda obj: obj, setter=lambda obj: obj, cache_transformer=lambda obj: obj)

def preview_url_cache_key(url: str) -> typing.Text:
    return f'preview_url:{hashlib.sha1(url.encode()).hexdigest()}'

def display_recipient_cache_key(recipient_id: Union[int, str]) -> typing.Text:
    return f'display_recipient_dict:{recipient_id}'

def single_user_display_recipient_cache_key(user_id: Union[int, str]) -> typing.Text:
    return f'single_user_display_recipient:{user_id}'

def user_profile_by_email_realm_id_cache_key(email: Union[str, int, list[int]], realm_id: Union[str, int, list[int]]) -> typing.Text:
    return f'user_profile:{hashlib.sha1(email.strip().encode()).hexdigest()}:{realm_id}'

def user_profile_by_email_realm_cache_key(email: Union[str, Realm], realm: Union[str, Realm]):
    return user_profile_by_email_realm_id_cache_key(email, realm.id)

def user_profile_delivery_email_cache_key(delivery_email: Union[bytes, int, str], realm_id: Union[bytes, int, str]) -> typing.Text:
    return f'user_profile_by_delivery_email:{hashlib.sha1(delivery_email.strip().encode()).hexdigest()}:{realm_id}'

def bot_profile_cache_key(email: Union[str, int], realm_id: Union[int, str]) -> typing.Text:
    return f'bot_profile:{hashlib.sha1(email.strip().encode()).hexdigest()}'

def user_profile_by_id_cache_key(user_profile_id: Union[int, zerver.models.Realm, zerver.models.UserProfile]) -> typing.Text:
    return f'user_profile_by_id:{user_profile_id}'

def user_profile_narrow_by_id_cache_key(user_profile_id: Union[int, zerver.models.Realm, list[int]]) -> typing.Text:
    return f'user_profile_narrow_by_id:{user_profile_id}'

def user_profile_by_api_key_cache_key(api_key: Union[str, list[str]]) -> typing.Text:
    return f'user_profile_by_api_key:{api_key}'

def get_cross_realm_dicts_key() -> typing.Text:
    emails = list(settings.CROSS_REALM_BOT_EMAILS)
    raw_key = ','.join(sorted(emails))
    digest = hashlib.sha1(raw_key.encode()).hexdigest()
    return f'get_cross_realm_dicts:{digest}'
realm_user_dict_fields = ['id', 'full_name', 'email', 'avatar_source', 'avatar_version', 'is_active', 'role', 'is_billing_admin', 'is_bot', 'timezone', 'date_joined', 'bot_owner_id', 'delivery_email', 'bot_type', 'long_term_idle', 'email_address_visibility']

def realm_user_dicts_cache_key(realm_id: int) -> typing.Text:
    return f'realm_user_dicts:{realm_id}'

def get_muting_users_cache_key(muted_user_id: Union[list[int], int, dict[str, tuple]]) -> typing.Text:
    return f'muting_users_list:{muted_user_id}'

def get_realm_used_upload_space_cache_key(realm_id: int) -> typing.Text:
    return f'realm_used_upload_space:{realm_id}'

def get_realm_seat_count_cache_key(realm_id: int) -> typing.Text:
    return f'realm_seat_count:{realm_id}'

def active_user_ids_cache_key(realm_id: int) -> typing.Text:
    return f'active_user_ids:{realm_id}'

def active_non_guest_user_ids_cache_key(realm_id: Union[int, list[int]]) -> typing.Text:
    return f'active_non_guest_user_ids:{realm_id}'
bot_dict_fields = ['api_key', 'avatar_source', 'avatar_version', 'bot_owner_id', 'bot_type', 'default_all_public_streams', 'default_events_register_stream__name', 'default_sending_stream__name', 'email', 'full_name', 'id', 'is_active', 'realm_id']

def bot_dicts_in_realm_cache_key(realm_id: int) -> typing.Text:
    return f'bot_dicts_in_realm:{realm_id}'

def delete_user_profile_caches(user_profiles: Union[zerver.models.Realm, zerver.models.UserProfile], realm_id: Union[int, list, list[dict[str, typing.Any]]]) -> None:
    from zerver.models.users import is_cross_realm_bot_email
    keys = []
    for user_profile in user_profiles:
        keys.append(user_profile_by_id_cache_key(user_profile.id))
        keys.append(user_profile_narrow_by_id_cache_key(user_profile.id))
        keys.append(user_profile_by_api_key_cache_key(user_profile.api_key))
        keys.append(user_profile_by_email_realm_id_cache_key(user_profile.email, realm_id))
        keys.append(user_profile_delivery_email_cache_key(user_profile.delivery_email, realm_id))
        if user_profile.is_bot and is_cross_realm_bot_email(user_profile.email):
            keys.append(bot_profile_cache_key(user_profile.email, realm_id))
            keys.append(get_cross_realm_dicts_key())
    cache_delete_many(keys)

def delete_display_recipient_cache(user_profile: Union[zerver.models.UserProfile, django.db.models.query.QuerySet]) -> None:
    from zerver.models import Subscription
    recipient_ids = Subscription.objects.filter(user_profile=user_profile).values_list('recipient_id', flat=True)
    keys = [display_recipient_cache_key(rid) for rid in recipient_ids]
    keys.append(single_user_display_recipient_cache_key(user_profile.id))
    cache_delete_many(keys)

def changed(update_fields: Union[dict[str, typing.Any], AbstractSetIntStr, MappingIntStrAny], fields: Union[T, str]) -> bool:
    if update_fields is None:
        return True
    update_fields_set = set(update_fields)
    return any((f in update_fields_set for f in fields))

def flush_user_profile(*, instance: Union[dict[str, typing.Any], None, bool, dict], update_fields: Union[None, users.models.User, list[str], bool]=None, **kwargs) -> None:
    user_profile = instance
    delete_user_profile_caches([user_profile], user_profile.realm_id)
    if changed(update_fields, realm_user_dict_fields):
        cache_delete(realm_user_dicts_cache_key(user_profile.realm_id))
    if changed(update_fields, ['is_active']):
        cache_delete(active_user_ids_cache_key(user_profile.realm_id))
        cache_delete(active_non_guest_user_ids_cache_key(user_profile.realm_id))
    if changed(update_fields, ['role']):
        cache_delete(active_non_guest_user_ids_cache_key(user_profile.realm_id))
    if changed(update_fields, ['email', 'full_name', 'id', 'is_mirror_dummy']):
        delete_display_recipient_cache(user_profile)
    if user_profile.is_bot and changed(update_fields, bot_dict_fields):
        cache_delete(bot_dicts_in_realm_cache_key(user_profile.realm_id))

def flush_muting_users_cache(*, instance: Union[core.models.Ingredient, core.models.Step, esm.models.ServiceInstance], **kwargs) -> None:
    mute_object = instance
    cache_delete(get_muting_users_cache_key(mute_object.muted_user_id))

def flush_realm(*, instance: Union[list[dict[str, str]], bool, list[str]], update_fields: Union[CustomerGroupReference, list[dict[str, str]], CustomerGroupResourceIdentifier]=None, from_deletion: bool=False, **kwargs) -> None:
    realm = instance
    users = realm.get_active_users()
    delete_user_profile_caches(users, realm.id)
    if from_deletion or realm.deactivated or (update_fields is not None and 'string_id' in update_fields):
        cache_delete(realm_user_dicts_cache_key(realm.id))
        cache_delete(active_user_ids_cache_key(realm.id))
        cache_delete(bot_dicts_in_realm_cache_key(realm.id))
        cache_delete(realm_alert_words_cache_key(realm.id))
        cache_delete(realm_alert_words_automaton_cache_key(realm.id))
        cache_delete(active_non_guest_user_ids_cache_key(realm.id))
        cache_delete(realm_rendered_description_cache_key(realm))
        cache_delete(realm_text_description_cache_key(realm))
    elif changed(update_fields, ['description']):
        cache_delete(realm_rendered_description_cache_key(realm))
        cache_delete(realm_text_description_cache_key(realm))

def realm_alert_words_cache_key(realm_id: Union[int, list[dict]]) -> typing.Text:
    return f'realm_alert_words:{realm_id}'

def realm_alert_words_automaton_cache_key(realm_id: Union[int, list[dict]]) -> typing.Text:
    return f'realm_alert_words_automaton:{realm_id}'

def realm_rendered_description_cache_key(realm: Union[Realm, daylighdb.models.User]) -> typing.Text:
    return f'realm_rendered_description:{realm.string_id}'

def realm_text_description_cache_key(realm: Union[Realm, daylighdb.models.User]) -> typing.Text:
    return f'realm_text_description:{realm.string_id}'

def flush_stream(*, instance: dict, update_fields: Union[abilian.core.models.Model, zam_repondeur.models.User]=None, **kwargs) -> None:
    from zerver.models import UserProfile
    stream = instance
    if update_fields is None or ('name' in update_fields and UserProfile.objects.filter(Q(default_sending_stream=stream) | Q(default_events_register_stream=stream)).exists()):
        cache_delete(bot_dicts_in_realm_cache_key(stream.realm_id))

def flush_used_upload_space_cache(*, instance: Union[abilian.services.security.models.Permission, None, tartare.core.models.Contributor, django.db.models.Model], created: bool=True, **kwargs) -> None:
    attachment = instance
    if created:
        cache_delete(get_realm_used_upload_space_cache_key(attachment.owner.realm_id))

def to_dict_cache_key_id(message_id: Union[int, None]) -> typing.Text:
    return f'message_dict:{message_id}'

def to_dict_cache_key(message: int, realm_id: Union[None, int, zam_repondeur.models.Lecture, typing.Iterator]=None):
    return to_dict_cache_key_id(message.id)

def open_graph_description_cache_key(content: Union[str, None, django.http.HttpRequest], request_url: Union[str, bytes]) -> typing.Text:
    return f'open_graph_description_path:{hashlib.sha1(request_url.encode()).hexdigest()}'

def zoom_server_access_token_cache_key(account_id: Union[str, int, None]) -> typing.Text:
    return f'zoom_server_to_server_access_token:{account_id}'

def flush_zoom_server_access_token_cache(account_id: Union[str, int]) -> None:
    cache_delete(zoom_server_access_token_cache_key(account_id))

def flush_message(*, instance: Union[SpeciesNameComplex, pykechain.models.sidebar.sidebar_button.SideBarButton, bool], **kwargs) -> None:
    message = instance
    cache_delete(to_dict_cache_key_id(message.id))

def flush_submessage(*, instance: Union[dict[str, typing.Callable], typing.Type], **kwargs) -> None:
    submessage = instance
    message_id = submessage.message_id
    cache_delete(to_dict_cache_key_id(message_id))

class IgnoreUnhashableLruCacheWrapper(Generic[ParamT, ReturnT]):

    def __init__(self, function: Union[typing.Callable, typing.Type], cached_function: Union[typing.Callable, static_frame.core.util.IndexConstructor]) -> None:
        self.key_prefix = KEY_PREFIX
        self.function = function
        self.cached_function = cached_function
        self.cache_info = cached_function.cache_info
        self.cache_clear = cached_function.cache_clear

    def __call__(self, *args, **kwargs):
        if settings.DEVELOPMENT and (not settings.TEST_SUITE):
            return self.function(*args, **kwargs)
        if self.key_prefix != KEY_PREFIX:
            self.cache_clear()
            self.key_prefix = KEY_PREFIX
        try:
            return self.cached_function(*args, **kwargs)
        except TypeError:
            pass
        return self.function(*args, **kwargs)

def ignore_unhashable_lru_cache(maxsize: int=128, typed: bool=False):
    """
    This is a wrapper over lru_cache function. It adds following features on
    top of lru_cache:

        * It will not cache result of functions with unhashable arguments.
        * It will clear cache whenever zerver.lib.cache.KEY_PREFIX changes.
    """
    internal_decorator = lru_cache(maxsize=maxsize, typed=typed)

    def decorator(user_function):
        return IgnoreUnhashableLruCacheWrapper(user_function, internal_decorator(user_function))
    return decorator

def dict_to_items_tuple(user_function: Union[typing.Callable, list[str]]):
    """Wrapper that converts any dict args to dict item tuples."""

    def dict_to_tuple(arg: Any) -> Union[tuple[list], dict]:
        if isinstance(arg, dict):
            return tuple(sorted(arg.items()))
        return arg

    def wrapper(*args, **kwargs):
        new_args = (dict_to_tuple(arg) for arg in args)
        return user_function(*new_args, **kwargs)
    return wrapper

def items_tuple_to_dict(user_function: Union[typing.Callable, typing.Type, list[str]]):
    """Wrapper that converts any dict items tuple args to dicts."""

    def dict_items_to_dict(arg: Any) -> tuple:
        if isinstance(arg, tuple):
            try:
                return dict(arg)
            except TypeError:
                pass
        return arg

    def wrapper(*args, **kwargs):
        new_args = (dict_items_to_dict(arg) for arg in args)
        new_kwargs = {key: dict_items_to_dict(val) for key, val in kwargs.items()}
        return user_function(*new_args, **new_kwargs)
    return wrapper