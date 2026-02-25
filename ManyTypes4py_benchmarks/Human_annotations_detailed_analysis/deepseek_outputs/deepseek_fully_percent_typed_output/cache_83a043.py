# See https://zulip.readthedocs.io/en/latest/subsystems/caching.html for docs
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
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Optional, Union, Dict, List, Tuple, Set, cast

from django.conf import settings
from django.core.cache import caches
from django.core.cache.backends.base import BaseCache
from django.db.models import Q, QuerySet
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    # These modules have to be imported for type annotations but
    # they cannot be imported at runtime due to cyclic dependency.
    from zerver.models import Attachment, Message, MutedUser, Realm, Stream, SubMessage, UserProfile

MEMCACHED_MAX_KEY_LENGTH = 250

ParamT = ParamSpec("ParamT")
ReturnT = TypeVar("ReturnT")

logger: logging.Logger = logging.getLogger()

remote_cache_time_start: float = 0.0
remote_cache_total_time: float = 0.0
remote_cache_total_requests: int = 0


def get_remote_cache_time() -> float:
    return remote_cache_total_time


def get_remote_cache_requests() -> int:
    return remote_cache_total_requests


def remote_cache_stats_start() -> None:
    global remote_cache_time_start
    remote_cache_time_start = time.time()


def remote_cache_stats_finish() -> None:
    global remote_cache_total_time, remote_cache_total_requests
    remote_cache_total_requests += 1
    remote_cache_total_time += time.time() - remote_cache_time_start


def get_or_create_key_prefix() -> str:
    if settings.PUPPETEER_TESTS:
        return "puppeteer_tests:"
    elif settings.TEST_SUITE:
        return "django_tests_unused:"

    os.makedirs(os.path.join(settings.DEPLOY_ROOT, "var"), exist_ok=True)

    filename: str = os.path.join(settings.DEPLOY_ROOT, "var", "remote_cache_prefix")
    prefix: str = ""
    try:
        with open(filename, "x") as f:
            prefix = secrets.token_hex(16) + ":"
            f.write(prefix + "\n")
    except FileExistsError:
        tries: int = 1
        while tries < 10:
            with open(filename) as f:
                prefix = f.readline().removesuffix("\n")
            if len(prefix) == 33:
                break
            tries += 1
            prefix = ""
            time.sleep(0.5)

    if not prefix:
        print("Could not read remote cache key prefix file")
        sys.exit(1)

    return prefix


KEY_PREFIX: str = get_or_create_key_prefix()


def bounce_key_prefix_for_testing(test_name: str) -> None:
    global KEY_PREFIX
    KEY_PREFIX = test_name + ":" + str(os.getpid()) + ":"
    KEY_PREFIX = hashlib.sha1(KEY_PREFIX.encode()).hexdigest() + ":"


def get_cache_backend(cache_name: Optional[str]) -> BaseCache:
    if cache_name is None:
        cache_name = "default"
    return caches[cache_name]


def cache_with_key(
    keyfunc: Callable[ParamT, str],
    cache_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Callable[[Callable[ParamT, ReturnT]], Callable[ParamT, ReturnT]]:
    def decorator(func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
        @wraps(func)
        def func_with_caching(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
            key: str = keyfunc(*args, **kwargs)

            try:
                val: Any = cache_get(key, cache_name=cache_name)
            except InvalidCacheKeyError:
                stack_trace: str = traceback.format_exc()
                log_invalid_cache_keys(stack_trace, [key])
                return func(*args, **kwargs)

            if val is not None:
                return val[0]

            val = func(*args, **kwargs)
            if isinstance(val, QuerySet):
                logging.error(
                    "cache_with_key attempted to store a full QuerySet object -- declining to cache",
                    stack_info=True,
                )
            else:
                cache_set(key, val, cache_name=cache_name, timeout=timeout)

            return val

        return func_with_caching

    return decorator


class InvalidCacheKeyError(Exception):
    pass


def log_invalid_cache_keys(stack_trace: str, key: List[str]) -> None:
    logger.warning(
        "Invalid cache key used: %s\nStack trace: %s\n",
        key,
        stack_trace,
    )


def validate_cache_key(key: str) -> None:
    if not key.startswith(KEY_PREFIX):
        key = KEY_PREFIX + key

    if not bool(re.fullmatch(r"([!-~])+", key)):
        raise InvalidCacheKeyError("Invalid characters in the cache key: " + key)
    if len(key) > MEMCACHED_MAX_KEY_LENGTH:
        raise InvalidCacheKeyError(f"Cache key too long: {key} Length: {len(key)}")


def cache_set(
    key: str, val: Any, cache_name: Optional[str] = None, timeout: Optional[int] = None
) -> None:
    final_key: str = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    cache_backend: BaseCache = get_cache_backend(cache_name)
    cache_backend.set(final_key, (val,), timeout=timeout)
    remote_cache_stats_finish()


def cache_get(key: str, cache_name: Optional[str] = None) -> Any:
    final_key: str = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    cache_backend: BaseCache = get_cache_backend(cache_name)
    ret: Any = cache_backend.get(final_key)
    remote_cache_stats_finish()
    return ret


def cache_get_many(keys: List[str], cache_name: Optional[str] = None) -> Dict[str, Any]:
    keys = [KEY_PREFIX + key for key in keys]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    ret: Dict[str, Any] = get_cache_backend(cache_name).get_many(keys)
    remote_cache_stats_finish()
    return {key.removeprefix(KEY_PREFIX): value for key, value in ret.items()}


def safe_cache_get_many(keys: List[str], cache_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        return cache_get_many(keys, cache_name)
    except InvalidCacheKeyError:
        stack_trace: str = traceback.format_exc()
        good_keys: List[str]
        bad_keys: List[str]
        good_keys, bad_keys = filter_good_and_bad_keys(keys)

        log_invalid_cache_keys(stack_trace, bad_keys)
        return cache_get_many(good_keys, cache_name)


def cache_set_many(
    items: Dict[str, Any], cache_name: Optional[str] = None, timeout: Optional[int] = None
) -> None:
    new_items: Dict[str, Any] = {}
    for key, item in items.items():
        new_key: str = KEY_PREFIX + key
        validate_cache_key(new_key)
        new_items[new_key] = item
    items = new_items
    remote_cache_stats_start()
    get_cache_backend(cache_name).set_many(items, timeout=timeout)
    remote_cache_stats_finish()


def safe_cache_set_many(
    items: Dict[str, Any], cache_name: Optional[str] = None, timeout: Optional[int] = None
) -> None:
    try:
        return cache_set_many(items, cache_name, timeout)
    except InvalidCacheKeyError:
        stack_trace: str = traceback.format_exc()

        good_keys: List[str]
        bad_keys: List[str]
        good_keys, bad_keys = filter_good_and_bad_keys(list(items.keys()))
        log_invalid_cache_keys(stack_trace, bad_keys)

        good_items: Dict[str, Any] = {key: items[key] for key in good_keys}
        return cache_set_many(good_items, cache_name, timeout)


def cache_delete(key: str, cache_name: Optional[str] = None) -> None:
    final_key: str = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    get_cache_backend(cache_name).delete(final_key)
    remote_cache_stats_finish()


def cache_delete_many(items: Iterable[str], cache_name: Optional[str] = None) -> None:
    keys: List[str] = [KEY_PREFIX + item for item in items]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    get_cache_backend(cache_name).delete_many(keys)
    remote_cache_stats_finish()


def filter_good_and_bad_keys(keys: List[str]) -> Tuple[List[str], List[str]]:
    good_keys: List[str] = []
    bad_keys: List[str] = []
    for key in keys:
        try:
            validate_cache_key(key)
            good_keys.append(key)
        except InvalidCacheKeyError:
            bad_keys.append(key)

    return good_keys, bad_keys


ObjKT = TypeVar("ObjKT")
ItemT = TypeVar("ItemT")
CacheItemT = TypeVar("CacheItemT")
CompressedItemT = TypeVar("CompressedItemT")


def generic_bulk_cached_fetch(
    cache_key_function: Callable[[ObjKT], str],
    query_function: Callable[[List[ObjKT]], Iterable[ItemT]],
    object_ids: Sequence[ObjKT],
    *,
    extractor: Callable[[CompressedItemT], CacheItemT],
    setter: Callable[[CacheItemT], CompressedItemT],
    id_fetcher: Callable[[ItemT], ObjKT],
    cache_transformer: Callable[[ItemT], CacheItemT],
) -> Dict[ObjKT, CacheItemT]:
    if len(object_ids) == 0:
        return {}

    cache_keys: Dict[ObjKT, str] = {}
    for object_id in object_ids:
        cache_keys[object_id] = cache_key_function(object_id)

    cached_objects_compressed: Dict[str, Tuple[CompressedItemT]] = safe_cache_get_many(
        [cache_keys[object_id] for object_id in object_ids],
    )

    cached_objects: Dict[str, CacheItemT] = {key: extractor(val[0]) for key, val in cached_objects_compressed.items()}
    needed_ids: List[ObjKT] = [
        object_id for object_id in object_ids if cache_keys[object_id] not in cached_objects
    ]

    if len(needed_ids) > 0:
        db_objects: Iterable[ItemT] = query_function(needed_ids)
    else:
        db_objects = []

    items_for_remote_cache: Dict[str, Tuple[CompressedItemT]] = {}
    for obj in db_objects:
        key: str = cache_keys[id_fetcher(obj)]
        item: CacheItemT = cache_transformer(obj)
        items_for_remote_cache[key] = (setter(item),)
        cached_objects[key] = item
    if len(items_for_remote_cache) > 0:
        safe_cache_set_many(items_for_remote_cache)
    return {
        object_id: cached_objects[cache_keys[object_id]]
        for object_id in object_ids
        if cache_keys[object_id] in cached_objects
    }


def bulk_cached_fetch(
    cache_key_function: Callable[[ObjKT], str],
    query_function: Callable[[List[ObjKT]], Iterable[ItemT]],
    object_ids: Sequence[ObjKT],
    *,
    id_fetcher: Callable[[ItemT], ObjKT],
) -> Dict[ObjKT, ItemT]:
    return generic_bulk_cached_fetch(
        cache_key_function,
        query_function,
        object_ids,
        id_fetcher=id_fetcher,
        extractor=lambda obj: obj,
        setter=lambda obj: obj,
        cache_transformer=lambda obj: obj,
    )


def preview_url_cache_key(url: str) -> str:
    return f"preview_url:{hashlib.sha1(url.encode()).hexdigest()}"


def display_recipient_cache_key(recipient_id: int) -> str:
    return f"display_recipient_dict:{recipient_id}"


def single_user_display_recipient_cache_key(user_id: int) -> str:
    return f"single_user_display_recipient:{user_id}"


def user_profile_by_email_realm_id_cache_key(email: str, realm_id: int) -> str:
    return f"user_profile:{hashlib.sha1(email.strip().encode()).hexdigest()}:{realm_id}"


def user_profile_by_email_realm_cache_key(email: str, realm: "Realm") -> str:
    return user_profile_by_email_realm_id_cache_key(email, realm.id)


def user_profile_delivery_email_cache_key(delivery_email: str, realm_id: int) -> str:
    return f"user_profile_by_delivery_email:{hashlib.sha1(delivery_email.strip().encode()).hexdigest()}:{realm_id}"


def bot_profile_cache_key(email: str, realm_id: int) -> str:
    return f"bot_profile:{hashlib.sha1(email.strip().encode()).hexdigest()}"


def user_profile_by_id_cache_key(user_profile_id: int) -> str:
    return f"user_profile_by_id:{user_profile_id}"


def user_profile_narrow_by_id_cache_key(user_profile_id: int) -> str:
    return f"user_profile_narrow_by_id:{user_profile_id}"


def user_profile_by_api_key_cache_key(api_key: str) -> str:
    return f"user_profile_by_api_key:{api_key}"


def get_cross_realm_dicts_key() -> str:
    emails: List[str] = list(settings.CROSS_REALM_BOT_EMAILS)
    raw_key: str = ",".join(sorted(emails))
    digest: str = hashlib.sha1(raw_key.encode()).hexdigest()
    return f"get_cross_realm_dicts:{digest}"


realm_user_dict_fields: List[str] = [
    "id",
    "full_name",
    "email",
    "avatar_source",
    "avatar_version",
    "is_active",
    "role",
    "is_billing_admin",
    "is_bot",
    "timezone",
    "date_joined",
    "bot_owner_id",
    "delivery_email",
    "bot_type",
    "long_term_idle",
    "email_address_visibility",
]


def realm_user_dicts_cache_key(realm_id: int) -> str:
    return f"realm_user_dicts:{realm_id}"


def get_muting_users_cache_key(muted_user_id: int) -> str:
    return f"muting_users_list:{muted_user_id}"


def get_realm_used_upload_space_cache_key(realm_id: int) -> str:
    return f"realm_used_upload_space:{realm_id}"


def get_realm_seat_count_cache_key(realm_id: int) -> str:
    return f"realm_seat_count:{realm_id}"


def active_user_ids_cache_key(realm_id: int) -> str:
    return f"active_user_ids:{realm_id}"


def active_non_guest_user_ids_cache_key(realm_id: int) -> str:
    return f"active_non_guest_user_ids:{realm_id}"


bot_dict_fields: List[str] = [
    "api_key",
    "avatar_source",
    "avatar_version",
    "bot_owner_id",
    "bot_type",
    "default_all_public_streams",
    "default_events_register_stream__name",
    "default_sending_stream__name",
    "email",
    "full_name",
    "id",
    "is_active",
    "realm_id",
]


def bot_dicts_in_realm_cache_key(realm_id: int) -> str:
    return f"bot_dicts_in_realm:{realm_id}"


def delete_user_profile_caches(user_profiles: Iterable["UserProfile"], realm_id: int) -> None:
    from zerver.models.users import is_cross_realm_bot_email

    keys: List[str] = []
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


def delete_display_recipient_cache(user_profile: "UserProfile") -> None:
    from zerver.models import Subscription

    recipient_ids: QuerySet[int] = Subscription.objects.filter(user_profile=user_profile).values_list(
        "recipient_id", flat=True
    )
    keys: List[str] = [display_recipient_cache_key(rid) for rid in recipient_ids]
    keys.append(single_user_display_recipient_cache_key(user_profile.id))
    cache_delete_many(keys)


def changed(update_fields: Optional[Sequence[str]], fields: List[str]) -> bool:
    if update_fields is None:
        return True

    update_fields_set: Set[str] = set(update_fields)
    return any(f in update_fields_set for f in fields)


def flush_user_profile(
    *,
    instance: "UserProfile",
    update_fields: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> None:
    user_profile: "UserProfile" = instance
    delete_user_profile_caches([user_profile], user_profile.realm_id)

    if changed(update_fields, realm_user