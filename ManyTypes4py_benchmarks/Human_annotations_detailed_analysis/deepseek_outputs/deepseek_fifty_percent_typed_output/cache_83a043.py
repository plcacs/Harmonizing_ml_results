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
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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

logger = logging.getLogger()

remote_cache_time_start = 0.0
remote_cache_total_time = 0.0
remote_cache_total_requests = 0


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
        # This sets the prefix for the benefit of the Puppeteer tests.
        #
        # Having a fixed key is OK since we don't support running
        # multiple copies of the Puppeteer tests at the same time anyway.
        return "puppeteer_tests:"
    elif settings.TEST_SUITE:
        # The Python tests overwrite KEY_PREFIX on each test, but use
        # this codepath as well, just to save running the more complex
        # code below for reading the normal key prefix.
        return "django_tests_unused:"

    # directory `var` should exist in production
    os.makedirs(os.path.join(settings.DEPLOY_ROOT, "var"), exist_ok=True)

    filename = os.path.join(settings.DEPLOY_ROOT, "var", "remote_cache_prefix")
    try:
        with open(filename, "x") as f:
            prefix = secrets.token_hex(16) + ":"
            f.write(prefix + "\n")
    except FileExistsError:
        tries = 1
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
    # We are taking the hash of the KEY_PREFIX to decrease the size of the key.
    # Memcached keys should have a length of less than 250.
    KEY_PREFIX = hashlib.sha1(KEY_PREFIX.encode()).hexdigest() + ":"


def get_cache_backend(cache_name: str | None = None) -> BaseCache:
    if cache_name is None:
        cache_name = "default"
    return caches[cache_name]


def cache_with_key(
    keyfunc: Callable[..., str],
    cache_name: str | None = None,
    timeout: int | None = None,
) -> Callable[[Callable[ParamT, ReturnT]], Callable[ParamT, ReturnT]]:
    """Decorator which applies Django caching to a function.

    Decorator argument is a function which computes a cache key
    from the original function's arguments.  You are responsible
    for avoiding collisions with other uses of this decorator or
    other uses of caching."""

    def decorator(func: Callable[ParamT, ReturnT]) -> Callable[ParamT, ReturnT]:
        @wraps(func)
        def func_with_caching(*args: ParamT.args, **kwargs: ParamT.kwargs) -> ReturnT:
            key = keyfunc(*args, **kwargs)

            try:
                val = cache_get(key, cache_name=cache_name)
            except InvalidCacheKeyError:
                stack_trace = traceback.format_exc()
                log_invalid_cache_keys(stack_trace, [key])
                return func(*args, **kwargs)

            # Values are singleton tuples so that we can distinguish
            # a result of None from a missing key.
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


def log_invalid_cache_keys(stack_trace: str, key: list[str]) -> None:
    logger.warning(
        "Invalid cache key used: %s\nStack trace: %s\n",
        key,
        stack_trace,
    )


def validate_cache_key(key: str) -> None:
    if not key.startswith(KEY_PREFIX):
        key = KEY_PREFIX + key

    # Theoretically memcached can handle non-ascii characters
    # and only "control" characters are strictly disallowed, see:
    # https://github.com/memcached/memcached/blob/master/doc/protocol.txt
    # However, limiting the characters we allow in keys simiplifies things,
    # and anyway we use a hash function when forming some keys to ensure
    # the resulting keys fit the regex below.
    # The regex checks "all characters between ! and ~ in the ascii table",
    # which happens to be the set of all "nice" ascii characters.
    if not bool(re.fullmatch(r"([!-~])+", key)):
        raise InvalidCacheKeyError("Invalid characters in the cache key: " + key)
    if len(key) > MEMCACHED_MAX_KEY_LENGTH:
        raise InvalidCacheKeyError(f"Cache key too long: {key} Length: {len(key)}")


def cache_set(
    key: str, val: Any, cache_name: str | None = None, timeout: int | None = None
) -> None:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    cache_backend = get_cache_backend(cache_name)
    cache_backend.set(final_key, (val,), timeout=timeout)
    remote_cache_stats_finish()


def cache_get(key: str, cache_name: str | None = None) -> Any:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    cache_backend = get_cache_backend(cache_name)
    ret = cache_backend.get(final_key)
    remote_cache_stats_finish()
    return ret


def cache_get_many(keys: list[str], cache_name: str | None = None) -> dict[str, Any]:
    keys = [KEY_PREFIX + key for key in keys]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    ret = get_cache_backend(cache_name).get_many(keys)
    remote_cache_stats_finish()
    return {key.removeprefix(KEY_PREFIX): value for key, value in ret.items()}


def safe_cache_get_many(keys: list[str], cache_name: str | None = None) -> dict[str, Any]:
    """Variant of cache_get_many that drops any keys that fail
    validation, rather than throwing an exception visible to the
    caller."""
    try:
        # Almost always the keys will all be correct, so we just try
        # to do normal cache_get_many to avoid the overhead of
        # validating all the keys here.
        return cache_get_many(keys, cache_name)
    except InvalidCacheKeyError:
        stack_trace = traceback.format_exc()
        good_keys, bad_keys = filter_good_and_bad_keys(keys)

        log_invalid_cache_keys(stack_trace, bad_keys)
        return cache_get_many(good_keys, cache_name)


def cache_set_many(
    items: dict[str, Any], cache_name: str | None = None, timeout: int | None = None
) -> None:
    new_items = {}
    for key, item in items.items():
        new_key = KEY_PREFIX + key
        validate_cache_key(new_key)
        new_items[new_key] = item
    items = new_items
    remote_cache_stats_start()
    get_cache_backend(cache_name).set_many(items, timeout=timeout)
    remote_cache_stats_finish()


def safe_cache_set_many(
    items: dict[str, Any], cache_name: str | None = None, timeout: int | None = None
) -> None:
    """Variant of cache_set_many that drops saving any keys that fail
    validation, rather than throwing an exception visible to the
    caller."""
    try:
        # Almost always the keys will all be correct, so we just try
        # to do normal cache_set_many to avoid the overhead of
        # validating all the keys here.
        return cache_set_many(items, cache_name, timeout)
    except InvalidCacheKeyError:
        stack_trace = traceback.format_exc()

        good_keys, bad_keys = filter_good_and_bad_keys(list(items.keys()))
        log_invalid_cache_keys(stack_trace, bad_keys)

        good_items = {key: items[key] for key in good_keys}
        return cache_set_many(good_items, cache_name, timeout)


def cache_delete(key: str, cache_name: str | None = None) -> None:
    final_key = KEY_PREFIX + key
    validate_cache_key(final_key)

    remote_cache_stats_start()
    get_cache_backend(cache_name).delete(final_key)
    remote_cache_stats_finish()


def cache_delete_many(items: list[str], cache_name: str | None = None) -> None:
    keys = [KEY_PREFIX + item for item in items]
    for key in keys:
        validate_cache_key(key)
    remote_cache_stats_start()
    get_cache_backend(cache_name).delete_many(keys)
    remote_cache_stats_finish()


def filter_good_and_bad_keys(keys: list[str]) -> tuple[list[str], list[str]]:
    good_keys: list[str] = []
    bad_keys: list[str] = []
    for key in keys:
        try:
            validate_cache_key(key)
            good_keys.append(key)
        except InvalidCacheKeyError:
            bad_keys.append(key)

    return good_keys, bad_keys


# Generic_bulk_cached fetch and its helpers.  We start with declaring
# a few type variables that help define its interface.

# Type for the cache's keys; will typically be int or str.
ObjKT = TypeVar("ObjKT")

# Type for items to be fetched from the database (e.g. a Django model object)
ItemT = TypeVar("ItemT")

# Type for items to be stored in the cache (e.g. a dictionary serialization).
# Will equal ItemT unless a cache_transformer is specified.
CacheItemT = TypeVar("CacheItemT")

# Type for compressed items for storage in the cache.  For
# serializable objects, will be the object; if encoded, bytes.
CompressedItemT = TypeVar("CompressedItemT")


# Required arguments are as follows:
# * object_ids: The list of object ids to look up
# * cache_key_function: object_id => cache key
# * query_function: [object_ids] => [objects from database]
# * setter: Function to call before storing items to cache (e.g. compression)
# * extractor: Function to call on items returned from cache
#   (e.g. decompression).  Should be the inverse of the setter
#   function.
# * id_fetcher: Function mapping an object from database => object_id
#   (in case we're using a key more complex than obj.id)
# * cache_transformer: Function mapping an object from database =>
#   value for cache (in case the values that we're caching are some
#   function of the objects, not the objects themselves)
def generic_bulk_cached_fetch(
    cache_key_function: Callable[[ObjKT], str],
    query_function: Callable[[list[ObjKT]], Iterable[ItemT]],
    object_ids: list[ObjKT],
    *,
    extractor: Callable[[CompressedItemT], CacheItemT],
    setter: Callable[[CacheItemT], CompressedItemT],
    id_fetcher: Callable[[ItemT], ObjKT],
    cache_transformer: Callable[[ItemT], CacheItemT],
) -> dict[ObjKT, CacheItemT]:
    if len(object_ids) == 0:
        # Nothing to fetch.
        return {}

    cache_keys: dict[ObjKT, str] = {}
    for object_id in object_ids:
        cache_keys[object_id] = cache_key_function(object_id)

    cached_objects_compressed: dict[str, tuple[CompressedItemT]] = safe_cache_get_many(
        [cache_keys[object_id] for object_id in object_ids],
    )

    cached_objects = {key: extractor(val[0]) for key, val in cached_objects_compressed.items()}
    needed_ids = [
        object_id for object_id in object_ids if cache_keys[object_id] not in cached_objects
    ]

    # Only call query_function if there are some ids to fetch from the database:
    if len(needed_ids) > 0:
        db_objects = query_function(needed_ids)
    else:
        db_objects = []

    items_for_remote_cache: dict[str, tuple[CompressedItemT]] = {}
    for obj in db_objects:
        key = cache_keys[id_fetcher(obj)]
        item = cache_transformer(obj)
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
    query_function: Callable[[list[ObjKT]], Iterable[ItemT]],
    object_ids: Sequence[ObjKT],
    *,
    id_fetcher: Callable[[ItemT], ObjKT],
) -> dict[ObjKT, ItemT]:
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
    emails = list(settings.CROSS_REALM_BOT_EMAILS)
    raw_key = ",".join(sorted(emails))
    digest = hashlib.sha1(raw_key.encode()).hexdigest()
    return f"get_cross_realm_dicts:{digest}"


realm_user_dict_fields: list[str] = [
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
    return f