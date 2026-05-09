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
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    ParamSpec,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from django.conf import settings
from django.core.cache import caches
from django.core.cache.backends.base import BaseCache
from django.db.models import Q, QuerySet
from typing_extensions import ParamSpec

logger = logging.getLogger()
remote_cache_time_start: float
remote_cache_total_time: float
remote_cache_total_requests: int

class InvalidCacheKeyError(Exception):
    pass

def get_remote_cache_time() -> float:
    ...

def get_remote_cache_requests() -> int:
    ...

def remote_cache_stats_start() -> None:
    ...

def remote_cache_stats_finish() -> None:
    ...

def get_or_create_key_prefix() -> str:
    ...

def bounce_key_prefix_for_testing(test_name: str) -> None:
    ...

def get_cache_backend(cache_name: Optional[str] = None) -> BaseCache:
    ...

def cache_with_key(
    keyfunc: Callable[..., str],
    cache_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Callable[[Callable], Callable]:
    ...

def log_invalid_cache_keys(stack_trace: str, key: Union[str, List[str]]) -> None:
    ...

def validate_cache_key(key: str) -> None:
    ...

def cache_set(
    key: str,
    val: Any,
    cache_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    ...

def cache_get(key: str, cache_name: Optional[str] = None) -> Optional[Tuple[Any]]:
    ...

def cache_get_many(keys: List[str], cache_name: Optional[str] = None) -> Dict[str, Tuple[Any]]:
    ...

def safe_cache_get_many(keys: List[str], cache_name: Optional[str] = None) -> Dict[str, Any]:
    ...

def cache_set_many(
    items: Dict[str, Any],
    cache_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    ...

def safe_cache_set_many(
    items: Dict[str, Any],
    cache_name: Optional[str] = None,
    timeout: Optional[int] = None,
) -> None:
    ...

def cache_delete(key: str, cache_name: Optional[str] = None) -> None:
    ...

def cache_delete_many(items: List[str], cache_name: Optional[str] = None) -> None:
    ...

def filter_good_and_bad_keys(keys: List[str]) -> Tuple[List[str], List[str]]:
    ...

def generic_bulk_cached_fetch(
    cache_key_function: Callable[[Any], str],
    query_function: Callable[[List[Any]], Iterable[Any]],
    object_ids: List[Any],
    extractor: Callable[[Any], Any],
    setter: Callable[[Any], Any],
    id_fetcher: Callable[[Any], Any],
    cache_transformer: Callable[[Any], Any],
) -> Dict[Any, Any]:
    ...

def bulk_cached_fetch(
    cache_key_function: Callable[[Any], str],
    query_function: Callable[[List[Any]], Iterable[Any]],
    object_ids: List[Any],
    id_fetcher: Callable[[Any], Any],
) -> Dict[Any, Any]:
    ...

def preview_url_cache_key(url: str) -> str:
    ...

def display_recipient_cache_key(recipient_id: int) -> str:
    ...

def single_user_display_recipient_cache_key(user_id: int) -> str:
    ...

def user_profile_by_email_realm_id_cache_key(email: str, realm_id: int) -> str:
    ...

def user_profile_by_email_realm_cache_key(email: str, realm: Any) -> str:
    ...

def user_profile_delivery_email_cache_key(delivery_email: str, realm_id: int) -> str:
    ...

def bot_profile_cache_key(email: str, realm_id: int) -> str:
    ...

def user_profile_by_id_cache_key(user_profile_id: int) -> str:
    ...

def user_profile_narrow_by_id_cache_key(user_profile_id: int) -> str:
    ...

def user_profile_by_api_key_cache_key(api_key: str) -> str:
    ...

def get_cross_realm_dicts_key() -> str:
    ...

def realm_user_dicts_cache_key(realm_id: int) -> str:
    ...

def get_muting_users_cache_key(muted_user_id: int) -> str:
    ...

def get_realm_used_upload_space_cache_key(realm_id: int) -> str:
    ...

def get_realm_seat_count_cache_key(realm_id: int) -> str:
    ...

def active_user_ids_cache_key(realm_id: int) -> str:
    ...

def active_non_guest_user_ids_cache_key(realm_id: int) -> str:
    ...

def bot_dicts_in_realm_cache_key(realm_id: int) -> str:
    ...

def delete_user_profile_caches(user_profiles: List[Any], realm_id: int) -> None:
    ...

def delete_display_recipient_cache(user_profile: Any) -> None:
    ...

def flush_user_profile(*args: Any, **kwargs: Any) -> None:
    ...

def flush_muting_users_cache(*args: Any, **kwargs: Any) -> None:
    ...

def flush_realm(*args: Any, **kwargs: Any) -> None:
    ...

def realm_alert_words_cache_key(realm_id: int) -> str:
    ...

def realm_alert_words_automaton_cache_key(realm_id: int) -> str:
    ...

def realm_rendered_description_cache_key(realm: Any) -> str:
    ...

def realm_text_description_cache_key(realm: Any) -> str:
    ...

def flush_stream(*args: Any, **kwargs: Any) -> None:
    ...

def flush_used_upload_space_cache(*args: Any, **kwargs: Any) -> None:
    ...

def to_dict_cache_key_id(message_id: int) -> str:
    ...

def to_dict_cache_key(message: Any, realm_id: Optional[int] = None) -> str:
    ...

def open_graph_description_cache_key(content: str, request_url: str) -> str:
    ...

def zoom_server_access_token_cache_key(account_id: Any) -> str:
    ...

def flush_zoom_server_access_token_cache(account_id: Any) -> None:
    ...

def flush_message(*args: Any, **kwargs: Any) -> None:
    ...

def flush_submessage(*args: Any, **kwargs: Any) -> None:
    ...

class IgnoreUnhashableLruCacheWrapper(Generic[ParamSpec, Any]):
    def __init__(self, function: Callable, cached_function: _lru_cache_wrapper) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

def ignore_unhashable_lru_cache(
    maxsize: int = 128,
    typed: bool = False,
) -> Callable[[Callable], IgnoreUnhashableLruCacheWrapper]:
    ...

def dict_to_items_tuple(user_function: Callable) -> Callable:
    ...

def items_tuple_to_dict(user_function: Callable) -> Callable:
    ...