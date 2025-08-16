from typing import Any, Dict, List, Tuple

def get_users() -> QuerySet[UserProfile]:
    ...

def user_cache_items(items_for_remote_cache: Dict[str, Tuple[UserProfile]]) -> None:
    ...

def get_narrow_users() -> QuerySet[UserProfile]:
    ...

def user_narrow_cache_items(items_for_remote_cache: Dict[str, Tuple[UserProfile]]) -> None:
    ...

def client_cache_items(items_for_remote_cache: Dict[str, Tuple[Client]]) -> None:
    ...

def session_cache_items(items_for_remote_cache: Dict[str, Any], session: Session) -> None:
    ...

def get_active_realm_ids() -> List[int]:
    ...

cache_fillers: Dict[str, Tuple[Callable, Callable, int, int]] = {
    'user': (get_users, user_cache_items, 3600 * 24 * 7, 10000),
    'user_narrow': (get_narrow_users, user_narrow_cache_items, 3600 * 24 * 7, 10000),
    'client': (Client.objects.all, client_cache_items, 3600 * 24 * 7, 10000),
    'session': (Session.objects.all, session_cache_items, 3600 * 24 * 7, 10000)
}

class SQLQueryCounter:
    def __init__(self) -> None:
        ...

    def __call__(self, execute, sql, params, many, context) -> Any:
        ...

def fill_remote_cache(cache: str) -> None:
    ...
