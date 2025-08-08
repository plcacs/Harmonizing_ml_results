from typing import Optional, cast, List, Tuple, Set

class RateLimitedObject(ABC):

    def __init__(self, backend: Optional[Type[RateLimiterBackend]] = None) -> None:
        ...

    def rate_limit(self) -> Tuple[bool, float]:
        ...

    def rate_limit_request(self, request: HttpRequest) -> None:
        ...

    def block_access(self, seconds: int) -> None:
        ...

    def unblock_access(self) -> None:
        ...

    def clear_history(self) -> None:
        ...

    def max_api_calls(self) -> int:
        ...

    def max_api_window(self) -> int:
        ...

    def api_calls_left(self) -> Tuple[int, float]:
        ...

    def get_rules(self) -> List[Tuple[int, int]]:
        ...

    @abstractmethod
    def key(self) -> str:
        ...

    @abstractmethod
    def rules(self) -> List[Tuple[int, int]]:
        ...

class RateLimitedUser(RateLimitedObject):

    def __init__(self, user: UserProfile, domain: str = 'api_by_user') -> None:
        ...

    def key(self) -> str:
        ...

    def rules(self) -> List[Tuple[int, int]]:
        ...

class RateLimitedIPAddr(RateLimitedObject):

    def __init__(self, ip_addr: str, domain: str = 'api_by_ip') -> None:
        ...

    def key(self) -> str:
        ...

    def rules(self) -> List[Tuple[int, int]]:
        ...

class RateLimitedEndpoint(RateLimitedObject):

    def __init__(self, endpoint_name: str) -> None:
        ...

    def key(self) -> str:
        ...

    def rules(self) -> List[Tuple[int, int]]:
        ...

class RateLimiterBackend(ABC):

    @classmethod
    @abstractmethod
    def block_access(cls, entity_key: str, seconds: int) -> None:
        ...

    @classmethod
    @abstractmethod
    def unblock_access(cls, entity_key: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def clear_history(cls, entity_key: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def get_api_calls_left(cls, entity_key: str, range_seconds: int, max_calls: int) -> Tuple[int, float]:
        ...

    @classmethod
    @abstractmethod
    def rate_limit_entity(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int, max_api_window: int) -> Tuple[bool, float]

class TornadoInMemoryRateLimiterBackend(RateLimiterBackend):

    @classmethod
    def _garbage_collect_for_rule(cls, now: float, time_window: int, max_count: int) -> None:
        ...

    @classmethod
    def need_to_limit(cls, entity_key: str, time_window: int, max_count: int) -> Tuple[bool, float]:
        ...

    @classmethod
    def get_api_calls_left(cls, entity_key: str, range_seconds: int, max_calls: int) -> Tuple[int, float]:
        ...

    @classmethod
    def block_access(cls, entity_key: str, seconds: int) -> None:
        ...

    @classmethod
    def unblock_access(cls, entity_key: str) -> None:
        ...

    @classmethod
    def clear_history(cls, entity_key: str) -> None:
        ...

    @classmethod
    def rate_limit_entity(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int, max_api_window: int) -> Tuple[bool, float]

class RedisRateLimiterBackend(RateLimiterBackend):

    @classmethod
    def get_keys(cls, entity_key: str) -> List[str]:
        ...

    @classmethod
    def block_access(cls, entity_key: str, seconds: int) -> None:
        ...

    @classmethod
    def unblock_access(cls, entity_key: str) -> None:
        ...

    @classmethod
    def clear_history(cls, entity_key: str) -> None:
        ...

    @classmethod
    def get_api_calls_left(cls, entity_key: str, range_seconds: int, max_calls: int) -> Tuple[int, float]:
        ...

    @classmethod
    def is_ratelimited(cls, entity_key: str, rules: List[Tuple[int, int]]) -> Tuple[bool, float]:
        ...

    @classmethod
    def incr_ratelimit(cls, entity_key: str, max_api_calls: int, max_api_window: int) -> None:
        ...

    @classmethod
    def rate_limit_entity(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int, max_api_window: int) -> Tuple[bool, float]

class RateLimitResult:

    def __init__(self, entity: RateLimitedObject, secs_to_freedom: float, over_limit: bool, remaining: int) -> None:
        ...

class RateLimitedSpectatorAttachmentAccessByFile(RateLimitedObject):

    def __init__(self, path_id: str) -> None:
        ...

    def key(self) -> str:
        ...

    def rules(self) -> List[Tuple[int, int]]:
        ...

def rate_limit_spectator_attachment_access_by_file(path_id: str) -> None:
    ...

def is_local_addr(addr: str) -> bool:
    ...

def get_tor_ips() -> Set[str]:
    ...

def client_is_exempt_from_rate_limiting(request: HttpRequest) -> bool:
    ...

def rate_limit_user(request: HttpRequest, user: UserProfile, domain: str) -> None:
    ...

def rate_limit_request_by_ip(request: HttpRequest, domain: str) -> None:
    ...

def rate_limit_endpoint_absolute(endpoint_name: str) -> None:
    ...

def should_rate_limit(request: HttpRequest) -> bool:
    ...
