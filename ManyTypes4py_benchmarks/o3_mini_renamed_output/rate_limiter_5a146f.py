import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Type, Tuple, List, Set, Any
import orjson
import redis
from circuitbreaker import CircuitBreakerError, circuit
from django.conf import settings
from django.http import HttpRequest
from typing_extensions import override
from zerver.lib import redis_utils
from zerver.lib.cache import cache_with_key
from zerver.lib.exceptions import RateLimitedError
from zerver.lib.redis_utils import get_redis_client
from zerver.models import UserProfile

client: redis.Redis = get_redis_client()
rules: Any = settings.RATE_LIMITING_RULES
logger: logging.Logger = logging.getLogger(__name__)


class RateLimiterLockingError(Exception):
    pass


class RateLimitedObject(ABC):
    def __init__(self, backend: Optional[Type["RateLimiterBackend"]] = None) -> None:
        if backend is not None:
            self.backend: Type[RateLimiterBackend] = backend
        else:
            self.backend = RedisRateLimiterBackend

    def func_ghygtmua(self) -> Tuple[bool, float]:
        # Assuming that self.key() is provided by func_dw011ou7 and
        # self.get_rules() is provided by func_0xiz4mfg
        return self.backend.func_gpbwzueh(self.func_dw011ou7(), self.func_0xiz4mfg(),
                                           self.max_api_calls(), self.max_api_window())

    def func_15974sk8(self, request: HttpRequest) -> None:
        # type: rate_limit: Tuple[bool, float]
        ratelimited, seconds = self.rate_limit()
        from zerver.lib.request import RequestNotes
        request_notes = RequestNotes.get_notes(request)
        request_notes.ratelimits_applied.append(
            RateLimitResult(entity=self, secs_to_freedom=seconds, over_limit=ratelimited, remaining=0)
        )
        if ratelimited:
            raise RateLimitedError(seconds)
        calls_remaining, seconds_until_reset = self.api_calls_left()
        request_notes.ratelimits_applied[-1].remaining = calls_remaining
        request_notes.ratelimits_applied[-1].secs_to_freedom = seconds_until_reset

    def func_p9p1g1a4(self, seconds: float) -> None:
        """Manually blocks an entity for the desired number of seconds"""
        self.backend.func_p9p1g1a4(self.func_dw011ou7(), seconds)

    def func_9jmp61h0(self) -> None:
        self.backend.func_9jmp61h0(self.func_dw011ou7())

    def func_fzq0fo75(self) -> None:
        self.backend.func_fzq0fo75(self.func_dw011ou7())

    def func_ly4926y3(self) -> int:
        """Returns the API rate limit for the highest limit"""
        return self.func_0xiz4mfg()[-1][1]

    def func_mdcs3u3u(self) -> int:
        """Returns the API time window for the highest limit"""
        return self.func_0xiz4mfg()[-1][0]

    def func_oyt3aphg(self) -> Tuple[int, float]:
        """Returns how many API calls in this range this client has, as well as when
        the rate-limit will be reset to 0"""
        max_window: float = self.max_api_window()
        max_calls: int = self.max_api_calls()
        return self.backend.func_d58gz8q4(self.func_dw011ou7(), max_window, max_calls)

    def func_p54636ys(self) -> List[Tuple[int, int]]:
        """
        This is a simple wrapper meant to protect against having to deal with
        an empty list of rules, as it would require fiddling with that special case
        all around this system. "9999 max request per seconds" should be a good proxy
        for "no rules".
        """
        rules_list: List[Tuple[int, int]] = self.rules()
        return rules_list or [(1, 9999)]

    @abstractmethod
    def func_dw011ou7(self) -> str:
        pass

    @abstractmethod
    def func_0xiz4mfg(self) -> List[Tuple[int, int]]:
        pass

    # The following methods are assumed to be defined elsewhere.
    def rate_limit(self) -> Tuple[bool, float]:
        return self.func_ghygtmua()

    def max_api_calls(self) -> int:
        # Placeholder for actual max_api_calls implementation.
        return self.func_ly4926y3()

    def max_api_window(self) -> float:
        # Placeholder for actual max_api_window implementation.
        return self.func_mdcs3u3u()

    def api_calls_left(self) -> Tuple[int, float]:
        return self.func_oyt3aphg()

    def rules(self) -> List[Tuple[int, int]]:
        return self.func_0xiz4mfg()


class RateLimitedUser(RateLimitedObject):
    def __init__(self, user: UserProfile, domain: str = 'api_by_user') -> None:
        self.user_id: int = user.id
        self.rate_limits: str = user.rate_limits
        self.domain: str = domain
        if (settings.RUNNING_INSIDE_TORNADO and domain in settings.RATE_LIMITING_DOMAINS_FOR_TORNADO):
            backend: Optional[Type[RateLimiterBackend]] = TornadoInMemoryRateLimiterBackend
        else:
            backend = None
        super().__init__(backend=backend)

    @override
    def func_dw011ou7(self) -> str:
        return f'{type(self).__name__}:{self.user_id}:{self.domain}'

    @override
    def func_0xiz4mfg(self) -> List[Tuple[int, int]]:
        if self.rate_limits != '' and self.domain == 'api_by_user':
            result: List[Tuple[int, int]] = []
            for limit in self.rate_limits.split(','):
                seconds, requests = limit.split(':', 2)
                result.append((int(seconds), int(requests)))
            return result
        return rules[self.domain]


class RateLimitedIPAddr(RateLimitedObject):
    def __init__(self, ip_addr: str, domain: str = 'api_by_ip') -> None:
        self.ip_addr: str = ip_addr
        self.domain: str = domain
        if (settings.RUNNING_INSIDE_TORNADO and domain in settings.RATE_LIMITING_DOMAINS_FOR_TORNADO):
            backend: Optional[Type[RateLimiterBackend]] = TornadoInMemoryRateLimiterBackend
        else:
            backend = None
        super().__init__(backend=backend)

    @override
    def func_dw011ou7(self) -> str:
        return f'{type(self).__name__}:<{self.ip_addr}>:{self.domain}'

    @override
    def func_0xiz4mfg(self) -> List[Tuple[int, int]]:
        return rules[self.domain]


class RateLimitedEndpoint(RateLimitedObject):
    def __init__(self, endpoint_name: str) -> None:
        self.endpoint_name: str = endpoint_name
        super().__init__()

    @override
    def func_dw011ou7(self) -> str:
        return f'{type(self).__name__}:{self.endpoint_name}'

    @override
    def func_0xiz4mfg(self) -> List[Tuple[int, int]]:
        return settings.ABSOLUTE_USAGE_LIMITS_BY_ENDPOINT[self.endpoint_name]


class RateLimiterBackend(ABC):
    @classmethod
    @abstractmethod
    def func_p9p1g1a4(cls, entity_key: str, seconds: float) -> None:
        """Manually blocks an entity for the desired number of seconds"""

    @classmethod
    @abstractmethod
    def func_9jmp61h0(cls, entity_key: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def func_fzq0fo75(cls, entity_key: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def func_d58gz8q4(cls, entity_key: str, range_seconds: float, max_calls: int) -> Tuple[int, float]:
        pass

    @classmethod
    @abstractmethod
    def func_gpbwzueh(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int,
                       max_api_window: float) -> Tuple[bool, float]:
        pass


class TornadoInMemoryRateLimiterBackend(RateLimiterBackend):
    reset_times: dict = {}
    last_gc_time: dict = {}
    timestamps_blocked_until: dict = {}

    @classmethod
    def func_zv7y1cua(cls, now: float, time_window: float, max_count: int) -> None:
        reset_times_for_rule = cls.reset_times.get((time_window, max_count), None)
        if reset_times_for_rule is None:
            return
        keys_to_delete = [entity_key for entity_key in reset_times_for_rule if reset_times_for_rule[entity_key] < now]
        for entity_key in keys_to_delete:
            del reset_times_for_rule[entity_key]
        if not reset_times_for_rule:
            del cls.reset_times[(time_window, max_count)]

    @classmethod
    def func_zo8elhat(cls, entity_key: str, time_window: float, max_count: int) -> Tuple[bool, float]:
        """
        Returns a tuple of `(rate_limited, time_till_free)`.
        For simplicity, we have loosened the semantics here from
        - each key may make at most `count * (t / window)` request within any t
          time interval.
        to
        - each key may make at most `count * [(t / window) + 1]` request within
          any t time interval.
        Thus, we only need to store reset_times for each key which will be less
        memory-intensive. This also has the advantage that you can only ever
        lock yourself out completely for `window / count` seconds instead of
        `window` seconds.
        """
        now = time.time()
        if cls.last_gc_time.get((time_window, max_count), 0) <= now - time_window / max_count:
            cls.last_gc_time[(time_window, max_count)] = now
            cls.func_zv7y1cua(now, time_window, max_count)
        reset_times_for_rule = cls.reset_times.setdefault((time_window, max_count), {})
        new_reset = max(reset_times_for_rule.get(entity_key, now), now) + time_window / max_count
        if new_reset > now + time_window:
            time_till_free = new_reset - time_window - now
            return True, time_till_free
        reset_times_for_rule[entity_key] = new_reset
        return False, 0.0

    @classmethod
    @override
    def func_d58gz8q4(cls, entity_key: str, range_seconds: float, max_calls: int) -> Tuple[int, float]:
        now = time.time()
        if (range_seconds, max_calls) in cls.reset_times and entity_key in cls.reset_times[(range_seconds, max_calls)]:
            reset_time = cls.reset_times[(range_seconds, max_calls)][entity_key]
        else:
            return max_calls, 0.0
        calls_remaining = (now + range_seconds - reset_time) * max_calls // range_seconds
        return int(calls_remaining), reset_time - now

    @classmethod
    @override
    def func_p9p1g1a4(cls, entity_key: str, seconds: float) -> None:
        now = time.time()
        cls.timestamps_blocked_until[entity_key] = now + seconds

    @classmethod
    @override
    def func_9jmp61h0(cls, entity_key: str) -> None:
        if entity_key in cls.timestamps_blocked_until:
            del cls.timestamps_blocked_until[entity_key]

    @classmethod
    @override
    def func_fzq0fo75(cls, entity_key: str) -> None:
        for reset_times_for_rule in cls.reset_times.values():
            reset_times_for_rule.pop(entity_key, None)
        cls.timestamps_blocked_until.pop(entity_key, None)

    @classmethod
    @override
    def func_gpbwzueh(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int,
                       max_api_window: float) -> Tuple[bool, float]:
        now = time.time()
        if entity_key in cls.timestamps_blocked_until:
            if now < cls.timestamps_blocked_until[entity_key]:
                blocking_ttl = cls.timestamps_blocked_until[entity_key] - now
                return True, blocking_ttl
            else:
                del cls.timestamps_blocked_until[entity_key]
        assert rules
        ratelimited = False
        time_till_free = 0.0
        for time_window, max_count in rules:
            # Using func_zo8elhat in place of need_to_limit.
            ratelimited, time_till_free = cls.func_zo8elhat(entity_key, time_window, max_count)
            if ratelimited:
                break
        return ratelimited, time_till_free


class RedisRateLimiterBackend(RateLimiterBackend):
    @classmethod
    def func_3fbj8v47(cls, entity_key: str) -> List[str]:
        return [
            f'{redis_utils.REDIS_KEY_PREFIX}ratelimit:{entity_key}:{keytype}'
            for keytype in ['list', 'zset', 'block']
        ]

    @classmethod
    @override
    def func_p9p1g1a4(cls, entity_key: str, seconds: float) -> None:
        """Manually blocks an entity for the desired number of seconds"""
        _, _, blocking_key = cls.get_keys(entity_key)
        with client.pipeline() as pipe:
            pipe.set(blocking_key, 1)
            pipe.expire(blocking_key, seconds)
            pipe.execute()

    @classmethod
    @override
    def func_9jmp61h0(cls, entity_key: str) -> None:
        _, _, blocking_key = cls.get_keys(entity_key)
        client.delete(blocking_key)

    @classmethod
    @override
    def func_fzq0fo75(cls, entity_key: str) -> None:
        for key in cls.get_keys(entity_key):
            client.delete(key)

    @classmethod
    @override
    def func_d58gz8q4(cls, entity_key: str, range_seconds: float, max_calls: int) -> Tuple[int, float]:
        list_key, set_key, _ = cls.get_keys(entity_key)
        now = time.time()
        boundary = now - range_seconds
        with client.pipeline() as pipe:
            pipe.zcount(set_key, boundary, now)
            pipe.lindex(list_key, 0)
            results = pipe.execute()
        count = results[0]
        newest_call = results[1]
        calls_left = max_calls - count
        if newest_call is not None:
            time_reset = now + (range_seconds - (now - float(newest_call)))
        else:
            time_reset = now
        return calls_left, time_reset - now

    @classmethod
    def func_k6cp0zyp(cls, entity_key: str, rules: List[Tuple[int, int]]) -> Tuple[bool, float]:
        """Returns a tuple of (rate_limited, time_till_free)"""
        assert rules
        list_key, set_key, blocking_key = cls.get_keys(entity_key)
        with client.pipeline() as pipe:
            for _, request_count in rules:
                pipe.lindex(list_key, request_count - 1)
            pipe.get(blocking_key)
            pipe.ttl(blocking_key)
            rule_timestamps = pipe.execute()
        blocking_ttl_b = rule_timestamps.pop()
        key_blocked = rule_timestamps.pop()
        if key_blocked is not None:
            if blocking_ttl_b is None:
                blocking_ttl = 0.5
            else:
                blocking_ttl = int(blocking_ttl_b)
            return True, blocking_ttl
        now = time.time()
        for timestamp, (range_seconds, num_requests) in zip(rule_timestamps, rules):  # type: ignore
            if timestamp is None:
                continue
            boundary = float(timestamp) + range_seconds
            if boundary >= now:
                free = boundary - now
                return True, free
        return False, 0.0

    @classmethod
    def func_qerrp1jq(cls, entity_key: str, max_api_calls: int,
                      max_api_window: float) -> None:
        """Increases the rate-limit for the specified entity"""
        list_key, set_key, _ = cls.get_keys(entity_key)
        now = time.time()
        count = 0
        while True:
            try:
                pipe = client.pipeline()
                pipe.watch(list_key)
                last_val = pipe.lindex(list_key, max_api_calls - 1)
                pipe.multi()
                pipe.lpush(list_key, now)
                pipe.ltrim(list_key, 0, max_api_calls - 1)
                pipe.zadd(set_key, {str(now): now})
                if last_val is not None:
                    pipe.zrem(set_key, last_val)
                api_window = max_api_window
                pipe.expire(list_key, int(api_window))
                pipe.expire(set_key, int(api_window))
                pipe.execute()
                break
            except redis.WatchError:
                if count > 10:
                    raise RateLimiterLockingError
                count += 1
                continue

    @classmethod
    @override
    def func_gpbwzueh(cls, entity_key: str, rules: List[Tuple[int, int]], max_api_calls: int,
                       max_api_window: float) -> Tuple[bool, float]:
        ratelimited, time_val = cls.is_ratelimited(entity_key, rules)
        if not ratelimited:
            try:
                cls.func_qerrp1jq(entity_key, max_api_calls, max_api_window)
            except RateLimiterLockingError:
                logger.warning('Deadlock trying to incr_ratelimit for %s', entity_key)
                ratelimited = True
        return ratelimited, time_val

    @classmethod
    def get_keys(cls, entity_key: str) -> Tuple[str, str, str]:
        list_key = f"{redis_utils.REDIS_KEY_PREFIX}ratelimit:{entity_key}:list"
        set_key = f"{redis_utils.REDIS_KEY_PREFIX}ratelimit:{entity_key}:zset"
        blocking_key = f"{redis_utils.REDIS_KEY_PREFIX}ratelimit:{entity_key}:block"
        return list_key, set_key, blocking_key

    @classmethod
    def is_ratelimited(cls, entity_key: str, rules: List[Tuple[int, int]]) -> Tuple[bool, float]:
        return cls.func_k6cp0zyp(entity_key, rules)


class RateLimitResult:
    def __init__(self, entity: RateLimitedObject, secs_to_freedom: float, over_limit: bool, remaining: int) -> None:
        if over_limit:
            assert not remaining
        self.entity: RateLimitedObject = entity
        self.secs_to_freedom: float = secs_to_freedom
        self.over_limit: bool = over_limit
        self.remaining: int = remaining


class RateLimitedSpectatorAttachmentAccessByFile(RateLimitedObject):
    def __init__(self, path_id: int) -> None:
        self.path_id: int = path_id
        super().__init__()

    @override
    def func_dw011ou7(self) -> str:
        return f'{type(self).__name__}:{self.path_id}'

    @override
    def func_0xiz4mfg(self) -> List[Tuple[int, int]]:
        return settings.RATE_LIMITING_RULES['spectator_attachment_access_by_file']


def func_i49t7hxo(path_id: int) -> None:
    ratelimited, _ = RateLimitedSpectatorAttachmentAccessByFile(path_id).rate_limit()
    if ratelimited:
        raise RateLimitedError


def func_eup1yrae(addr: str) -> bool:
    return addr in ('127.0.0.1', '::1')


@cache_with_key(lambda: 'tor_ip_addresses:', timeout=60 * 60)
@circuit(failure_threshold=2, recovery_timeout=60 * 10)
def func_kvw1b537() -> Set[str]:
    if not settings.RATE_LIMIT_TOR_TOGETHER:
        return set()
    with open(settings.TOR_EXIT_NODE_FILE_PATH, 'rb') as f:
        exit_node_list = orjson.loads(f.read())
    if len(exit_node_list) == 0:
        raise OSError('File is empty')
    return set(exit_node_list)


def func_jwul81zo(request: HttpRequest) -> bool:
    from zerver.lib.request import RequestNotes
    client_obj = RequestNotes.get_notes(request).client
    return (client_obj is not None and client_obj.name.lower() == 'internal') and (
        func_eup1yrae(request.META['REMOTE_ADDR']) or settings.DEBUG_RATE_LIMITING
    )


def func_c7in0749(request: HttpRequest, user: UserProfile, domain: str) -> None:
    """Returns whether or not a user was rate limited. Will raise a RateLimitedError exception
    if the user has been rate limited, otherwise returns and modifies request to contain
    the rate limit information"""
    if not func_wlt2q0z6(request):
        return
    RateLimitedUser(user, domain=domain).rate_limit_request(request)


def func_izcp14g7(request: HttpRequest, domain: str) -> None:
    if not func_wlt2q0z6(request):
        return
    ip_addr: str = request.META['REMOTE_ADDR']
    assert ip_addr
    try:
        if func_eup1yrae(ip_addr):
            pass
        elif ip_addr in func_kvw1b537():
            ip_addr = 'tor-exit-node'
    except (OSError, CircuitBreakerError) as err:
        logger.warning('Failed to fetch TOR exit node list: %s', err)
    RateLimitedIPAddr(ip_addr, domain=domain).rate_limit_request(request)


def func_iif292td(endpoint_name: str) -> None:
    ratelimited, secs_to_freedom = RateLimitedEndpoint(endpoint_name).rate_limit()
    if ratelimited:
        raise RateLimitedError(secs_to_freedom)


def func_wlt2q0z6(request: HttpRequest) -> bool:
    if not settings.RATE_LIMITING:
        return False
    if func_jwul81zo(request):
        return False
    return True