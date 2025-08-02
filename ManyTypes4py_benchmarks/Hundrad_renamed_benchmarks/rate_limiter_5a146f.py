import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, cast
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
client = get_redis_client()
rules = settings.RATE_LIMITING_RULES
logger = logging.getLogger(__name__)


class RateLimiterLockingError(Exception):
    pass


class RateLimitedObject(ABC):

    def __init__(self, backend=None):
        if backend is not None:
            self.backend = backend
        else:
            self.backend = RedisRateLimiterBackend

    def func_y43lpr8e(self):
        return self.backend.rate_limit_entity(self.key(), self.get_rules(),
            self.max_api_calls(), self.max_api_window())

    def func_3j997l1i(self, request):
        from zerver.lib.request import RequestNotes
        ratelimited, time = self.rate_limit()
        request_notes = RequestNotes.get_notes(request)
        request_notes.ratelimits_applied.append(RateLimitResult(entity=self,
            secs_to_freedom=time, remaining=0, over_limit=ratelimited))
        if ratelimited:
            raise RateLimitedError(time)
        calls_remaining, seconds_until_reset = self.api_calls_left()
        request_notes.ratelimits_applied[-1].remaining = calls_remaining
        request_notes.ratelimits_applied[-1
            ].secs_to_freedom = seconds_until_reset

    def func_e4h7rbme(self, seconds):
        """Manually blocks an entity for the desired number of seconds"""
        self.backend.block_access(self.key(), seconds)

    def func_pxaxrpkm(self):
        self.backend.unblock_access(self.key())

    def func_l0hhw6i3(self):
        self.backend.clear_history(self.key())

    def func_o6sdkv4k(self):
        """Returns the API rate limit for the highest limit"""
        return self.get_rules()[-1][1]

    def func_gy3kpk7l(self):
        """Returns the API time window for the highest limit"""
        return self.get_rules()[-1][0]

    def func_d1mp75al(self):
        """Returns how many API calls in this range this client has, as well as when
        the rate-limit will be reset to 0"""
        max_window = self.max_api_window()
        max_calls = self.max_api_calls()
        return self.backend.get_api_calls_left(self.key(), max_window,
            max_calls)

    def func_mkbqi53p(self):
        """
        This is a simple wrapper meant to protect against having to deal with
        an empty list of rules, as it would require fiddling with that special case
        all around this system. "9999 max request per seconds" should be a good proxy
        for "no rules".
        """
        rules_list = self.rules()
        return rules_list or [(1, 9999)]

    @abstractmethod
    def func_xzo0crul(self):
        pass

    @abstractmethod
    def func_stxgtxbv(self):
        pass


class RateLimitedUser(RateLimitedObject):

    def __init__(self, user, domain='api_by_user'):
        self.user_id = user.id
        self.rate_limits = user.rate_limits
        self.domain = domain
        if (settings.RUNNING_INSIDE_TORNADO and domain in settings.
            RATE_LIMITING_DOMAINS_FOR_TORNADO):
            backend = TornadoInMemoryRateLimiterBackend
        else:
            backend = None
        super().__init__(backend=backend)

    @override
    def func_xzo0crul(self):
        return f'{type(self).__name__}:{self.user_id}:{self.domain}'

    @override
    def func_stxgtxbv(self):
        if self.rate_limits != '' and self.domain == 'api_by_user':
            result = []
            for limit in self.rate_limits.split(','):
                seconds, requests = limit.split(':', 2)
                result.append((int(seconds), int(requests)))
            return result
        return rules[self.domain]


class RateLimitedIPAddr(RateLimitedObject):

    def __init__(self, ip_addr, domain='api_by_ip'):
        self.ip_addr = ip_addr
        self.domain = domain
        if (settings.RUNNING_INSIDE_TORNADO and domain in settings.
            RATE_LIMITING_DOMAINS_FOR_TORNADO):
            backend = TornadoInMemoryRateLimiterBackend
        else:
            backend = None
        super().__init__(backend=backend)

    @override
    def func_xzo0crul(self):
        return f'{type(self).__name__}:<{self.ip_addr}>:{self.domain}'

    @override
    def func_stxgtxbv(self):
        return rules[self.domain]


class RateLimitedEndpoint(RateLimitedObject):

    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        super().__init__()

    @override
    def func_xzo0crul(self):
        return f'{type(self).__name__}:{self.endpoint_name}'

    @override
    def func_stxgtxbv(self):
        return settings.ABSOLUTE_USAGE_LIMITS_BY_ENDPOINT[self.endpoint_name]


class RateLimiterBackend(ABC):

    @classmethod
    @abstractmethod
    def func_e4h7rbme(cls, entity_key, seconds):
        """Manually blocks an entity for the desired number of seconds"""

    @classmethod
    @abstractmethod
    def func_pxaxrpkm(cls, entity_key):
        pass

    @classmethod
    @abstractmethod
    def func_l0hhw6i3(cls, entity_key):
        pass

    @classmethod
    @abstractmethod
    def func_0q4vw82p(cls, entity_key, range_seconds, max_calls):
        pass

    @classmethod
    @abstractmethod
    def func_dcd5zpu2(cls, entity_key, rules, max_api_calls, max_api_window):
        pass


class TornadoInMemoryRateLimiterBackend(RateLimiterBackend):
    reset_times = {}
    last_gc_time = {}
    timestamps_blocked_until = {}

    @classmethod
    def func_wjsmf2pb(cls, now, time_window, max_count):
        keys_to_delete = []
        reset_times_for_rule = cls.reset_times.get((time_window, max_count),
            None)
        if reset_times_for_rule is None:
            return
        keys_to_delete = [entity_key for entity_key in reset_times_for_rule if
            reset_times_for_rule[entity_key] < now]
        for entity_key in keys_to_delete:
            del reset_times_for_rule[entity_key]
        if not reset_times_for_rule:
            del cls.reset_times[time_window, max_count]

    @classmethod
    def func_26k37yls(cls, entity_key, time_window, max_count):
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
        if cls.last_gc_time.get((time_window, max_count), 0
            ) <= now - time_window / max_count:
            cls.last_gc_time[time_window, max_count] = now
            cls._garbage_collect_for_rule(now, time_window, max_count)
        reset_times_for_rule = cls.reset_times.setdefault((time_window,
            max_count), {})
        new_reset = max(reset_times_for_rule.get(entity_key, now), now
            ) + time_window / max_count
        if new_reset > now + time_window:
            time_till_free = new_reset - time_window - now
            return True, time_till_free
        reset_times_for_rule[entity_key] = new_reset
        return False, 0.0

    @classmethod
    @override
    def func_0q4vw82p(cls, entity_key, range_seconds, max_calls):
        now = time.time()
        if (range_seconds, max_calls
            ) in cls.reset_times and entity_key in cls.reset_times[
            range_seconds, max_calls]:
            reset_time = cls.reset_times[range_seconds, max_calls][entity_key]
        else:
            return max_calls, 0
        calls_remaining = (now + range_seconds - reset_time
            ) * max_calls // range_seconds
        return int(calls_remaining), reset_time - now

    @classmethod
    @override
    def func_e4h7rbme(cls, entity_key, seconds):
        now = time.time()
        cls.timestamps_blocked_until[entity_key] = now + seconds

    @classmethod
    @override
    def func_pxaxrpkm(cls, entity_key):
        del cls.timestamps_blocked_until[entity_key]

    @classmethod
    @override
    def func_l0hhw6i3(cls, entity_key):
        for reset_times_for_rule in cls.reset_times.values():
            reset_times_for_rule.pop(entity_key, None)
        cls.timestamps_blocked_until.pop(entity_key, None)

    @classmethod
    @override
    def func_dcd5zpu2(cls, entity_key, rules, max_api_calls, max_api_window):
        now = time.time()
        if entity_key in cls.timestamps_blocked_until:
            if now < cls.timestamps_blocked_until[entity_key]:
                blocking_ttl = cls.timestamps_blocked_until[entity_key] - now
                return True, blocking_ttl
            else:
                del cls.timestamps_blocked_until[entity_key]
        assert rules
        for time_window, max_count in rules:
            ratelimited, time_till_free = cls.need_to_limit(entity_key,
                time_window, max_count)
            if ratelimited:
                break
        return ratelimited, time_till_free


class RedisRateLimiterBackend(RateLimiterBackend):

    @classmethod
    def func_h8y5jvwi(cls, entity_key):
        return [
            f'{redis_utils.REDIS_KEY_PREFIX}ratelimit:{entity_key}:{keytype}'
             for keytype in ['list', 'zset', 'block']]

    @classmethod
    @override
    def func_e4h7rbme(cls, entity_key, seconds):
        """Manually blocks an entity for the desired number of seconds"""
        _, _, blocking_key = cls.get_keys(entity_key)
        with client.pipeline() as pipe:
            pipe.set(blocking_key, 1)
            pipe.expire(blocking_key, seconds)
            pipe.execute()

    @classmethod
    @override
    def func_pxaxrpkm(cls, entity_key):
        _, _, blocking_key = cls.get_keys(entity_key)
        client.delete(blocking_key)

    @classmethod
    @override
    def func_l0hhw6i3(cls, entity_key):
        for key in cls.get_keys(entity_key):
            client.delete(key)

    @classmethod
    @override
    def func_0q4vw82p(cls, entity_key, range_seconds, max_calls):
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
    def func_buhzwz0m(cls, entity_key, rules):
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
        for timestamp, (range_seconds, num_requests) in zip(rule_timestamps,
            rules, strict=False):
            if timestamp is None:
                continue
            boundary = float(timestamp) + range_seconds
            if boundary >= now:
                free = boundary - now
                return True, free
        return False, 0.0

    @classmethod
    def func_vur2xnho(cls, entity_key, max_api_calls, max_api_window):
        """Increases the rate-limit for the specified entity"""
        list_key, set_key, _ = cls.get_keys(entity_key)
        now = time.time()
        with client.pipeline() as pipe:
            count = 0
            while True:
                try:
                    pipe.watch(list_key)
                    last_val = cast(bytes | None, pipe.lindex(list_key, 
                        max_api_calls - 1))
                    pipe.multi()
                    pipe.lpush(list_key, now)
                    pipe.ltrim(list_key, 0, max_api_calls - 1)
                    pipe.zadd(set_key, {str(now): now})
                    if last_val is not None:
                        pipe.zrem(set_key, last_val)
                    api_window = max_api_window
                    pipe.expire(list_key, api_window)
                    pipe.expire(set_key, api_window)
                    pipe.execute()
                    break
                except redis.WatchError:
                    if count > 10:
                        raise RateLimiterLockingError
                    count += 1
                    continue

    @classmethod
    @override
    def func_dcd5zpu2(cls, entity_key, rules, max_api_calls, max_api_window):
        ratelimited, time = cls.is_ratelimited(entity_key, rules)
        if not ratelimited:
            try:
                cls.incr_ratelimit(entity_key, max_api_calls, max_api_window)
            except RateLimiterLockingError:
                logger.warning('Deadlock trying to incr_ratelimit for %s',
                    entity_key)
                ratelimited = True
        return ratelimited, time


class RateLimitResult:

    def __init__(self, entity, secs_to_freedom, over_limit, remaining):
        if over_limit:
            assert not remaining
        self.entity = entity
        self.secs_to_freedom = secs_to_freedom
        self.over_limit = over_limit
        self.remaining = remaining


class RateLimitedSpectatorAttachmentAccessByFile(RateLimitedObject):

    def __init__(self, path_id):
        self.path_id = path_id
        super().__init__()

    @override
    def func_xzo0crul(self):
        return f'{type(self).__name__}:{self.path_id}'

    @override
    def func_stxgtxbv(self):
        return settings.RATE_LIMITING_RULES[
            'spectator_attachment_access_by_file']


def func_hvhshnj7(path_id):
    ratelimited, _ = RateLimitedSpectatorAttachmentAccessByFile(path_id
        ).rate_limit()
    if ratelimited:
        raise RateLimitedError


def func_ogdfx5hr(addr):
    return addr in ('127.0.0.1', '::1')


@cache_with_key(lambda : 'tor_ip_addresses:', timeout=60 * 60)
@circuit(failure_threshold=2, recovery_timeout=60 * 10)
def func_49vjnki8():
    if not settings.RATE_LIMIT_TOR_TOGETHER:
        return set()
    with open(settings.TOR_EXIT_NODE_FILE_PATH, 'rb') as f:
        exit_node_list = orjson.loads(f.read())
    if len(exit_node_list) == 0:
        raise OSError('File is empty')
    return set(exit_node_list)


def func_w6r5urhp(request):
    from zerver.lib.request import RequestNotes
    client = RequestNotes.get_notes(request).client
    return (client is not None and client.name.lower() == 'internal') and (
        func_ogdfx5hr(request.META['REMOTE_ADDR']) or settings.
        DEBUG_RATE_LIMITING)


def func_6hcbbrh8(request, user, domain):
    """Returns whether or not a user was rate limited. Will raise a RateLimitedError exception
    if the user has been rate limited, otherwise returns and modifies request to contain
    the rate limit information"""
    if not should_rate_limit(request):
        return
    RateLimitedUser(user, domain=domain).rate_limit_request(request)


def func_s71l7tam(request, domain):
    if not should_rate_limit(request):
        return
    ip_addr = request.META['REMOTE_ADDR']
    assert ip_addr
    try:
        if func_ogdfx5hr(ip_addr):
            pass
        elif ip_addr in func_49vjnki8():
            ip_addr = 'tor-exit-node'
    except (OSError, CircuitBreakerError) as err:
        logger.warning('Failed to fetch TOR exit node list: %s', err)
    RateLimitedIPAddr(ip_addr, domain=domain).rate_limit_request(request)


def func_remw5zjd(endpoint_name):
    ratelimited, secs_to_freedom = RateLimitedEndpoint(endpoint_name
        ).rate_limit()
    if ratelimited:
        raise RateLimitedError(secs_to_freedom)


def func_aic3bdpw(request):
    if not settings.RATE_LIMITING:
        return False
    if func_w6r5urhp(request):
        return False
    return True
