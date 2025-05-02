import calendar
import contextlib
import datetime
import heapq
import itertools
import os
import pathlib
import pickle
import re
import time
import warnings
from collections import defaultdict
from http.cookies import BaseCookie, Morsel, SimpleCookie
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from yarl import URL
from .abc import AbstractCookieJar, ClearCookiePredicate
from .helpers import is_ip_address
from .typedefs import LooseCookies, PathLike, StrOrURL

__all__ = ('CookieJar', 'DummyCookieJar')

CookieItem = Union[str, 'Morsel[str]']
_FORMAT_PATH = '{}/{}'.format
_FORMAT_DOMAIN_REVERSED = '{1}.{0}'.format
_MIN_SCHEDULED_COOKIE_EXPIRATION = 100

class CookieJar(AbstractCookieJar):
    """Implements cookie storage adhering to RFC 6265."""
    DATE_TOKENS_RE: re.Pattern = re.compile('[\\x09\\x20-\\x2F\\x3B-\\x40\\x5B-\\x60\\x7B-\\x7E]*(?P<token>[\\x00-\\x08\\x0A-\\x1F\\d:a-zA-Z\\x7F-\\xFF]+)')
    DATE_HMS_TIME_RE: re.Pattern = re.compile('(\\d{1,2}):(\\d{1,2}):(\\d{1,2})')
    DATE_DAY_OF_MONTH_RE: re.Pattern = re.compile('(\\d{1,2})')
    DATE_MONTH_RE: re.Pattern = re.compile('(jan)|(feb)|(mar)|(apr)|(may)|(jun)|(jul)|(aug)|(sep)|(oct)|(nov)|(dec)', re.I)
    DATE_YEAR_RE: re.Pattern = re.compile('(\\d{2,4})')
    MAX_TIME: int = int(datetime.datetime.max.replace(tzinfo=datetime.timezone.utc).timestamp()) - 1
    try:
        calendar.timegm(time.gmtime(MAX_TIME))
    except (OSError, ValueError):
        MAX_TIME = calendar.timegm((3000, 12, 31, 23, 59, 59, -1, -1, -1))
    except OverflowError:
        MAX_TIME = 2 ** 31 - 1
    SUB_MAX_TIME: int = MAX_TIME - 1

    def __init__(
        self,
        *,
        unsafe: bool = False,
        quote_cookie: bool = True,
        treat_as_secure_origin: Optional[Union[URL, str, Iterable[Union[URL, str]]] = None
    ) -> None:
        self._cookies: DefaultDict[Tuple[str, str], SimpleCookie] = defaultdict(SimpleCookie)
        self._morsel_cache: DefaultDict[Tuple[str, str], Dict[str, Morsel[str]]] = defaultdict(dict)
        self._host_only_cookies: Set[Tuple[str, str]] = set()
        self._unsafe: bool = unsafe
        self._quote_cookie: bool = quote_cookie
        if treat_as_secure_origin is None:
            self._treat_as_secure_origin: FrozenSet[URL] = frozenset()
        elif isinstance(treat_as_secure_origin, URL):
            self._treat_as_secure_origin = frozenset({treat_as_secure_origin.origin()})
        elif isinstance(treat_as_secure_origin, str):
            self._treat_as_secure_origin = frozenset({URL(treat_as_secure_origin).origin()})
        else:
            self._treat_as_secure_origin = frozenset({URL(url).origin() if isinstance(url, str) else url.origin() for url in treat_as_secure_origin})
        self._expire_heap: List[Tuple[float, Tuple[str, str, str]]] = []
        self._expirations: Dict[Tuple[str, str, str], float] = {}

    @property
    def quote_cookie(self) -> bool:
        return self._quote_cookie

    def save(self, file_path: PathLike) -> None:
        file_path = pathlib.Path(file_path)
        with file_path.open(mode='wb') as f:
            pickle.dump(self._cookies, f, pickle.HIGHEST_PROTOCOL)

    def load(self, file_path: PathLike) -> None:
        file_path = pathlib.Path(file_path)
        with file_path.open(mode='rb') as f:
            self._cookies = pickle.load(f)

    def clear(self, predicate: Optional[ClearCookiePredicate] = None) -> None:
        if predicate is None:
            self._expire_heap.clear()
            self._cookies.clear()
            self._morsel_cache.clear()
            self._host_only_cookies.clear()
            self._expirations.clear()
            return
        now = time.time()
        to_del = [key for (domain, path), cookie in self._cookies.items() for name, morsel in cookie.items() if (key := (domain, path, name)) in self._expirations and self._expirations[key] <= now or predicate(morsel)]
        if to_del:
            self._delete_cookies(to_del)

    def clear_domain(self, domain: str) -> None:
        self.clear(lambda x: self._is_domain_match(domain, x['domain']))

    def __iter__(self) -> Iterator[Morsel[str]]:
        self._do_expiration()
        for val in self._cookies.values():
            yield from val.values()

    def __len__(self) -> int:
        """Return number of cookies.

        This function does not iterate self to avoid unnecessary expiration
        checks.
        """
        return sum((len(cookie.values()) for cookie in self._cookies.values()))

    def _do_expiration(self) -> None:
        """Remove expired cookies."""
        if not (expire_heap_len := len(self._expire_heap)):
            return
        if expire_heap_len > _MIN_SCHEDULED_COOKIE_EXPIRATION and expire_heap_len > len(self._expirations) * 2:
            self._expire_heap = [entry for entry in self._expire_heap if self._expirations.get(entry[1]) == entry[0]]
            heapq.heapify(self._expire_heap)
        now = time.time()
        to_del = []
        while self._expire_heap:
            when, cookie_key = self._expire_heap[0]
            if when > now:
                break
            heapq.heappop(self._expire_heap)
            if self._expirations.get(cookie_key) == when:
                to_del.append(cookie_key)
        if to_del:
            self._delete_cookies(to_del)

    def _delete_cookies(self, to_del: List[Tuple[str, str, str]]) -> None:
        for domain, path, name in to_del:
            self._host_only_cookies.discard((domain, name))
            self._cookies[domain, path].pop(name, None)
            self._morsel_cache[domain, path].pop(name, None)
            self._expirations.pop((domain, path, name), None)

    def _expire_cookie(self, when: float, domain: str, path: str, name: str) -> None:
        cookie_key = (domain, path, name)
        if self._expirations.get(cookie_key) == when:
            return
        heapq.heappush(self._expire_heap, (when, cookie_key))
        self._expirations[cookie_key] = when

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        """Update cookies."""
        hostname = response_url.raw_host
        if not self._unsafe and is_ip_address(hostname):
            return
        if isinstance(cookies, Mapping):
            cookies = cookies.items()
        for name, cookie in cookies:
            if not isinstance(cookie, Morsel):
                tmp = SimpleCookie()
                tmp[name] = cookie
                cookie = tmp[name]
            domain = cookie['domain']
            if domain and domain[-1] == '.':
                domain = ''
                del cookie['domain']
            if not domain and hostname is not None:
                self._host_only_cookies.add((hostname, name))
                domain = cookie['domain'] = hostname
            if domain and domain[0] == '.':
                domain = domain[1:]
                cookie['domain'] = domain
            if hostname and (not self._is_domain_match(domain, hostname)):
                continue
            path = cookie['path']
            if not path or path[0] != '/':
                path = response_url.path
                if not path.startswith('/'):
                    path = '/'
                else:
                    path = '/' + path[1:path.rfind('/')]
                cookie['path'] = path
            path = path.rstrip('/')
            if (max_age := cookie['max-age']):
                try:
                    delta_seconds = int(max_age)
                    max_age_expiration = min(time.time() + delta_seconds, self.MAX_TIME)
                    self._expire_cookie(max_age_expiration, domain, path, name)
                except ValueError:
                    cookie['max-age'] = ''
            elif (expires := cookie['expires']):
                if (expire_time := self._parse_date(expires)):
                    self._expire_cookie(expire_time, domain, path, name)
                else:
                    cookie['expires'] = ''
            key = (domain, path)
            if self._cookies[key].get(name) != cookie:
                self._cookies[key][name] = cookie
                self._morsel_cache[key].pop(name, None)
        self._do_expiration()

    def filter_cookies(self, request_url: StrOrURL) -> Union[SimpleCookie, BaseCookie]:
        """Returns this jar's cookies filtered by their attributes."""
        if not isinstance(request_url, URL):
            warnings.warn('The method accepts yarl.URL instances only, got {}'.format(type(request_url)), DeprecationWarning)
            request_url = URL(request_url)
        filtered = SimpleCookie() if self._quote_cookie else BaseCookie()
        if not self._cookies:
            return filtered
        self._do_expiration()
        if not self._cookies:
            return filtered
        hostname = request_url.raw_host or ''
        is_not_secure = request_url.scheme not in ('https', 'wss')
        if is_not_secure and self._treat_as_secure_origin:
            request_origin = URL()
            with contextlib.suppress(ValueError):
                request_origin = request_url.origin()
            is_not_secure = request_origin not in self._treat_as_secure_origin
        for c in self._cookies['', ''].values():
            filtered[c.key] = c.value
        if is_ip_address(hostname):
            if not self._unsafe:
                return filtered
            domains = (hostname,)
        else:
            domains = itertools.accumulate(reversed(hostname.split('.')), _FORMAT_DOMAIN_REVERSED)
        paths = itertools.accumulate(request_url.path.split('/'), _FORMAT_PATH)
        pairs = itertools.product(domains, paths)
        path_len = len(request_url.path)
        for p in pairs:
            for name, cookie in self._cookies[p].items():
                domain = cookie['domain']
                if (domain, name) in self._host_only_cookies and domain != hostname:
                    continue
                if len(cookie['path']) > path_len:
                    continue
                if is_not_secure and cookie['secure']:
                    continue
                if name in self._morsel_cache[p]:
                    filtered[name] = self._morsel_cache[p][name]
                    continue
                mrsl_val = cast('Morsel[str]', cookie.get(cookie.key, Morsel()))
                mrsl_val.set(cookie.key, cookie.value, cookie.coded_value)
                self._morsel_cache[p][name] = mrsl_val
                filtered[name] = mrsl_val
        return filtered

    @staticmethod
    def _is_domain_match(domain: str, hostname: str) -> bool:
        """Implements domain matching adhering to RFC 6265."""
        if hostname == domain:
            return True
        if not hostname.endswith(domain):
            return False
        non_matching = hostname[:-len(domain)]
        if not non_matching.endswith('.'):
            return False
        return not is_ip_address(hostname)

    @classmethod
    def _parse_date(cls, date_str: str) -> Optional[float]:
        """Implements date string parsing adhering to RFC 6265."""
        if not date_str:
            return None
        found_time = False
        found_day = False
        found_month = False
        found_year = False
        hour = minute = second = 0
        day = 0
        month = 0
        year = 0
        for token_match in cls.DATE_TOKENS_RE.finditer(date_str):
            token = token_match.group('token')
            if not found_time:
                time_match = cls.DATE_HMS_TIME_RE.match(token)
                if time_match:
                    found_time = True
                    hour, minute, second = (int(s) for s in time_match.groups())
                    continue
            if not found_day:
                day_match = cls.DATE_DAY_OF_MONTH_RE.match(token)
                if day_match:
                    found_day = True
                    day = int(day_match.group())
                    continue
            if not found_month:
                month_match = cls.DATE_MONTH_RE.match(token)
                if month_match:
                    found_month = True
                    assert month_match.lastindex is not None
                    month = month_match.lastindex
                    continue
            if not found_year:
                year_match = cls.DATE_YEAR_RE.match(token)
                if year_match:
                    found_year = True
                    year = int(year_match.group())
        if 70 <= year <= 99:
            year += 1900
        elif 0 <= year <= 69:
            year += 2000
        if False in (found_day, found_month, found_year, found_time):
            return None
        if not 1 <= day <= 31:
            return None
        if year < 1601 or hour > 23 or minute > 59 or (second > 59):
            return None
        return calendar.timegm((year, month, day, hour, minute, second, -1, -1, -1))

class DummyCookieJar(AbstractCookieJar):
    """Implements a dummy cookie storage.

    It can be used with the ClientSession when no cookie processing is needed.

    """

    def __iter__(self) -> Iterator[None]:
        while False:
            yield None

    def __len__(self) -> int:
        return 0

    @property
    def quote_cookie(self) -> bool:
        return True

    def clear(self, predicate: Optional[ClearCookiePredicate] = None) -> None:
        pass

    def clear_domain(self, domain: str) -> None:
        pass

    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        pass

    def filter_cookies(self, request_url: StrOrURL) -> SimpleCookie:
        return SimpleCookie()
