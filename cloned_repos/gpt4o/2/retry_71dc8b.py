from __future__ import absolute_import
import time
import logging
from collections import namedtuple
from itertools import takewhile
import email
import re
from ..exceptions import ConnectTimeoutError, MaxRetryError, ProtocolError, ReadTimeoutError, ResponseError, InvalidHeader
from ..packages import six
from typing import Optional, Union, Iterable, Tuple

log = logging.getLogger(__name__)
RequestHistory = namedtuple('RequestHistory', ['method', 'url', 'error', 'status', 'redirect_location'])

class Retry(object):
    DEFAULT_METHOD_WHITELIST = frozenset(['HEAD', 'GET', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])
    RETRY_AFTER_STATUS_CODES = frozenset([413, 429, 503])
    BACKOFF_MAX = 120

    def __init__(self, total: Optional[Union[int, bool]] = 10, connect: Optional[int] = None, read: Optional[int] = None, redirect: Optional[Union[int, bool]] = None, status: Optional[int] = None, method_whitelist: Optional[Iterable[str]] = DEFAULT_METHOD_WHITELIST, status_forcelist: Optional[Iterable[int]] = None, backoff_factor: float = 0, raise_on_redirect: bool = True, raise_on_status: bool = True, history: Optional[Tuple[RequestHistory, ...]] = None, respect_retry_after_header: bool = True):
        self.total = total
        self.connect = connect
        self.read = read
        self.status = status
        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False
        self.redirect = redirect
        self.status_forcelist = status_forcelist or set()
        self.method_whitelist = method_whitelist
        self.backoff_factor = backoff_factor
        self.raise_on_redirect = raise_on_redirect
        self.raise_on_status = raise_on_status
        self.history = history or tuple()
        self.respect_retry_after_header = respect_retry_after_header

    def new(self, **kw) -> 'Retry':
        params = dict(total=self.total, connect=self.connect, read=self.read, redirect=self.redirect, status=self.status, method_whitelist=self.method_whitelist, status_forcelist=self.status_forcelist, backoff_factor=self.backoff_factor, raise_on_redirect=self.raise_on_redirect, raise_on_status=self.raise_on_status, history=self.history)
        params.update(kw)
        return type(self)(**params)

    @classmethod
    def from_int(cls, retries: Optional[Union[int, 'Retry']], redirect: bool = True, default: Optional['Retry'] = None) -> 'Retry':
        if retries is None:
            retries = default if default is not None else cls.DEFAULT
        if isinstance(retries, Retry):
            return retries
        redirect = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect)
        log.debug('Converted retries value: %r -> %r', retries, new_retries)
        return new_retries

    def get_backoff_time(self) -> float:
        consecutive_errors_len = len(list(takewhile(lambda x: x.redirect_location is None, reversed(self.history))))
        if consecutive_errors_len <= 1:
            return 0
        backoff_value = self.backoff_factor * 2 ** (consecutive_errors_len - 1)
        return min(self.BACKOFF_MAX, backoff_value)

    def parse_retry_after(self, retry_after: str) -> int:
        if re.match('^\\s*[0-9]+\\s*$', retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader('Invalid Retry-After header: %s' % retry_after)
            retry_date = time.mktime(retry_date_tuple)
            seconds = retry_date - time.time()
        if seconds < 0:
            seconds = 0
        return seconds

    def get_retry_after(self, response) -> Optional[int]:
        retry_after = response.getheader('Retry-After')
        if retry_after is None:
            return None
        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response=None) -> bool:
        retry_after = self.get_retry_after(response)
        if retry_after:
            time.sleep(retry_after)
            return True
        return False

    def _sleep_backoff(self) -> None:
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def sleep(self, response=None) -> None:
        if response:
            slept = self.sleep_for_retry(response)
            if slept:
                return
        self._sleep_backoff()

    def _is_connection_error(self, err) -> bool:
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err) -> bool:
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def _is_method_retryable(self, method: str) -> bool:
        if self.method_whitelist and method.upper() not in self.method_whitelist:
            return False
        return True

    def is_retry(self, method: str, status_code: int, has_retry_after: bool = False) -> bool:
        if not self._is_method_retryable(method):
            return False
        if self.status_forcelist and status_code in self.status_forcelist:
            return True
        return self.total and self.respect_retry_after_header and has_retry_after and (status_code in self.RETRY_AFTER_STATUS_CODES)

    def is_exhausted(self) -> bool:
        retry_counts = (self.total, self.connect, self.read, self.redirect, self.status)
        retry_counts = list(filter(None, retry_counts))
        if not retry_counts:
            return False
        return min(retry_counts) < 0

    def increment(self, method: Optional[str] = None, url: Optional[str] = None, response=None, error: Optional[Exception] = None, _pool=None, _stacktrace=None) -> 'Retry':
        if self.total is False and error:
            raise six.reraise(type(error), error, _stacktrace)
        total = self.total
        if total is not None:
            total -= 1
        connect = self.connect
        read = self.read
        redirect = self.redirect
        status_count = self.status
        cause = 'unknown'
        status = None
        redirect_location = None
        if error and self._is_connection_error(error):
            if connect is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1
        elif error and self._is_read_error(error):
            if read is False or not self._is_method_retryable(method):
                raise six.reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1
        elif response and response.get_redirect_location():
            if redirect is not None:
                redirect -= 1
            cause = 'too many redirects'
            redirect_location = response.get_redirect_location()
            status = response.status
        else:
            cause = ResponseError.GENERIC_ERROR
            if response and response.status:
                if status_count is not None:
                    status_count -= 1
                cause = ResponseError.SPECIFIC_ERROR.format(status_code=response.status)
                status = response.status
        history = self.history + (RequestHistory(method, url, error, status, redirect_location),)
        new_retry = self.new(total=total, connect=connect, read=read, redirect=redirect, status=status_count, history=history)
        if new_retry.is_exhausted():
            raise MaxRetryError(_pool, url, error or ResponseError(cause))
        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)
        return new_retry

    def __repr__(self) -> str:
        return '{cls.__name__}(total={self.total}, connect={self.connect}, read={self.read}, redirect={self.redirect}, status={self.status})'.format(cls=type(self), self=self)

Retry.DEFAULT = Retry(3)
