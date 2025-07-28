from __future__ import absolute_import
import time
import logging
from collections import namedtuple
from itertools import takewhile
import email
import re
from typing import Any, Optional, Union, Collection, Tuple, ClassVar
from ..exceptions import (
    ConnectTimeoutError,
    MaxRetryError,
    ProtocolError,
    ReadTimeoutError,
    ResponseError,
    InvalidHeader,
)
from ..packages import six

log = logging.getLogger(__name__)
RequestHistory = namedtuple('RequestHistory', ['method', 'url', 'error', 'status', 'redirect_location'])


class Retry(object):
    DEFAULT_METHOD_WHITELIST: ClassVar[frozenset] = frozenset(['HEAD', 'GET', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])
    RETRY_AFTER_STATUS_CODES: ClassVar[frozenset] = frozenset([413, 429, 503])
    BACKOFF_MAX: ClassVar[int] = 120

    def __init__(
        self,
        total: Union[int, bool] = 10,
        connect: Optional[int] = None,
        read: Optional[int] = None,
        redirect: Optional[Union[int, bool]] = None,
        status: Optional[int] = None,
        method_whitelist: Union[Collection[str], bool] = DEFAULT_METHOD_WHITELIST,
        status_forcelist: Optional[Collection[int]] = None,
        backoff_factor: float = 0,
        raise_on_redirect: bool = True,
        raise_on_status: bool = True,
        history: Optional[Tuple[RequestHistory, ...]] = None,
        respect_retry_after_header: bool = True,
    ) -> None:
        self.total: Union[int, bool] = total
        self.connect: Optional[int] = connect
        self.read: Optional[int] = read
        self.status: Optional[int] = status
        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False
        self.redirect: Optional[Union[int, bool]] = redirect
        self.status_forcelist: Collection[int] = set(status_forcelist) if status_forcelist is not None else set()
        self.method_whitelist: Union[Collection[str], bool] = method_whitelist
        self.backoff_factor: float = backoff_factor
        self.raise_on_redirect: bool = raise_on_redirect
        self.raise_on_status: bool = raise_on_status
        self.history: Tuple[RequestHistory, ...] = history or tuple()
        self.respect_retry_after_header: bool = respect_retry_after_header

    def new(self, **kw: Any) -> "Retry":
        params = dict(
            total=self.total,
            connect=self.connect,
            read=self.read,
            redirect=self.redirect,
            status=self.status,
            method_whitelist=self.method_whitelist,
            status_forcelist=self.status_forcelist,
            backoff_factor=self.backoff_factor,
            raise_on_redirect=self.raise_on_redirect,
            raise_on_status=self.raise_on_status,
            history=self.history,
            respect_retry_after_header=self.respect_retry_after_header,
        )
        params.update(kw)
        return type(self)(**params)

    @classmethod
    def from_int(cls, retries: Union[int, "Retry", None], redirect: bool = True, default: Optional[int] = None) -> "Retry":
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
            seconds = int(retry_date - time.time())
        if seconds < 0:
            seconds = 0
        return seconds

    def get_retry_after(self, response: Any) -> Optional[int]:
        retry_after = response.getheader('Retry-After')  # type: Optional[str]
        if retry_after is None:
            return None
        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response: Optional[Any] = None) -> bool:
        retry_after = self.get_retry_after(response) if response is not None else None
        if retry_after:
            time.sleep(retry_after)
            return True
        return False

    def _sleep_backoff(self) -> None:
        backoff = self.get_backoff_time()
        if backoff <= 0:
            return
        time.sleep(backoff)

    def sleep(self, response: Optional[Any] = None) -> None:
        if response:
            slept = self.sleep_for_retry(response)
            if slept:
                return
        self._sleep_backoff()

    def _is_connection_error(self, err: Exception) -> bool:
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err: Exception) -> bool:
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
        return bool(self.total) and self.respect_retry_after_header and has_retry_after and (status_code in self.RETRY_AFTER_STATUS_CODES)

    def is_exhausted(self) -> bool:
        retry_counts = (self.total, self.connect, self.read, self.redirect, self.status)
        retry_counts = list(filter(None, retry_counts))
        if not retry_counts:
            return False
        return min(retry_counts) < 0

    def increment(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        response: Optional[Any] = None,
        error: Optional[Exception] = None,
        _pool: Optional[Any] = None,
        _stacktrace: Optional[Any] = None,
    ) -> "Retry":
        if self.total is False and error:
            raise six.reraise(type(error), error, _stacktrace)
        total: Union[int, bool] = self.total  # type: ignore
        if total is not None and total is not False:
            total -= 1
        connect: Optional[int] = self.connect
        read: Optional[int] = self.read
        redirect: Optional[Union[int, bool]] = self.redirect
        status_count: Optional[int] = self.status
        cause = 'unknown'
        status: Optional[int] = None
        redirect_location: Optional[str] = None
        if error and self._is_connection_error(error):
            if connect is False:
                raise six.reraise(type(error), error, _stacktrace)
            elif connect is not None:
                connect -= 1
        elif error and self._is_read_error(error):
            if read is False or not self._is_method_retryable(method or ""):
                raise six.reraise(type(error), error, _stacktrace)
            elif read is not None:
                read -= 1
        elif response and response.get_redirect_location():
            if redirect is not None and redirect is not False:
                redirect -= 1  # type: ignore
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
        return '{cls.__name__}(total={self.total}, connect={self.connect}, read={self.read}, redirect={self.redirect}, status={self.status})'.format(
            cls=type(self), self=self
        )


Retry.DEFAULT = Retry(3)