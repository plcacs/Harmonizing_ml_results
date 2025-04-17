from __future__ import absolute_import
import time
import logging
from collections import namedtuple
from itertools import takewhile
import email
import re
from typing import Optional, Iterable, Set, Tuple, Type, Any, Union, ClassVar

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
# Data structure for representing the metadata of requests that result in a retry.
RequestHistory = namedtuple(
    "RequestHistory", ["method", "url", "error", "status", "redirect_location"]
)


class Retry(object):
    DEFAULT_METHOD_WHITELIST: ClassVar[Set[str]] = frozenset(
        ["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    RETRY_AFTER_STATUS_CODES: ClassVar[Set[int]] = frozenset([413, 429, 503])
    # : Maximum backoff time.
    BACKOFF_MAX: ClassVar[int] = 120

    def __init__(
        self,
        total: Union[int, bool] = 10,
        connect: Optional[int] = None,
        read: Optional[int] = None,
        redirect: Optional[Union[int, bool]] = None,
        status: Optional[int] = None,
        method_whitelist: Optional[Iterable[str]] = DEFAULT_METHOD_WHITELIST,
        status_forcelist: Optional[Iterable[int]] = None,
        backoff_factor: float = 0,
        raise_on_redirect: bool = True,
        raise_on_status: bool = True,
        history: Optional[Tuple[RequestHistory, ...]] = None,
        respect_retry_after_header: bool = True,
    ) -> None:
        self.total = total
        self.connect = connect
        self.read = read
        self.status = status
        if redirect is False or total is False:
            redirect = 0
            raise_on_redirect = False
        self.redirect = redirect
        self.status_forcelist: Set[int] = set(status_forcelist) if status_forcelist else set()
        self.method_whitelist: Optional[Iterable[str]] = method_whitelist
        self.backoff_factor: float = backoff_factor
        self.raise_on_redirect: bool = raise_on_redirect
        self.raise_on_status: bool = raise_on_status
        self.history: Tuple[RequestHistory, ...] = history or tuple()
        self.respect_retry_after_header: bool = respect_retry_after_header

    def new(self, **kw: Any) -> 'Retry':
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
        )
        params.update(kw)
        return type(self)(**params)

    @classmethod
    def from_int(
        cls, retries: Optional[Union[int, 'Retry']], redirect: bool = True, default: Optional['Retry'] = None
    ) -> 'Retry':
        """ Backwards-compatibility for the old retries format."""
        if retries is None:
            retries = default if default is not None else cls.DEFAULT
        if isinstance(retries, Retry):
            return retries

        redirect_value = bool(redirect) and None
        new_retries = cls(retries, redirect=redirect_value)
        log.debug("Converted retries value: %r -> %r", retries, new_retries)
        return new_retries

    def get_backoff_time(self) -> float:
        """ Formula for computing the current backoff

        :rtype: float
        """
        # We want to consider only the last consecutive errors sequence (Ignore redirects).
        consecutive_errors_len = len(
            list(
                takewhile(lambda x: x.redirect_location is None, reversed(self.history))
            )
        )
        if consecutive_errors_len <= 1:
            return 0.0

        backoff_value = self.backoff_factor * (2 ** (consecutive_errors_len - 1))
        return min(self.BACKOFF_MAX, backoff_value)

    def parse_retry_after(self, retry_after: str) -> float:
        # Whitespace: https://tools.ietf.org/html/rfc7230#section-3.2.4
        if re.match(r"^\s*[0-9]+\s*$", retry_after):
            seconds = int(retry_after)
        else:
            retry_date_tuple = email.utils.parsedate(retry_after)
            if retry_date_tuple is None:
                raise InvalidHeader(f"Invalid Retry-After header: {retry_after}")

            retry_date = time.mktime(retry_date_tuple)
            seconds = retry_date - time.time()
        if seconds < 0:
            seconds = 0.0
        return float(seconds)

    def get_retry_after(self, response: Any) -> Optional[float]:
        """ Get the value of Retry-After in seconds. """
        retry_after = response.getheader("Retry-After")
        if retry_after is None:
            return None

        return self.parse_retry_after(retry_after)

    def sleep_for_retry(self, response: Optional[Any] = None) -> bool:
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

    def sleep(self, response: Optional[Any] = None) -> None:
        """ Sleep between retry attempts.

        This method will respect a server's ``Retry-After`` response header
        and sleep the duration of the time requested. If that is not present, it
        will use an exponential backoff. By default, the backoff factor is 0 and
        this method will return immediately.
        """
        if response:
            slept = self.sleep_for_retry(response)
            if slept:
                return

        self._sleep_backoff()

    def _is_connection_error(self, err: Exception) -> bool:
        """ Errors when we're fairly sure that the server did not receive the
        request, so it should be safe to retry.
        """
        return isinstance(err, ConnectTimeoutError)

    def _is_read_error(self, err: Exception) -> bool:
        """ Errors that occur after the request has been started, so we should
        assume that the server began processing it.
        """
        return isinstance(err, (ReadTimeoutError, ProtocolError))

    def _is_method_retryable(self, method: Optional[str]) -> bool:
        """ Checks if a given HTTP method should be retried upon, depending if
        it is included on the method whitelist.
        """
        if self.method_whitelist and method is not None and method.upper() not in self.method_whitelist:
            return False

        return True

    def is_retry(
        self, method: Optional[str], status_code: Optional[int], has_retry_after: bool = False
    ) -> bool:
        """ Is this method/status code retryable? (Based on whitelists and control
        variables such as the number of total retries to allow, whether to
        respect the Retry-After header, whether this header is present, and
        whether the returned status code is on the list of status codes to
        be retried upon on the presence of the aforementioned header)
        """
        if not self._is_method_retryable(method):
            return False

        if status_code in self.status_forcelist:
            return True

        return (
            self.total
            and self.respect_retry_after_header
            and has_retry_after
            and (status_code in self.RETRY_AFTER_STATUS_CODES)
        )

    def is_exhausted(self) -> bool:
        """ Are we out of retries? """
        retry_counts = [c for c in (self.total, self.connect, self.read, self.redirect, self.status) if c is not None]
        if not retry_counts:
            return False

        return min(retry_counts) < 0

    def increment(
        self,
        method: Optional[str] = None,
        url: Optional[str] = None,
        response: Optional[Any] = None,
        error: Optional[Exception] = None,
        _pool: Any = None,
        _stacktrace: Any = None,
    ) -> 'Retry':
        """ Return a new Retry object with incremented retry counters.

        :param response: A response object, or None, if the server did not
            return a response.
        :type response: :class:`~urllib3.response.HTTPResponse`
        :param Exception error: An error encountered during the request, or
            None if the response was received successfully.

        :return: A new ``Retry`` object.
        """
        if self.total is False and error:
            # Disabled, indicate to re-raise the error.
            six.reraise(type(error), error, _stacktrace)

        total = self.total
        if isinstance(total, int):
            total -= 1
        connect = self.connect
        read = self.read
        redirect = self.redirect
        status_count = self.status
        cause = "unknown"
        status = None
        redirect_location = None
        if error and self._is_connection_error(error):
            # Connect retry?
            if connect is False:
                six.reraise(type(error), error, _stacktrace)

            elif isinstance(connect, int):
                connect -= 1
        elif error and self._is_read_error(error):
            # Read retry?
            if read is False or not self._is_method_retryable(method):
                six.reraise(type(error), error, _stacktrace)

            elif isinstance(read, int):
                read -= 1
        elif response and response.get_redirect_location():
            # Redirect retry?
            if isinstance(redirect, int):
                redirect -= 1
            cause = "too many redirects"
            redirect_location = response.get_redirect_location()
            status = response.status
        else:
            # Incrementing because of a server error like a 500 in
            # status_forcelist and a the given method is in the whitelist
            cause = ResponseError.GENERIC_ERROR
            if response and hasattr(response, 'status'):
                if isinstance(status_count, int):
                    status_count -= 1
                cause = ResponseError.SPECIFIC_ERROR.format(status_code=response.status)
                status = response.status
        history = self.history + (
            RequestHistory(method, url, error, status, redirect_location),
        )
        new_retry = self.new(
            total=total,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status_count,
            history=history,
        )
        if new_retry.is_exhausted():
            raise MaxRetryError(_pool, url, error or ResponseError(cause))

        log.debug("Incremented Retry for (url='%s'): %r", url, new_retry)
        return new_retry

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(total={self.total}, connect={self.connect}, "
            f"read={self.read}, redirect={self.redirect}, status={self.status})"
        )


# For backwards compatibility (equivalent to pre-v1.9):
Retry.DEFAULT: Retry = Retry(3)
