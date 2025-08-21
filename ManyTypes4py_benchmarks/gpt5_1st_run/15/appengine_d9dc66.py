from __future__ import absolute_import
import logging
import os
import warnings
from typing import Any, Mapping, Optional, Union

from ..packages.six.moves.urllib.parse import urljoin
from ..exceptions import HTTPError, HTTPWarning, MaxRetryError, ProtocolError, TimeoutError, SSLError
from ..packages.six import BytesIO
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.timeout import Timeout
from ..util.retry import Retry

try:
    from google.appengine.api import urlfetch as _urlfetch  # type: ignore
    urlfetch: Any = _urlfetch
except ImportError:
    urlfetch: Any = None

log: logging.Logger = logging.getLogger(__name__)


class AppEnginePlatformWarning(HTTPWarning):
    pass


class AppEnginePlatformError(HTTPError):
    pass


TimeoutType = Union[Timeout, float, int, object]


class AppEngineManager(RequestMethods):
    """
    Connection manager for Google App Engine sandbox applications.

    This manager uses the URLFetch service directly instead of using the
    emulated httplib, and is subject to URLFetch limitations as described in
    the App Engine documentation `here
    <https://cloud.google.com/appengine/docs/python/urlfetch>`_.

    Notably it will raise an :class:`AppEnginePlatformError` if:
        * URLFetch is not available.
        * If you attempt to use this on App Engine Flexible, as full socket
          support is available.
        * If a request size is more than 10 megabytes.
        * If a response size is more than 32 megabtyes.
        * If you use an unsupported request method such as OPTIONS.

    Beyond those cases, it will raise normal urllib3 errors.
    """

    def __init__(
        self,
        headers: Optional[Mapping[str, str]] = None,
        retries: Optional[Retry] = None,
        validate_certificate: bool = True,
        urlfetch_retries: bool = True,
    ) -> None:
        if not urlfetch:
            raise AppEnginePlatformError('URLFetch is not available in this environment.')
        if is_prod_appengine_mvms():
            raise AppEnginePlatformError('Use normal urllib3.PoolManager instead of AppEngineManageron Managed VMs, as using URLFetch is not necessary in this environment.')
        warnings.warn('urllib3 is using URLFetch on Google App Engine sandbox instead of sockets. To use sockets directly instead of URLFetch see https://urllib3.readthedocs.io/en/latest/reference/urllib3.contrib.html.', AppEnginePlatformWarning)
        RequestMethods.__init__(self, headers)
        self.validate_certificate: bool = validate_certificate
        self.urlfetch_retries: bool = urlfetch_retries
        self.retries: Retry = retries or Retry.DEFAULT

    def __enter__(self) -> "AppEngineManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def urlopen(
        self,
        method: str,
        url: str,
        body: Optional[Union[bytes, str]] = None,
        headers: Optional[Mapping[str, str]] = None,
        retries: Optional[Union[Retry, int, bool]] = None,
        redirect: bool = True,
        timeout: TimeoutType = Timeout.DEFAULT_TIMEOUT,
        **response_kw: Any
    ) -> HTTPResponse:
        retries = self._get_retries(retries, redirect)
        try:
            follow_redirects: bool = bool(redirect and retries.redirect != 0 and retries.total)
            response = urlfetch.fetch(
                url,
                payload=body,
                method=method,
                headers=headers or {},
                allow_truncated=False,
                follow_redirects=self.urlfetch_retries and follow_redirects,
                deadline=self._get_absolute_timeout(timeout),
                validate_certificate=self.validate_certificate,
            )
        except urlfetch.DeadlineExceededError as e:  # type: ignore[attr-defined]
            raise TimeoutError(self, e)
        except urlfetch.InvalidURLError as e:  # type: ignore[attr-defined]
            if 'too large' in str(e):
                raise AppEnginePlatformError('URLFetch request too large, URLFetch only supports requests up to 10mb in size.', e)
            raise ProtocolError(e)
        except urlfetch.DownloadError as e:  # type: ignore[attr-defined]
            if 'Too many redirects' in str(e):
                raise MaxRetryError(self, url, reason=e)
            raise ProtocolError(e)
        except urlfetch.ResponseTooLargeError as e:  # type: ignore[attr-defined]
            raise AppEnginePlatformError('URLFetch response too large, URLFetch only supportsresponses up to 32mb in size.', e)
        except urlfetch.SSLCertificateError as e:  # type: ignore[attr-defined]
            raise SSLError(e)
        except urlfetch.InvalidMethodError as e:  # type: ignore[attr-defined]
            raise AppEnginePlatformError('URLFetch does not support method: %s' % method, e)
        http_response = self._urlfetch_response_to_http_response(response, retries=retries, **response_kw)
        redirect_location = redirect and http_response.get_redirect_location()
        if redirect_location:
            if self.urlfetch_retries and retries.raise_on_redirect:
                raise MaxRetryError(self, url, 'too many redirects')
            else:
                if http_response.status == 303:
                    method = 'GET'
                try:
                    retries = retries.increment(method, url, response=http_response, _pool=self)
                except MaxRetryError:
                    if retries.raise_on_redirect:
                        raise MaxRetryError(self, url, 'too many redirects')
                    return http_response
                retries.sleep_for_retry(http_response)
                log.debug('Redirecting %s -> %s', url, redirect_location)
                redirect_url = urljoin(url, redirect_location)
                return self.urlopen(method, redirect_url, body, headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)
        has_retry_after = bool(http_response.getheader('Retry-After'))
        if retries.is_retry(method, http_response.status, has_retry_after):
            retries = retries.increment(method, url, response=http_response, _pool=self)
            log.debug('Retry: %s', url)
            retries.sleep(http_response)
            return self.urlopen(method, url, body=body, headers=headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)
        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp: Any, **response_kw: Any) -> HTTPResponse:
        if is_prod_appengine():
            content_encoding = urlfetch_resp.headers.get('content-encoding')
            if content_encoding == 'deflate':
                del urlfetch_resp.headers['content-encoding']
        transfer_encoding = urlfetch_resp.headers.get('transfer-encoding')
        if transfer_encoding == 'chunked':
            encodings = transfer_encoding.split(',')
            encodings.remove('chunked')
            urlfetch_resp.headers['transfer-encoding'] = ','.join(encodings)
        return HTTPResponse(
            body=BytesIO(urlfetch_resp.content),
            headers=urlfetch_resp.headers,
            status=urlfetch_resp.status_code,
            **response_kw
        )

    def _get_absolute_timeout(self, timeout: TimeoutType) -> Optional[float]:
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return None
        if isinstance(timeout, Timeout):
            if timeout._read is not None or timeout._connect is not None:
                warnings.warn('URLFetch does not support granular timeout settings, reverting to total or default URLFetch timeout.', AppEnginePlatformWarning)
            return timeout.total
        return float(timeout)  # type: ignore[arg-type]

    def _get_retries(self, retries: Optional[Union[Retry, int, bool]], redirect: bool) -> Retry:
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)
        if retries.connect or retries.read or retries.redirect:
            warnings.warn('URLFetch only supports total retries and does not recognize connect, read, or redirect retry parameters.', AppEnginePlatformWarning)
        return retries


def is_appengine() -> bool:
    return is_local_appengine() or is_prod_appengine() or is_prod_appengine_mvms()


def is_appengine_sandbox() -> bool:
    return is_appengine() and (not is_prod_appengine_mvms())


def is_local_appengine() -> bool:
    return 'APPENGINE_RUNTIME' in os.environ and 'Development/' in os.environ['SERVER_SOFTWARE']


def is_prod_appengine() -> bool:
    return 'APPENGINE_RUNTIME' in os.environ and 'Google App Engine/' in os.environ['SERVER_SOFTWARE'] and (not is_prod_appengine_mvms())


def is_prod_appengine_mvms() -> bool:
    return os.environ.get('GAE_VM', False) == 'true'