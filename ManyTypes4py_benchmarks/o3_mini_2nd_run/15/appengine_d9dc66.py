#!/usr/bin/env python
"""
This module provides a pool manager that uses Google App Engine's
`URLFetch Service <https://cloud.google.com/appengine/docs/python/urlfetch>`_.

Example usage::

    from urllib3 import PoolManager
    from urllib3.contrib.appengine import AppEngineManager, is_appengine_sandbox

    if is_appengine_sandbox():
        # AppEngineManager uses AppEngine's URLFetch API behind the scenes
        http = AppEngineManager()
    else:
        # PoolManager uses a socket-level API behind the scenes
        http = PoolManager()

    r = http.request('GET', 'https://google.com/')

There are `limitations <https://cloud.google.com/appengine/docs/python/urlfetch/#Python_Quotas_and_limits>`_ to the URLFetch service and it may not be
the best choice for your application. There are three options for using
urllib3 on Google App Engine:

1. You can use :class:`AppEngineManager` with URLFetch. URLFetch is
   cost-effective in many circumstances as long as your usage is within the
   limitations.
2. You can use a normal :class:`~urllib3.PoolManager` by enabling sockets.
   Sockets also have `limitations and restrictions
   <https://cloud.google.com/appengine/docs/python/sockets/   #limitations-and-restrictions>`_ and have a lower free quota than URLFetch.
   To use sockets, be sure to specify the following in your ``app.yaml``::

        env_variables:
            GAE_USE_SOCKETS_HTTPLIB : 'true'

3. If you are using `App Engine Flexible
<https://cloud.google.com/appengine/docs/flexible/>`_, you can use the standard
:class:`PoolManager` without any configuration or special environment variables.
"""

from __future__ import absolute_import
import logging
import os
import warnings
from typing import Optional, Union, Dict, Any, Iterator

from ..packages.six.moves.urllib.parse import urljoin
from ..exceptions import HTTPError, HTTPWarning, MaxRetryError, ProtocolError, TimeoutError, SSLError
from ..packages.six import BytesIO
from ..request import RequestMethods
from ..response import HTTPResponse
from ..util.timeout import Timeout
from ..util.retry import Retry

try:
    from google.appengine.api import urlfetch
except ImportError:
    urlfetch = None

log = logging.getLogger(__name__)


class AppEnginePlatformWarning(HTTPWarning):
    pass


class AppEnginePlatformError(HTTPError):
    pass


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

    def __init__(self, 
                 headers: Optional[Dict[str, str]] = None, 
                 retries: Optional[Union[int, Retry]] = None, 
                 validate_certificate: bool = True, 
                 urlfetch_retries: bool = True) -> None:
        if not urlfetch:
            raise AppEnginePlatformError('URLFetch is not available in this environment.')
        if is_prod_appengine_mvms():
            raise AppEnginePlatformError('Use normal urllib3.PoolManager instead of AppEngineManager on Managed VMs, as using URLFetch is not necessary in this environment.')
        warnings.warn('urllib3 is using URLFetch on Google App Engine sandbox instead of sockets. To use sockets directly instead of URLFetch see https://urllib3.readthedocs.io/en/latest/reference/urllib3.contrib.html.', AppEnginePlatformWarning)
        RequestMethods.__init__(self, headers)
        self.validate_certificate: bool = validate_certificate
        self.urlfetch_retries: bool = urlfetch_retries
        self.retries: Union[int, Retry] = retries or Retry.DEFAULT

    def __enter__(self) -> "AppEngineManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    def urlopen(self, 
                method: str, 
                url: str, 
                body: Optional[Union[str, bytes]] = None, 
                headers: Optional[Dict[str, str]] = None, 
                retries: Optional[Union[int, Retry]] = None, 
                redirect: bool = True, 
                timeout: Union[Timeout, float, int] = Timeout.DEFAULT_TIMEOUT, 
                **response_kw: Any) -> HTTPResponse:
        retries_obj: Retry = self._get_retries(retries, redirect)
        try:
            follow_redirects: bool = redirect and retries_obj.redirect != 0 and retries_obj.total
            response: Any = urlfetch.fetch(
                url,
                payload=body,
                method=method,
                headers=headers or {},
                allow_truncated=False,
                follow_redirects=self.urlfetch_retries and follow_redirects,
                deadline=self._get_absolute_timeout(timeout),
                validate_certificate=self.validate_certificate)
        except urlfetch.DeadlineExceededError as e:
            raise TimeoutError(self, e)
        except urlfetch.InvalidURLError as e:
            if 'too large' in str(e):
                raise AppEnginePlatformError('URLFetch request too large, URLFetch only supports requests up to 10mb in size.', e)
            raise ProtocolError(e)
        except urlfetch.DownloadError as e:
            if 'Too many redirects' in str(e):
                raise MaxRetryError(self, url, reason=e)
            raise ProtocolError(e)
        except urlfetch.ResponseTooLargeError as e:
            raise AppEnginePlatformError('URLFetch response too large, URLFetch only supports responses up to 32mb in size.', e)
        except urlfetch.SSLCertificateError as e:
            raise SSLError(e)
        except urlfetch.InvalidMethodError as e:
            raise AppEnginePlatformError('URLFetch does not support method: %s' % method, e)
        http_response: HTTPResponse = self._urlfetch_response_to_http_response(response, retries=retries_obj, **response_kw)
        redirect_location: Optional[str] = redirect and http_response.get_redirect_location()
        if redirect_location:
            if self.urlfetch_retries and retries_obj.raise_on_redirect:
                raise MaxRetryError(self, url, 'too many redirects')
            else:
                if http_response.status == 303:
                    method = 'GET'
                try:
                    retries_obj = retries_obj.increment(method, url, response=http_response, _pool=self)
                except MaxRetryError:
                    if retries_obj.raise_on_redirect:
                        raise MaxRetryError(self, url, 'too many redirects')
                    return http_response
                retries_obj.sleep_for_retry(http_response)
                log.debug('Redirecting %s -> %s', url, redirect_location)
                redirect_url: str = urljoin(url, redirect_location)
                return self.urlopen(method, redirect_url, body, headers, retries=retries_obj, redirect=redirect, timeout=timeout, **response_kw)
        has_retry_after: bool = bool(http_response.getheader('Retry-After'))
        if retries_obj.is_retry(method, http_response.status, has_retry_after):
            retries_obj = retries_obj.increment(method, url, response=http_response, _pool=self)
            log.debug('Retry: %s', url)
            retries_obj.sleep(http_response)
            return self.urlopen(method, url, body=body, headers=headers, retries=retries_obj, redirect=redirect, timeout=timeout, **response_kw)
        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp: Any, **response_kw: Any) -> HTTPResponse:
        if is_prod_appengine():
            content_encoding: Optional[str] = urlfetch_resp.headers.get('content-encoding')
            if content_encoding == 'deflate':
                del urlfetch_resp.headers['content-encoding']
        transfer_encoding: Optional[str] = urlfetch_resp.headers.get('transfer-encoding')
        if transfer_encoding == 'chunked':
            encodings = transfer_encoding.split(',')
            encodings.remove('chunked')
            urlfetch_resp.headers['transfer-encoding'] = ','.join(encodings)
        return HTTPResponse(body=BytesIO(urlfetch_resp.content), headers=urlfetch_resp.headers, status=urlfetch_resp.status_code, **response_kw)

    def _get_absolute_timeout(self, timeout: Union[Timeout, float, int]) -> Optional[float]:
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return None
        if isinstance(timeout, Timeout):
            if timeout._read is not None or timeout._connect is not None:
                warnings.warn('URLFetch does not support granular timeout settings, reverting to total or default URLFetch timeout.', AppEnginePlatformWarning)
            return timeout.total  # type: ignore
        return timeout  # type: ignore

    def _get_retries(self, retries: Optional[Union[int, Retry]], redirect: bool) -> Retry:
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)  # type: ignore
        if retries.connect or retries.read or retries.redirect:
            warnings.warn('URLFetch only supports total retries and does not recognize connect, read, or redirect retry parameters.', AppEnginePlatformWarning)
        return retries


def is_appengine() -> bool:
    return is_local_appengine() or is_prod_appengine() or is_prod_appengine_mvms()


def is_appengine_sandbox() -> bool:
    return is_appengine() and (not is_prod_appengine_mvms())


def is_local_appengine() -> bool:
    return 'APPENGINE_RUNTIME' in os.environ and 'Development/' in os.environ.get('SERVER_SOFTWARE', '')


def is_prod_appengine() -> bool:
    return 'APPENGINE_RUNTIME' in os.environ and 'Google App Engine/' in os.environ.get('SERVER_SOFTWARE', '') and (not is_prod_appengine_mvms())


def is_prod_appengine_mvms() -> bool:
    return os.environ.get('GAE_VM', False) == 'true'
