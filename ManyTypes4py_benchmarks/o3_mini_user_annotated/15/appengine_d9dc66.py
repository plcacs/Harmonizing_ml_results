from __future__ import absolute_import
import logging
import os
import warnings
from typing import Optional, Union, Any, Dict, IO, Type
from io import BytesIO as StdBytesIO

from ..packages.six.moves.urllib.parse import urljoin

from ..exceptions import (
    HTTPError,
    HTTPWarning,
    MaxRetryError,
    ProtocolError,
    TimeoutError,
    SSLError,
)
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

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, int]] = None,
        validate_certificate: bool = True,
        urlfetch_retries: bool = True,
    ) -> None:
        if not urlfetch:
            raise AppEnginePlatformError(
                "URLFetch is not available in this environment."
            )

        if is_prod_appengine_mvms():
            raise AppEnginePlatformError(
                "Use normal urllib3.PoolManager instead of AppEngineManager"
                " on Managed VMs, as using URLFetch is not necessary in "
                "this environment."
            )

        warnings.warn(
            "urllib3 is using URLFetch on Google App Engine sandbox instead "
            "of sockets. To use sockets directly instead of URLFetch see "
            "https://urllib3.readthedocs.io/en/latest/reference/urllib3.contrib.html.",
            AppEnginePlatformWarning,
        )
        RequestMethods.__init__(self, headers)
        self.validate_certificate: bool = validate_certificate
        self.urlfetch_retries: bool = urlfetch_retries
        self.retries: Union[Retry, int] = retries or Retry.DEFAULT

    def __enter__(self) -> "AppEngineManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        # Return False to re-raise any potential exceptions
        return False

    def urlopen(
        self,
        method: str,
        url: str,
        body: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[Union[Retry, int]] = None,
        redirect: bool = True,
        timeout: Union[Timeout, float] = Timeout.DEFAULT_TIMEOUT,
        **response_kw: Any
    ) -> HTTPResponse:
        retries = self._get_retries(retries, redirect)
        try:
            follow_redirects: bool = redirect and retries.redirect != 0 and retries.total  # type: ignore
            response = urlfetch.fetch(
                url,
                payload=body,
                method=method,
                headers=headers or {},
                allow_truncated=False,
                follow_redirects=self.urlfetch_retries and bool(follow_redirects),
                deadline=self._get_absolute_timeout(timeout),
                validate_certificate=self.validate_certificate,
            )
        except urlfetch.DeadlineExceededError as e:
            raise TimeoutError(self, e)

        except urlfetch.InvalidURLError as e:
            if "too large" in str(e):
                raise AppEnginePlatformError(
                    "URLFetch request too large, URLFetch only "
                    "supports requests up to 10mb in size.",
                    e,
                )

            raise ProtocolError(e)

        except urlfetch.DownloadError as e:
            if "Too many redirects" in str(e):
                raise MaxRetryError(self, url, reason=e)

            raise ProtocolError(e)

        except urlfetch.ResponseTooLargeError as e:
            raise AppEnginePlatformError(
                "URLFetch response too large, URLFetch only supports"
                " responses up to 32mb in size.",
                e,
            )

        except urlfetch.SSLCertificateError as e:
            raise SSLError(e)

        except urlfetch.InvalidMethodError as e:
            raise AppEnginePlatformError(
                "URLFetch does not support method: %s" % method, e
            )

        http_response: HTTPResponse = self._urlfetch_response_to_http_response(
            response, **response_kw, retries=retries
        )
        # Handle redirect?
        redirect_location: Optional[str] = redirect and http_response.get_redirect_location()  # type: ignore
        if redirect_location:
            # Check for redirect response
            if self.urlfetch_retries and retries.raise_on_redirect:  # type: ignore
                raise MaxRetryError(self, url, "too many redirects")
            else:
                if http_response.status == 303:
                    method = "GET"
                try:
                    retries = retries.increment(
                        method, url, response=http_response, _pool=self  # type: ignore
                    )
                except MaxRetryError:
                    if retries.raise_on_redirect:  # type: ignore
                        raise MaxRetryError(self, url, "too many redirects")
                    return http_response

                retries.sleep_for_retry(http_response)  # type: ignore
                log.debug("Redirecting %s -> %s", url, redirect_location)
                redirect_url: str = urljoin(url, redirect_location)
                return self.urlopen(
                    method,
                    redirect_url,
                    body,
                    headers,
                    retries=retries,
                    redirect=redirect,
                    timeout=timeout,
                    **response_kw
                )

        # Check if we should retry the HTTP response.
        has_retry_after: bool = bool(http_response.getheader("Retry-After"))
        if retries.is_retry(method, http_response.status, has_retry_after):  # type: ignore
            retries = retries.increment(method, url, response=http_response, _pool=self)  # type: ignore
            log.debug("Retry: %s", url)
            retries.sleep(http_response)  # type: ignore
            return self.urlopen(
                method,
                url,
                body=body,
                headers=headers,
                retries=retries,
                redirect=redirect,
                timeout=timeout,
                **response_kw
            )

        return http_response

    def _urlfetch_response_to_http_response(self, urlfetch_resp: Any, **response_kw: Any) -> HTTPResponse:
        if is_prod_appengine():
            # Production GAE handles deflate encoding automatically, but does
            # not remove the encoding header.
            content_encoding: Optional[str] = urlfetch_resp.headers.get("content-encoding")
            if content_encoding == "deflate":
                del urlfetch_resp.headers["content-encoding"]
        transfer_encoding: Optional[str] = urlfetch_resp.headers.get("transfer-encoding")
        # We have a full response's content,
        # so let's make sure we don't report ourselves as chunked data.
        if transfer_encoding == "chunked":
            encodings = transfer_encoding.split(",")
            encodings.remove("chunked")
            urlfetch_resp.headers["transfer-encoding"] = ",".join(encodings)
        return HTTPResponse(
            # In order for decoding to work, we must present the content as
            # a file-like object.
            body=StdBytesIO(urlfetch_resp.content),
            headers=urlfetch_resp.headers,
            status=urlfetch_resp.status_code,
            **response_kw
        )

    def _get_absolute_timeout(self, timeout: Union[Timeout, float]) -> Optional[float]:
        if timeout is Timeout.DEFAULT_TIMEOUT:
            return None  # Defer to URLFetch's default.

        if isinstance(timeout, Timeout):
            if timeout._read is not None or timeout._connect is not None:  # type: ignore
                warnings.warn(
                    "URLFetch does not support granular timeout settings, "
                    "reverting to total or default URLFetch timeout.",
                    AppEnginePlatformWarning,
                )
            return timeout.total  # type: ignore

        return timeout

    def _get_retries(
        self, retries: Optional[Union[Retry, int]], redirect: bool
    ) -> Retry:
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)  # type: ignore
        if retries.connect or retries.read or retries.redirect:  # type: ignore
            warnings.warn(
                "URLFetch only supports total retries and does not "
                "recognize connect, read, or redirect retry parameters.",
                AppEnginePlatformWarning,
            )
        return retries


def is_appengine() -> bool:
    return is_local_appengine() or is_prod_appengine() or is_prod_appengine_mvms()


def is_appengine_sandbox() -> bool:
    return is_appengine() and not is_prod_appengine_mvms()


def is_local_appengine() -> bool:
    return (
        "APPENGINE_RUNTIME" in os.environ
        and "Development/" in os.environ.get("SERVER_SOFTWARE", "")
    )


def is_prod_appengine() -> bool:
    return (
        "APPENGINE_RUNTIME" in os.environ
        and "Google App Engine/" in os.environ.get("SERVER_SOFTWARE", "")
        and not is_prod_appengine_mvms()
    )


def is_prod_appengine_mvms() -> bool:
    return os.environ.get("GAE_VM", False) == "true"