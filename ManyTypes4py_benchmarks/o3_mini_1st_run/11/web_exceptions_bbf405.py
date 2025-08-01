import warnings
from http import HTTPStatus
from typing import Any, Iterable, Optional, Set, Tuple, Union
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
from . import hdrs
from .helpers import CookieMixin
from .typedefs import LooseHeaders, StrOrURL

__all__ = (
    'HTTPException', 'HTTPError', 'HTTPRedirection', 'HTTPSuccessful', 'HTTPOk', 'HTTPCreated', 'HTTPAccepted', 
    'HTTPNonAuthoritativeInformation', 'HTTPNoContent', 'HTTPResetContent', 'HTTPPartialContent', 'HTTPMove', 
    'HTTPMultipleChoices', 'HTTPMovedPermanently', 'HTTPFound', 'HTTPSeeOther', 'HTTPNotModified', 'HTTPUseProxy', 
    'HTTPTemporaryRedirect', 'HTTPPermanentRedirect', 'HTTPClientError', 'HTTPBadRequest', 'HTTPUnauthorized', 
    'HTTPPaymentRequired', 'HTTPForbidden', 'HTTPNotFound', 'HTTPMethodNotAllowed', 'HTTPNotAcceptable', 
    'HTTPProxyAuthenticationRequired', 'HTTPRequestTimeout', 'HTTPConflict', 'HTTPGone', 'HTTPLengthRequired', 
    'HTTPPreconditionFailed', 'HTTPRequestEntityTooLarge', 'HTTPRequestURITooLong', 'HTTPUnsupportedMediaType', 
    'HTTPRequestRangeNotSatisfiable', 'HTTPExpectationFailed', 'HTTPMisdirectedRequest', 'HTTPUnprocessableEntity', 
    'HTTPFailedDependency', 'HTTPUpgradeRequired', 'HTTPPreconditionRequired', 'HTTPTooManyRequests', 
    'HTTPRequestHeaderFieldsTooLarge', 'HTTPUnavailableForLegalReasons', 'HTTPServerError', 'HTTPInternalServerError', 
    'HTTPNotImplemented', 'HTTPBadGateway', 'HTTPServiceUnavailable', 'HTTPGatewayTimeout', 'HTTPVersionNotSupported', 
    'HTTPVariantAlsoNegotiates', 'HTTPInsufficientStorage', 'HTTPNotExtended', 'HTTPNetworkAuthenticationRequired'
)

class NotAppKeyWarning(UserWarning):
    """Warning when not using AppKey in Application."""

class HTTPException(CookieMixin, Exception):
    status_code: int = -1
    empty_body: bool = False
    default_reason: str = ''

    def __init__(
        self,
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> None:
        if reason is None:
            reason = self.default_reason
        elif '\n' in reason:
            raise ValueError('Reason cannot contain \\n')
        if text is None:
            if not self.empty_body:
                text = f'{self.status_code}: {reason}'
        elif self.empty_body:
            warnings.warn(
                'text argument is deprecated for HTTP status {} since 4.0 and scheduled for removal in 5.0 (#3462),'
                'the response should be provided without a body'.format(self.status_code),
                DeprecationWarning,
                stacklevel=2
            )
        if headers is not None:
            real_headers: CIMultiDict = CIMultiDict(headers)
        else:
            real_headers = CIMultiDict()
        if content_type is not None:
            if not text:
                warnings.warn(
                    'content_type without text is deprecated since 4.0 and scheduled for removal in 5.0 (#3462)',
                    DeprecationWarning,
                    stacklevel=2
                )
            real_headers[hdrs.CONTENT_TYPE] = content_type
        elif hdrs.CONTENT_TYPE not in real_headers and text:
            real_headers[hdrs.CONTENT_TYPE] = 'text/plain'
        self._reason: str = reason
        self._text: Optional[str] = text
        self._headers: CIMultiDict = real_headers
        self.args: Tuple[Any, ...] = ()

    def __bool__(self) -> bool:
        return True

    @property
    def status(self) -> int:
        return self.status_code

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def text(self) -> Optional[str]:
        return self._text

    @property
    def headers(self) -> CIMultiDict:
        return self._headers

    def __str__(self) -> str:
        return self.reason

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}: {self.reason}>'

    __reduce__ = object.__reduce__

    def __getnewargs__(self) -> Tuple[Any, ...]:
        return self.args

class HTTPError(HTTPException):
    """Base class for exceptions with status codes in the 400s and 500s."""

class HTTPRedirection(HTTPException):
    """Base class for exceptions with status codes in the 300s."""

class HTTPSuccessful(HTTPException):
    """Base class for exceptions with status codes in the 200s."""

class HTTPOk(HTTPSuccessful):
    status_code: int = 200

class HTTPCreated(HTTPSuccessful):
    status_code: int = 201

class HTTPAccepted(HTTPSuccessful):
    status_code: int = 202

class HTTPNonAuthoritativeInformation(HTTPSuccessful):
    status_code: int = 203

class HTTPNoContent(HTTPSuccessful):
    status_code: int = 204
    empty_body: bool = True

class HTTPResetContent(HTTPSuccessful):
    status_code: int = 205
    empty_body: bool = True

class HTTPPartialContent(HTTPSuccessful):
    status_code: int = 206

class HTTPMove(HTTPRedirection):
    def __init__(
        self,
        location: Union[StrOrURL, str],
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> None:
        if not location:
            raise ValueError('HTTP redirects need a location to redirect to.')
        super().__init__(headers=headers, reason=reason, text=text, content_type=content_type)
        self._location: URL = URL(location)
        self.headers['Location'] = str(self.location)

    @property
    def location(self) -> URL:
        return self._location

class HTTPMultipleChoices(HTTPMove):
    status_code: int = 300

class HTTPMovedPermanently(HTTPMove):
    status_code: int = 301

class HTTPFound(HTTPMove):
    status_code: int = 302

class HTTPSeeOther(HTTPMove):
    status_code: int = 303

class HTTPNotModified(HTTPRedirection):
    status_code: int = 304
    empty_body: bool = True

class HTTPUseProxy(HTTPMove):
    status_code: int = 305

class HTTPTemporaryRedirect(HTTPMove):
    status_code: int = 307

class HTTPPermanentRedirect(HTTPMove):
    status_code: int = 308

class HTTPClientError(HTTPError):
    pass

class HTTPBadRequest(HTTPClientError):
    status_code: int = 400

class HTTPUnauthorized(HTTPClientError):
    status_code: int = 401

class HTTPPaymentRequired(HTTPClientError):
    status_code: int = 402

class HTTPForbidden(HTTPClientError):
    status_code: int = 403

class HTTPNotFound(HTTPClientError):
    status_code: int = 404

class HTTPMethodNotAllowed(HTTPClientError):
    status_code: int = 405

    def __init__(
        self,
        method: str,
        allowed_methods: Iterable[str],
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> None:
        allow: str = ','.join(sorted(allowed_methods))
        super().__init__(headers=headers, reason=reason, text=text, content_type=content_type)
        self.headers['Allow'] = allow
        self._allowed: Set[str] = set(allowed_methods)
        self._method: str = method

    @property
    def allowed_methods(self) -> Set[str]:
        return self._allowed

    @property
    def method(self) -> str:
        return self._method

class HTTPNotAcceptable(HTTPClientError):
    status_code: int = 406

class HTTPProxyAuthenticationRequired(HTTPClientError):
    status_code: int = 407

class HTTPRequestTimeout(HTTPClientError):
    status_code: int = 408

class HTTPConflict(HTTPClientError):
    status_code: int = 409

class HTTPGone(HTTPClientError):
    status_code: int = 410

class HTTPLengthRequired(HTTPClientError):
    status_code: int = 411

class HTTPPreconditionFailed(HTTPClientError):
    status_code: int = 412

class HTTPRequestEntityTooLarge(HTTPClientError):
    status_code: int = 413

    def __init__(self, max_size: int, actual_size: int, **kwargs: Any) -> None:
        kwargs.setdefault('text', 'Maximum request body size {} exceeded, actual body size {}'.format(max_size, actual_size))
        super().__init__(**kwargs)

class HTTPRequestURITooLong(HTTPClientError):
    status_code: int = 414

class HTTPUnsupportedMediaType(HTTPClientError):
    status_code: int = 415

class HTTPRequestRangeNotSatisfiable(HTTPClientError):
    status_code: int = 416

class HTTPExpectationFailed(HTTPClientError):
    status_code: int = 417

class HTTPMisdirectedRequest(HTTPClientError):
    status_code: int = 421

class HTTPUnprocessableEntity(HTTPClientError):
    status_code: int = 422

class HTTPFailedDependency(HTTPClientError):
    status_code: int = 424

class HTTPUpgradeRequired(HTTPClientError):
    status_code: int = 426

class HTTPPreconditionRequired(HTTPClientError):
    status_code: int = 428

class HTTPTooManyRequests(HTTPClientError):
    status_code: int = 429

class HTTPRequestHeaderFieldsTooLarge(HTTPClientError):
    status_code: int = 431

class HTTPUnavailableForLegalReasons(HTTPClientError):
    status_code: int = 451

    def __init__(
        self,
        link: Optional[Union[StrOrURL, str]],
        *,
        headers: Optional[LooseHeaders] = None,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> None:
        super().__init__(headers=headers, reason=reason, text=text, content_type=content_type)
        self._link: Optional[URL] = None
        if link:
            self._link = URL(link)
            self.headers['Link'] = f'<{str(self._link)}>; rel="blocked-by"'

    @property
    def link(self) -> Optional[URL]:
        return self._link

class HTTPServerError(HTTPError):
    pass

class HTTPInternalServerError(HTTPServerError):
    status_code: int = 500

class HTTPNotImplemented(HTTPServerError):
    status_code: int = 501

class HTTPBadGateway(HTTPServerError):
    status_code: int = 502

class HTTPServiceUnavailable(HTTPServerError):
    status_code: int = 503

class HTTPGatewayTimeout(HTTPServerError):
    status_code: int = 504

class HTTPVersionNotSupported(HTTPServerError):
    status_code: int = 505

class HTTPVariantAlsoNegotiates(HTTPServerError):
    status_code: int = 506

class HTTPInsufficientStorage(HTTPServerError):
    status_code: int = 507

class HTTPNotExtended(HTTPServerError):
    status_code: int = 510

class HTTPNetworkAuthenticationRequired(HTTPServerError):
    status_code: int = 511

def _initialize_default_reason() -> None:
    for obj in globals().values():
        if isinstance(obj, type) and issubclass(obj, HTTPException):
            if obj.status_code >= 0:
                try:
                    status = HTTPStatus(obj.status_code)
                    obj.default_reason = status.phrase
                except ValueError:
                    pass

_initialize_default_reason()
del _initialize_default_reason