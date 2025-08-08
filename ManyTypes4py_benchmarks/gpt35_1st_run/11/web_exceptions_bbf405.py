import warnings
from http import HTTPStatus
from typing import Any, Iterable, Optional, Set, Tuple
from multidict import CIMultiDict
from yarl import URL
from . import hdrs
from .helpers import CookieMixin
from .typedefs import LooseHeaders, StrOrURL

__all__: Tuple[str, ...] = ('HTTPException', 'HTTPError', 'HTTPRedirection', 'HTTPSuccessful', 'HTTPOk', 'HTTPCreated', 'HTTPAccepted', 'HTTPNonAuthoritativeInformation', 'HTTPNoContent', 'HTTPResetContent', 'HTTPPartialContent', 'HTTPMove', 'HTTPMultipleChoices', 'HTTPMovedPermanently', 'HTTPFound', 'HTTPSeeOther', 'HTTPNotModified', 'HTTPUseProxy', 'HTTPTemporaryRedirect', 'HTTPPermanentRedirect', 'HTTPClientError', 'HTTPBadRequest', 'HTTPUnauthorized', 'HTTPPaymentRequired', 'HTTPForbidden', 'HTTPNotFound', 'HTTPMethodNotAllowed', 'HTTPNotAcceptable', 'HTTPProxyAuthenticationRequired', 'HTTPRequestTimeout', 'HTTPConflict', 'HTTPGone', 'HTTPLengthRequired', 'HTTPPreconditionFailed', 'HTTPRequestEntityTooLarge', 'HTTPRequestURITooLong', 'HTTPUnsupportedMediaType', 'HTTPRequestRangeNotSatisfiable', 'HTTPExpectationFailed', 'HTTPMisdirectedRequest', 'HTTPUnprocessableEntity', 'HTTPFailedDependency', 'HTTPUpgradeRequired', 'HTTPPreconditionRequired', 'HTTPTooManyRequests', 'HTTPRequestHeaderFieldsTooLarge', 'HTTPUnavailableForLegalReasons', 'HTTPServerError', 'HTTPInternalServerError', 'HTTPNotImplemented', 'HTTPBadGateway', 'HTTPServiceUnavailable', 'HTTPGatewayTimeout', 'HTTPVersionNotSupported', 'HTTPVariantAlsoNegotiates', 'HTTPInsufficientStorage', 'HTTPNotExtended', 'HTTPNetworkAuthenticationRequired')

class NotAppKeyWarning(UserWarning):
    """Warning when not using AppKey in Application."""

class HTTPException(CookieMixin, Exception):
    status_code: int = -1
    empty_body: bool = False
    default_reason: str = ''

    def __init__(self, *, headers: Optional[LooseHeaders] = None, reason: Optional[str] = None, text: Optional[str] = None, content_type: Optional[str] = None) -> None:
        ...

    def __bool__(self) -> bool:
        ...

    @property
    def status(self) -> int:
        ...

    @property
    def reason(self) -> str:
        ...

    @property
    def text(self) -> Optional[str]:
        ...

    @property
    def headers(self) -> CIMultiDict:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __getnewargs__(self) -> Tuple:
        ...

class HTTPError(HTTPException):
    ...

class HTTPRedirection(HTTPException):
    ...

class HTTPSuccessful(HTTPException):
    ...

class HTTPOk(HTTPSuccessful):
    ...

class HTTPCreated(HTTPSuccessful):
    ...

class HTTPAccepted(HTTPSuccessful):
    ...

class HTTPNonAuthoritativeInformation(HTTPSuccessful):
    ...

class HTTPNoContent(HTTPSuccessful):
    ...

class HTTPResetContent(HTTPSuccessful):
    ...

class HTTPPartialContent(HTTPSuccessful):
    ...

class HTTPMove(HTTPRedirection):
    ...

class HTTPMultipleChoices(HTTPMove):
    ...

class HTTPMovedPermanently(HTTPMove):
    ...

class HTTPFound(HTTPMove):
    ...

class HTTPSeeOther(HTTPMove):
    ...

class HTTPNotModified(HTTPRedirection):
    ...

class HTTPUseProxy(HTTPMove):
    ...

class HTTPTemporaryRedirect(HTTPMove):
    ...

class HTTPPermanentRedirect(HTTPMove):
    ...

class HTTPClientError(HTTPError):
    ...

class HTTPBadRequest(HTTPClientError):
    ...

class HTTPUnauthorized(HTTPClientError):
    ...

class HTTPPaymentRequired(HTTPClientError):
    ...

class HTTPForbidden(HTTPClientError):
    ...

class HTTPNotFound(HTTPClientError):
    ...

class HTTPMethodNotAllowed(HTTPClientError):
    ...

class HTTPNotAcceptable(HTTPClientError):
    ...

class HTTPProxyAuthenticationRequired(HTTPClientError):
    ...

class HTTPRequestTimeout(HTTPClientError):
    ...

class HTTPConflict(HTTPClientError):
    ...

class HTTPGone(HTTPClientError):
    ...

class HTTPLengthRequired(HTTPClientError):
    ...

class HTTPPreconditionFailed(HTTPClientError):
    ...

class HTTPRequestEntityTooLarge(HTTPClientError):
    ...

class HTTPRequestURITooLong(HTTPClientError):
    ...

class HTTPUnsupportedMediaType(HTTPClientError):
    ...

class HTTPRequestRangeNotSatisfiable(HTTPClientError):
    ...

class HTTPExpectationFailed(HTTPClientError):
    ...

class HTTPMisdirectedRequest(HTTPClientError):
    ...

class HTTPUnprocessableEntity(HTTPClientError):
    ...

class HTTPFailedDependency(HTTPClientError):
    ...

class HTTPUpgradeRequired(HTTPClientError):
    ...

class HTTPPreconditionRequired(HTTPClientError):
    ...

class HTTPTooManyRequests(HTTPClientError):
    ...

class HTTPRequestHeaderFieldsTooLarge(HTTPClientError):
    ...

class HTTPUnavailableForLegalReasons(HTTPClientError):
    ...

class HTTPServerError(HTTPError):
    ...

class HTTPInternalServerError(HTTPServerError):
    ...

class HTTPNotImplemented(HTTPServerError):
    ...

class HTTPBadGateway(HTTPServerError):
    ...

class HTTPServiceUnavailable(HTTPServerError):
    ...

class HTTPGatewayTimeout(HTTPServerError):
    ...

class HTTPVersionNotSupported(HTTPServerError):
    ...

class HTTPVariantAlsoNegotiates(HTTPServerError):
    ...

class HTTPInsufficientStorage(HTTPServerError):
    ...

class HTTPNotExtended(HTTPServerError):
    ...

class HTTPNetworkAuthenticationRequired(HTTPServerError):
    ...

def _initialize_default_reason() -> None:
    ...

_initialize_default_reason()
del _initialize_default_reason
