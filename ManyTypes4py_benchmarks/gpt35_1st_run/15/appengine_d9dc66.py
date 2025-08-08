from typing import Optional, Union
from urllib3.exceptions import HTTPError, HTTPWarning, MaxRetryError, ProtocolError, TimeoutError, SSLError
from urllib3.request import RequestMethods
from urllib3.response import HTTPResponse
from urllib3.util.timeout import Timeout
from urllib3.util.retry import Retry

class AppEngineManager(RequestMethods):
    def __init__(self, headers: Optional[dict] = None, retries: Optional[Union[Retry, int]] = None, validate_certificate: bool = True, urlfetch_retries: bool = True) -> None:
        ...

    def __enter__(self) -> 'AppEngineManager':
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        ...

    def urlopen(self, method: str, url: str, body: Optional[bytes] = None, headers: Optional[dict] = None, retries: Optional[Union[Retry, int]] = None, redirect: bool = True, timeout: Union[float, Timeout] = Timeout.DEFAULT_TIMEOUT, **response_kw) -> HTTPResponse:
        ...

    def _urlfetch_response_to_http_response(self, urlfetch_resp, **response_kw) -> HTTPResponse:
        ...

    def _get_absolute_timeout(self, timeout: Union[float, Timeout]) -> Optional[float]:
        ...

    def _get_retries(self, retries: Union[Retry, int], redirect: bool) -> Retry:
        ...

def is_appengine() -> bool:
    ...

def is_appengine_sandbox() -> bool:
    ...

def is_local_appengine() -> bool:
    ...

def is_prod_appengine() -> bool:
    ...

def is_prod_appengine_mvms() -> bool:
    ...
