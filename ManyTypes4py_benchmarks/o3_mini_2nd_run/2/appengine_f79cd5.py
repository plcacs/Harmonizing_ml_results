from typing import Any, Optional, Union, Dict
import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout

class AppEngineMROHack(adapters.HTTPAdapter):
    _initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self._initialized:
            self._initialized = True
            super(AppEngineMROHack, self).__init__(*args, **kwargs)

class AppEngineAdapter(AppEngineMROHack, adapters.HTTPAdapter):
    __attrs__ = adapters.HTTPAdapter.__attrs__ + ['_validate_certificate']
    _validate_certificate: bool

    def __init__(self, validate_certificate: bool = True, *args: Any, **kwargs: Any) -> None:
        _check_version()
        self._validate_certificate = validate_certificate
        super(AppEngineAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False) -> None:
        self.poolmanager = _AppEnginePoolManager(self._validate_certificate)

class InsecureAppEngineAdapter(AppEngineAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop('validate_certificate', False):
            warnings.warn(
                'Certificate validation cannot be specified on the InsecureAppEngineAdapter, but was present. '
                'This will be ignored and certificate validation will remain off.',
                exc.IgnoringGAECertificateValidation
            )
        super(InsecureAppEngineAdapter, self).__init__(*args, validate_certificate=False, **kwargs)

class _AppEnginePoolManager:
    def __init__(self, validate_certificate: bool = True) -> None:
        self.appengine_manager: Any = gaecontrib.AppEngineManager(validate_certificate=validate_certificate)

    def connection_from_url(self, url: str) -> '_AppEngineConnection':
        return _AppEngineConnection(self.appengine_manager, url)

    def clear(self) -> None:
        pass

class _AppEngineConnection:
    def __init__(self, appengine_manager: Any, url: str) -> None:
        self.appengine_manager: Any = appengine_manager
        self.url: str = url

    def urlopen(self,
                method: str,
                url: str,
                body: Optional[Union[str, bytes]] = None,
                headers: Optional[Dict[str, str]] = None,
                retries: Any = None,
                redirect: bool = True,
                assert_same_host: bool = True,
                timeout: timeout.Timeout = timeout.Timeout.DEFAULT_TIMEOUT,
                pool_timeout: Optional[float] = None,
                release_conn: Any = None,
                **response_kw: Any) -> Any:
        if not timeout.total:
            timeout.total = timeout._read or timeout._connect
        return self.appengine_manager.urlopen(method, self.url, body=body, headers=headers, retries=retries,
                                              redirect=redirect, timeout=timeout, **response_kw)

def monkeypatch(validate_certificate: bool = True) -> None:
    _check_version()
    adapter: Any = AppEngineAdapter
    if not validate_certificate:
        adapter = InsecureAppEngineAdapter
    sessions.HTTPAdapter = adapter
    adapters.HTTPAdapter = adapter

def _check_version() -> None:
    if gaecontrib is None:
        raise exc.VersionMismatchError(
            'The toolbelt requires at least Requests 2.10.0 to be installed. Version {} was found instead.'.format(
                requests.__version__
            )
        )