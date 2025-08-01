"""The App Engine Transport Adapter for requests.

.. versionadded:: 0.6.0

This requires a version of requests >= 2.10.0 and Python 2.

There are two ways to use this library:

#. If you're using requests directly, you can use code like:

   .. code-block:: python

       >>> import requests
       >>> import ssl
       >>> import requests.packages.urllib3.contrib.appengine as ul_appengine
       >>> from requests_toolbelt.adapters import appengine
       >>> s = requests.Session()
       >>> if ul_appengine.is_appengine_sandbox():
       ...    s.mount('http://', appengine.AppEngineAdapter())
       ...    s.mount('https://', appengine.AppEngineAdapter())

#. If you depend on external libraries which use requests, you can use code
   like:

   .. code-block:: python

       >>> from requests_toolbelt.adapters import appengine
       >>> appengine.monkeypatch()

which will ensure all requests.Session objects use AppEngineAdapter properly.

You are also able to :ref:`disable certificate validation <insecure_appengine>`
when monkey-patching.
"""
import requests
import warnings
from typing import Any, Optional, Dict, Union, Type
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout

class AppEngineMROHack(adapters.HTTPAdapter):
    """Resolves infinite recursion when monkeypatching.

    This works by injecting itself as the base class of both the
    :class:`AppEngineAdapter` and Requests' default HTTPAdapter, which needs to
    be done because default HTTPAdapter's MRO is recompiled when we
    monkeypatch, at which point this class becomes HTTPAdapter's base class.
    In addition, we use an instantiation flag to avoid infinite recursion.
    """
    _initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not self._initialized:
            self._initialized = True
            super(AppEngineMROHack, self).__init__(*args, **kwargs)

class AppEngineAdapter(AppEngineMROHack, adapters.HTTPAdapter):
    """The transport adapter for Requests to use urllib3's GAE support.

    Implements Requests's HTTPAdapter API.

    When deploying to Google's App Engine service, some of Requests'
    functionality is broken. There is underlying support for GAE in urllib3.
    This functionality, however, is opt-in and needs to be enabled explicitly
    for Requests to be able to use it.
    """
    __attrs__ = adapters.HTTPAdapter.__attrs__ + ['_validate_certificate']
    _validate_certificate: bool

    def __init__(self, validate_certificate: bool = True, *args: Any, **kwargs: Any) -> None:
        _check_version()
        self._validate_certificate = validate_certificate
        super(AppEngineAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False) -> None:
        self.poolmanager = _AppEnginePoolManager(self._validate_certificate)

class InsecureAppEngineAdapter(AppEngineAdapter):
    """An always-insecure GAE adapter for Requests.

    This is a variant of the the transport adapter for Requests to use
    urllib3's GAE support that does not validate certificates. Use with
    caution!

    .. note::
        The ``validate_certificate`` keyword argument will not be honored here
        and is not part of the signature because we always force it to
        ``False``.

    See :class:`AppEngineAdapter` for further details.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.pop('validate_certificate', False):
            warnings.warn('Certificate validation cannot be specified on the InsecureAppEngineAdapter, but was present. This will be ignored and certificate validation will remain off.', exc.IgnoringGAECertificateValidation)
        super(InsecureAppEngineAdapter, self).__init__(*args, validate_certificate=False, **kwargs)

class _AppEnginePoolManager:
    """Implements urllib3's PoolManager API expected by requests.

    While a real PoolManager map hostnames to reusable Connections,
    AppEngine has no concept of a reusable connection to a host.
    So instead, this class constructs a small Connection per request,
    that is returned to the Adapter and used to access the URL.
    """
    appengine_manager: Any

    def __init__(self, validate_certificate: bool = True) -> None:
        self.appengine_manager = gaecontrib.AppEngineManager(validate_certificate=validate_certificate)

    def connection_from_url(self, url: str) -> '_AppEngineConnection':
        return _AppEngineConnection(self.appengine_manager, url)

    def clear(self) -> None:
        pass

class _AppEngineConnection:
    """Implements urllib3's HTTPConnectionPool API's urlopen().

    This Connection's urlopen() is called with a host-relative path,
    so in order to properly support opening the URL, we need to store
    the full URL when this Connection is constructed from the PoolManager.

    This code wraps AppEngineManager.urlopen(), which exposes a different
    API than in the original urllib3 urlopen(), and thus needs this adapter.
    """
    appengine_manager: Any
    url: str

    def __init__(self, appengine_manager: Any, url: str) -> None:
        self.appengine_manager = appengine_manager
        self.url = url

    def urlopen(self, method: str, url: str, body: Optional[Any] = None, headers: Optional[Dict[str, str]] = None, retries: Optional[Any] = None, redirect: bool = True, assert_same_host: bool = True, timeout: Union[timeout.Timeout, float] = timeout.Timeout.DEFAULT_TIMEOUT, pool_timeout: Optional[float] = None, release_conn: Optional[bool] = None, **response_kw: Any) -> Any:
        if not timeout.total:
            timeout.total = timeout._read or timeout._connect
        return self.appengine_manager.urlopen(method, self.url, body=body, headers=headers, retries=retries, redirect=redirect, timeout=timeout, **response_kw)

def monkeypatch(validate_certificate: bool = True) -> None:
    """Sets up all Sessions to use AppEngineAdapter by default.

    If you don't want to deal with configuring your own Sessions,
    or if you use libraries that use requests directly (ie requests.post),
    then you may prefer to monkeypatch and auto-configure all Sessions.

    .. warning: :

        If ``validate_certificate`` is ``False``, certification validation will
        effectively be disabled for all requests.
    """
    _check_version()
    adapter: Type[AppEngineAdapter] = AppEngineAdapter
    if not validate_certificate:
        adapter = InsecureAppEngineAdapter
    sessions.HTTPAdapter = adapter
    adapters.HTTPAdapter = adapter

def _check_version() -> None:
    if gaecontrib is None:
        raise exc.VersionMismatchError('The toolbelt requires at least Requests 2.10.0 to be installed. Version {} was found instead.'.format(requests.__version__))
