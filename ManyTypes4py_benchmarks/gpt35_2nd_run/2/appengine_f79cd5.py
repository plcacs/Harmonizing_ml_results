from typing import Optional

class AppEngineMROHack(adapters.HTTPAdapter):
    _initialized: bool

    def __init__(self, *args, **kwargs):
        ...

class AppEngineAdapter(AppEngineMROHack, adapters.HTTPAdapter):
    __attrs__: list[str]

    def __init__(self, validate_certificate: bool = True, *args, **kwargs):
        ...

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False):
        ...

class InsecureAppEngineAdapter(AppEngineAdapter):
    def __init__(self, *args, **kwargs):
        ...

class _AppEnginePoolManager:
    def __init__(self, validate_certificate: bool = True):
        ...

    def connection_from_url(self, url: str):
        ...

    def clear(self):
        ...

class _AppEngineConnection:
    def __init__(self, appengine_manager, url: str):
        ...

    def urlopen(self, method, url, body=None, headers=None, retries=None, redirect=True, assert_same_host=True, timeout=timeout.Timeout.DEFAULT_TIMEOUT, pool_timeout=None, release_conn=None, **response_kw):
        ...

def monkeypatch(validate_certificate: bool = True):
    ...

def _check_version():
    ...
