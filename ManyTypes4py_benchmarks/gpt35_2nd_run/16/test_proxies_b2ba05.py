from typing import Optional, Dict, Any, List, Tuple, Union

def url_to_origin(url: str) -> httpcore.URL:
    ...

def test_socks_proxy():
    ...

PROXY_URL: str = 'http://[::1]'

def test_transport_for_request(url: str, proxies: Dict[str, str], expected: Optional[str]):
    ...

async def test_async_proxy_close():
    ...

def test_sync_proxy_close():
    ...

def test_unsupported_proxy_scheme():
    ...

def test_proxies_environ(monkeypatch, client_class, url: str, env: Dict[str, str], expected: Optional[str]):
    ...

def test_for_deprecated_proxy_params(proxies: Dict[str, str], is_valid: bool):
    ...

def test_proxy_with_mounts():
    ...
