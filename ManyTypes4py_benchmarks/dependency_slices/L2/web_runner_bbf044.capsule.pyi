from typing import Any

# === Internal dependency: aiohttp.abc ===
class AbstractAccessLogger(ABC): ...

# === Internal dependency: aiohttp.web_app ===
class Application(MutableMapping[Union[str, AppKey[Any]], Any]): ...

# === Internal dependency: aiohttp.web_log ===
class AccessLogger(AbstractAccessLogger): ...

# === Internal dependency: aiohttp.web_request ===
class BaseRequest(MutableMapping[str, Any], HeadersMixin): ...
class Request(BaseRequest): ...

# === Internal dependency: aiohttp.web_server ===
class Server(Generic[_Request]): ...

# === Third-party dependency: yarl ===
# Used symbols: URL