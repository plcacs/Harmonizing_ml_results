from typing import Any

# === Unresolved dependency: PIL ===
# Used unresolved symbols: Image

# === Third-party dependency: flask ===
# Used symbols: current_app

# === Third-party dependency: flask_caching ===
class Cache: ...

# === Internal dependency: superset ===
# re-export: from superset.extensions import feature_flag_manager
thumbnail_cache: LocalProxy

# === Internal dependency: superset.exceptions ===
class ScreenshotImageNotAvailableException(SupersetException): ...

# === Internal dependency: superset.extensions ===
event_logger: LocalProxy

# === Internal dependency: superset.utils.hashing ===
def hash_from_dict(obj: dict[Any, Any], ignore_nan: bool = ..., default: Optional[Callable[[Any], Any]] = ..., algorithm: Optional[HashAlgorithm] = ...) -> str: ...

# === Internal dependency: superset.utils.urls ===
def modify_url_query(url: str, **kwargs: Any) -> str: ...

# === Internal dependency: superset.utils.webdriver ===
class DashboardStandaloneMode(Enum): ...
class ChartStandaloneMode(Enum): ...
class WebDriverPlaywright(WebDriverProxy): ...
class WebDriverSelenium(WebDriverProxy): ...
# re-export: from selenium.webdriver.remote.webdriver import WebDriver
WindowSize: Any

# === Third-party dependency: werkzeug.local ===
class LocalProxy(t.Generic[T]): ...