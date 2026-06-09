# === Unresolved dependency: PIL ===
# Used unresolved symbols: Image

# === Third-party dependency: flask ===
# Used symbols: current_app

# === Third-party dependency: flask_caching ===
class Cache: ...

# === Internal dependency: superset ===
from superset.extensions import feature_flag_manager
thumbnail_cache = LocalProxy(...)

# === Internal dependency: superset.exceptions ===
class ScreenshotImageNotAvailableException(SupersetException): ...

# === Internal dependency: superset.extensions ===
_event_logger = {}
event_logger = LocalProxy(...)

# === Internal dependency: superset.utils.hashing ===
def hash_from_dict(obj, ignore_nan=..., default=..., algorithm=...): ...

# === Internal dependency: superset.utils.urls ===
def modify_url_query(url, **kwargs): ...

# === Internal dependency: superset.utils.webdriver ===
class DashboardStandaloneMode(Enum): ...
class ChartStandaloneMode(Enum): ...
class WebDriverPlaywright(WebDriverProxy): ...
class WebDriverSelenium(WebDriverProxy): ...
from selenium.webdriver.remote.webdriver import WebDriver
WindowSize = tuple[int, int]

# === Third-party dependency: werkzeug.local ===
class LocalProxy(t.Generic[T]): ...