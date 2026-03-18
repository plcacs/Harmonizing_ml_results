```pyi
from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, TypedDict, cast

from flask import Flask
from superset.dashboards.permalink.types import DashboardPermalinkState
from superset.extensions import EventLogger
from superset.utils.webdriver import ChartStandaloneMode, DashboardStandaloneMode, WebDriver, WebDriverPlaywright, WebDriverSelenium, WindowSize

logger: logging.Logger

DEFAULT_SCREENSHOT_WINDOW_SIZE: tuple[int, int]
DEFAULT_SCREENSHOT_THUMBNAIL_SIZE: tuple[int, int]
DEFAULT_CHART_WINDOW_SIZE: tuple[int, int]
DEFAULT_CHART_THUMBNAIL_SIZE: tuple[int, int]
DEFAULT_DASHBOARD_WINDOW_SIZE: tuple[int, int]
DEFAULT_DASHBOARD_THUMBNAIL_SIZE: tuple[int, int]

class StatusValues(Enum):
    PENDING: str
    COMPUTING: str
    UPDATED: str
    ERROR: str

class ScreenshotCachePayloadType(TypedDict):
    pass

class ScreenshotCachePayload:
    _image: bytes | None
    _timestamp: str
    status: StatusValues
    def __init__(self, image: bytes | None = None, status: StatusValues = ..., timestamp: str = ...) -> None: ...
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScreenshotCachePayload: ...
    def to_dict(self) -> dict[str, Any]: ...
    def update_timestamp(self) -> None: ...
    def pending(self) -> None: ...
    def computing(self) -> None: ...
    def update(self, image: bytes) -> None: ...
    def error(self) -> None: ...
    def get_image(self) -> BytesIO | None: ...
    def get_timestamp(self) -> str: ...
    def get_status(self) -> str: ...
    def is_error_cache_ttl_expired(self) -> bool: ...
    def should_trigger_task(self, force: bool = ...) -> bool: ...

class BaseScreenshot:
    driver_type: str
    thumbnail_type: str
    element: str
    window_size: tuple[int, int]
    thumb_size: tuple[int, int]
    cache: Any
    digest: str
    url: str
    screenshot: bytes | None
    def __init__(self, url: str, digest: str) -> None: ...
    def driver(self, window_size: tuple[int, int] | None = None) -> WebDriver: ...
    def get_screenshot(self, user: Any = None, window_size: tuple[int, int] | None = None) -> bytes: ...
    def get_cache_key(self, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None) -> str: ...
    def get_from_cache(self, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None) -> ScreenshotCachePayload | None: ...
    @classmethod
    def get_from_cache_key(cls, cache_key: str) -> ScreenshotCachePayload | None: ...
    def compute_and_cache(self, force: bool, user: Any = None, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None, cache_key: str | None = None) -> None: ...
    @classmethod
    def resize_image(cls, img_bytes: bytes, output: str = ..., thumb_size: tuple[int, int] | None = None, crop: bool = ...) -> bytes: ...

class ChartScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str
    def __init__(self, url: str, digest: str, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None) -> None: ...

class DashboardScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str
    def __init__(self, url: str, digest: str, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None) -> None: ...
    def get_cache_key(self, window_size: tuple[int, int] | None = None, thumb_size: tuple[int, int] | None = None, dashboard_state: DashboardPermalinkState | None = None) -> str: ...
```