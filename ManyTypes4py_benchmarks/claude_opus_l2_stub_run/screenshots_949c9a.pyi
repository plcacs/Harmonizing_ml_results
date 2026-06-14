from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, TypedDict

from superset.dashboards.permalink.types import DashboardPermalinkState
from superset.utils.webdriver import (
    ChartStandaloneMode,
    DashboardStandaloneMode,
    WebDriver,
    WebDriverPlaywright,
    WebDriverSelenium,
    WindowSize,
)

if TYPE_CHECKING:
    from flask_appbuilder.security.sqla.models import User
    from flask_caching import Cache

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

    def __init__(
        self,
        image: bytes | None = ...,
        status: StatusValues = ...,
        timestamp: str = ...,
    ) -> None: ...
    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ScreenshotCachePayload: ...
    def to_dict(self) -> dict[str, object]: ...
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
    cache: Cache
    digest: str
    url: str
    screenshot: bytes | None

    def __init__(self, url: str, digest: str) -> None: ...
    def driver(
        self, window_size: WindowSize | None = ...
    ) -> WebDriverPlaywright | WebDriverSelenium: ...
    def get_screenshot(
        self, user: User | None = ..., window_size: WindowSize | None = ...
    ) -> bytes | None: ...
    def get_cache_key(
        self,
        window_size: WindowSize | None = ...,
        thumb_size: tuple[int, int] | None = ...,
    ) -> str: ...
    def get_from_cache(
        self,
        window_size: WindowSize | None = ...,
        thumb_size: tuple[int, int] | None = ...,
    ) -> ScreenshotCachePayload | None: ...
    @classmethod
    def get_from_cache_key(cls, cache_key: str) -> ScreenshotCachePayload | None: ...
    def compute_and_cache(
        self,
        force: bool,
        user: User | None = ...,
        window_size: WindowSize | None = ...,
        thumb_size: tuple[int, int] | None = ...,
        cache_key: str | None = ...,
    ) -> None: ...
    @classmethod
    def resize_image(
        cls,
        img_bytes: bytes,
        output: str = ...,
        thumb_size: tuple[int, int] | None = ...,
        crop: bool = ...,
    ) -> bytes: ...

class ChartScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str

    def __init__(
        self,
        url: str,
        digest: str,
        window_size: tuple[int, int] | None = ...,
        thumb_size: tuple[int, int] | None = ...,
    ) -> None: ...

class DashboardScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str

    def __init__(
        self,
        url: str,
        digest: str,
        window_size: tuple[int, int] | None = ...,
        thumb_size: tuple[int, int] | None = ...,
    ) -> None: ...
    def get_cache_key(  # type: ignore[override]
        self,
        window_size: WindowSize | None = ...,
        thumb_size: tuple[int, int] | None = ...,
        dashboard_state: DashboardPermalinkState | None = ...,
    ) -> str: ...