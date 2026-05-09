from __future__ import annotations
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from PIL import Image
from flask_caching import Cache
from superset.dashboards.permalink.types import DashboardPermalinkState
from flask_appbuilder.security.sqla.models import User

DEFAULT_SCREENSHOT_WINDOW_SIZE: Tuple[int, int]
DEFAULT_SCREENSHOT_THUMBNAIL_SIZE: Tuple[int, int]
DEFAULT_CHART_WINDOW_SIZE: Tuple[int, int]
DEFAULT_CHART_THUMBNAIL_SIZE: Tuple[int, int]
DEFAULT_DASHBOARD_WINDOW_SIZE: Tuple[int, int]
DEFAULT_DASHBOARD_THUMBNAIL_SIZE: Tuple[int, int]

class StatusValues(Enum):
    PENDING: str
    COMPUTING: str
    UPDATED: str
    ERROR: str

class ScreenshotCachePayloadType(TypedDict):
    ...

class ScreenshotCachePayload:
    def __init__(self, image: Optional[bytes] = None, status: StatusValues = StatusValues.PENDING, timestamp: str = ''):
        ...

    @classmethod
    def from_dict(cls, payload: ScreenshotCachePayloadType) -> ScreenshotCachePayload:
        ...

    def to_dict(self) -> Dict[str, Union[bytes, str]]:
        ...

    def update_timestamp(self) -> None:
        ...

    def pending(self) -> None:
        ...

    def computing(self) -> None:
        ...

    def update(self, image: bytes) -> None:
        ...

    def error(self) -> None:
        ...

    def get_image(self) -> Optional[BytesIO]:
        ...

    def get_timestamp(self) -> str:
        ...

    def get_status(self) -> str:
        ...

    def is_error_cache_ttl_expired(self) -> bool:
        ...

    def should_trigger_task(self, force: bool = False) -> bool:
        ...

class BaseScreenshot:
    driver_type: str
    thumbnail_type: str
    element: str
    window_size: Tuple[int, int]
    thumb_size: Tuple[int, int]
    cache: Cache

    def __init__(self, url: str, digest: str) -> None:
        ...

    def driver(self, window_size: Optional[Tuple[int, int]] = None) -> Any:
        ...

    def get_screenshot(self, user: Optional[User] = None, window_size: Optional[Tuple[int, int]] = None) -> bytes:
        ...

    def get_cache_key(self, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None) -> str:
        ...

    def get_from_cache(self, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None) -> Optional[ScreenshotCachePayload]:
        ...

    @classmethod
    def get_from_cache_key(cls, cache_key: str) -> Optional[ScreenshotCachePayload]:
        ...

    def compute_and_cache(self, force: bool, user: Optional[User] = None, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None, cache_key: Optional[str] = None) -> None:
        ...

    @classmethod
    def resize_image(cls, img_bytes: bytes, output: str = 'png', thumb_size: Optional[Tuple[int, int]] = None, crop: bool = True) -> bytes:
        ...

class ChartScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str

    def __init__(self, url: str, digest: str, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None) -> None:
        ...

class DashboardScreenshot(BaseScreenshot):
    thumbnail_type: str
    element: str

    def __init__(self, url: str, digest: str, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None) -> None:
        ...

    def get_cache_key(self, window_size: Optional[Tuple[int, int]] = None, thumb_size: Optional[Tuple[int, int]] = None, dashboard_state: Optional[DashboardPermalinkState] = None) -> str:
        ...