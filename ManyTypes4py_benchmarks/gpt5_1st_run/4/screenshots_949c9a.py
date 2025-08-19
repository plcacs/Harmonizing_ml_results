from __future__ import annotations
import logging
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, ClassVar, Optional, Tuple, cast, TYPE_CHECKING, TypedDict
from flask import current_app
from superset import app, feature_flag_manager, thumbnail_cache
from superset.dashboards.permalink.types import DashboardPermalinkState
from superset.extensions import event_logger
from superset.utils.hashing import md5_sha_from_dict
from superset.utils.urls import modify_url_query
from superset.utils.webdriver import ChartStandaloneMode, DashboardStandaloneMode, WebDriver, WebDriverPlaywright, WebDriverSelenium, WindowSize

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_SCREENSHOT_WINDOW_SIZE: Tuple[int, int] = (800, 600)
DEFAULT_SCREENSHOT_THUMBNAIL_SIZE: Tuple[int, int] = (400, 300)
DEFAULT_CHART_WINDOW_SIZE: Tuple[int, int] = (800, 600)
DEFAULT_CHART_THUMBNAIL_SIZE: Tuple[int, int] = (800, 600)
DEFAULT_DASHBOARD_WINDOW_SIZE: Tuple[int, int] = (1600, 1200)
DEFAULT_DASHBOARD_THUMBNAIL_SIZE: Tuple[int, int] = (800, 600)

try:
    from PIL import Image
except ModuleNotFoundError:
    logger.info('No PIL installation found')

if TYPE_CHECKING:
    from flask_appbuilder.security.sqla.models import User
    from flask_caching import Cache


class StatusValues(Enum):
    PENDING = 'Pending'
    COMPUTING = 'Computing'
    UPDATED = 'Updated'
    ERROR = 'Error'


class ScreenshotCachePayloadType(TypedDict):
    image: Optional[bytes]
    timestamp: str
    status: str


class ScreenshotCachePayload:
    _image: Optional[bytes]
    _timestamp: str
    status: StatusValues

    def __init__(
        self,
        image: Optional[bytes] = None,
        status: StatusValues = StatusValues.PENDING,
        timestamp: str = '',
    ) -> None:
        self._image = image
        self._timestamp = timestamp or datetime.now().isoformat()
        self.status = StatusValues.UPDATED if image else status

    @classmethod
    def from_dict(cls, payload: ScreenshotCachePayloadType) -> ScreenshotCachePayload:
        return cls(
            image=payload['image'],
            status=StatusValues(payload['status']),
            timestamp=payload['timestamp'],
        )

    def to_dict(self) -> ScreenshotCachePayloadType:
        return {'image': self._image, 'timestamp': self._timestamp, 'status': self.status.value}

    def update_timestamp(self) -> None:
        self._timestamp = datetime.now().isoformat()

    def pending(self) -> None:
        self.update_timestamp()
        self._image = None
        self.status = StatusValues.PENDING

    def computing(self) -> None:
        self.update_timestamp()
        self._image = None
        self.status = StatusValues.COMPUTING

    def update(self, image: bytes) -> None:
        self.update_timestamp()
        self.status = StatusValues.UPDATED
        self._image = image

    def error(self) -> None:
        self.update_timestamp()
        self.status = StatusValues.ERROR

    def get_image(self) -> Optional[BytesIO]:
        if not self._image:
            return None
        return BytesIO(self._image)

    def get_timestamp(self) -> str:
        return self._timestamp

    def get_status(self) -> str:
        return self.status.value

    def is_error_cache_ttl_expired(self) -> bool:
        error_cache_ttl = app.config['THUMBNAIL_ERROR_CACHE_TTL']
        return (datetime.now() - datetime.fromisoformat(self.get_timestamp())).total_seconds() > error_cache_ttl

    def should_trigger_task(self, force: bool = False) -> bool:
        return (
            force
            or self.status == StatusValues.PENDING
            or (self.status == StatusValues.ERROR and self.is_error_cache_ttl_expired())
        )


class BaseScreenshot:
    driver_type: ClassVar[str] = cast(str, current_app.config['WEBDRIVER_TYPE'])
    thumbnail_type: ClassVar[str] = ''
    element: ClassVar[str] = ''
    window_size: Tuple[int, int] = DEFAULT_SCREENSHOT_WINDOW_SIZE
    thumb_size: Tuple[int, int] = DEFAULT_SCREENSHOT_THUMBNAIL_SIZE
    cache: ClassVar['Cache'] = thumbnail_cache

    def __init__(self, url: str, digest: str) -> None:
        self.digest: str = digest
        self.url: str = url
        self.screenshot: Optional[bytes] = None

    def driver(self, window_size: Optional[Tuple[int, int]] = None) -> WebDriver:
        window_size = window_size or self.window_size
        if feature_flag_manager.is_feature_enabled('PLAYWRIGHT_REPORTS_AND_THUMBNAILS'):
            return WebDriverPlaywright(self.driver_type, window_size)
        return WebDriverSelenium(self.driver_type, window_size)

    def get_screenshot(self, user: Optional['User'], window_size: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
        driver = self.driver(window_size)
        self.screenshot = driver.get_screenshot(self.url, self.element, user)
        return self.screenshot

    def get_cache_key(
        self,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
    ) -> str:
        window_size = window_size or self.window_size
        thumb_size = thumb_size or self.thumb_size
        args: dict[str, Any] = {
            'thumbnail_type': self.thumbnail_type,
            'digest': self.digest,
            'type': 'thumb',
            'window_size': window_size,
            'thumb_size': thumb_size,
        }
        return md5_sha_from_dict(args)

    def get_from_cache(
        self,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[ScreenshotCachePayload]:
        cache_key = self.get_cache_key(window_size, thumb_size)
        return self.get_from_cache_key(cache_key)

    @classmethod
    def get_from_cache_key(cls, cache_key: str) -> Optional[ScreenshotCachePayload]:
        logger.info('Attempting to get from cache: %s', cache_key)
        if (payload := cls.cache.get(cache_key)):
            if isinstance(payload, bytes):
                payload = ScreenshotCachePayload(payload)
            elif isinstance(payload, ScreenshotCachePayload):
                pass
            elif isinstance(payload, dict):
                payload = cast(ScreenshotCachePayloadType, payload)
                payload = ScreenshotCachePayload.from_dict(payload)
            return cast(ScreenshotCachePayload, payload)
        logger.info('Failed at getting from cache: %s', cache_key)
        return None

    def compute_and_cache(
        self,
        force: bool,
        user: Optional['User'] = None,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        """
        Computes the thumbnail and caches the result

        :param user: If no user is given will use the current context
        :param cache: The cache to keep the thumbnail payload
        :param window_size: The window size from which will process the thumb
        :param thumb_size: The final thumbnail size
        :param force: Will force the computation even if it's already cached
        :return: Image payload
        """
        cache_key = cache_key or self.get_cache_key(window_size, thumb_size)
        cache_payload = self.get_from_cache_key(cache_key) or ScreenshotCachePayload()
        if cache_payload.status in [StatusValues.COMPUTING, StatusValues.UPDATED] and (not force):
            logger.info('Skipping compute - already processed for thumbnail: %s', cache_key)
            return
        window_size = window_size or self.window_size
        thumb_size = thumb_size or self.thumb_size
        logger.info('Processing url for thumbnail: %s', cache_key)
        cache_payload.computing()
        self.cache.set(cache_key, cache_payload.to_dict())
        image: Optional[bytes] = None
        try:
            logger.info('trying to generate screenshot')
            with event_logger.log_context(f'screenshot.compute.{self.thumbnail_type}'):
                image = self.get_screenshot(user=user, window_size=window_size)
        except Exception as ex:
            logger.warning('Failed at generating thumbnail %s', ex, exc_info=True)
            cache_payload.error()
        if image and window_size != thumb_size:
            try:
                image = self.resize_image(image, thumb_size=thumb_size)
            except Exception as ex:
                logger.warning('Failed at resizing thumbnail %s', ex, exc_info=True)
                cache_payload.error()
                image = None
        if image:
            logger.info('Caching thumbnail: %s', cache_key)
            with event_logger.log_context(f'screenshot.cache.{self.thumbnail_type}'):
                cache_payload.update(image)
        self.cache.set(cache_key, cache_payload.to_dict())
        logger.info('Updated thumbnail cache; Status: %s', cache_payload.get_status())
        return

    @classmethod
    def resize_image(
        cls,
        img_bytes: bytes,
        output: str = 'png',
        thumb_size: Optional[Tuple[int, int]] = None,
        crop: bool = True,
    ) -> bytes:
        thumb_size = thumb_size or cls.thumb_size
        img = Image.open(BytesIO(img_bytes))
        logger.debug('Selenium image size: %s', str(img.size))
        if crop and img.size[1] != cls.window_size[1]:
            desired_ratio = float(cls.window_size[1]) / cls.window_size[0]
            desired_width = int(img.size[0] * desired_ratio)
            logger.debug('Cropping to: %s*%s', str(img.size[0]), str(desired_width))
            img = img.crop((0, 0, img.size[0], desired_width))
        logger.debug('Resizing to %s', str(thumb_size))
        img = img.resize(thumb_size, Image.Resampling.LANCZOS)
        new_img = BytesIO()
        if output != 'png':
            img = img.convert('RGB')
        img.save(new_img, output)
        new_img.seek(0)
        return new_img.read()


class ChartScreenshot(BaseScreenshot):
    thumbnail_type: ClassVar[str] = 'chart'
    element: ClassVar[str] = 'chart-container'

    def __init__(
        self,
        url: str,
        digest: str,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        url = modify_url_query(url, standalone=ChartStandaloneMode.HIDE_NAV.value)
        super().__init__(url, digest)
        self.window_size = window_size or DEFAULT_CHART_WINDOW_SIZE
        self.thumb_size = thumb_size or DEFAULT_CHART_THUMBNAIL_SIZE


class DashboardScreenshot(BaseScreenshot):
    thumbnail_type: ClassVar[str] = 'dashboard'
    element: ClassVar[str] = 'standalone'

    def __init__(
        self,
        url: str,
        digest: str,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        url = modify_url_query(url, standalone=DashboardStandaloneMode.REPORT.value)
        super().__init__(url, digest)
        self.window_size = window_size or DEFAULT_DASHBOARD_WINDOW_SIZE
        self.thumb_size = thumb_size or DEFAULT_DASHBOARD_THUMBNAIL_SIZE

    def get_cache_key(
        self,
        window_size: Optional[Tuple[int, int]] = None,
        thumb_size: Optional[Tuple[int, int]] = None,
        dashboard_state: Optional[DashboardPermalinkState] = None,
    ) -> str:
        window_size = window_size or self.window_size
        thumb_size = thumb_size or self.thumb_size
        args: dict[str, Any] = {
            'thumbnail_type': self.thumbnail_type,
            'digest': self.digest,
            'type': 'thumb',
            'window_size': window_size,
            'thumb_size': thumb_size,
            'dashboard_state': dashboard_state,
        }
        return md5_sha_from_dict(args)