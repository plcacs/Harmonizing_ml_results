import asyncio
import concurrent.futures
from typing import Any, Awaitable, Dict, List, Union
from pyppeteer import helper
from pyppeteer.errors import TimeoutError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.util import merge_dict

class NavigatorWatcher:
    def __init__(self, frameManager: FrameManager, frame: Frame, timeout: int, options: Dict[str, Any] = None, **kwargs: Any) -> None:
    def _validate_options(self, options: Dict[str, Any]) -> None:
    def _createTimeoutPromise(self) -> Awaitable[None]:
    def navigationPromise(self) -> Awaitable[None]:
    def _navigatedWithinDocument(self, frame: Frame = None) -> None:
    def _checkLifecycleComplete(self, frame: Frame = None) -> None:
    def _checkLifecycle(self, frame: Frame, expectedLifecycle: List[str]) -> bool:
    def cancel(self) -> None:
    def _cleanup(self) -> None:
