#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Navigator Watcher module."""

import asyncio
import concurrent.futures
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from pyppeteer import helper
from pyppeteer.errors import TimeoutError
from pyppeteer.frame_manager import FrameManager, Frame
from pyppeteer.util import merge_dict

pyppeteerToProtocolLifecycle: Dict[str, str] = {
    'load': 'load',
    'domcontentloaded': 'DOMContentLoaded',
    'documentloaded': 'DOMContentLoaded',
    'networkidle0': 'networkIdle',
    'networkidle2': 'networkAlmostIdle',
}


class NavigatorWatcher:
    """NavigatorWatcher class."""

    def __init__(
        self,
        frameManager: FrameManager,
        frame: Frame,
        timeout: int,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Make new navigator watcher."""
        options = merge_dict(options, kwargs)  # type: ignore
        self._validate_options(options)
        self._frameManager: FrameManager = frameManager
        self._frame: Frame = frame
        self._initialLoaderId: str = frame._loaderId
        self._timeout: int = timeout
        self._hasSameDocumentNavigation: bool = False
        self._eventListeners: List[Any] = [
            helper.addEventListener(
                self._frameManager,
                FrameManager.Events.LifecycleEvent,
                self._checkLifecycleComplete,
            ),
            helper.addEventListener(
                self._frameManager,
                FrameManager.Events.FrameNavigatedWithinDocument,
                self._navigatedWithinDocument,
            ),
            helper.addEventListener(
                self._frameManager,
                FrameManager.Events.FrameDetached,
                self._checkLifecycleComplete,
            ),
        ]
        self._loop: asyncio.AbstractEventLoop = self._frameManager._client._loop
        self._lifecycleCompletePromise: asyncio.Future[None] = self._loop.create_future()

        timeout_promise: Awaitable[None] = self._createTimeoutPromise()
        self._navigationPromise: asyncio.Task[Any] = self._loop.create_task(
            asyncio.wait(
                [self._lifecycleCompletePromise, timeout_promise],
                return_when=concurrent.futures.FIRST_COMPLETED
            )
        )
        self._navigationPromise.add_done_callback(lambda fut: self._cleanup())

        # Expected lifecycle events will be set in _validate_options.
        self._expectedLifecycle: List[str] = []

    def _validate_options(self, options: Dict[str, Any]) -> None:  # noqa: C901
        if 'networkIdleTimeout' in options:
            raise ValueError('`networkIdleTimeout` option is no longer supported.')
        if 'networkIdleInflight' in options:
            raise ValueError('`networkIdleInflight` option is no longer supported.')
        if options.get('waitUntil') == 'networkidle':
            raise ValueError(
                '`networkidle` option is no logner supported. '
                'Use `networkidle2` instead.'
            )
        if options.get('waitUntil') == 'documentloaded':
            logging.getLogger(__name__).warning(
                '`documentloaded` option is no longer supported. '
                'Use `domcontentloaded` instead.'
            )
        _waitUntil: Union[str, List[str]] = options.get('waitUntil', 'load')
        if isinstance(_waitUntil, list):
            waitUntil: List[str] = _waitUntil
        elif isinstance(_waitUntil, str):
            waitUntil = [_waitUntil]
        else:
            raise TypeError(
                '`waitUntil` option should be str or list of str, '
                f'but got type {type(_waitUntil)}'
            )
        self._expectedLifecycle = []  # type: List[str]
        for value in waitUntil:
            protocolEvent: Optional[str] = pyppeteerToProtocolLifecycle.get(value)
            if protocolEvent is None:
                raise ValueError(f'Unknown value for options.waitUntil: {value}')
            self._expectedLifecycle.append(protocolEvent)

    def _createTimeoutPromise(self) -> Awaitable[None]:
        self._maximumTimer: asyncio.Future[None] = self._loop.create_future()
        if self._timeout:
            errorMessage: str = f'Navigation Timeout Exceeded: {self._timeout} ms exceeded.'

            async def _timeout_func() -> None:
                await asyncio.sleep(self._timeout / 1000)
                if not self._maximumTimer.done():
                    self._maximumTimer.set_exception(TimeoutError(errorMessage))

            self._timeout_timer: Union[asyncio.Task[Any], asyncio.Future[Any]] = self._loop.create_task(_timeout_func())
        else:
            self._timeout_timer = self._loop.create_future()
        return self._maximumTimer

    def navigationPromise(self) -> asyncio.Task[Any]:
        """Return navigation promise."""
        return self._navigationPromise

    def _navigatedWithinDocument(self, frame: Optional[Frame] = None) -> None:
        if frame != self._frame:
            return
        self._hasSameDocumentNavigation = True
        self._checkLifecycleComplete()

    def _checkLifecycleComplete(self, frame: Optional[Frame] = None) -> None:
        if (self._frame._loaderId == self._initialLoaderId and
                not self._hasSameDocumentNavigation):
            return
        if not self._checkLifecycle(self._frame, self._expectedLifecycle):
            return

        if not self._lifecycleCompletePromise.done():
            self._lifecycleCompletePromise.set_result(None)

    def _checkLifecycle(self, frame: Frame, expectedLifecycle: List[str]) -> bool:
        for event in expectedLifecycle:
            if event not in frame._lifecycleEvents:
                return False
        for child in frame.childFrames:
            if not self._checkLifecycle(child, expectedLifecycle):
                return False
        return True

    def cancel(self) -> None:
        """Cancel navigation."""
        self._cleanup()

    def _cleanup(self) -> None:
        helper.removeEventListeners(self._eventListeners)
        self._lifecycleCompletePromise.cancel()
        self._maximumTimer.cancel()
        self._timeout_timer.cancel()