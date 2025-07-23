"""
Main Freqtrade worker class.
"""
import logging
import time
import traceback
from collections.abc import Callable
from os import getpid
from typing import Any, Optional
import sdnotify
from freqtrade import __version__
from freqtrade.configuration import Configuration
from freqtrade.constants import PROCESS_THROTTLE_SECS, RETRY_TIMEOUT, Config
from freqtrade.enums import RPCMessageType, State
from freqtrade.exceptions import OperationalException, TemporaryError
from freqtrade.exchange import timeframe_to_next_date
from freqtrade.freqtradebot import FreqtradeBot

logger = logging.getLogger(__name__)

class Worker:
    """
    Freqtradebot worker class
    """

    def __init__(self, args: Any, config: Optional[Config] = None) -> None:
        """
        Init all variables and objects the bot needs to work
        """
        logger.info(f'Starting worker {__version__}')
        self._args: Any = args
        self._config: Optional[Config] = config
        self._init(False)
        self._heartbeat_msg: float = 0
        self._notify('READY=1')

    def _init(self, reconfig: bool) -> None:
        """
        Also called from the _reconfigure() method (with reconfig=True).
        """
        if reconfig or self._config is None:
            self._config = Configuration(self._args, None).get_config()
        self.freqtrade: FreqtradeBot = FreqtradeBot(self._config)
        internals_config: dict = self._config.get('internals', {})
        self._throttle_secs: float = internals_config.get('process_throttle_secs', PROCESS_THROTTLE_SECS)
        self._heartbeat_interval: Optional[float] = internals_config.get('heartbeat_interval', 60)
        self._sd_notify: Optional[sdnotify.SystemdNotifier] = sdnotify.SystemdNotifier() if self._config.get('internals', {}).get('sd_notify', False) else None

    def _notify(self, message: str) -> None:
        """
        Removes the need to verify in all occurrences if sd_notify is enabled
        :param message: Message to send to systemd if it's enabled.
        """
        if self._sd_notify:
            logger.debug(f'sd_notify: {message}')
            self._sd_notify.notify(message)

    def run(self) -> None:
        state: Optional[State] = None
        while True:
            state = self._worker(old_state=state)
            if state == State.RELOAD_CONFIG:
                self._reconfigure()

    def _worker(self, old_state: Optional[State]) -> State:
        """
        The main routine that runs each throttling iteration and handles the states.
        :param old_state: the previous service state from the previous call
        :return: current service state
        """
        state: State = self.freqtrade.state
        if state != old_state:
            if old_state != State.RELOAD_CONFIG:
                self.freqtrade.notify_status(f'{state.name.lower()}')
            logger.info(f'Changing state{(f" from {old_state.name}" if old_state else "")} to: {state.name}')
            if state == State.RUNNING:
                self.freqtrade.startup()
            if state == State.STOPPED:
                self.freqtrade.check_for_open_trades()
            self._heartbeat_msg = 0
        if state == State.STOPPED:
            self._notify('WATCHDOG=1\nSTATUS=State: STOPPED.')
            self._throttle(func=self._process_stopped, throttle_secs=self._throttle_secs)
        elif state == State.RUNNING:
            self._notify('WATCHDOG=1\nSTATUS=State: RUNNING.')
            self._throttle(
                func=self._process_running,
                throttle_secs=self._throttle_secs,
                timeframe=self._config['timeframe'] if self._config else None,
                timeframe_offset=1
            )
        if self._heartbeat_interval:
            now: float = time.time()
            if now - self._heartbeat_msg > self._heartbeat_interval:
                version: str = __version__
                strategy_version: Optional[str] = self.freqtrade.strategy.version()
                if strategy_version is not None:
                    version += ', strategy_version: ' + strategy_version
                logger.info(f"Bot heartbeat. PID={getpid()}, version='{version}', state='{state.name}'")
                self._heartbeat_msg = now
        return state

    def _throttle(
        self,
        func: Callable[..., Any],
        throttle_secs: float,
        timeframe: Optional[str] = None,
        timeframe_offset: float = 1.0,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Throttles the given callable that it
        takes at least `min_secs` to finish execution.
        :param func: Any callable
        :param throttle_secs: throttling iteration execution time limit in seconds
        :param timeframe: ensure iteration is executed at the beginning of the next candle.
        :param timeframe_offset: offset in seconds to apply to the next candle time.
        :return: Any (result of execution of func)
        """
        last_throttle_start_time: float = time.time()
        logger.debug('========================================')
        result: Any = func(*args, **kwargs)
        time_passed: float = time.time() - last_throttle_start_time
        sleep_duration: float = throttle_secs - time_passed
        if timeframe:
            next_tf: Any = timeframe_to_next_date(timeframe)
            next_tft: float = next_tf.timestamp() - time.time()
            next_tf_with_offset: float = next_tft + timeframe_offset
            if next_tft < sleep_duration and sleep_duration < next_tf_with_offset:
                sleep_duration = next_tf_with_offset
            sleep_duration = min(sleep_duration, next_tf_with_offset)
        sleep_duration = max(sleep_duration, 0.0)
        logger.debug(f"Throttling with '{func.__name__}()': sleep for {sleep_duration:.2f} s, last iteration took {time_passed:.2f} s.")
        self._sleep(sleep_duration)
        return result

    @staticmethod
    def _sleep(sleep_duration: float) -> None:
        """Local sleep method - to improve testability"""
        time.sleep(sleep_duration)

    def _process_stopped(self) -> None:
        self.freqtrade.process_stopped()

    def _process_running(self) -> None:
        try:
            self.freqtrade.process()
        except TemporaryError as error:
            logger.warning(f'Error: {error}, retrying in {RETRY_TIMEOUT} seconds...')
            time.sleep(RETRY_TIMEOUT)
        except OperationalException:
            tb: str = traceback.format_exc()
            hint: str = 'Issue `/start` if you think it is safe to restart.'
            self.freqtrade.notify_status(
                f'*OperationalException:*\n