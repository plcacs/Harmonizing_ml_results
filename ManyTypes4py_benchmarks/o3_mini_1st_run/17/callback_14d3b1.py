import sys
import uuid
import logging
from threading import RLock
from functools import partial
import sublime
from ..anaconda_lib import aenum as enum
from ._typing import Callable, Any, Union

from typing import Optional, Dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)


@enum.unique
class CallbackStatus(enum.Enum):
    unfired = 'unfired'
    succeeded = 'succeeded'
    failed = 'failed'
    timed_out = 'timed_out'


class Callback(object):
    def __init__(self, 
                 on_success: Optional[Callable[..., Any]] = None, 
                 on_failure: Optional[Callable[..., Any]] = None, 
                 on_timeout: Optional[Callable[..., Any]] = None, 
                 timeout: Union[int, float] = 0) -> None:
        self._lock: RLock = RLock()
        self._timeout: Union[int, float] = 0
        self.uid: uuid.UUID = uuid.uuid4()
        self.waiting_for_timeout: bool = False
        self._status: CallbackStatus = CallbackStatus.unfired
        self.callbacks: Dict[str, Optional[Callable[..., Any]]] = {
            'succeeded': on_success,
            'failed': on_failure
        }
        if on_timeout is not None and callable(on_timeout):
            self.callbacks['timed_out'] = on_timeout
        self.timeout = timeout
        if on_timeout is not None and timeout > 0:
            self.initialize_timeout()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._infere_status_from_data(*args, **kwargs)
            return self._fire_callback(*args, **kwargs)

    @property
    def id(self) -> uuid.UUID:
        return self.uid

    @property
    def hexid(self) -> str:
        return self.uid.hex

    @property
    def timeout(self) -> Union[int, float]:
        return self._timeout

    @timeout.setter
    def timeout(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise RuntimeError('Callback.timeout must be integer or float!')
        self._timeout = value

    @property
    def status(self) -> CallbackStatus:
        return self._status

    @status.setter
    def status(self, status: Union[CallbackStatus, str]) -> None:
        with self._lock:
            if self._status != CallbackStatus.unfired:
                if self._status != CallbackStatus.timed_out:
                    raise RuntimeError('Callback {} already fired!'.format(self.hexid))
                else:
                    logger.info("Calback {} came back with data but it's status was `timed_out` already".format(self.hexid))
                    return
            if isinstance(status, CallbackStatus):
                self._status = status
            else:
                temp_status: Optional[CallbackStatus] = CallbackStatus._member_map_.get(status)
                if temp_status is not None:
                    self._status = temp_status
                else:
                    raise RuntimeError('Status {} does not exists!'.format(status))

    def initialize_timeout(self) -> None:
        def _timeout_callback(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError('Timeout occurred on {}'.format(self.hexid))

        def _on_timeout(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
            if self._status is CallbackStatus.unfired:
                self.status = CallbackStatus.timed_out
                func(*args, **kwargs)
        if self.timeout > 0:
            self.waiting_for_timeout = True
            callback: Callable[..., Any] = self.callbacks.get('timed_out', _timeout_callback)  # type: ignore
            sublime.set_timeout(partial(_on_timeout, callback), int(self.timeout * 1000))

    def on(self, 
           success: Optional[Callable[..., Any]] = None, 
           error: Optional[Callable[..., Any]] = None, 
           timeout: Optional[Callable[..., Any]] = None) -> None:
        if success is not None and self.callbacks.get('succeeded') is None:
            self.callbacks['succeeded'] = success
        if error is not None and self.callbacks.get('failed') is None:
            self.callbacks['failed'] = error
        if timeout is not None and self.callbacks.get('timed_out') is None:
            if callable(timeout):
                self.callbacks['timed_out'] = timeout
                self.initialize_timeout()

    def _infere_status_from_data(self, *args: Any, **kwargs: Any) -> None:
        data: Any = kwargs.get('data') or (args[0] if args else {})
        if isinstance(data, dict):
            if 'status' in data:
                self.status = data['status']
            elif 'success' in data:
                smap = {True: 'succeeded', False: 'failed'}
                self.status = smap[data['success']]
            else:
                self.status = 'succeeded'
        else:
            self.status = 'succeeded'

    def _fire_callback(self, *args: Any, **kwargs: Any) -> Any:
        def _panic(*args: Any, **kwargs: Any) -> Any:
            if self.status is CallbackStatus.failed:
                callback: Optional[Callable[..., Any]] = self.callbacks.get('succeeded')
                if callback is not None:
                    return callback(*args, **kwargs)
            raise RuntimeError('We tried to call non existing callback {}!'.format(self.status.value))
        callback: Optional[Callable[..., Any]] = self.callbacks.get(self._status.value, _panic)
        return callback(*args, **kwargs) if callback else None