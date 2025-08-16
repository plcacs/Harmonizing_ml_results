import os
import threading
import time
from http import HTTPStatus
from typing import Any, Dict, Iterator, NamedTuple, Optional, Set, Tuple, Type, Union
import pytest
from _pytest.assertion.util import _compare_eq_dict, _compare_eq_set
from aiohttp.client import ClientError, ClientSession
from aiohttp.web import Response
from mode.utils.futures import all_tasks
from mode.utils.mocks import (
    AsyncContextManagerMock,
    AsyncMock,
    MagicMock,
    Mock,
    patch,
)
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from pytest import FixtureRequest as PytestFixtureRequest

sentinel = object()


class DirtyTest(Exception):
    ...


@pytest.fixture()
def patching(monkeypatch: MonkeyPatch, request: FixtureRequest) -> '_patching':
    """Monkeypath.setattr shortcut.

    Example:
        .. sourcecode:: python

        def test_foo(patching):
            # execv value here will be mock.MagicMock by default.
            execv = patching('os.execv')

            patching('sys.platform', 'darwin')  # set concrete value
            patching.setenv('DJANGO_SETTINGS_MODULE', 'x.settings')

            # val will be of type mock.MagicMock by default
            val = patching.setitem('path.to.dict', 'KEY')
    """
    return _patching(monkeypatch, request)


@pytest.fixture()
def loop(event_loop: Any) -> Any:
    return event_loop


class _patching(object):

    def __init__(self, monkeypatch: MonkeyPatch, request: FixtureRequest) -> None:
        self.monkeypatch = monkeypatch
        self.request = request

    def __getattr__(self, name: str) -> Any:
        return getattr(self.monkeypatch, name)

    def __call__(
        self,
        path: str,
        value: Any = sentinel,
        name: Optional[str] = None,
        new: Type[Any] = MagicMock,
        **kwargs: Any
    ) -> Any:
        value = self._value_or_mock(value, new, name, path, **kwargs)
        self.monkeypatch.setattr(path, value)
        return value

    def _value_or_mock(
        self,
        value: Any,
        new: Type[Any],
        name: Optional[str],
        path: str,
        **kwargs: Any
    ) -> Any:
        if value is sentinel:
            value = new(name=name or path.rpartition('.')[2])
        for k, v in kwargs.items():
            setattr(value, k, v)
        return value

    def setattr(
        self,
        target: str,
        name: Any = sentinel,
        value: Any = sentinel,
        **kwargs: Any
    ) -> Any:
        # alias to __call__ with the interface of pytest.monkeypatch.setattr
        if value is sentinel:
            value, name = name, None
        return self(target, value, name=name, **kwargs)

    def setitem(
        self,
        dic: Dict[str, Any],
        name: str,
        value: Any = sentinel,
        new: Type[Any] = MagicMock,
        **kwargs: Any
    ) -> Any:
        # same as pytest.monkeypatch.setattr but default value is MagicMock
        value = self._value_or_mock(value, new, name, dic, **kwargs)
        self.monkeypatch.setitem(dic, name, value)
        return value


class TimeMarks(NamedTuple):
    time: float = None
    monotonic: float = None


@pytest.yield_fixture()
def freeze_time(event_loop: Any, request: FixtureRequest) -> Iterator[TimeMarks]:
    marks = request.node.get_closest_marker('time')
    timestamp = time.time()
    monotimestamp = time.monotonic()

    with patch('time.time') as time_:
        with patch('time.monotonic') as monotonic_:
            options = TimeMarks(**{
                **{'time': timestamp,
                   'monotonic': monotimestamp},
                **((marks.kwargs or {}) if marks else {}),
            })
            time_.return_value = options.time
            monotonic_.return_value = options.monotonic
            yield options


class SessionMarker(NamedTuple):
    status_code: int
    text: bytes
    json: Any
    json_iterator: Any
    max_failures: Optional[int]


@pytest.fixture()
def mock_http_client(
    *,
    app: Any,
    monkeypatch: MonkeyPatch,
    request: FixtureRequest,
) -> ClientSession:
    marker = request.node.get_closest_marker('http_session')
    options = SessionMarker(**{
        **{
            'status_code': HTTPStatus.OK,
            'text': b'',
            'json': None,
            'json_iterator': None,
            'max_failures': None,
        },
        **(marker.kwargs or {} if marker else {}),
    })

    def raise_for_status() -> None:
        if options.max_failures is not None:
            if session.request.call_count >= options.max_failures:
                return
        if 400 <= options.status_code:
            raise ClientError()

    response = AsyncMock(
        autospec=Response,
        text=AsyncMock(return_value=options.text),
        read=AsyncMock(return_value=options.text),
        json=AsyncMock(
            return_value=options.json,
            side_effect=options.json_iterator,
        ),
        status_code=options.status_code,
        raise_for_status=raise_for_status,
    )
    session = Mock(
        name='http_client',
        autospec=ClientSession,
        request=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        get=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        post=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        put=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        delete=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        patch=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        options=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
        head=Mock(
            return_value=AsyncContextManagerMock(
                return_value=response,
            ),
        ),
    )
    session.marks = options
    monkeypatch.setattr(app, '_http_client', session)
    return session


@pytest.fixture(scope='session', autouse=True)
def _collected_environ() -> Dict[str, str]:
    return dict(os.environ)


@pytest.yield_fixture(autouse=True)
def _verify_environ(_collected_environ: Dict[str, str]) -> Iterator[None]:
    try:
        yield
    finally:
        new_environ = dict(os.environ)
        current_test = new_environ.pop('PYTEST_CURRENT_TEST', None)
        old_environ = dict(_collected_environ)
        old_environ.pop('PYTEST_CURRENT_TEST', None)
        if new_environ != old_environ:
            raise DirtyTest(
                'Left over environment variables',
                current_test,
                _compare_eq_dict(new_environ, old_environ, verbose=2))


def alive_threads() -> Set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.is_alive()}


@pytest.fixture(scope='session', autouse=True)
def _recorded_threads_at_startup(request: PytestFixtureRequest) -> Set[threading.Thread]:
    try:
        request.session._threads_at_startup
    except AttributeError:
        request.session._threads_at_startup = alive_threads()
    return request.session._threads_at_startup


@pytest.fixture(autouse=True)
def threads_not_lingering(request: PytestFixtureRequest) -> Iterator[None]:
    try:
        yield
    finally:
        threads_then = request.session._threads_at_startup
        threads_now = alive_threads()
        if threads_then != threads_now:
            request.session._threads_at_startup = threads_now
            raise DirtyTest(
                'Left over threads',
                os.environ.get('PYTEST_CURRENT_TEST'),
                _compare_eq_set(threads_now, threads_then, verbose=2))


@pytest.fixture(autouse=True)
def _recorded_tasks_at_startup(request: PytestFixtureRequest, loop: Any) -> Set[Any]:
    try:
        request.node._tasks_at_startup
    except AttributeError:
        request.node._tasks_at_startup = set(all_tasks(loop=loop))
    return request.node._tasks_at_startup


@pytest.fixture(autouse=True)
def tasks_not_lingering(
    request: PytestFixtureRequest,
    loop: Any,
    event_loop: Any,
    _recorded_tasks_at_startup: Set[Any],
) -> Iterator[None]:
    allow_lingering_tasks = False
    allow_count = 0
    marks = request.node.get_closest_marker('allow_lingering_tasks')
    if marks:
        allow_lingering_tasks = True
        allow_count = marks.kwargs.get('count', 0)
    try:
        yield
    finally:
        tasks_then = request.node._tasks_at_startup
        tasks_now = set(all_tasks(loop=loop))
        if tasks_then != tasks_now:
            request.node._tasks_at_startup = tasks_now
            pending = {task for task in tasks_now if task and not task.done()}
            if pending:
                diff = len(pending - tasks_then)
                if not allow_lingering_tasks or diff > allow_count:
                    raise DirtyTest(
                        'Left over tasks',
                        os.environ.get('PYTEST_CURRENT_TEST'),
                        _compare_eq_set(tasks_now, tasks_then, verbose=2))


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        'markers',
        'allow_lingering_tasks: Allow test to start background tasks',
    )
    config.addinivalue_line(
        'markers',
        'allow_lingering_tasks: Allow test to start background tasks',
    )
    config.addinivalue_line(
        'markers',
        'time: Set the current time',
    )
    config.addinivalue_line(
        'markers',
        'http_session: Set mock aiohttp session result',
    )
