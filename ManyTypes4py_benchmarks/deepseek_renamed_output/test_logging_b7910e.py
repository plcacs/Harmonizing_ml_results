import json
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from functools import partial
from io import StringIO
from typing import Type, Any, Dict, List, Optional, Tuple, Union, Generator, Iterator, ContextManager
from unittest import mock
from unittest.mock import ANY, MagicMock
import pendulum
import pytest
from rich.color import Color, ColorType
from rich.console import Console
from rich.highlighter import NullHighlighter, ReprHighlighter
from rich.style import Style
import prefect
import prefect.logging.configuration
import prefect.settings
from prefect import flow, task
from prefect._internal.concurrency.api import create_call, from_sync
from prefect.context import FlowRunContext, TaskRunContext
from prefect.exceptions import MissingContextError
from prefect.logging import LogEavesdropper
from prefect.logging.configuration import DEFAULT_LOGGING_SETTINGS_PATH, load_logging_config, setup_logging
from prefect.logging.filters import ObfuscateApiKeyFilter
from prefect.logging.formatters import JsonFormatter
from prefect.logging.handlers import APILogHandler, APILogWorker, PrefectConsoleHandler, WorkerAPILogHandler
from prefect.logging.highlighters import PrefectConsoleHighlighter
from prefect.logging.loggers import PrefectLogAdapter, disable_logger, disable_run_logger, flow_run_logger, get_logger, get_run_logger, get_worker_logger, patch_print, task_run_logger
from prefect.server.schemas.actions import LogCreate
from prefect.settings import PREFECT_API_KEY, PREFECT_LOGGING_COLORS, PREFECT_LOGGING_EXTRA_LOGGERS, PREFECT_LOGGING_LEVEL, PREFECT_LOGGING_MARKUP, PREFECT_LOGGING_SETTINGS_PATH, PREFECT_LOGGING_TO_API_BATCH_INTERVAL, PREFECT_LOGGING_TO_API_BATCH_SIZE, PREFECT_LOGGING_TO_API_ENABLED, PREFECT_LOGGING_TO_API_MAX_LOG_SIZE, PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW, PREFECT_TEST_MODE, temporary_settings
from prefect.testing.cli import temporary_console_width
from prefect.testing.utilities import AsyncMock
from prefect.utilities.names import obfuscate
from prefect.workers.base import BaseJobConfiguration, BaseWorker


@pytest.fixture
def func_56k5i3d7(monkeypatch: pytest.MonkeyPatch) -> Generator[MagicMock, None, None]:
    mock = MagicMock()
    monkeypatch.setattr('logging.config.dictConfig', mock)
    old = prefect.logging.configuration.PROCESS_LOGGING_CONFIG
    prefect.logging.configuration.PROCESS_LOGGING_CONFIG = None
    yield mock
    prefect.logging.configuration.PROCESS_LOGGING_CONFIG = old


@pytest.fixture
async def func_2sl7l6sl(prefect_client: Any) -> Any:
    """
    A deployment with a flow that returns information about the given loggers
    """

    @prefect.flow
    def func_40r8vjcx(loggers: List[str] = ['foo', 'bar', 'prefect']) -> Dict[str, Dict[str, Any]]:
        import logging
        settings = {}
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            settings[logger_name] = {'handlers': [handler.name for handler in
                logger.handlers], 'level': logger.level}
            logger.info(f'Hello from {logger_name}')
        return settings
    flow_id = await prefect_client.create_flow(my_flow)
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id,
        name='logger_test_deployment')
    return deployment_id


def func_6xody286(tmp_path: Any, dictConfigMock: MagicMock) -> None:
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: tmp_path.
        joinpath('does-not-exist.yaml')}):
        expected_config = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)
        expected_config['incremental'] = False
        setup_logging()
    func_56k5i3d7.assert_called_once_with(expected_config)


def func_tiqihao8(dictConfigMock: MagicMock) -> None:
    setup_logging()
    assert dictConfigMock.call_count == 1
    setup_logging()
    assert dictConfigMock.call_count == 2
    assert dictConfigMock.mock_calls[0][1][0]['incremental'] is False
    assert dictConfigMock.mock_calls[1][1][0]['incremental'] is True


def func_80kiqgza(tmp_path: Any, dictConfigMock: MagicMock) -> None:
    config_file = tmp_path.joinpath('exists.yaml')
    config_file.write_text('foo: bar')
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: config_file}):
        setup_logging()
        expected_config = load_logging_config(tmp_path.joinpath('exists.yaml'))
        expected_config['incremental'] = False
    func_56k5i3d7.assert_called_once_with(expected_config)


def func_1csnsb04(tmp_path: Any, dictConfigMock: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: tmp_path.
        joinpath('does-not-exist.yaml')}):
        expected_config = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)
    env = {}
    expected_config['incremental'] = False
    env['PREFECT_LOGGING_HANDLERS_API_LEVEL'] = 'API_LEVEL_VAL'
    expected_config['handlers']['api']['level'] = 'API_LEVEL_VAL'
    env['PREFECT_LOGGING_ROOT_LEVEL'] = 'ROOT_LEVEL_VAL'
    expected_config['root']['level'] = 'ROOT_LEVEL_VAL'
    env['PREFECT_LOGGING_FORMATTERS_STANDARD_FLOW_RUN_FMT'
        ] = 'UNDERSCORE_KEY_VAL'
    expected_config['formatters']['standard']['flow_run_fmt'
        ] = 'UNDERSCORE_KEY_VAL'
    env['PREFECT_LOGGING_LOGGERS_PREFECT_EXTRA_LEVEL'] = 'VAL'
    expected_config['loggers']['prefect.extra']['level'] = 'VAL'
    env['PREFECT_LOGGING_FOO'] = 'IGNORED'
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: tmp_path.
        joinpath('does-not-exist.yaml')}):
        setup_logging()
    func_56k5i3d7.assert_called_once_with(expected_config)


@pytest.mark.parametrize('name', ['default', None, ''])
def func_zuahq586(name: Optional[str]) -> None:
    if name == 'default':
        logger = get_logger()
    else:
        logger = get_logger(name)
    assert logger.name == 'prefect'


def func_olqhd902() -> None:
    logger = get_logger('foo')
    assert logger.name == 'prefect.foo'


def func_ziqkfpts() -> None:
    logger = get_logger('prefect.foo')
    assert logger.name == 'prefect.foo'


def func_kvl1lpal(dictConfigMock: MagicMock) -> None:
    with temporary_settings({PREFECT_LOGGING_LEVEL: 'WARNING',
        PREFECT_TEST_MODE: False}):
        expected_config = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)
        expected_config['incremental'] = False
        assert expected_config['loggers']['prefect']['level'] == 'WARNING'
        assert expected_config['loggers']['prefect.extra']['level'
            ] == 'WARNING'
        setup_logging()
    func_56k5i3d7.assert_called_once_with(expected_config)


@pytest.fixture()
def func_oudnwapf(request: Any) -> Generator[Tuple[str, int, bool], None, None]:
    name, level = request.param
    logger = logging.getLogger(name)
    old_level, old_propagate = logger.level, logger.propagate
    assert logger.level == logging.NOTSET, 'Logger should start with NOTSET level'
    assert logger.handlers == [], 'Logger should start with no handlers'
    logger.setLevel(level)
    yield name, level, old_propagate
    logger.setLevel(old_level)
    logger.propagate = old_propagate
    logger.handlers = []


@pytest.mark.parametrize('external_logger_setup', [('foo', logging.DEBUG),
    ('foo.child', logging.DEBUG), ('foo', logging.INFO), ('foo.child',
    logging.INFO), ('foo', logging.WARNING), ('foo.child', logging.WARNING),
    ('foo', logging.ERROR), ('foo.child', logging.ERROR), ('foo', logging.
    CRITICAL), ('foo.child', logging.CRITICAL)], indirect=True, ids=lambda
    x: f"logger='{x[0]}'-level='{logging.getLevelName(x[1])}'")
def func_ot32lv8w(dictConfigMock: MagicMock, external_logger_setup: Tuple[str, int, bool]) -> None:
    ext_name, ext_level, ext_propagate = external_logger_setup
    with temporary_settings({PREFECT_LOGGING_LEVEL: 'WARNING',
        PREFECT_TEST_MODE: False, PREFECT_LOGGING_EXTRA_LOGGERS: ext_name}):
        expected_config = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)
        expected_config['incremental'] = False
        setup_logging()
    func_56k5i3d7.assert_called_once_with(expected_config)
    external_logger = logging.getLogger(ext_name)
    assert external_logger.level == ext_level, 'External logger level was not preserved'
    if ext_level > logging.NOTSET:
        assert external_logger.isEnabledFor(ext_level
            ), 'External effective level was not preserved'
    assert external_logger.propagate == ext_propagate, 'External logger propagate was not preserved'


@pytest.fixture
def func_99gdk93s(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr('prefect.logging.handlers.APILogWorker', mock)
    return mock


@pytest.mark.enable_api_log_handler
class TestAPILogHandler:

    @pytest.fixture
    def func_app5ij1k(self) -> Generator[APILogHandler, None, None]:
        yield APILogHandler()

    @pytest.fixture
    def func_8ixz7h5g(self, handler: logging.Handler) -> Generator[logging.Logger, None, None]:
        logger = logging.getLogger(__name__)
        func_8ixz7h5g.setLevel(logging.DEBUG)
        func_8ixz7h5g.addHandler(handler)
        yield logger
        func_8ixz7h5g.removeHandler(handler)

    def func_j2uy2poz(self, mock_log_worker: MagicMock) -> None:
        handler = APILogHandler()
        func_app5ij1k.close()
        mock_log_worker.drain_all.assert_not_called()

    async def func_tbl02mu5(self, logger: logging.Logger, handler: logging.Handler, flow_run: Any, prefect_client: Any) -> None:
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.close()
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_4bahmrwl(self, logger: logging.Logger, handler: logging.Handler, flow_run: Any, prefect_client: Any) -> None:
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_vomhwyku(self, logger: logging.Logger, handler: logging.Handler, flow_run: Any, prefect_client: Any) -> None:
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.flush()
        time.sleep(2)
        logs = await prefect_client.read_logs()
        assert len(logs) == 1

    def func_wxkki436(self, logger: logging.Logger, handler: logging.Handler, flow_run: Any) -> None:
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        with pytest.raises(RuntimeError, match='would block'):
            from_sync.call_soon_in_loop_thread(create_call(handler.flush)
                ).result()

    def func_1j9bwhho(self, logger: logging.Logger, handler: logging.Handler, flow_run: Any) -> None:
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.flush()

    def func_s3mdzozf(self, logger: logging.Logger, mock_log_worker: MagicMock, task_run: Any) -> None:
        with TaskRunContext.model_construct(task_run=task_run):
            func_8ixz7h5g.info('test-task')
        expected = LogCreate.model_construct(flow_run_id=task_run.
            flow_run_id, task_run_id=task_run.id, name=logger.name, level=
            logging.INFO, message='test-task').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    def func_ti9wp1n8(self, logger: logging.Logger, mock_log_worker: MagicMock, flow_run: Any) -> None:
        with FlowRunContext.model_construct(flow_run=flow_run):
            func_8ixz7h5g.info('test-flow')
        expected = LogCreate.model_construct(flow_run_id=flow_run.id,
            task_run_id=None, name=logger.name, level=logging.INFO, message
            ='test-flow').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    @pytest.mark.parametrize('with_context', [True, False])
    def func_hkcb87vj(self, logger: logging.Logger, mock_log_worker: MagicMock, flow_run: Any, with_context: bool) -> None:
        flow_run_id = uuid.uuid4()
        context = FlowRunContext.model_construct(flow_run=flow_run
            ) if with_context else nullcontext()
        with context:
            func_8ixz7h5g.info('test-task', extra={'flow_run_id': flow_run_id})
        expected = LogCreate.model_construct(flow_run_id=flow_run_id,
            task_run_id=None, name=logger.name, level=logging.INFO, message
            ='test-task').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    @pytest.mark.parametrize('with_context', [True, False])
    def func_m1om9w5n(self, logger: logging.Logger, mock_log_worker: MagicMock, flow_run: Any, with_context: bool,
        task_run: Any) -> None:
        task_run_id = uuid.uuid4()
        context = TaskRunContext.model_construct(task_run=task_run
            ) if with_context else nullcontext()
        with FlowRunContext.model_construct(flow_run=flow_run):
            with context:
                func_8ixz7h5g.warning('test-task', extra={'task_run_id':
                    task_run_id})
        expected = LogCreate.model_construct(flow_run_id=flow_run.id,
            task_run_id=task_run_id, name=logger.name, level=logging.
            WARNING, message='test-task').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    def func_ewoi70k0(self, logger: logging.Logger, mock_log_worker: MagicMock) -> None:
        func_8ixz7h5g.setLevel(logging.WARNING)
        func_8ixz7h5g.info('test-task', extra={'flow_run_id': uuid.uuid4()})
        func_99gdk93s.instance().send.assert_not_called()

    def func_vi1zakr5(self, logger: logging.Logger, mock_log_worker: MagicMock) -> None:
        task_run_id = uuid.uuid4()
        with pytest.warns(UserWarning, match=
            'attempted to send logs .* without a flow run id'):
            func_8ixz7h5g.info('test-task', extra={'task_run_id': task_run_id})
        func_99gdk93s.instance().send.assert_not_called()

    def func_i99usa9d(self, logger: logging.Logger, mock_log_worker: MagicMock, flow_run: Any, handler: logging.Handler) -> None:
       