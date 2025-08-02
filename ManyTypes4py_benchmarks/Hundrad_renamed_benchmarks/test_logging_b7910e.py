import json
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from functools import partial
from io import StringIO
from typing import Type
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
def func_56k5i3d7(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr('logging.config.dictConfig', mock)
    old = prefect.logging.configuration.PROCESS_LOGGING_CONFIG
    prefect.logging.configuration.PROCESS_LOGGING_CONFIG = None
    yield mock
    prefect.logging.configuration.PROCESS_LOGGING_CONFIG = old


@pytest.fixture
async def func_2sl7l6sl(prefect_client):
    """
    A deployment with a flow that returns information about the given loggers
    """

    @prefect.flow
    def func_40r8vjcx(loggers=['foo', 'bar', 'prefect']):
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


def func_6xody286(tmp_path, dictConfigMock):
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: tmp_path.
        joinpath('does-not-exist.yaml')}):
        expected_config = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)
        expected_config['incremental'] = False
        setup_logging()
    func_56k5i3d7.assert_called_once_with(expected_config)


def func_tiqihao8(dictConfigMock):
    setup_logging()
    assert dictConfigMock.call_count == 1
    setup_logging()
    assert dictConfigMock.call_count == 2
    assert dictConfigMock.mock_calls[0][1][0]['incremental'] is False
    assert dictConfigMock.mock_calls[1][1][0]['incremental'] is True


def func_80kiqgza(tmp_path, dictConfigMock):
    config_file = tmp_path.joinpath('exists.yaml')
    config_file.write_text('foo: bar')
    with temporary_settings({PREFECT_LOGGING_SETTINGS_PATH: config_file}):
        setup_logging()
        expected_config = load_logging_config(tmp_path.joinpath('exists.yaml'))
        expected_config['incremental'] = False
    func_56k5i3d7.assert_called_once_with(expected_config)


def func_1csnsb04(tmp_path, dictConfigMock, monkeypatch):
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
def func_zuahq586(name):
    if name == 'default':
        logger = get_logger()
    else:
        logger = get_logger(name)
    assert logger.name == 'prefect'


def func_olqhd902():
    logger = get_logger('foo')
    assert logger.name == 'prefect.foo'


def func_ziqkfpts():
    logger = get_logger('prefect.foo')
    assert logger.name == 'prefect.foo'


def func_kvl1lpal(dictConfigMock):
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
def func_oudnwapf(request):
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
def func_ot32lv8w(dictConfigMock, external_logger_setup):
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
def func_99gdk93s(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr('prefect.logging.handlers.APILogWorker', mock)
    return mock


@pytest.mark.enable_api_log_handler
class TestAPILogHandler:

    @pytest.fixture
    def func_app5ij1k(self):
        yield APILogHandler()

    @pytest.fixture
    def func_8ixz7h5g(self, handler):
        logger = logging.getLogger(__name__)
        func_8ixz7h5g.setLevel(logging.DEBUG)
        func_8ixz7h5g.addHandler(handler)
        yield logger
        func_8ixz7h5g.removeHandler(handler)

    def func_j2uy2poz(self, mock_log_worker):
        handler = APILogHandler()
        func_app5ij1k.close()
        mock_log_worker.drain_all.assert_not_called()

    async def func_tbl02mu5(self, logger, handler, flow_run, prefect_client):
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.close()
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_4bahmrwl(self, logger, handler, flow_run, prefect_client):
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        await func_app5ij1k.aflush()
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_vomhwyku(self, logger, handler, flow_run, prefect_client):
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.flush()
        time.sleep(2)
        logs = await prefect_client.read_logs()
        assert len(logs) == 1

    def func_wxkki436(self, logger, handler, flow_run):
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        with pytest.raises(RuntimeError, match='would block'):
            from_sync.call_soon_in_loop_thread(create_call(handler.flush)
                ).result()

    def func_1j9bwhho(self, logger, handler, flow_run):
        func_8ixz7h5g.info('Test', extra={'flow_run_id': flow_run.id})
        func_app5ij1k.flush()

    def func_s3mdzozf(self, logger, mock_log_worker, task_run):
        with TaskRunContext.model_construct(task_run=task_run):
            func_8ixz7h5g.info('test-task')
        expected = LogCreate.model_construct(flow_run_id=task_run.
            flow_run_id, task_run_id=task_run.id, name=logger.name, level=
            logging.INFO, message='test-task').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    def func_ti9wp1n8(self, logger, mock_log_worker, flow_run):
        with FlowRunContext.model_construct(flow_run=flow_run):
            func_8ixz7h5g.info('test-flow')
        expected = LogCreate.model_construct(flow_run_id=flow_run.id,
            task_run_id=None, name=logger.name, level=logging.INFO, message
            ='test-flow').model_dump(mode='json')
        expected['timestamp'] = ANY
        expected['__payload_size__'] = ANY
        func_99gdk93s.instance().send.assert_called_once_with(expected)

    @pytest.mark.parametrize('with_context', [True, False])
    def func_hkcb87vj(self, logger, mock_log_worker, flow_run, with_context):
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
    def func_m1om9w5n(self, logger, mock_log_worker, flow_run, with_context,
        task_run):
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

    def func_ewoi70k0(self, logger, mock_log_worker):
        func_8ixz7h5g.setLevel(logging.WARNING)
        func_8ixz7h5g.info('test-task', extra={'flow_run_id': uuid.uuid4()})
        func_99gdk93s.instance().send.assert_not_called()

    def func_vi1zakr5(self, logger, mock_log_worker):
        task_run_id = uuid.uuid4()
        with pytest.warns(UserWarning, match=
            'attempted to send logs .* without a flow run id'):
            func_8ixz7h5g.info('test-task', extra={'task_run_id': task_run_id})
        func_99gdk93s.instance().send.assert_not_called()

    def func_i99usa9d(self, logger, mock_log_worker, flow_run, handler):
        handler.emit = MagicMock(side_effect=handler.emit)
        with FlowRunContext.model_construct(flow_run=flow_run):
            func_8ixz7h5g.info('test-flow')
        record = handler.emit.call_args[0][0]
        log_dict = func_99gdk93s.instance().send.call_args[0][0]
        assert log_dict['timestamp'] == pendulum.from_timestamp(record.created
            ).to_iso8601_string()

    def func_9tbvid1i(self, logger, mock_log_worker, flow_run, handler,
        monkeypatch):

        def func_j72cdd3x(emit, record):
            record.created = None
            return emit(record)
        handler.emit = MagicMock(side_effect=partial(drop_created_and_emit,
            handler.emit))
        now = time.time()
        monkeypatch.setattr('time.time', lambda : now)
        with FlowRunContext.model_construct(flow_run=flow_run):
            func_8ixz7h5g.info('test-flow')
        log_dict = func_99gdk93s.instance().send.call_args[0][0]
        assert log_dict['timestamp'] == pendulum.from_timestamp(now
            ).to_iso8601_string()

    def func_4vdgn9lg(self, logger, mock_log_worker, task_run):
        with TaskRunContext.model_construct(task_run=task_run):
            func_8ixz7h5g.info('test', extra={'send_to_api': False})
        func_99gdk93s.instance().send.assert_not_called()

    def func_jwnis7xy(self, logger, mock_log_worker, task_run):
        with temporary_settings(updates={PREFECT_LOGGING_TO_API_ENABLED:
            'False'}):
            with TaskRunContext.model_construct(task_run=task_run):
                func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()

    def func_vmugemko(self, logger, mock_log_worker, capsys):
        with pytest.warns(UserWarning, match=
            'attempted to send logs .* without a flow run id'):
            func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert output.err == ''

    def func_1cgv2avi(self, logger):
        with pytest.warns(UserWarning, match=
            "Logger 'tests.test_logging' attempted to send logs to the API without a flow run id."
            ):
            func_8ixz7h5g.info('test')

    def func_g0r3xf9x(self, logger, mock_log_worker, capsys):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'error'}):
            with pytest.raises(MissingContextError, match=
                'attempted to send logs .* without a flow run id'):
                func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert output.err == ''

    def func_8gwwsl5d(self, logger):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'error'}):
            with pytest.raises(MissingContextError, match=
                "Logger 'tests.test_logging' attempted to send logs to the API without a flow run id."
                ):
                func_8ixz7h5g.info('test')

    def func_jh275how(self, logger, mock_log_worker, capsys):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'ignore'}):
            func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert output.err == ''

    def func_g5sh7h4h(self, logger):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'ignore'}):
            func_8ixz7h5g.info('test')

    def func_ctlx04qe(self, logger, mock_log_worker, capsys):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'warn'}):
            with pytest.warns(UserWarning, match=
                'attempted to send logs .* without a flow run id'):
                func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert output.err == ''

    def func_huexceff(self, logger):
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW: 'warn'}):
            with pytest.warns(UserWarning, match=
                "Logger 'tests.test_logging' attempted to send logs to the API without a flow run id."
                ):
                func_8ixz7h5g.info('test')

    def func_qnc7chio(self, logger, mock_log_worker):
        from inspect import currentframe, getframeinfo
        with pytest.warns(UserWarning, match=
            'attempted to send logs .* without a flow run id') as warnings:
            func_8ixz7h5g.info('test')
            lineno = getframeinfo(currentframe()).lineno - 1
        func_99gdk93s.instance().send.assert_not_called()
        assert warnings.pop().lineno == lineno

    def func_5k0vdgm1(self, logger, mock_log_worker, capsys, monkeypatch):
        monkeypatch.setattr('prefect.logging.handlers.APILogHandler.prepare',
            MagicMock(side_effect=RuntimeError('Oh no!')))
        func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert 'RuntimeError: Oh no!' in output.err

    def func_w4k4rxv8(self, logger, mock_log_worker, capsys):
        func_8ixz7h5g.info('test', extra={'send_to_api': False})
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert 'RuntimeError: Attempted to send logs to the API without a flow run id.' not in output.err

    async def func_43quug34(self, task_run, logger, capsys, mock_log_worker):
        with TaskRunContext.model_construct(task_run=task_run):
            with temporary_settings(updates={
                PREFECT_LOGGING_TO_API_MAX_LOG_SIZE: '1'}):
                func_8ixz7h5g.info('test')
        func_99gdk93s.instance().send.assert_not_called()
        output = capsys.readouterr()
        assert 'ValueError' in output.err
        assert 'is greater than the max size of 1' in output.err

    def func_3ykg823i(self):
        dict_log = {'name': 'prefect.flow_runs', 'level': 20, 'message':
            'Finished in state Completed()', 'timestamp':
            '2023-02-08T17:55:52.993831+00:00', 'flow_run_id':
            '47014fb1-9202-4a78-8739-c993d8c24415', 'task_run_id': None}
        log_size = len(json.dumps(dict_log))
        assert log_size == 211
        handler = APILogHandler()
        assert func_app5ij1k._get_payload_size(dict_log) == log_size


WORKER_ID = uuid.uuid4()


class TestWorkerLogging:


    class CloudWorkerTestImpl(BaseWorker):
        type = 'cloud_logging_test'
        job_configuration = BaseJobConfiguration

        async def func_xjj8lidv(self, *_, **__):
            """
            Workers only return an ID here if they're connected to Cloud,
            so this simulates the worker being connected to Cloud.
            """
            return WORKER_ID

        async def func_f8ennr3i(self, *_, **__):
            pass


    class ServerWorkerTestImpl(BaseWorker):
        type = 'server_logging_test'
        job_configuration = BaseJobConfiguration

        async def func_f8ennr3i(self, *_, **__):
            pass

        async def func_xjj8lidv(self, *_, **__):
            """
            Workers only return an ID here if they're connected to Cloud,
            so this simulates the worker not being connected to Cloud.
            """
            return None

    @pytest.fixture
    def func_a24kgyem(self):
        with temporary_settings(updates={PREFECT_LOGGING_TO_API_ENABLED: True}
            ):
            yield

    @pytest.fixture
    def func_90iqcx89(self):
        yield WorkerAPILogHandler()

    @pytest.fixture
    def func_8ixz7h5g(self, worker_handler):
        logger = logging.getLogger(__name__)
        func_8ixz7h5g.setLevel(logging.DEBUG)
        func_8ixz7h5g.addHandler(worker_handler)
        yield logger
        func_8ixz7h5g.removeHandler(worker_handler)

    async def func_45yova0i(self):
        async with self.CloudWorkerTestImpl(name='test', work_pool_name=
            'test-work-pool') as worker:
            logger = get_worker_logger(worker)
            assert logger.name == 'prefect.workers.cloud_logging_test.test'

    async def func_asjhyznq(self):
        async with self.CloudWorkerTestImpl(name='test', work_pool_name=
            'test-work-pool') as worker:
            await worker.sync_with_backend()
            logger = get_worker_logger(worker)
            assert logger.name == 'prefect.workers.cloud_logging_test.test'
            assert logger.extra['worker_id'] == str(WORKER_ID)

    async def func_37xzmm4i(self, caplog):
        async with self.CloudWorkerTestImpl(name='test', work_pool_name=
            'test-work-pool') as worker:
            await worker.sync_with_backend()
            worker._logger.info('testing_with_extras')
            record_with_extras = [r for r in caplog.records if 
                'testing_with_extras' in r.message]
            assert 'testing_with_extras' in caplog.text
            assert record_with_extras[0].worker_id == str(worker.backend_id)
            assert worker._logger.extra['worker_id'] == str(worker.backend_id)

    async def func_wm9tlxw3(self, mock_log_worker, worker_handler,
        logging_to_api_enabled):
        async with self.CloudWorkerTestImpl(name='test', work_pool_name=
            'test-work-pool') as worker:
            await worker.sync_with_backend()
            worker._logger.debug('test-worker-log')
        log_statement = [log for call in func_99gdk93s.instance().send.
            call_args_list for log in call.args if log['name'] == worker.
            _logger.name and log['message'] == 'test-worker-log']
        assert len(log_statement) == 1
        assert log_statement[0]['worker_id'] == str(worker.backend_id)

    async def func_rj39so2a(self, mock_log_worker, worker_handler,
        logging_to_api_enabled):
        async with self.ServerWorkerTestImpl(name='test', work_pool_name=
            'test-work-pool') as worker:
            assert isinstance(worker._logger, logging.Logger)
            worker._logger.debug('test-worker-log')
        func_99gdk93s.instance().send.assert_not_called()


class TestAPILogWorker:

    @pytest.fixture
    async def func_dsx45z8g(self):
        return APILogWorker.instance()

    @pytest.fixture
    def func_yjh1g9ae(self):
        return LogCreate(flow_run_id=uuid.uuid4(), task_run_id=uuid.uuid4(),
            name='test.logger', level=10, timestamp=pendulum.now('utc'),
            message='hello').model_dump(mode='json')

    async def func_2miau0hz(self, log_dict, prefect_client, worker):
        func_dsx45z8g.send(log_dict)
        await func_dsx45z8g.drain()
        logs = await prefect_client.read_logs()
        assert len(logs) == 1
        assert logs[0].model_dump(include=func_yjh1g9ae.keys(), mode='json'
            ) == log_dict

    async def func_4rilr2jf(self, log_dict, prefect_client, worker):
        count = prefect.settings.PREFECT_API_DEFAULT_LIMIT.value()
        func_yjh1g9ae.pop('message')
        for i in range(count):
            new_log = func_yjh1g9ae.copy()
            new_log['message'] = str(i)
            func_dsx45z8g.send(new_log)
        await func_dsx45z8g.drain()
        logs = await prefect_client.read_logs()
        assert len(logs) == count
        for log in logs:
            assert log.model_dump(include=func_yjh1g9ae.keys(), exclude={
                'message'}, mode='json') == log_dict
        assert len(set(log.message for log in logs)
            ) == count, 'Each log is unique'

    async def func_iwbgxhnh(self, log_dict, capsys, monkeypatch, worker):
        monkeypatch.setattr(
            'prefect.client.orchestration.PrefectClient.create_logs',
            MagicMock(side_effect=ValueError('Test')))
        func_dsx45z8g.send(log_dict)
        await func_dsx45z8g.drain()
        err = capsys.readouterr().err
        assert '--- Error logging to API ---' in err
        assert 'ValueError: Test' in err

    async def func_c4brru50(self, log_dict, monkeypatch):
        mock_create_logs = AsyncMock()
        monkeypatch.setattr(
            'prefect.client.orchestration.PrefectClient.create_logs',
            mock_create_logs)
        log_size = APILogHandler()._get_payload_size(log_dict)
        with temporary_settings(updates={PREFECT_LOGGING_TO_API_BATCH_SIZE:
            log_size + 1, PREFECT_LOGGING_TO_API_MAX_LOG_SIZE: log_size}):
            worker = APILogWorker.instance()
            func_dsx45z8g.send(log_dict)
            func_dsx45z8g.send(log_dict)
            func_dsx45z8g.send(log_dict)
            await func_dsx45z8g.drain()
        assert mock_create_logs.call_count == 3

    async def func_2ch5e3i7(self, log_dict, prefect_client):
        start_time = time.time()
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_BATCH_INTERVAL: '10'}):
            worker = APILogWorker.instance()
            func_dsx45z8g.send(log_dict)
            func_dsx45z8g.send(log_dict)
            await func_dsx45z8g.drain()
        end_time = time.time()
        assert end_time - start_time < 5
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_izg98fzm(self, log_dict, prefect_client, worker):
        start_time = time.time()
        with temporary_settings(updates={
            PREFECT_LOGGING_TO_API_BATCH_INTERVAL: '10'}):
            func_dsx45z8g.send(log_dict)
            func_dsx45z8g.send(log_dict)
            await func_dsx45z8g.drain()
        end_time = time.time()
        assert end_time - start_time < 5
        logs = await prefect_client.read_logs()
        assert len(logs) == 2

    async def func_fwrhsuju(self, worker, log_dict, prefect_client):
        worker_id = str(uuid.uuid4())
        log_dict['worker_id'] = worker_id
        with mock.patch(
            'prefect.client.orchestration.PrefectClient.create_logs',
            autospec=True) as mock_create_logs:
            func_dsx45z8g.send(log_dict)
            await func_dsx45z8g.drain()
            assert mock_create_logs.call_count == 1
            logs = mock_create_logs.call_args.args[1]
            assert len(logs) == 1
            assert logs[0]['worker_id'] == worker_id


def func_zcb6b3u6(flow_run):
    logger = flow_run_logger(flow_run)
    assert logger.name == 'prefect.flow_runs'
    assert logger.extra == {'flow_run_name': flow_run.name, 'flow_run_id':
        str(flow_run.id), 'flow_name': '<unknown>'}


def func_f9vk5ihc(flow_run):

    @flow(name='foo')
    def func_sls5svah():
        pass
    logger = flow_run_logger(flow_run, test_flow)
    assert logger.extra['flow_name'] == 'foo'


def func_4tes3tft(flow_run):
    logger = flow_run_logger(flow_run, foo='test', flow_run_name='bar')
    assert logger.extra['foo'] == 'test'
    assert logger.extra['flow_run_name'] == 'bar'


def func_0eyofqqu(task_run):
    logger = task_run_logger(task_run)
    assert logger.name == 'prefect.task_runs'
    assert logger.extra == {'task_run_name': task_run.name, 'task_run_id':
        str(task_run.id), 'flow_run_id': str(task_run.flow_run_id),
        'flow_run_name': '<unknown>', 'flow_name': '<unknown>', 'task_name':
        '<unknown>'}


def func_mkgyyqaq(task_run):

    @task(name='task_run_logger_with_task')
    def func_gitw4tre():
        pass
    logger = task_run_logger(task_run, test_task)
    assert logger.extra['task_name'] == 'task_run_logger_with_task'


def func_0s6u301c(task_run, flow_run):
    logger = task_run_logger(task_run, flow_run=flow_run)
    assert logger.extra['flow_run_id'] == str(task_run.flow_run_id)
    assert logger.extra['flow_run_name'] == flow_run.name


def func_gybgn1x9(task_run):

    @flow(name='foo')
    def func_sls5svah():
        pass
    logger = task_run_logger(task_run, flow=test_flow)
    assert logger.extra['flow_name'] == 'foo'


def func_dmvnr2nh(task_run, flow_run):

    @flow(name='foo')
    def func_sls5svah():
        pass
    with FlowRunContext.model_construct(flow_run=flow_run, flow=test_flow):
        logger = task_run_logger(task_run)
        assert logger.extra['flow_run_id'] == str(task_run.flow_run_id) == str(
            flow_run.id)
        assert logger.extra['flow_run_name'] == flow_run.name
        assert logger.extra['flow_name'] == test_flow.name == 'foo'


def func_r8vo1pxq(caplog):
    """Test that get_run_logger works when called from a constructed FlowRunContext"""
    with FlowRunContext.model_construct(flow_run=None, flow=None):
        logger = get_run_logger()
        with caplog.at_level(logging.INFO):
            func_8ixz7h5g.info('test3141592')
        assert 'prefect.flow_runs' in caplog.text
        assert 'test3141592' in caplog.text
        assert logger.extra['flow_run_id'] == '<unknown>'
        assert logger.extra['flow_run_name'] == '<unknown>'
        assert logger.extra['flow_name'] == '<unknown>'


async def func_8om00dvk(prefect_client, caplog):
    """Test that get_run_logger works when passed a constructed TaskRunContext"""

    @task
    def func_s9byzy5w():
        pass
    task_run = await prefect_client.create_task_run(foo, flow_run_id=None,
        dynamic_key='')
    task_run_context = TaskRunContext.model_construct(task=foo, task_run=
        task_run, client=prefect_client)
    logger = get_run_logger(task_run_context)
    with caplog.at_level(logging.INFO):
        func_8ixz7h5g.info('test3141592')
    assert 'prefect.task_runs' in caplog.text
    assert 'test3141592' in caplog.text


def func_7ix8wecu(task_run):
    logger = task_run_logger(task_run, foo='test', task_run_name='bar')
    assert logger.extra['foo'] == 'test'
    assert logger.extra['task_run_name'] == 'bar'


def func_nfl29gso():
    with pytest.raises(MissingContextError, match=
        'no active flow or task run context'):
        get_run_logger()


async def func_pweg1yej():
    with pytest.raises(TypeError, match=
        "Received unexpected type 'str' for context."):
        get_run_logger('my man!')


async def func_h9ks2uyx(prefect_client, flow_run, local_filesystem):

    @task
    def func_s9byzy5w():
        pass
    task_run = await prefect_client.create_task_run(foo, flow_run.id,
        dynamic_key='')
    context = TaskRunContext.model_construct(task=foo, task_run=task_run,
        client=prefect_client)
    logger = get_run_logger(context)
    assert logger.name == 'prefect.task_runs'
    assert logger.extra == {'task_name': foo.name, 'task_run_id': str(
        task_run.id), 'task_run_name': task_run.name, 'flow_run_id': str(
        flow_run.id), 'flow_name': '<unknown>', 'flow_run_name': '<unknown>'}


async def func_n2pih61f(prefect_client, flow_run, local_filesystem):

    @task
    def func_s9byzy5w():
        pass

    @task
    def func_qghuaeec():
        pass
    task_run = await prefect_client.create_task_run(foo, flow_run.id,
        dynamic_key='')
    context = TaskRunContext.model_construct(task=bar, task_run=task_run,
        client=prefect_client)
    logger = get_run_logger(context)
    assert logger.extra['task_name'] == bar.name


async def func_4fgpsz4o(prefect_client):

    @flow
    def func_sls5svah():
        return get_run_logger()
    state = func_sls5svah(return_state=True)
    flow_run = await prefect_client.read_flow_run(state.state_details.
        flow_run_id)
    logger = await state.result()
    assert logger.name == 'prefect.flow_runs'
    assert logger.extra == {'flow_name': test_flow.name, 'flow_run_id': str
        (flow_run.id), 'flow_run_name': flow_run.name}


async def func_dhqekm19(prefect_client):

    @flow
    def func_sls5svah():
        return get_run_logger(foo='test', flow_name='bar')
    state = func_sls5svah(return_state=True)
    flow_run = await prefect_client.read_flow_run(state.state_details.
        flow_run_id)
    logger = await state.result()
    assert logger.name == 'prefect.flow_runs'
    assert logger.extra == {'flow_name': 'bar', 'foo': 'test',
        'flow_run_id': str(flow_run.id), 'flow_run_name': flow_run.name}


async def func_yz5jdvzm(prefect_client):

    @flow
    def func_2fl7rzj1():
        return get_run_logger()

    @flow
    def func_sls5svah():
        return func_2fl7rzj1(return_state=True)
    child_state = await func_sls5svah(return_state=True).result()
    flow_run = await prefect_client.read_flow_run(child_state.state_details
        .flow_run_id)
    logger = await child_state.result()
    assert logger.name == 'prefect.flow_runs'
    assert logger.extra == {'flow_name': child_flow.name, 'flow_run_id':
        str(flow_run.id), 'flow_run_name': flow_run.name}


async def func_biebuvyi(prefect_client, events_pipeline):

    @task
    def func_gitw4tre():
        return get_run_logger()

    @flow
    def func_sls5svah():
        return func_gitw4tre(return_state=True)
    flow_state = func_sls5svah(return_state=True)
    flow_run = await prefect_client.read_flow_run(flow_state.state_details.
        flow_run_id)
    task_state = await flow_state.result()
    await events_pipeline.process_events()
    task_run = await prefect_client.read_task_run(task_state.state_details.
        task_run_id)
    logger = await task_state.result()
    assert logger.name == 'prefect.task_runs'
    assert logger.extra == {'task_name': test_task.name, 'task_run_id': str
        (task_run.id), 'task_run_name': task_run.name, 'flow_name':
        test_flow.name, 'flow_run_id': str(flow_run.id), 'flow_run_name':
        flow_run.name}


class TestPrefectConsoleHandler:

    @pytest.fixture
    def func_app5ij1k(self):
        yield PrefectConsoleHandler()

    @pytest.fixture
    def func_8ixz7h5g(self, handler):
        logger = logging.getLogger(__name__)
        func_8ixz7h5g.setLevel(logging.DEBUG)
        func_8ixz7h5g.addHandler(handler)
        yield logger
        func_8ixz7h5g.removeHandler(handler)

    def func_b6xtuupv(self):
        handler = PrefectConsoleHandler()
        console = handler.console
        assert isinstance(console, Console)
        assert isinstance(console.highlighter, PrefectConsoleHighlighter)
        assert console._theme_stack._entries == [{}]
        assert handler.level == logging.NOTSET

    def func_vmd8avst(self):
        with temporary_settings({PREFECT_LOGGING_COLORS: False}):
            handler = PrefectConsoleHandler()
            console = handler.console
            assert isinstance(console, Console)
            assert isinstance(console.highlighter, NullHighlighter)
            assert console._theme_stack._entries == [{}]
            assert handler.level == logging.NOTSET

    def func_7ookao3t(self):
        handler = PrefectConsoleHandler(highlighter=ReprHighlighter, styles
            ={'number': 'red'}, level=logging.DEBUG)
        console = handler.console
        assert isinstance(console, Console)
        assert isinstance(console.highlighter, ReprHighlighter)
        assert console._theme_stack._entries == [{'number': Style(color=
            Color('red', ColorType.STANDARD, number=1))}]
        assert handler.level == logging.DEBUG

    def func_mlq9xg5y(self, capsys):
        logger = get_logger(uuid.uuid4().hex)
        logger.handlers = [PrefectConsoleHandler()]
        func_8ixz7h5g.info('Test!')
        stdout, stderr = capsys.readouterr()
        assert stdout == ''
        assert 'Test!' in stderr

    def func_plm3m7dc(self, capsys):
        logger = get_logger(uuid.uuid4().hex)
        logger.handlers = [PrefectConsoleHandler(stream=sys.stdout)]
        func_8ixz7h5g.info('Test!')
        stdout, stderr = capsys.readouterr()
        assert stderr == ''
        assert 'Test!' in stdout

    def func_tzk7jn1z(self, capsys):
        logger = get_logger(uuid.uuid4().hex)
        logger.handlers = [PrefectConsoleHandler()]
        try:
            raise ValueError('oh my')
        except Exception:
            func_8ixz7h5g.exception('Helpful context!')
        _, stderr = capsys.readouterr()
        assert 'Helpful context!' in stderr
        assert 'Traceback' in stderr
        assert 'raise ValueError("oh my")' in stderr
        assert 'ValueError: oh my' in stderr

    def func_819gvq6f(self, capsys):
        logger = get_logger(uuid.uuid4().hex)
        handler = PrefectConsoleHandler()
        logger.handlers = [handler]
        with temporary_console_width(handler.console, 10):
            func_8ixz7h5g.info('x' * 1000)
        _, stderr = capsys.readouterr()
        assert 'x' * 1000 in stderr

    def func_wowhxs7n(self, capsys):
        logger = get_logger(uuid.uuid4().hex)
        handler = PrefectConsoleHandler()
        logger.handlers = [handler]
        msg = 'DROP TABLE [dbo].[SomeTable];'
        func_8ixz7h5g.info(msg)
        _, stderr = capsys.readouterr()
        assert msg in stderr

    def func_p9hs2l2d(self, capsys):
        with temporary_settings({PREFECT_LOGGING_MARKUP: True}):
            logger = get_logger(uuid.uuid4().hex)
            handler = PrefectConsoleHandler()
            logger.handlers = [handler]
            msg = 'this applies [red]style[/red]!;'
            func_8ixz7h5g.info(msg)
            _, stderr = capsys.readouterr()
            assert 'this applies style' in stderr


class TestJsonFormatter:

    def func_2hktqmrs(self):
        formatter = JsonFormatter('default', None, '%')
        record = logging.LogRecord(name='Test Log', level=1, pathname=
            '/path/file.py', lineno=1, msg='log message', args=None,
            exc_info=None)
        formatted = formatter.format(record)
        deserialized = json.loads(formatted)
        assert deserialized['name'] == 'Test Log'
        assert deserialized['levelname'] == 'Level 1'
        assert deserialized['filename'] == 'file.py'
        assert deserialized['lineno'] == 1

    def func_g1t50d4w(self):
        exc_info = None
        try:
            raise Exception('test exception')
        except Exception as exc:
            exc_info = sys.exc_info()
        formatter = JsonFormatter('default', None, '%')
        record = logging.LogRecord(name='Test Log', level=1, pathname=
            '/path/file.py', lineno=1, msg='log message', args=None,
            exc_info=exc_info)
        formatted = formatter.format(record)
        deserialized = json.loads(formatted)
        assert deserialized['name'] == 'Test Log'
        assert deserialized['levelname'] == 'Level 1'
        assert deserialized['filename'] == 'file.py'
        assert deserialized['lineno'] == 1
        assert deserialized['exc_info'] is not None
        assert deserialized['exc_info']['type'] == 'Exception'
        assert deserialized['exc_info']['message'] == 'test exception'
        assert deserialized['exc_info']['traceback'] is not None
        assert len(deserialized['exc_info']['traceback']) > 0


class TestObfuscateApiKeyFilter:

    def func_5s0wdm4n(self):
        test_api_key = 'hi-hello-im-an-api-key'
        with temporary_settings({PREFECT_API_KEY: test_api_key}):
            filter = ObfuscateApiKeyFilter()
            record = logging.LogRecord(name='Test Log', level=1, pathname=
                '/path/file.py', lineno=1, msg=test_api_key, args=None,
                exc_info=None)
            filter.filter(record)
        assert test_api_key not in record.getMessage()
        assert obfuscate(test_api_key) in record.getMessage()

    def func_ih6wgpww(self, caplog):
        test_api_key = (
            'hot-dog-theres-a-logger-this-is-my-big-chance-for-stardom')
        with temporary_settings({PREFECT_API_KEY: test_api_key}):
            logger = get_logger('test')
            func_8ixz7h5g.info(test_api_key)
        assert test_api_key not in caplog.text
        assert obfuscate(test_api_key) in caplog.text

    def func_eelv876l(self, caplog):
        test_api_key = (
            'i-am-a-plaintext-api-key-and-i-dream-of-being-logged-one-day')
        with temporary_settings({PREFECT_API_KEY: test_api_key}):

            @flow
            def func_sls5svah():
                logger = get_run_logger()
                func_8ixz7h5g.info(test_api_key)
            func_sls5svah()
        assert test_api_key not in caplog.text
        assert obfuscate(test_api_key) in caplog.text

    def func_oikxnz1s(self, caplog):
        test_api_key = 'i-am-a-sneaky-little-api-key'
        with temporary_settings({PREFECT_API_KEY: test_api_key}):

            @flow(log_prints=True)
            def func_sls5svah():
                print(test_api_key)
            func_sls5svah()
        assert test_api_key not in caplog.text
        assert obfuscate(test_api_key) in caplog.text

    def func_3vzhtlic(self, caplog):
        test_api_key = 'i-am-jacks-security-risk'
        with temporary_settings({PREFECT_API_KEY: test_api_key}):

            @task
            def func_gitw4tre():
                logger = get_run_logger()
                func_8ixz7h5g.info(test_api_key)

            @flow
            def func_sls5svah():
                func_gitw4tre()
            func_sls5svah()
        assert test_api_key not in caplog.text
        assert obfuscate(test_api_key) in caplog.text

    @pytest.mark.parametrize('raw_log_record,expected_log_record', [([
        'super-mega-admin-key', 'in', 'a', 'list'], ['********', 'in', 'a',
        'list']), ({'super-mega-admin-key': 'in', 'a': 'dict'}, {'********':
        'in', 'a': 'dict'}), ({'key1': 'some_value', 'key2': [{'nested_key':
        'api_key: super-mega-admin-key'}, 'another_value']}, {'key1':
        'some_value', 'key2': [{'nested_key': 'api_key: ********'},
        'another_value']})])
    def func_a47jvg8u(self, caplog, raw_log_record, expected_log_record):
        """
        This is a regression test for https://github.com/PrefectHQ/prefect/issues/12139
        """

        @flow()
        def func_j0kzze3f():
            logger = get_run_logger()
            func_8ixz7h5g.info(raw_log_record)
        with temporary_settings({PREFECT_API_KEY: 'super-mega-admin-key'}):
            func_j0kzze3f()
        assert str(expected_log_record) in caplog.text


def func_aca4vwlh(caplog):
    msg = 'Hello world!'

    @flow
    def func_sls5svah():
        logger = get_run_logger()
        func_8ixz7h5g.warning(msg)
    func_sls5svah()
    for record in caplog.records:
        if record.msg == msg:
            assert record.levelno == logging.WARNING
            break
    else:
        raise AssertionError(
            f'{msg} was not found in records: {caplog.records}')


def func_osodjp64(caplog):
    msg = 'Hello world!'

    @task
    def func_gitw4tre():
        logger = get_run_logger()
        func_8ixz7h5g.warning(msg)

    @flow
    def func_sls5svah():
        func_gitw4tre()
    func_sls5svah()
    for record in caplog.records:
        if record.msg == msg:
            assert record.levelno == logging.WARNING
            break
    else:
        raise AssertionError(f'{msg} was not found in records')


def func_yn8g0s68(caplog):
    """
    Sanity test to double check whether caplog actually works
    so can be more confident in the asserts in test_disable_logger.
    """
    logger = logging.getLogger('griffe.agents.nodes')

    def func_akoliuut(logger):
        assert not logger.disabled
        func_8ixz7h5g.critical("it's enabled!")
        return 42
    func_akoliuut(logger)
    assert not logger.disabled
    assert ('griffe.agents.nodes', 50, "it's enabled!") in caplog.record_tuples


def func_qemfiblt(caplog):
    logger = logging.getLogger('griffe.agents.nodes')

    def func_akoliuut(logger):
        func_8ixz7h5g.critical("I know this is critical, but it's disabled!")
        return 42
    with disable_logger(logger.name):
        assert logger.disabled
        func_akoliuut(logger)
    assert not logger.disabled
    assert caplog.record_tuples == []


def func_nepe5olk(caplog):

    @task
    def func_visns8c2():
        logger = get_run_logger()
        func_8ixz7h5g.critical("won't show")
        return 42
    flow_run_logger = get_logger('prefect.flow_run')
    task_run_logger = get_logger('prefect.task_run')
    task_run_logger.disabled = True
    with disable_run_logger():
        num = func_visns8c2.fn()
        assert num == 42
        assert flow_run_logger.disabled
        assert task_run_logger.disabled
    assert not flow_run_logger.disabled
    assert task_run_logger.disabled
    assert caplog.record_tuples == [('null', logging.CRITICAL, "won't show")]


def func_5onfq9gf(caplog, capsys):
    with patch_print():
        print('foo')
    assert 'foo' in capsys.readouterr().out
    assert 'foo' not in caplog.text


@pytest.mark.parametrize('run_context_cls', [TaskRunContext, FlowRunContext])
def func_ldollvo3(caplog, capsys, run_context_cls):
    with patch_print():
        with run_context_cls.model_construct(log_prints=False):
            print('foo')
    assert 'foo' in capsys.readouterr().out
    assert 'foo' not in caplog.text


def func_9k9yu3gl(caplog, capsys, task_run):
    string_io = StringIO()

    @task
    def func_2bgixxc0():
        pass
    with patch_print():
        with TaskRunContext.model_construct(log_prints=True, task_run=
            task_run, task=my_task):
            print('foo', file=string_io)
    assert 'foo' not in caplog.text
    assert 'foo' not in capsys.readouterr().out
    assert string_io.getvalue().rstrip() == 'foo'


def func_5hgbipzo(caplog, capsys, task_run):

    @task
    def func_2bgixxc0():
        pass
    with patch_print():
        with TaskRunContext.model_construct(log_prints=True, task_run=
            task_run, task=my_task):
            print('foo')
    assert 'foo' not in capsys.readouterr().out
    assert 'foo' in caplog.text
    for record in caplog.records:
        if record.message == 'foo':
            break
    assert record.levelname == 'INFO'
    assert record.name == 'prefect.task_runs'
    assert record.task_run_id == str(task_run.id)
    assert record.task_name == my_task.name


@pytest.mark.parametrize('file', ['stdout', 'stderr'])
def func_vtlkts40(caplog, capsys, task_run, file):

    @task
    def func_2bgixxc0():
        pass
    with patch_print():
        with TaskRunContext.model_construct(log_prints=True, task_run=
            task_run, task=my_task):
            print('foo', file=getattr(sys, file))
    out, err = capsys.readouterr()
    assert 'foo' not in out
    assert 'foo' not in err
    assert 'foo' in caplog.text
    for record in caplog.records:
        if record.message == 'foo':
            break
    assert record.levelname == 'INFO'
    assert record.name == 'prefect.task_runs'
    assert record.task_run_id == str(task_run.id)
    assert record.task_name == my_task.name


def func_ijl0sv3d(caplog, capsys, flow_run):

    @flow
    def func_40r8vjcx():
        pass
    with patch_print():
        with FlowRunContext.model_construct(log_prints=True, flow_run=
            flow_run, flow=my_flow):
            print('foo')
    assert 'foo' not in capsys.readouterr().out
    assert 'foo' in caplog.text
    for record in caplog.records:
        if record.message == 'foo':
            break
    assert record.levelname == 'INFO'
    assert record.name == 'prefect.flow_runs'
    assert record.flow_run_id == str(flow_run.id)
    assert record.flow_name == my_flow.name


def func_qf6qp9ns(flow_run):
    logger = PrefectLogAdapter(get_logger('prefect.parent'), {'hello': 'world'}
        )
    assert logger.extra == {'hello': 'world'}
    child_logger = func_8ixz7h5g.getChild('child', {'goodnight': 'moon'})
    assert child_logger.logger.name == 'prefect.parent.child'
    assert child_logger.extra == {'hello': 'world', 'goodnight': 'moon'}


def func_w6ocbtg8():
    logging.getLogger('my_logger').debug('This is before the context')
    with LogEavesdropper('my_logger', level=logging.INFO) as eavesdropper:
        logging.getLogger('my_logger').info('Hello, world!')
        logging.getLogger('my_logger.child_module').warning('Another one!')
        logging.getLogger('my_logger').debug('Not this one!')
    logging.getLogger('my_logger').debug('This is after the context')
    assert eavesdropper.text(
        ) == '[INFO]: Hello, world!\n[WARNING]: Another one!'
