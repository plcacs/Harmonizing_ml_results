from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from contextlib import nullcontext
from datetime import datetime
from io import StringIO
from unittest.mock import (
    ANY,
    MagicMock,
    Mock,
    patch,
)
from uuid import UUID

import pytest
from pytest import (
    fixture,
    mark,
    param,
)
from pytest_mock import MockFixture
from prefect import flow, task
from prefect._internal.concurrency.api import from_sync
from prefect.context import (
    FlowRunContext,
    TaskRunContext,
)
from prefect.exceptions import MissingContextError
from prefect.server.schemas.actions import LogCreate
from prefect.settings import (
    PREFECT_API_KEY,
    PREFECT_LOGGING_LEVEL,
    Settings,
)
from prefect.testing.cli import temporary_console_width
from prefect.workers.base import BaseJobConfiguration, BaseWorker

if TYPE_CHECKING:
    from prefect import FlowRun, TaskRun
    from prefect.client.orchestration import PrefectClient
    from prefect.logging import (
        LogEavesdropper,
        PrefectConsoleHandler,
        PrefectLogAdapter,
    )
    from prefect.logging.handlers import (
        APILogHandler,
        APILogWorker,
        WorkerAPILogHandler,
    )
    from prefect.logging.highlighters import PrefectConsoleHighlighter
    from prefect.logging.loggers import (
        disable_logger,
        disable_run_logger,
        flow_run_logger,
        get_logger,
        get_run_logger,
        get_worker_logger,
        patch_print,
        task_run_logger,
    )

@fixture
def dictConfigMock(monkeypatch: MockFixture) -> MagicMock:
    ...

@fixture
async def logger_test_deployment(prefect_client: PrefectClient) -> UUID:
    ...

def test_setup_logging_uses_default_path(tmp_path: str, dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_sets_incremental_on_repeated_calls(dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_uses_settings_path_if_exists(tmp_path: str, dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_uses_env_var_overrides(tmp_path: str, dictConfigMock: MagicMock, monkeypatch: MockFixture) -> None:
    ...

@mark.parametrize('name', ['default', None, ''])
def test_get_logger_returns_prefect_logger_by_default(name: Optional[str]) -> None:
    ...

def test_get_logger_returns_prefect_child_logger() -> None:
    ...

def test_get_logger_does_not_duplicate_prefect_prefix() -> None:
    ...

def test_default_level_is_applied_to_interpolated_yaml_values(dictConfigMock: MagicMock) -> None:
    ...

@fixture()
def external_logger_setup(request: pytest.FixtureRequest) -> Tuple[str, int, bool]:
    ...

@mark.parametrize(
    'external_logger_setup',
    [
        ('foo', logging.DEBUG),
        ('foo.child', logging.DEBUG),
        ('foo', logging.INFO),
        ('foo.child', logging.INFO),
        ('foo', logging.WARNING),
        ('foo.child', logging.WARNING),
        ('foo', logging.ERROR),
        ('foo.child', logging.ERROR),
        ('foo', logging.CRITICAL),
        ('foo.child', logging.CRITICAL),
    ],
    indirect=True,
    ids=lambda x: f"logger='{x[0]}'-level='{logging.getLevelName(x[1])}'"
)
def test_setup_logging_extra_loggers_does_not_modify_external_logger_level(
    dictConfigMock: MagicMock,
    external_logger_setup: Tuple[str, int, bool]
) -> None:
    ...

@fixture
def mock_log_worker(monkeypatch: MockFixture) -> MagicMock:
    ...

@mark.enable_api_log_handler
class TestAPILogHandler:
    ...

@pytest.fixture
def handler() -> APILogHandler:
    ...

@pytest.fixture
def logger(handler: APILogHandler) -> logging.Logger:
    ...

def test_worker_is_not_flushed_on_handler_close(mock_log_worker: MagicMock) -> None:
    ...

async def test_logs_can_still_be_sent_after_close(
    logger: logging.Logger,
    handler: APILogHandler,
    flow_run: FlowRun,
    prefect_client: PrefectClient
) -> None:
    ...

async def test_logs_can_still_be_sent_after_flush(
    logger: logging.Logger,
    handler: APILogHandler,
    flow_run: FlowRun,
    prefect_client: PrefectClient
) -> None:
    ...

def test_sync_flush_from_global_event_loop(
    logger: logging.Logger,
    handler: APILogHandler,
    flow_run: FlowRun
) -> None:
    ...

def test_sync_flush_from_sync_context(
    logger: logging.Logger,
    handler: APILogHandler,
    flow_run: FlowRun
) -> None:
    ...

def test_sends_task_run_log_to_worker(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    task_run: TaskRun
) -> None:
    ...

def test_sends_flow_run_log_to_worker(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    flow_run: FlowRun
) -> None:
    ...

@mark.parametrize('with_context', [True, False])
def test_respects_explicit_flow_run_id(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    flow_run: FlowRun,
    with_context: bool
) -> None:
    ...

@mark.parametrize('with_context', [True, False])
def test_respects_explicit_task_run_id(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    flow_run: FlowRun,
    with_context: bool,
    task_run: TaskRun
) -> None:
    ...

def test_does_not_emit_logs_below_level(
    logger: logging.Logger,
    mock_log_worker: MagicMock
) -> None:
    ...

def test_explicit_task_run_id_still_requires_flow_run_id(
    logger: logging.Logger,
    mock_log_worker: MagicMock
) -> None:
    ...

def test_sets_timestamp_from_record_created_time(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    flow_run: FlowRun,
    handler: APILogHandler
) -> None:
    ...

def test_sets_timestamp_from_time_if_missing_from_record(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    flow_run: FlowRun,
    handler: APILogHandler,
    monkeypatch: MockFixture
) -> None:
    ...

def test_does_not_send_logs_that_opt_out(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    task_run: TaskRun
) -> None:
    ...

def test_does_not_send_logs_when_handler_is_disabled(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    task_run: TaskRun
) -> None:
    ...

def test_does_not_send_logs_outside_of_run_context_with_default_setting(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture
) -> None:
    ...

def test_does_not_raise_when_logger_outside_of_run_context_with_default_setting(
    logger: logging.Logger
) -> None:
    ...

def test_does_not_send_logs_outside_of_run_context_with_error_setting(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture
) -> None:
    ...

def test_does_not_warn_when_logger_outside_of_run_context_with_error_setting(
    logger: logging.Logger
) -> None:
    ...

def test_does_not_send_logs_outside_of_run_context_with_ignore_setting(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture
) -> None:
    ...

def test_does_not_raise_or_warn_when_logger_outside_of_run_context_with_ignore_setting(
    logger: logging.Logger
) -> None:
    ...

def test_does_not_send_logs_outside_of_run_context_with_warn_setting(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture
) -> None:
    ...

def test_does_not_raise_when_logger_outside_of_run_context_with_warn_setting(
    logger: logging.Logger
) -> None:
    ...

def test_missing_context_warning_refers_to_caller_lineno(
    logger: logging.Logger,
    mock_log_worker: MagicMock
) -> None:
    ...

def test_writes_logging_errors_to_stderr(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture,
    monkeypatch: MockFixture
) -> None:
    ...

def test_does_not_write_error_for_logs_outside_run_context_that_opt_out(
    logger: logging.Logger,
    mock_log_worker: MagicMock,
    capsys: pytest.CaptureFixture
) -> None:
    ...

async def test_does_not_enqueue_logs_that_are_too_big(
    task_run: TaskRun,
    logger: logging.Logger,
    capsys: pytest.CaptureFixture,
    mock_log_worker: MagicMock
) -> None:
    ...

def test_handler_knows_how_large_logs_are() -> None:
    ...