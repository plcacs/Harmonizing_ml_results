import json
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from functools import partial
from io import StringIO
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import ANY, MagicMock
from uuid import UUID

import pendulum
import pytest
from rich.color import Color
from rich.console import Console
from rich.highlighter import Highlighter
from rich.style import Style
from prefect import flow, task
from prefect._internal.concurrency.api import create_call, from_sync
from prefect.context import FlowRunContext, TaskRunContext
from prefect.exceptions import MissingContextError
from prefect.logging import LogEavesdropper
from prefect.logging.configuration import DEFAULT_LOGGING_SETTINGS_PATH
from prefect.settings import (
    PREFECT_API_KEY,
    PREFECT_LOGGING_COLORS,
    PREFECT_LOGGING_EXTRA_LOGGERS,
    PREFECT_LOGGING_LEVEL,
    PREFECT_LOGGING_MARKUP,
    PREFECT_LOGGING_SETTINGS_PATH,
    PREFECT_LOGGING_TO_API_BATCH_INTERVAL,
    PREFECT_LOGGING_TO_API_BATCH_SIZE,
    PREFECT_LOGGING_TO_API_ENABLED,
    PREFECT_LOGGING_TO_API_MAX_LOG_SIZE,
    PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW,
    PREFECT_TEST_MODE,
    temporary_settings,
)
from prefect.server.schemas.actions import LogCreate
from prefect.workers.base import BaseJobConfiguration, BaseWorker
from pytest import fixture

WORKER_ID: UUID = ...

@fixture
def dictConfigMock(monkeypatch: Any) -> MagicMock:
    ...

@fixture
async def logger_test_deployment(prefect_client: Any) -> str:
    ...

def test_setup_logging_uses_default_path(tmp_path: Any, dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_sets_incremental_on_repeated_calls(dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_uses_settings_path_if_exists(tmp_path: Any, dictConfigMock: MagicMock) -> None:
    ...

def test_setup_logging_uses_env_var_overrides(tmp_path: Any, dictConfigMock: MagicMock, monkeypatch: Any) -> None:
    ...

@pytest.mark.parametrize('name', ['default', None, ''])
def test_get_logger_returns_prefect_logger_by_default(name: Optional[str]) -> None:
    ...

def test_get_logger_returns_prefect_child_logger() -> None:
    ...

def test_get_logger_does_not_duplicate_prefect_prefix() -> None:
    ...

def test_default_level_is_applied_to_interpolated_yaml_values(dictConfigMock: MagicMock) -> None:
    ...

@pytest.fixture()
def external_logger_setup(request: Any) -> tuple[str, int, bool]:
    ...

@pytest.mark.parametrize('external_logger_setup', [('foo', logging.DEBUG), ('foo.child', logging.DEBUG), ('foo', logging.INFO), ('foo.child', logging.INFO), ('foo', logging.WARNING), ('foo.child', logging.WARNING), ('foo', logging.ERROR), ('foo.child', logging.ERROR), ('foo', logging.CRITICAL), ('foo.child', logging.CRITICAL)], indirect=True, ids=lambda x: f"logger='{x[0]}'-level='{logging.getLevelName(x[1])}'")
def test_setup_logging_extra_loggers_does_not_modify_external_logger_level(dictConfigMock: MagicMock, external_logger_setup: tuple[str, int, bool]) -> None:
    ...

@pytest.fixture
def mock_log_worker(monkeypatch: Any) -> MagicMock:
    ...

@pytest.mark.enable_api_log_handler
class TestAPILogHandler:
    @pytest.fixture
    def handler(self) -> Any:
        ...

    @pytest.fixture
    def logger(self, handler: Any) -> Any:
        ...

    def test_worker_is_not_flushed_on_handler_close(self, mock_log_worker: MagicMock) -> None:
        ...

    async def test_logs_can_still_be_sent_after_close(self, logger: Any, handler: Any, flow_run: Any, prefect_client: Any) -> None:
        ...

    async def test_logs_can_still_be_sent_after_flush(self, logger: Any, handler: Any, flow_run: Any, prefect_client: Any) -> None:
        ...

    def test_sync_flush_from_global_event_loop(self, logger: Any, handler: Any, flow_run: Any) -> None:
        ...

    def test_sync_flush_from_sync_context(self, logger: Any, handler: Any, flow_run: Any) -> None:
        ...

    def test_sends_task_run_log_to_worker(self, logger: Any, mock_log_worker: MagicMock, task_run: Any) -> None:
        ...

    def test_sends_flow_run_log_to_worker(self, logger: Any, mock_log_worker: MagicMock, flow_run: Any) -> None:
        ...

    @pytest.mark.parametrize('with_context', [True, False])
    def test_respects_explicit_flow_run_id(self, logger: Any, mock_log_worker: MagicMock, flow_run: Any, with_context: bool) -> None:
        ...

    @pytest.mark.parametrize('with_context', [True, False])
    def test_respects_explicit_task_run_id(self, logger: Any, mock_log_worker: MagicMock, flow_run: Any, with_context: bool, task_run: Any) -> None:
        ...

    def test_does_not_emit_logs_below_level(self, logger: Any, mock_log_worker: MagicMock) -> None:
        ...

    def test_explicit_task_run_id_still_requires_flow_run_id(self, logger: Any, mock_log_worker: MagicMock) -> None:
        ...

    def test_sets_timestamp_from_record_created_time(self, logger: Any, mock_log_worker: MagicMock, flow_run: Any, handler: Any) -> None:
        ...

    def test_sets_timestamp_from_time_if_missing_from_recrod(self, logger: Any, mock_log_worker: MagicMock, flow_run: Any, handler: Any, monkeypatch: Any) -> None:
        ...

    def test_does_not_send_logs_that_opt_out(self, logger: Any, mock_log_worker: MagicMock, task_run: Any) -> None:
        ...

    def test_does_not_send_logs_when_handler_is_disabled(self, logger: Any, mock_log_worker: MagicMock, task_run: Any) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_default_setting(self, logger: Any, mock_log_worker: MagicMock, capsys: Any) -> None:
        ...

    def test_does_not_raise_when_logger_outside_of_run_context_with_default_setting(self, logger: Any) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_error_setting(self, logger: Any, mock_log_worker: MagicMock, capsys: Any) -> None:
        ...

    def test_does_not_warn_when_logger_outside_of_run_context_with_error_setting(self, logger: Any) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_ignore_setting(self, logger: Any, mock_log_worker: MagicMock, capsys: Any) -> None:
        ...

    def test_does_not_raise_or_warn_when_logger_outside_of_run_context_with_ignore_setting(self, logger: Any) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_warn_setting(self, logger: Any, mock_log_worker: MagicMock, capsys: Any) -> None:
        ...

    def test_does_not_raise_when_logger_outside_of_run_context_with_warn_setting(self, logger: Any) -> None:
        ...

    def test_missing_context_warning_refers_to_caller_lineno(self, logger: Any, mock_log_worker: MagicMock) -> None:
        ...

    def test_writes_logging_errors_to_stderr(self, logger: Any, mock_log_worker: MagicMock, capsys: Any, monkeypatch: Any) -> None:
        ...

    def test_does_not_write_error_for_logs_outside_run_context_that_opt_out(self, logger: Any, mock_log_worker: MagicMock, capsys: Any) -> None:
        ...

    async def test_does_not_enqueue_logs_that_are_too_big(self, task_run: Any, logger: Any, capsys: Any, mock_log_worker: MagicMock) -> None:
        ...

    def test_handler_knows_how_large_logs_are(self) -> None:
        ...

class TestWorkerLogging:
    class CloudWorkerTestImpl(BaseWorker):
        type = 'cloud_logging_test'
        job_configuration = BaseJobConfiguration

        async def _send_worker_heartbeat(self, *_, **__) -> UUID:
            ...

        async def run(self, *_, **__) -> None:
            ...

    class ServerWorkerTestImpl(BaseWorker):
        type = 'server_logging_test'
        job_configuration = BaseJobConfiguration

        async def run(self, *_, **__) -> None:
            ...

        async def _send_worker_heartbeat(self, *_, **__) -> None:
            ...

    @pytest.fixture
    def logging_to_api_enabled(self) -> None:
        ...

    @pytest.fixture
    def worker_handler(self) -> Any:
        ...

    @pytest.fixture
    def logger(self, worker_handler: Any) -> Any:
        ...

    async def test_get_worker_logger_works_with_no_backend_id(self) -> None:
        ...

    async def test_get_worker_logger_works_with_backend_id(self) -> None:
        ...

    async def test_worker_emits_logs_with_worker_id(self, caplog: Any) -> None:
        ...

    async def test_worker_logger_sends_log_to_api_worker_when_connected_to_cloud(self, mock_log_worker: MagicMock, worker_handler: Any, logging_to_api_enabled: Any) -> None:
        ...

    async def test_worker_logger_does_not_send_logs_when_not_connected_to_cloud(self, mock_log_worker: MagicMock, worker_handler: Any, logging_to_api_enabled: Any) -> None:
        ...

class TestAPILogWorker:
    @pytest.fixture
    async def worker(self) -> Any:
        ...

    @pytest.fixture
    def log_dict(self) -> Dict[str, Any]:
        ...

    async def test_send_logs_single_record(self, log_dict: Dict[str, Any], prefect_client: Any, worker: Any) -> None:
        ...

    async def test_send_logs_many_records(self, log_dict: Dict[str, Any], prefect_client: Any, worker: Any) -> None:
        ...

    async def test_send_logs_writes_exceptions_to_stderr(self, log_dict: Dict[str, Any], capsys: Any, monkeypatch: Any, worker: Any) -> None:
        ...

    async def test_send_logs_batches_by_size(self, log_dict: Dict[str, Any], monkeypatch: Any) -> None:
        ...

    async def test_logs_are_sent_immediately_when_stopped(self, log_dict: Dict[str, Any], prefect_client: Any) -> None:
        ...

    async def test_logs_are_sent_immediately_when_flushed(self, log_dict: Dict[str, Any], prefect_client: Any, worker: Any) -> None:
        ...

    async def test_logs_include_worker_id_if_available(self, worker: Any, log_dict: Dict[str, Any], prefect_client: Any) -> None:
        ...

def test_flow_run_logger(flow_run: Any) -> None:
    ...

def test_flow_run_logger_with_flow(flow_run: Any) -> None:
    ...

def test_flow_run_logger_with_kwargs(flow_run: Any) -> None:
    ...

def test_task_run_logger(task_run: Any) -> None:
    ...

def test_task_run_logger_with_task(task_run: Any) -> None:
    ...

def test_task_run_logger_with_flow_run(task_run: Any, flow_run: Any) -> None:
    ...

def test_task_run_logger_with_flow(task_run: Any) -> None:
    ...

def test_task_run_logger_with_flow_run_from_context(task_run: Any, flow_run: Any) -> None:
    ...

def test_run_logger_with_flow_run_context_without_parent_flow_run_id(caplog: Any) -> None:
    ...

async def test_run_logger_with_task_run_context_without_parent_flow_run_id(prefect_client: Any, caplog: Any) -> None:
    ...

def test_task_run_logger_with_kwargs(task_run: Any) -> None:
    ...

def test_run_logger_fails_outside_context() -> None:
    ...

async def test_run_logger_with_explicit_context_of_invalid_type() -> None:
    ...

async def test_run_logger_with_explicit_context(prefect_client: Any, flow_run: Any, local_filesystem: Any) -> None:
    ...

async def test_run_logger_with_explicit_context_overrides_existing(prefect_client: Any, flow_run: Any, local_filesystem: Any) -> None:
    ...

async def test_run_logger_in_flow(prefect_client: Any) -> None:
    ...

async def test_run_logger_extra_data(prefect_client: Any) -> None:
    ...

async def test_run_logger_in_nested_flow(prefect_client: Any) -> None:
    ...

async def test_run_logger_in_task(prefect_client: Any, events_pipeline: Any) -> None:
    ...

class TestPrefectConsoleHandler:
    @pytest.fixture
    def handler(self) -> Any:
        ...

    @pytest.fixture
    def logger(self, handler: Any) -> Any:
        ...

    def test_init_defaults(self) -> None:
        ...

    def test_init_styled_console_disabled(self) -> None:
        ...

    def test_init_override_kwargs(self) -> None:
        ...

    def test_uses_stderr_by_default(self, capsys: Any) -> None:
        ...

    def test_respects_given_stream(self, capsys: Any) -> None:
        ...

    def test_includes_tracebacks_during_exceptions(self, capsys: Any) -> None:
        ...

    def test_does_not_word_wrap_or_crop_messages(self, capsys: Any) -> None:
        ...

    def test_outputs_square_brackets_as_text(self, capsys: Any) -> None:
        ...

    def test_outputs_square_brackets_as_style(self, capsys: Any) -> None:
        ...

class TestJsonFormatter:
    def test_json_log_formatter(self) -> None:
        ...

    def test_json_log_formatter_with_exception(self) -> None:
        ...

class TestObfuscateApiKeyFilter:
    def test_filters_current_api_key(self) -> None:
        ...

    def test_current_api_key_is_not_logged(self, caplog: Any) -> None:
        ...

    def test_current_api_key_is_not_logged_from_flow(self, caplog: Any) -> None:
        ...

    def test_current_api_key_is_not_logged_from_flow_log_prints(self, caplog: Any) -> None:
        ...

    def test_current_api_key_is_not_logged_from_task(self, caplog: Any) -> None:
        ...

    @pytest.mark.parametrize('raw_log_record,expected_log_record', [(['super-mega-admin-key', 'in', 'a', 'list'], ['********', 'in', 'a', 'list']), ({'super-mega-admin-key': 'in', 'a': 'dict'}, {'********': 'in', 'a': 'dict'}), ({'key1': 'some_value', 'key2': [{'nested_key': 'api_key: super-mega-admin-key'}, 'another_value']}, {'key1': 'some_value', 'key2': [{'nested_key': 'api_key: ********'}, 'another_value']})])
    def test_redact_substr_from_collections(self, caplog: Any, raw_log_record: Any, expected_log_record: Any) -> None:
        ...

def test_log_in_flow(caplog: Any) -> None:
    ...

def test_log_in_task(caplog: Any) -> None:
    ...

def test_without_disable_logger(caplog: Any) -> None:
    ...

def test_disable_logger(caplog: Any) -> None:
    ...

def test_disable_run_logger(caplog: Any) -> None:
    ...

def test_patch_print_writes_to_stdout_without_run_context(caplog: Any, capsys: Any) -> None:
    ...

@pytest.mark.parametrize('run_context_cls', [TaskRunContext, FlowRunContext])
def test_patch_print_writes_to_stdout_with_run_context_and_no_log_prints(caplog: Any, capsys: Any, run_context_cls: Type[Union[TaskRunContext, FlowRunContext]]) -> None:
    ...

def test_patch_print_does_not_write_to_logger_with_custom_file(caplog: Any, capsys: Any, task_run: Any) -> None:
    ...

def test_patch_print_writes_to_logger_with_task_run_context(caplog: Any, capsys: Any, task_run: Any) -> None:
    ...

@pytest.mark.parametrize('file', ['stdout', 'stderr'])
def test_patch_print_writes_to_logger_with_explicit_file(caplog: Any, capsys: Any, task_run: Any, file: str) -> None:
    ...

def test_patch_print_writes_to_logger_with_flow_run_context(caplog: Any, capsys: Any, flow_run: Any) -> None:
    ...

def test_log_adapter_get_child(flow_run: Any) -> None:
    ...

def test_eavesdropping() -> None:
    ...