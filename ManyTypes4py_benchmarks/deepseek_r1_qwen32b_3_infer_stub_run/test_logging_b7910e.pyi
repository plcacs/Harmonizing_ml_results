from contextlib import nullcontext
from functools import partial
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import ANY, MagicMock
from uuid import UUID

import json
import logging
import pendulum
import pytest
import prefect
import pytest
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
def dictConfigMock(monkeypatch) -> MagicMock:
    ...

@pytest.fixture
async def logger_test_deployment(prefect_client) -> str:
    ...

@pytest.fixture()
def external_logger_setup(request) -> tuple[str, int, bool]:
    ...

class TestAPILogHandler:
    @pytest.fixture
    def handler(self) -> APILogHandler:
        ...

    @pytest.fixture
    def logger(self, handler) -> logging.Logger:
        ...

    def test_worker_is_not_flushed_on_handler_close(self, mock_log_worker) -> None:
        ...

    async def test_logs_can_still_be_sent_after_close(self, logger, handler, flow_run, prefect_client) -> None:
        ...

    async def test_logs_can_still_be_sent_after_flush(self, logger, handler, flow_run, prefect_client) -> None:
        ...

    def test_sync_flush_from_async_context(self, logger, handler, flow_run) -> None:
        ...

    def test_sync_flush_from_global_event_loop(self, logger, handler, flow_run) -> None:
        ...

    def test_sync_flush_from_sync_context(self, logger, handler, flow_run) -> None:
        ...

    def test_sends_task_run_log_to_worker(self, logger, mock_log_worker, task_run) -> None:
        ...

    def test_sends_flow_run_log_to_worker(self, logger, mock_log_worker, flow_run) -> None:
        ...

    @pytest.mark.parametrize('with_context', [True, False])
    def test_respects_explicit_flow_run_id(self, logger, mock_log_worker, flow_run, with_context) -> None:
        ...

    @pytest.mark.parametrize('with_context', [True, False])
    def test_respects_explicit_task_run_id(self, logger, mock_log_worker, flow_run, with_context, task_run) -> None:
        ...

    def test_does_not_emit_logs_below_level(self, logger, mock_log_worker) -> None:
        ...

    def test_explicit_task_run_id_still_requires_flow_run_id(self, logger, mock_log_worker) -> None:
        ...

    def test_sets_timestamp_from_record_created_time(self, logger, mock_log_worker, flow_run, handler) -> None:
        ...

    def test_sets_timestamp_from_time_if_missing_from_recrod(self, logger, mock_log_worker, flow_run, handler, monkeypatch) -> None:
        ...

    def test_does_not_send_logs_that_opt_out(self, logger, mock_log_worker, task_run) -> None:
        ...

    def test_does_not_send_logs_when_handler_is_disabled(self, logger, mock_log_worker, task_run) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_default_setting(self, logger, mock_log_worker, capsys) -> None:
        ...

    def test_does_not_raise_when_logger_outside_of_run_context_with_default_setting(self, logger) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_error_setting(self, logger, mock_log_worker, capsys) -> None:
        ...

    def test_does_not_warn_when_logger_outside_of_run_context_with_error_setting(self, logger) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_ignore_setting(self, logger, mock_log_worker, capsys) -> None:
        ...

    def test_does_not_raise_or_warn_when_logger_outside_of_run_context_with_ignore_setting(self, logger) -> None:
        ...

    def test_does_not_send_logs_outside_of_run_context_with_warn_setting(self, logger, mock_log_worker, capsys) -> None:
        ...

    def test_does_not_raise_when_logger_outside_of_run_context_with_warn_setting(self, logger) -> None:
        ...

    def test_missing_context_warning_refers_to_caller_lineno(self, logger, mock_log_worker) -> None:
        ...

    def test_writes_logging_errors_to_stderr(self, logger, mock_log_worker, capsys, monkeypatch) -> None:
        ...

    def test_does_not_write_error_for_logs_outside_run_context_that_opt_out(self, logger, mock_log_worker, capsys) -> None:
        ...

    async def test_does_not_enqueue_logs_that_are_too_big(self, task_run, logger, capsys, mock_log_worker) -> None:
        ...

    def test_handler_knows_how_large_logs_are(self) -> None:
        ...

class TestWorkerLogging:
    class CloudWorkerTestImpl(BaseWorker):
        ...

    class ServerWorkerTestImpl(BaseWorker):
        ...

    @pytest.fixture
    def logging_to_api_enabled(self) -> None:
        ...

    @pytest.fixture
    def worker_handler(self) -> WorkerAPILogHandler:
        ...

    @pytest.fixture
    def logger(self, worker_handler) -> logging.Logger:
        ...

    async def test_get_worker_logger_works_with_no_backend_id(self) -> None:
        ...

    async def test_get_worker_logger_works_with_backend_id(self) -> None:
        ...

    async def test_worker_emits_logs_with_worker_id(self, caplog) -> None:
        ...

    async def test_worker_logger_sends_log_to_api_worker_when_connected_to_cloud(self, mock_log_worker, worker_handler, logging_to_api_enabled) -> None:
        ...

    async def test_worker_logger_does_not_send_logs_when_not_connected_to_cloud(self, mock_log_worker, worker_handler, logging_to_api_enabled) -> None:
        ...

class TestAPILogWorker:
    @pytest.fixture
    async def worker(self) -> APILogWorker:
        ...

    @pytest.fixture
    def log_dict(self) -> dict:
        ...

    async def test_send_logs_single_record(self, log_dict, prefect_client, worker) -> None:
        ...

    async def test_send_logs_many_records(self, log_dict, prefect_client, worker) -> None:
        ...

    async def test_send_logs_writes_exceptions_to_stderr(self, log_dict, capsys, monkeypatch, worker) -> None:
        ...

    async def test_send_logs_batches_by_size(self, log_dict, monkeypatch) -> None:
        ...

    async def test_logs_are_sent_immediately_when_stopped(self, log_dict, prefect_client) -> None:
        ...

    async def test_logs_are_sent_immediately_when_flushed(self, log_dict, prefect_client, worker) -> None:
        ...

    async def test_logs_include_worker_id_if_available(self, worker, log_dict, prefect_client) -> None:
        ...

class TestPrefectConsoleHandler:
    @pytest.fixture
    def handler(self) -> PrefectConsoleHandler:
        ...

    @pytest.fixture
    def logger(self, handler) -> logging.Logger:
        ...

    def test_init_defaults(self) -> None:
        ...

    def test_init_styled_console_disabled(self) -> None:
        ...

    def test_init_override_kwargs(self) -> None:
        ...

    def test_uses_stderr_by_default(self, capsys) -> None:
        ...

    def test_respects_given_stream(self, capsys) -> None:
        ...

    def test_includes_tracebacks_during_exceptions(self, capsys) -> None:
        ...

    def test_does_not_word_wrap_or_crop_messages(self, capsys) -> None:
        ...

    def test_outputs_square_brackets_as_text(self, capsys) -> None:
        ...

    def test_outputs_square_brackets_as_style(self, capsys) -> None:
        ...

class TestJsonFormatter:
    def test_json_log_formatter(self) -> None:
        ...

    def test_json_log_formatter_with_exception(self) -> None:
        ...

class TestObfuscateApiKeyFilter:
    def test_filters_current_api_key(self) -> None:
        ...

    def test_current_api_key_is_not_logged(self, caplog) -> None:
        ...

    def test_current_api_key_is_not_logged_from_flow(self, caplog) -> None:
        ...

    def test_current_api_key_is_not_logged_from_flow_log_prints(self, caplog) -> None:
        ...

    def test_current_api_key_is_not_logged_from_task(self, caplog) -> None:
        ...

    @pytest.mark.parametrize('raw_log_record,expected_log_record', [(['super-mega-admin-key', 'in', 'a', 'list'], ['********', 'in', 'a', 'list']), ({'super-mega-admin-key': 'in', 'a': 'dict'}, {'********': 'in', 'a': 'dict'}), ({'key1': 'some_value', 'key2': [{'nested_key': 'api_key: super-mega-admin-key'}, 'another_value']}, {'key1': 'some_value', 'key2': [{'nested_key': 'api_key: ********'}, 'another_value']})])
    def test_redact_substr_from_collections(self, caplog, raw_log_record, expected_log_record) -> None:
        ...

def test_log_in_flow(caplog) -> None:
    ...

def test_log_in_task(caplog) -> None:
    ...

def test_without_disable_logger(caplog) -> None:
    ...

def test_disable_logger(caplog) -> None:
    ...

def test_disable_run_logger(caplog) -> None:
    ...

def test_patch_print_writes_to_stdout_without_run_context(caplog, capsys) -> None:
    ...

@pytest.mark.parametrize('run_context_cls', [TaskRunContext, FlowRunContext])
def test_patch_print_writes_to_stdout_with_run_context_and_no_log_prints(caplog, capsys, run_context_cls) -> None:
    ...

def test_patch_print_does_not_write_to_logger_with_custom_file(caplog, capsys, task_run) -> None:
    ...

def test_patch_print_writes_to_logger_with_task_run_context(caplog, capsys, task_run) -> None:
    ...

@pytest.mark.parametrize('file', ['stdout', 'stderr'])
def test_patch_print_writes_to_logger_with_explicit_file(caplog, capsys, task_run, file) -> None:
    ...

def test_patch_print_writes_to_logger_with_flow_run_context(caplog, capsys, flow_run) -> None:
    ...

def test_log_adapter_get_child(flow_run) -> None:
    ...

def test_eavesdropping() -> None:
    ...