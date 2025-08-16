import json
import logging
import sys
import time
import uuid
from contextlib import nullcontext
from functools import partial
from io import StringIO
from typing import Any, Dict, List, Optional, Type, Union
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
from prefect.logging.configuration import (
    DEFAULT_LOGGING_SETTINGS_PATH,
    load_logging_config,
    setup_logging,
)
from prefect.logging.filters import ObfuscateApiKeyFilter
from prefect.logging.formatters import JsonFormatter
from prefect.logging.handlers import (
    APILogHandler,
    APILogWorker,
    PrefectConsoleHandler,
    WorkerAPILogHandler,
)
from prefect.logging.highlighters import PrefectConsoleHighlighter
from prefect.logging.loggers import (
    PrefectLogAdapter,
    disable_logger,
    disable_run_logger,
    flow_run_logger,
    get_logger,
    get_run_logger,
    get_worker_logger,
    patch_print,
    task_run_logger,
)
from prefect.server.schemas.actions import LogCreate
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
from prefect.testing.cli import temporary_console_width
from prefect.testing.utilities import AsyncMock
from prefect.utilities.names import obfuscate
from prefect.workers.base import BaseJobConfiguration, BaseWorker
