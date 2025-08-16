import os
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Mapping
import pytest
import yaml
import dbt.flags as flags
from dbt.adapters.factory import get_adapter, get_adapter_by_type, register_adapter, reset_adapters
from dbt.config.runtime import RuntimeConfig
from dbt.context.providers import generate_runtime_macro_context
from dbt.events.logging import setup_event_logger
from dbt.mp_context import get_mp_context
from dbt.parser.manifest import ManifestLoader
from dbt.tests.util import TestProcessingException, get_connection, run_sql_with_adapter, write_file
from dbt_common.context import set_invocation_context
from dbt_common.events.event_manager_client import cleanup_event_logger
from dbt_common.exceptions import CompilationError, DbtDatabaseError
from dbt_common.tests import enable_test_caching

@pytest.fixture(scope='class')
def prefix() -> str:
    _randint: int = random.randint(0, 9999)
    _runtime_timedelta: datetime = datetime.utcnow() - datetime(1970, 1, 1, 0, 0, 0)
    _runtime: int = int(_runtime_timedelta.total_seconds() * 1000000.0) + _runtime_timedelta.microseconds
    prefix: str = f'test{_runtime}{_randint:04}'
    return prefix

# Add type annotations for other fixtures as well
