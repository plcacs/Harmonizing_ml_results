from contextlib import AbstractContextManager, nullcontext as does_not_raise
from datetime import datetime, timedelta
import os
from typing import Any
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from homeassistant.components.recorder.const import SupportedDialect
from homeassistant.components.recorder.util import RETRYABLE_MYSQL_ERRORS, database_job_retry_wrapper, retryable_database_job, retryable_database_job_method

def test_database_job_retry_wrapper(side_effect: Exception, dialect: SupportedDialect, retval: Any, expected_result: Any, num_calls: int):
    instance: Any = MagicMock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job = MagicMock(side_effect=side_effect)

    @database_job_retry_wrapper('test', 5)
    def job(instance: Any, *args: Any, **kwargs: Any) -> Any:
        mock_job()
        return retval
    with expected_result:
        assert job(instance) == retval
    assert len(mock_job.mock_calls) == num_calls

def test_retryable_database_job(side_effect: Exception, retval: Any, expected_result: Any, dialect: SupportedDialect):
    instance: Any = MagicMock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job = MagicMock(side_effect=side_effect)

    @retryable_database_job(description='test')
    def job(instance: Any, *args: Any, **kwargs: Any) -> Any:
        mock_job()
        return retval
    with expected_result:
        assert job(instance) == retval
    assert len(mock_job.mock_calls) == 1

def test_retryable_database_job_method(side_effect: Exception, retval: Any, expected_result: Any, dialect: SupportedDialect):
    instance: Any = MagicMock()
    instance.db_retry_wait = 0
    instance.engine.dialect.name = dialect
    mock_job = MagicMock(side_effect=side_effect)

    class Test:
        @retryable_database_job_method(description='test')
        def job(self, instance: Any, *args: Any, **kwargs: Any) -> Any:
            mock_job()
            return retval
    test: Any = Test()
    with expected_result:
        assert test.job(instance) == retval
    assert len(mock_job.mock_calls) == 1
