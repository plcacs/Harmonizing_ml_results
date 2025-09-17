from __future__ import annotations
import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, TYPE_CHECKING
from flask_babel import gettext as __
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import (
    SupersetErrorException,
    SupersetErrorsException,
    SupersetGenericDBErrorException,
    SupersetTimeoutException,
)
from superset.sqllab.command_status import SqlJsonExecutionStatus
from superset.utils import core as utils
from superset.utils.core import get_username
from superset.utils.dates import now_as_float

if TYPE_CHECKING:
    from superset.daos.query import QueryDAO
    from superset.sqllab.sqllab_execution_context import SqlJsonExecutionContext

QueryStatus = utils.QueryStatus
logger = logging.getLogger(__name__)
SqlResults = Dict[str, Any]
GetSqlResultsTask = Callable[..., SqlResults]


class SqlJsonExecutor:
    @abstractmethod
    def execute(
        self,
        execution_context: Any,
        rendered_query: str,
        log_params: Dict[str, Any],
    ) -> SqlJsonExecutionStatus:
        raise NotImplementedError()


class SqlJsonExecutorBase(SqlJsonExecutor, ABC):
    def __init__(self, query_dao: QueryDAO, get_sql_results_task: GetSqlResultsTask) -> None:
        self._query_dao: QueryDAO = query_dao
        self._get_sql_results_task: GetSqlResultsTask = get_sql_results_task


class SynchronousSqlJsonExecutor(SqlJsonExecutorBase):
    def __init__(
        self,
        query_dao: QueryDAO,
        get_sql_results_task: GetSqlResultsTask,
        timeout_duration_in_seconds: float,
        sqllab_backend_persistence_feature_enable: bool,
    ) -> None:
        super().__init__(query_dao, get_sql_results_task)
        self._timeout_duration_in_seconds: float = timeout_duration_in_seconds
        self._sqllab_backend_persistence_feature_enable: bool = sqllab_backend_persistence_feature_enable

    def execute(
        self,
        execution_context: SqlJsonExecutionContext,
        rendered_query: str,
        log_params: Dict[str, Any],
    ) -> SqlJsonExecutionStatus:
        query_id: int = execution_context.query.id
        try:
            data: SqlResults = self._get_sql_results_with_timeout(execution_context, rendered_query, log_params)
            execution_context.set_execution_result(data)
        except SupersetTimeoutException:
            raise
        except Exception as ex:
            logger.exception('Query %i failed unexpectedly', query_id)
            raise SupersetGenericDBErrorException(utils.error_msg_from_exception(ex)) from ex
        if data.get('status') == QueryStatus.FAILED:
            if data.get('errors'):
                raise SupersetErrorsException(
                    [SupersetError(**params) for params in data['errors']]
                )
            print(data)
            raise SupersetGenericDBErrorException(data['error'])
        return SqlJsonExecutionStatus.HAS_RESULTS

    def _get_sql_results_with_timeout(
        self,
        execution_context: SqlJsonExecutionContext,
        rendered_query: str,
        log_params: Dict[str, Any],
    ) -> SqlResults:
        with utils.timeout(seconds=self._timeout_duration_in_seconds, error_message=self._get_timeout_error_msg()):
            return self._get_sql_results(execution_context, rendered_query, log_params)

    def _get_sql_results(
        self,
        execution_context: SqlJsonExecutionContext,
        rendered_query: str,
        log_params: Dict[str, Any],
    ) -> SqlResults:
        return self._get_sql_results_task(
            execution_context.query.id,
            rendered_query,
            return_results=True,
            store_results=self._is_store_results(execution_context),
            username=get_username(),
            expand_data=execution_context.expand_data,
            log_params=log_params,
        )

    def _is_store_results(self, execution_context: SqlJsonExecutionContext) -> bool:
        return self._sqllab_backend_persistence_feature_enable and (not execution_context.select_as_cta)

    def _get_timeout_error_msg(self) -> str:
        return f'The query exceeded the {self._timeout_duration_in_seconds} seconds timeout.'


class ASynchronousSqlJsonExecutor(SqlJsonExecutorBase):
    def execute(
        self,
        execution_context: SqlJsonExecutionContext,
        rendered_query: str,
        log_params: Dict[str, Any],
    ) -> SqlJsonExecutionStatus:
        query_id: int = execution_context.query.id
        logger.info('Query %i: Running query on a Celery worker', query_id)
        try:
            task: Any = self._get_sql_results_task.delay(
                query_id,
                rendered_query,
                return_results=False,
                store_results=not execution_context.select_as_cta,
                username=get_username(),
                start_time=now_as_float(),
                expand_data=execution_context.expand_data,
                log_params=log_params,
            )
            try:
                task.forget()
            except NotImplementedError:
                logger.warning('Unable to forget Celery task as backenddoes not support this operation')
        except Exception as ex:
            logger.exception('Query %i: %s', query_id, str(ex))
            message: str = __('Failed to start remote query on a worker.')
            error: SupersetError = SupersetError(
                message=message,
                error_type=SupersetErrorType.ASYNC_WORKERS_ERROR,
                level=ErrorLevel.ERROR,
            )
            error_payload: Dict[str, Any] = dataclasses.asdict(error)
            query = execution_context.query
            query.set_extra_json_key('errors', [error_payload])
            query.status = QueryStatus.FAILED
            query.error_message = message
            raise SupersetErrorException(error) from ex
        return SqlJsonExecutionStatus.QUERY_IS_RUNNING