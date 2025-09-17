from abc import abstractmethod
from datetime import datetime
from typing import Any, Generic, NoReturn, TypeVar

import traceback

import dbt.exceptions
import dbt_common.exceptions.base
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.sql import (RemoteCompileResult, RemoteCompileResultMixin,
                               RemoteRunResult, ResultTable)
from dbt.events.types import SQLRunnerException
from dbt.task.compile import CompileRunner
from dbt_common.events.functions import fire_event

SQLResult = TypeVar('SQLResult', bound=RemoteCompileResultMixin)


class GenericSqlRunner(CompileRunner, Generic[SQLResult]):

    def __init__(self, config: Any, adapter: Any, node: Any, node_index: int, num_nodes: int) -> None:
        CompileRunner.__init__(self, config, adapter, node, node_index, num_nodes)

    def handle_exception(self, e: Exception, ctx: Any) -> Exception:
        fire_event(SQLRunnerException(exc=str(e), exc_info=traceback.format_exc(), node_info=self.node.node_info))
        if isinstance(e, dbt.exceptions.Exception):
            if isinstance(e, dbt_common.exceptions.DbtRuntimeError):
                e.add_node(ctx.node)
            return e
        return e  # in case not matching, still return the exception

    def before_execute(self) -> None:
        pass

    def after_execute(self, result: SQLResult) -> None:
        pass

    def compile(self, manifest: Manifest) -> Any:
        return self.compiler.compile_node(self.node, manifest, {}, write=False)

    @abstractmethod
    def execute(self, compiled_node: Any, manifest: Manifest) -> SQLResult:
        pass

    @abstractmethod
    def from_run_result(self, result: SQLResult, start_time: datetime, timing_info: Any) -> SQLResult:
        pass

    def error_result(self, node: Any, error: Exception, start_time: datetime, timing_info: Any) -> NoReturn:
        raise error

    def ephemeral_result(self, node: Any, start_time: datetime, timing_info: Any) -> NoReturn:
        raise dbt_common.exceptions.base.NotImplementedError('cannot execute ephemeral nodes remotely!')


class SqlCompileRunner(GenericSqlRunner[RemoteCompileResult]):

    def execute(self, compiled_node: Any, manifest: Manifest) -> RemoteCompileResult:
        return RemoteCompileResult(
            raw_code=compiled_node.raw_code,
            compiled_code=compiled_node.compiled_code,
            node=compiled_node,
            timing=[],
            generated_at=datetime.utcnow()
        )

    def from_run_result(self, result: RemoteCompileResult, start_time: datetime, timing_info: Any) -> RemoteCompileResult:
        return RemoteCompileResult(
            raw_code=result.raw_code,
            compiled_code=result.compiled_code,
            node=result.node,
            timing=timing_info,
            generated_at=datetime.utcnow()
        )


class SqlExecuteRunner(GenericSqlRunner[RemoteRunResult]):

    def execute(self, compiled_node: Any, manifest: Manifest) -> RemoteRunResult:
        _, execute_result = self.adapter.execute(compiled_node.compiled_code, fetch=True)
        table = ResultTable(
            column_names=list(execute_result.column_names),
            rows=[list(row) for row in execute_result]
        )
        return RemoteRunResult(
            raw_code=compiled_node.raw_code,
            compiled_code=compiled_node.compiled_code,
            node=compiled_node,
            table=table,
            timing=[],
            generated_at=datetime.utcnow()
        )

    def from_run_result(self, result: RemoteRunResult, start_time: datetime, timing_info: Any) -> RemoteRunResult:
        return RemoteRunResult(
            raw_code=result.raw_code,
            compiled_code=result.compiled_code,
            node=result.node,
            table=result.table,
            timing=timing_info,
            generated_at=datetime.utcnow()
        )