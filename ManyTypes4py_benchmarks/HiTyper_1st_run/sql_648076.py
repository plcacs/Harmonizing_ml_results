import traceback
from abc import abstractmethod
from datetime import datetime
from typing import Generic, TypeVar
import dbt.exceptions
import dbt_common.exceptions.base
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.sql import RemoteCompileResult, RemoteCompileResultMixin, RemoteRunResult, ResultTable
from dbt.events.types import SQLRunnerException
from dbt.task.compile import CompileRunner
from dbt_common.events.functions import fire_event
SQLResult = TypeVar('SQLResult', bound=RemoteCompileResultMixin)

class GenericSqlRunner(CompileRunner, Generic[SQLResult]):

    def __init__(self, config: Union[int, None], adapter: Union[int, None], node: Union[int, None], node_index: Union[int, None], num_nodes: Union[int, None]) -> None:
        CompileRunner.__init__(self, config, adapter, node, node_index, num_nodes)

    def handle_exception(self, e: Any, ctx: Any) -> Union[dbt_@_exceptions_@_Exception, dbt_common_@_exceptions_@_DbtRuntimeError]:
        fire_event(SQLRunnerException(exc=str(e), exc_info=traceback.format_exc(), node_info=self.node.node_info))
        if isinstance(e, dbt.exceptions.Exception):
            if isinstance(e, dbt_common.exceptions.DbtRuntimeError):
                e.add_node(ctx.node)
            return e

    def before_execute(self) -> None:
        pass

    def after_execute(self, result: Union[bool, dict]) -> None:
        pass

    def compile(self, manifest: Union[dbcontracts.graph.manifesManifest, dict, str]) -> str:
        return self.compiler.compile_node(self.node, manifest, {}, write=False)

    @abstractmethod
    def execute(self, compiled_node: Union[dbcontracts.graph.manifesManifest, dict[str, typing.Any], str], manifest: Union[mkdocs2.types.Env, dict[str, typing.Any], list[tuple[typing.Union[str,typing.Any]]]]) -> RemoteRunResult:
        pass

    @abstractmethod
    def from_run_result(self, result: Union[datetime.date, datetime.datetime, str], start_time: Union[int, datetime.datetime, None], timing_info: Union[datetime.date, datetime.datetime, str]) -> RemoteRunResult:
        pass

    def error_result(self, node: Union[str, datetime.datetime, int], error: Union[str, datetime.datetime, int], start_time: Union[str, datetime.datetime, int], timing_info: Union[str, datetime.datetime, int]) -> None:
        raise error

    def ephemeral_result(self, node: Union[datetime.datetime, str, None, int], start_time: Union[datetime.datetime, str, None, int], timing_info: Union[datetime.datetime, str, None, int]) -> None:
        raise dbt_common.exceptions.base.NotImplementedError('cannot execute ephemeral nodes remotely!')

class SqlCompileRunner(GenericSqlRunner[RemoteCompileResult]):

    def execute(self, compiled_node: Union[dbcontracts.graph.manifesManifest, dict[str, typing.Any], str], manifest: Union[mkdocs2.types.Env, dict[str, typing.Any], list[tuple[typing.Union[str,typing.Any]]]]) -> RemoteRunResult:
        return RemoteCompileResult(raw_code=compiled_node.raw_code, compiled_code=compiled_node.compiled_code, node=compiled_node, timing=[], generated_at=datetime.utcnow())

    def from_run_result(self, result: Union[datetime.date, datetime.datetime, str], start_time: Union[int, datetime.datetime, None], timing_info: Union[datetime.date, datetime.datetime, str]) -> RemoteRunResult:
        return RemoteCompileResult(raw_code=result.raw_code, compiled_code=result.compiled_code, node=result.node, timing=timing_info, generated_at=datetime.utcnow())

class SqlExecuteRunner(GenericSqlRunner[RemoteRunResult]):

    def execute(self, compiled_node: Union[dbcontracts.graph.manifesManifest, dict[str, typing.Any], str], manifest: Union[mkdocs2.types.Env, dict[str, typing.Any], list[tuple[typing.Union[str,typing.Any]]]]) -> RemoteRunResult:
        _, execute_result = self.adapter.execute(compiled_node.compiled_code, fetch=True)
        table = ResultTable(column_names=list(execute_result.column_names), rows=[list(row) for row in execute_result])
        return RemoteRunResult(raw_code=compiled_node.raw_code, compiled_code=compiled_node.compiled_code, node=compiled_node, table=table, timing=[], generated_at=datetime.utcnow())

    def from_run_result(self, result: Union[datetime.date, datetime.datetime, str], start_time: Union[int, datetime.datetime, None], timing_info: Union[datetime.date, datetime.datetime, str]) -> RemoteRunResult:
        return RemoteRunResult(raw_code=result.raw_code, compiled_code=result.compiled_code, node=result.node, table=result.table, timing=timing_info, generated_at=datetime.utcnow())