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

    def __init__(self, config, adapter, node, node_index, num_nodes):
        CompileRunner.__init__(self, config, adapter, node, node_index, num_nodes)

    def handle_exception(self, e, ctx):
        fire_event(SQLRunnerException(exc=str(e), exc_info=traceback.format_exc(), node_info=self.node.node_info))
        if isinstance(e, dbt.exceptions.Exception):
            if isinstance(e, dbt_common.exceptions.DbtRuntimeError):
                e.add_node(ctx.node)
            return e

    def before_execute(self):
        pass

    def after_execute(self, result):
        pass

    def compile(self, manifest):
        return self.compiler.compile_node(self.node, manifest, {}, write=False)

    @abstractmethod
    def execute(self, compiled_node, manifest):
        pass

    @abstractmethod
    def from_run_result(self, result, start_time, timing_info):
        pass

    def error_result(self, node, error, start_time, timing_info):
        raise error

    def ephemeral_result(self, node, start_time, timing_info):
        raise dbt_common.exceptions.base.NotImplementedError('cannot execute ephemeral nodes remotely!')

class SqlCompileRunner(GenericSqlRunner[RemoteCompileResult]):

    def execute(self, compiled_node, manifest):
        return RemoteCompileResult(raw_code=compiled_node.raw_code, compiled_code=compiled_node.compiled_code, node=compiled_node, timing=[], generated_at=datetime.utcnow())

    def from_run_result(self, result, start_time, timing_info):
        return RemoteCompileResult(raw_code=result.raw_code, compiled_code=result.compiled_code, node=result.node, timing=timing_info, generated_at=datetime.utcnow())

class SqlExecuteRunner(GenericSqlRunner[RemoteRunResult]):

    def execute(self, compiled_node, manifest):
        _, execute_result = self.adapter.execute(compiled_node.compiled_code, fetch=True)
        table = ResultTable(column_names=list(execute_result.column_names), rows=[list(row) for row in execute_result])
        return RemoteRunResult(raw_code=compiled_node.raw_code, compiled_code=compiled_node.compiled_code, node=compiled_node, table=table, timing=[], generated_at=datetime.utcnow())

    def from_run_result(self, result, start_time, timing_info):
        return RemoteRunResult(raw_code=result.raw_code, compiled_code=result.compiled_code, node=result.node, table=result.table, timing=timing_info, generated_at=datetime.utcnow())