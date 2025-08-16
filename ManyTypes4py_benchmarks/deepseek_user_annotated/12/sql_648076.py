import traceback
from abc import abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, Any, Tuple, List, Optional

import dbt.exceptions
import dbt_common.exceptions.base
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.sql import (
    RemoteCompileResult,
    RemoteCompileResultMixin,
    RemoteRunResult,
    ResultTable,
)
from dbt.events.types import SQLRunnerException
from dbt.task.compile import CompileRunner
from dbt_common.events.functions import fire_event
from dbt.adapters.base import BaseAdapter
from dbt.contracts.graph.nodes import CompiledNode
from dbt.contracts.results import TimingInfo

SQLResult = TypeVar("SQLResult", bound=RemoteCompileResultMixin)


class GenericSqlRunner(CompileRunner, Generic[SQLResult]):
    def __init__(self, config: Any, adapter: BaseAdapter, node: CompiledNode, node_index: int, num_nodes: int) -> None:
        CompileRunner.__init__(self, config, adapter, node, node_index, num_nodes)

    def handle_exception(self, e: Exception, ctx: Any) -> Optional[Exception]:
        fire_event(
            SQLRunnerException(
                exc=str(e), exc_info=traceback.format_exc(), node_info=self.node.node_info
            )
        )
        # REVIEW: This code is invalid and will always throw.
        if isinstance(e, dbt.exceptions.Exception):
            if isinstance(e, dbt_common.exceptions.DbtRuntimeError):
                e.add_node(ctx.node)
            return e
        return None

    def before_execute(self) -> None:
        pass

    def after_execute(self, result: Any) -> None:
        pass

    def compile(self, manifest: Manifest) -> CompiledNode:
        return self.compiler.compile_node(self.node, manifest, {}, write=False)

    @abstractmethod
    def execute(self, compiled_node: CompiledNode, manifest: Manifest) -> SQLResult:
        pass

    @abstractmethod
    def from_run_result(self, result: Any, start_time: datetime, timing_info: List[TimingInfo]) -> SQLResult:
        pass

    def error_result(self, node: CompiledNode, error: Exception, start_time: datetime, timing_info: List[TimingInfo]) -> None:
        raise error

    def ephemeral_result(self, node: CompiledNode, start_time: datetime, timing_info: List[TimingInfo]) -> None:
        raise dbt_common.exceptions.base.NotImplementedError(
            "cannot execute ephemeral nodes remotely!"
        )


class SqlCompileRunner(GenericSqlRunner[RemoteCompileResult]):
    def execute(self, compiled_node: CompiledNode, manifest: Manifest) -> RemoteCompileResult:
        return RemoteCompileResult(
            raw_code=compiled_node.raw_code,
            compiled_code=compiled_node.compiled_code,
            node=compiled_node,
            timing=[],  # this will get added later
            generated_at=datetime.utcnow(),
        )

    def from_run_result(self, result: Any, start_time: datetime, timing_info: List[TimingInfo]) -> RemoteCompileResult:
        return RemoteCompileResult(
            raw_code=result.raw_code,
            compiled_code=result.compiled_code,
            node=result.node,
            timing=timing_info,
            generated_at=datetime.utcnow(),
        )


class SqlExecuteRunner(GenericSqlRunner[RemoteRunResult]):
    def execute(self, compiled_node: CompiledNode, manifest: Manifest) -> RemoteRunResult:
        _, execute_result = self.adapter.execute(compiled_node.compiled_code, fetch=True)

        table = ResultTable(
            column_names=list(execute_result.column_names),
            rows=[list(row) for row in execute_result],
        )

        return RemoteRunResult(
            raw_code=compiled_node.raw_code,
            compiled_code=compiled_node.compiled_code,
            node=compiled_node,
            table=table,
            timing=[],
            generated_at=datetime.utcnow(),
        )

    def from_run_result(self, result: Any, start_time: datetime, timing_info: List[TimingInfo]) -> RemoteRunResult:
        return RemoteRunResult(
            raw_code=result.raw_code,
            compiled_code=result.compiled_code,
            node=result.node,
            table=result.table,
            timing=timing_info,
            generated_at=datetime.utcnow(),
        )
