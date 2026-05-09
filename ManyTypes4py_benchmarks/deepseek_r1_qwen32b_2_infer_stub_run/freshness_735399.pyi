import os
import threading
import time
from typing import AbstractSet, Dict, List, Optional, Type

from dbt.adapters.base import BaseAdapter
from dbt.adapters.base.impl import FreshnessResponse
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.capability import Capability
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.artifacts.schemas.freshness import (
    FreshnessResult,
    FreshnessStatus,
    PartialSourceFreshnessResult,
    SourceFreshnessResult,
)
from dbt.clients import jinja
from dbt.context.providers import RuntimeProvider, SourceContext
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import HookNode, SourceDefinition
from dbt.contracts.results import RunStatus
from dbt.events.types import FreshnessCheckComplete, LogFreshnessResult, LogStartLine
from dbt.graph import ResourceTypeSelector
from dbt.node_types import NodeType, RunHookType
from dbt_common.events.base_types import EventLevel
from dbt_common.exceptions import DbtInternalError, DbtRuntimeError
from .base import BaseRunner
from .printer import print_run_result_error
from .run import RunTask

RESULT_FILE_NAME: str = ...

class FreshnessRunner(BaseRunner):
    _metadata_freshness_cache: Dict[BaseRelation, Dict[str, float]]

    def __init__(self, config: object, adapter: BaseAdapter, node: object, node_index: int, num_nodes: int) -> None:
        ...

    def set_metadata_freshness_cache(self, metadata_freshness_cache: Dict[BaseRelation, Dict[str, float]]) -> None:
        ...

    def on_skip(self) -> None:
        ...

    def before_execute(self) -> None:
        ...

    def after_execute(self, result: object) -> None:
        ...

    def error_result(self, node: object, message: str, start_time: float, timing_info: List[Dict[str, float]]) -> PartialSourceFreshnessResult:
        ...

    def _build_run_result(self, node: object, start_time: float, status: FreshnessStatus, timing_info: List[Dict[str, float]], message: Optional[str]) -> PartialSourceFreshnessResult:
        ...

    def from_run_result(self, result: PartialSourceFreshnessResult, start_time: float, timing_info: List[Dict[str, float]]) -> PartialSourceFreshnessResult:
        ...

    def execute(self, compiled_node: SourceDefinition, manifest: Manifest) -> SourceFreshnessResult:
        ...

    def compile(self, manifest: Manifest) -> SourceDefinition:
        ...

class FreshnessSelector(ResourceTypeSelector):
    def node_is_match(self, node: object) -> bool:
        ...

class FreshnessTask(RunTask):
    _metadata_freshness_cache: Dict[BaseRelation, Dict[str, float]]

    def __init__(self, args: object, config: object, manifest: Manifest) -> None:
        ...

    def result_path(self) -> str:
        ...

    def raise_on_first_error(self) -> bool:
        ...

    def get_node_selector(self) -> FreshnessSelector:
        ...

    def before_run(self, adapter: BaseAdapter, selected_uids: AbstractSet[str]) -> RunStatus:
        ...

    def get_runner(self, node: object) -> FreshnessRunner:
        ...

    def get_runner_type(self, _) -> Type[FreshnessRunner]:
        ...

    def get_result(self, results: List[SourceFreshnessResult], elapsed_time: float, generated_at: float) -> FreshnessResult:
        ...

    def task_end_messages(self, results: List[SourceFreshnessResult]) -> None:
        ...

    def get_hooks_by_type(self, hook_type: RunHookType) -> List[HookNode]:
        ...

    def populate_metadata_freshness_cache(self, adapter: BaseAdapter, selected_uids: List[str]) -> RunStatus:
        ...

    def get_freshness_metadata_cache(self) -> Dict[BaseRelation, Dict[str, float]]:
        ...