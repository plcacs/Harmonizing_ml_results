import os
import threading
import time
from typing import AbstractSet, Dict, List, Optional, Type

from dbt.adapters.base import BaseAdapter
from dbt.adapters.base.impl import FreshnessResponse
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.artifacts.schemas.freshness import (
    FreshnessResult,
    FreshnessStatus,
    PartialSourceFreshnessResult,
    SourceFreshnessResult,
)
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import HookNode, SourceDefinition
from dbt.contracts.results import RunStatus
from dbt.graph import ResourceTypeSelector
from dbt.node_types import NodeType

from .base import BaseRunner
from .run import RunTask

RESULT_FILE_NAME: str = ...

class FreshnessRunner(BaseRunner):
    _metadata_freshness_cache: Dict[BaseRelation, FreshnessResponse]

    def __init__(
        self,
        config: object,
        adapter: BaseAdapter,
        node: SourceDefinition,
        node_index: int,
        num_nodes: int,
    ) -> None: ...
    def set_metadata_freshness_cache(
        self, metadata_freshness_cache: Dict[BaseRelation, FreshnessResponse]
    ) -> None: ...
    def on_skip(self) -> None: ...
    def before_execute(self) -> None: ...
    def after_execute(self, result: SourceFreshnessResult) -> None: ...
    def error_result(
        self,
        node: SourceDefinition,
        message: str,
        start_time: float,
        timing_info: List[object],
    ) -> PartialSourceFreshnessResult: ...
    def _build_run_result(
        self,
        node: SourceDefinition,
        start_time: float,
        status: FreshnessStatus,
        timing_info: List[object],
        message: Optional[str],
    ) -> PartialSourceFreshnessResult: ...
    def from_run_result(
        self,
        result: PartialSourceFreshnessResult,
        start_time: float,
        timing_info: List[object],
    ) -> PartialSourceFreshnessResult: ...
    def execute(
        self, compiled_node: SourceDefinition, manifest: Manifest
    ) -> SourceFreshnessResult: ...
    def compile(self, manifest: Manifest) -> SourceDefinition: ...

class FreshnessSelector(ResourceTypeSelector):
    def node_is_match(self, node: object) -> bool: ...

class FreshnessTask(RunTask):
    _metadata_freshness_cache: Dict[BaseRelation, FreshnessResponse]

    def __init__(self, args: object, config: object, manifest: Optional[Manifest]) -> None: ...
    def result_path(self) -> str: ...
    def raise_on_first_error(self) -> bool: ...
    def get_node_selector(self) -> FreshnessSelector: ...
    def before_run(
        self, adapter: BaseAdapter, selected_uids: AbstractSet[str]
    ) -> RunStatus: ...
    def get_runner(self, node: SourceDefinition) -> FreshnessRunner: ...
    def get_runner_type(self, _: object) -> Type[FreshnessRunner]: ...
    def get_result(
        self, results: List[object], elapsed_time: float, generated_at: object
    ) -> FreshnessResult: ...
    def task_end_messages(self, results: List[object]) -> None: ...
    def get_hooks_by_type(self, hook_type: RunHookType) -> List[HookNode]: ...
    def populate_metadata_freshness_cache(
        self, adapter: BaseAdapter, selected_uids: AbstractSet[str]
    ) -> RunStatus: ...
    def get_freshness_metadata_cache(self) -> Dict[BaseRelation, FreshnessResponse]: ...

from dbt.node_types import RunHookType as RunHookType