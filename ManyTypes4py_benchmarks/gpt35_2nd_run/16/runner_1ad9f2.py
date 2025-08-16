from dbt.contracts.graph.manifest import Manifest
from dbt.artifacts.resources.types import NodeType
from dbt.artifacts.schemas.results import FreshnessStatus, RunStatus, TestStatus
from dbt.artifacts.schemas.run import RunExecutionResult
from prefect import Client
from prefect._experimental.lineage import emit_external_resource_lineage
from prefect.events.related import related_resources_from_run_context
from prefect.events.schemas.events import RelatedResource
from prefect.exceptions import MissingContextError
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect_dbt.core.profiles import aresolve_profiles_yml, resolve_profiles_yml
from prefect_dbt.core.settings import PrefectDbtSettings
from typing import Any, Callable, Optional
import os

FAILURE_STATUSES: list = [RunStatus.Error, TestStatus.Error, TestStatus.Fail, FreshnessStatus.Error, FreshnessStatus.RuntimeErr]
FAILURE_MSG: str = '{resource_type} {resource_name} {status}ed with message: "{message}"'
REQUIRES_MANIFEST: list = ['build', 'compile', 'docs', 'list', 'ls', 'run', 'run-operation', 'seed', 'show', 'snapshot', 'source', 'test']
NODE_TYPES_TO_EMIT_LINEAGE: list = [NodeType.Model, NodeType.Seed, NodeType.Snapshot]

class PrefectDbtRunner:
    def __init__(self, manifest: Optional[Manifest] = None, settings: Optional[PrefectDbtSettings] = None, raise_on_failure: bool = True, client: Optional[Client] = None) -> None:
    def _get_manifest_depends_on_nodes(self, manifest_node: ManifestNode) -> Any:
    def _emit_lineage_event(self, manifest_node: ManifestNode, related_prefect_context: list) -> None:
    def _emit_node_event(self, manifest_node: ManifestNode, related_prefect_context: list, dbt_event: Any) -> None:
    def _get_dbt_event_msg(self, event: Any) -> str:
    def _create_logging_callback(self, log_level: Any) -> Callable[[Any], None]:
    def _get_dbt_event_node_id(self, event: Any) -> str:
    def _create_events_callback(self, related_prefect_context: list) -> Callable[[Any], None]:
    def parse(self, **kwargs: Any) -> None:
    async def ainvoke(self, args: list, **kwargs: Any) -> Any:
    def invoke(self, args: list, **kwargs: Any) -> Any:
    async def aemit_lineage_events(self) -> None:
    def emit_lineage_events(self) -> None:
