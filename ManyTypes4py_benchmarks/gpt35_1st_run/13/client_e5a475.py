from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, List, Dict, Union
import httpx
from typing_extensions import TypeVar
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
T = TypeVar('T')
R = TypeVar('R', covariant=True)
if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas import FlowRun, OrchestrationResult
    from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
    from prefect.client.schemas.objects import FlowRunInput, FlowRunPolicy
    from prefect.client.schemas.sorting import FlowRunSort
    from prefect.flows import Flow as FlowObject
    from prefect.states import State
    from prefect.types import KeyValueLabelsField

class FlowRunClient(BaseClient):

    def create_flow_run(self, flow: FlowObject, name: str = None, parameters: Dict[str, Any] = None, context: Dict[str, Any] = None, tags: List[str] = None, parent_task_run_id: UUID = None, state: State = None) -> FlowRun:
    
    def update_flow_run(self, flow_run_id: UUID, flow_version: str = None, parameters: Dict[str, Any] = None, name: str = None, tags: List[str] = None, empirical_policy: FlowRunPolicy = None, infrastructure_pid: str = None, job_variables: Dict[str, Any] = None) -> httpx.Response:
    
    def delete_flow_run(self, flow_run_id: UUID) -> None:
    
    def read_flow_run(self, flow_run_id: UUID) -> FlowRun:
    
    def resume_flow_run(self, flow_run_id: UUID, run_input: Any = None) -> OrchestrationResult:
    
    def read_flow_runs(self, flow_filter: FlowFilter = None, flow_run_filter: FlowRunFilter = None, task_run_filter: TaskRunFilter = None, deployment_filter: DeploymentFilter = None, work_pool_filter: WorkPoolFilter = None, work_queue_filter: WorkQueueFilter = None, sort: FlowRunSort = None, limit: int = None, offset: int = 0) -> List[FlowRun]:
    
    def set_flow_run_state(self, flow_run_id: UUID, state: State, force: bool = False) -> OrchestrationResult:
    
    def read_flow_run_states(self, flow_run_id: UUID) -> List[State]:
    
    def set_flow_run_name(self, flow_run_id: UUID, name: str) -> httpx.Response:
    
    def create_flow_run_input(self, flow_run_id: UUID, key: str, value: Any, sender: Any = None) -> None:
    
    def filter_flow_run_input(self, flow_run_id: UUID, key_prefix: str, limit: int, exclude_keys: List[str]) -> List[FlowRunInput]:
    
    def read_flow_run_input(self, flow_run_id: UUID, key: str) -> str:
    
    def delete_flow_run_input(self, flow_run_id: UUID, key: str) -> None:
    
    def update_flow_run_labels(self, flow_run_id: UUID, labels: Dict[str, str]) -> None:

class FlowRunAsyncClient(BaseAsyncClient):

    async def create_flow_run(self, flow: FlowObject, name: str = None, parameters: Dict[str, Any] = None, context: Dict[str, Any] = None, tags: List[str] = None, parent_task_run_id: UUID = None, state: State = None) -> FlowRun:
    
    async def update_flow_run(self, flow_run_id: UUID, flow_version: str = None, parameters: Dict[str, Any] = None, name: str = None, tags: List[str] = None, empirical_policy: FlowRunPolicy = None, infrastructure_pid: str = None, job_variables: Dict[str, Any] = None) -> httpx.Response:
    
    async def delete_flow_run(self, flow_run_id: UUID) -> None:
    
    async def read_flow_run(self, flow_run_id: UUID) -> FlowRun:
    
    async def resume_flow_run(self, flow_run_id: UUID, run_input: Any = None) -> OrchestrationResult:
    
    async def read_flow_runs(self, flow_filter: FlowFilter = None, flow_run_filter: FlowRunFilter = None, task_run_filter: TaskRunFilter = None, deployment_filter: DeploymentFilter = None, work_pool_filter: WorkPoolFilter = None, work_queue_filter: WorkQueueFilter = None, sort: FlowRunSort = None, limit: int = None, offset: int = 0) -> List[FlowRun]:
    
    async def set_flow_run_state(self, flow_run_id: UUID, state: State, force: bool = False) -> OrchestrationResult:
    
    async def read_flow_run_states(self, flow_run_id: UUID) -> List[State]:
    
    async def set_flow_run_name(self, flow_run_id: UUID, name: str) -> httpx.Response:
    
    async def create_flow_run_input(self, flow_run_id: UUID, key: str, value: Any, sender: Any = None) -> None:
    
    async def filter_flow_run_input(self, flow_run_id: UUID, key_prefix: str, limit: int, exclude_keys: List[str]) -> List[FlowRunInput]:
    
    async def read_flow_run_input(self, flow_run_id: UUID, key: str) -> str:
    
    async def delete_flow_run_input(self, flow_run_id: UUID, key: str) -> None:
    
    async def update_flow_run_labels(self, flow_run_id: UUID, labels: Dict[str, str]) -> None:
