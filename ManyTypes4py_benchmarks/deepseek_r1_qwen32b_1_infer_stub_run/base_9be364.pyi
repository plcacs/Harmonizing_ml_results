from __future__ import annotations
import abc
import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
from uuid import UUID
import anyio
import httpx
from prefect.client.schemas.objects import Flow, FlowRun, Deployment, WorkerFlowRunResponse
from prefect.client.schemas.objects import WorkPool, StateType, WorkerMetadata
from prefect.client.schemas.objects import Integration
from prefect.client.schemas.actions import WorkPoolCreate, WorkPoolUpdate
from prefect.events import Event, RelatedResource
from pydantic import BaseModel

C = TypeVar('C', bound='BaseJobConfiguration')
V = TypeVar('V', bound='BaseVariables')
R = TypeVar('R', bound='BaseWorkerResult')

class BaseJobConfiguration(BaseModel):
    command: Optional[str]
    env: Dict[str, Optional[str]]
    labels: Dict[str, str]
    name: Optional[str]
    _related_objects: Dict[str, Any]

    def prepare_for_flow_run(self, flow_run: FlowRun, deployment: Optional[Deployment] = None, flow: Optional[Flow] = None) -> None: ...

class BaseVariables(BaseModel):
    name: Optional[str]
    env: Dict[str, Optional[str]]
    labels: Dict[str, str]
    command: Optional[str]

class BaseWorkerResult(BaseModel):
    def __init__(self, status_code: int, stdout: Optional[str] = None, stderr: Optional[str] = None) -> None: ...

@register_base_type
class BaseWorker(abc.ABC, Generic[C, V, R]):
    job_configuration: Type[C]
    job_configuration_variables: Type[V]
    _documentation_url: str
    _logo_url: str
    _description: str

    def __init__(self, work_pool_name: str, work_queues: Optional[Set[str]] = None, name: Optional[str] = None, prefetch_seconds: Optional[int] = None, create_pool_if_not_found: bool = True, limit: Optional[int] = None, heartbeat_interval_seconds: Optional[int] = None, base_job_template: Optional[Dict] = None) -> None: ...

    @classmethod
    def get_worker_class_from_type(cls, type: str) -> Optional[Type['BaseWorker']]: ...

    async def start(self, run_once: bool = False, with_healthcheck: bool = False, printer: Callable[[str], None] = print) -> None: ...

    @abc.abstractmethod
    async def run(self, flow_run: FlowRun, configuration: C, task_status: Optional[anyio.abc.TaskStatus] = None) -> R: ...

    async def setup(self) -> None: ...
    async def teardown(self, *exc_info: Any) -> None: ...

    async def get_and_submit_flow_runs(self) -> List[FlowRun]: ...
    async def _get_scheduled_flow_runs(self) -> List[WorkerFlowRunResponse]: ...

    def get_status(self) -> Dict[str, Any]: ...

    async def _get_configuration(self, flow_run: FlowRun, deployment: Optional[Deployment] = None) -> C: ...

    async def _propose_pending_state(self, flow_run: FlowRun) -> bool: ...

    async def _propose_crashed_state(self, flow_run: FlowRun, message: str) -> None: ...

    async def _mark_flow_run_as_cancelled(self, flow_run: FlowRun, state_updates: Optional[Dict] = None) -> None: ...

    async def _give_worker_labels_to_flow_run(self, flow_run_id: UUID) -> None: ...

    async def __aenter__(self) -> 'BaseWorker': ...
    async def __aexit__(self, *exc_info: Any) -> None: ...

    def __repr__(self) -> str: ...