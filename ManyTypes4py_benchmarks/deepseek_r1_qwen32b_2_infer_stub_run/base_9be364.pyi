from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import datetime as DateTime
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from uuid import UUID

import anyio
import httpx
from pydantic import BaseModel
from prefect.client.schemas.objects import (
    Deployment,
    Flow,
    FlowRun,
    State,
    WorkPool,
    WorkerMetadata,
)
from prefect.client.schemas.responses import DeploymentResponse, WorkerFlowRunResponse
from prefect.events import Event
from prefect.exceptions import Abort, ObjectNotFound
from prefect.utilities.services import CriticalServiceLoop

C = TypeVar("C", bound=BaseModel)
V = TypeVar("V", bound=BaseModel)
R = TypeVar("R", bound=BaseModel)

class BaseJobConfiguration(BaseModel):
    command: Optional[str]
    env: Dict[str, Optional[str]]
    labels: Dict[str, str]
    name: Optional[str]
    _related_objects: Dict[str, Optional[Any]]

    @property
    def is_using_a_runner(self) -> bool:
        ...

    @classmethod
    def _coerce_command(cls, v: Any) -> Optional[str]:
        ...

    @classmethod
    def _coerce_env(cls, v: Any) -> Dict[str, Optional[str]]:
        ...

    @classmethod
    def _get_base_config_defaults(cls, variables: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @classmethod
    @inject_client
    async def from_template_and_values(
        cls, base_job_template: Dict[str, Any], values: Dict[str, Any], client: Optional[httpx.Client] = None
    ) -> BaseJobConfiguration:
        ...

    @classmethod
    def json_template(cls) -> Dict[str, str]:
        ...

    def prepare_for_flow_run(
        self, flow_run: Any, deployment: Optional[Any] = None, flow: Optional[Any] = None
    ) -> None:
        ...

    @staticmethod
    def _base_flow_run_command() -> str:
        ...

    @staticmethod
    def _base_flow_run_labels(flow_run: Any) -> Dict[str, str]:
        ...

    @classmethod
    def _base_environment(cls) -> Dict[str, str]:
        ...

    @staticmethod
    def _base_flow_run_environment(flow_run: Any) -> Dict[str, str]:
        ...

    @staticmethod
    def _base_deployment_labels(deployment: Any) -> Dict[str, str]:
        ...

    @staticmethod
    def _base_flow_labels(flow: Any) -> Dict[str, str]:
        ...

    def _related_resources(self) -> List[Any]:
        ...

class BaseVariables(BaseModel):
    name: Optional[str]
    env: Dict[str, str]
    labels: Dict[str, str]
    command: Optional[str]

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/definitions/{model}",
        schema_generator: Type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: Literal["validation", "serialization"] = "validation",
    ) -> Dict[str, Any]:
        ...

class BaseWorkerResult(BaseModel, ABC):
    def __bool__(self) -> bool:
        ...

class BaseWorker(ABC, Generic[C, V, R]):
    job_configuration: ClassVar[Type[C]]
    job_configuration_variables: ClassVar[Optional[Type[V]]]
    _documentation_url: ClassVar[str]
    _logo_url: ClassVar[str]
    _description: ClassVar[str]

    def __init__(
        self,
        work_pool_name: str,
        work_queues: Optional[Set[str]] = None,
        name: Optional[str] = None,
        prefetch_seconds: Optional[float] = None,
        create_pool_if_not_found: bool = True,
        limit: Optional[int] = None,
        heartbeat_interval_seconds: Optional[float] = None,
        base_job_template: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    @classmethod
    def get_documentation_url(cls) -> str:
        ...

    @classmethod
    def get_logo_url(cls) -> str:
        ...

    @classmethod
    def get_description(cls) -> str:
        ...

    @classmethod
    def get_default_base_job_template(cls) -> Dict[str, Any]:
        ...

    @staticmethod
    def get_worker_class_from_type(type: str) -> Optional[Type[BaseWorker]]:
        ...

    @staticmethod
    def get_all_available_worker_types() -> List[str]:
        ...

    def get_name_slug(self) -> str:
        ...

    def get_flow_run_logger(self, flow_run: Any) -> PrefectLogAdapter:
        ...

    async def start(
        self, run_once: bool = False, with_healthcheck: bool = False, printer: Callable[..., None] = print
    ) -> None:
        ...

    @abstractmethod
    async def run(self, flow_run: Any, configuration: C, task_status: Optional[AnyioTaskStatus] = None) -> R:
        ...

    async def setup(self) -> Any:
        ...

    async def teardown(self, *exc_info: Tuple[Type[BaseException], BaseException, traceback]) -> Any:
        ...

    def is_worker_still_polling(self, query_interval_seconds: float) -> bool:
        ...

    async def get_and_submit_flow_runs(self) -> List[Any]:
        ...

    async def _update_local_work_pool_info(self) -> None:
        ...

    async def _worker_metadata(self) -> Optional[WorkerMetadata]:
        ...

    async def _send_worker_heartbeat(self) -> Optional[UUID]:
        ...

    async def sync_with_backend(self) -> None:
        ...

    def _should_get_worker_id(self) -> bool:
        ...

    async def _get_scheduled_flow_runs(self) -> List[WorkerFlowRunResponse]:
        ...

    async def _submit_scheduled_flow_runs(
        self, flow_run_response: List[WorkerFlowRunResponse]
    ) -> List[Any]:
        ...

    async def _submit_run(self, flow_run: Any) -> None:
        ...

    async def _submit_run_and_capture_errors(
        self, flow_run: Any, task_status: Optional[AnyioTaskStatus] = None
    ) -> Optional[Exception]:
        ...

    def _release_limit_slot(self, flow_run_id: UUID) -> None:
        ...

    def get_status(self) -> Dict[str, Any]:
        ...

    async def _get_configuration(self, flow_run: Any, deployment: Optional[Any] = None) -> C:
        ...

    async def _propose_pending_state(self, flow_run: Any) -> bool:
        ...

    async def _propose_failed_state(self, flow_run: Any, exc: Exception) -> None:
        ...

    async def _propose_crashed_state(self, flow_run: Any, message: str) -> None:
        ...

    async def _mark_flow_run_as_cancelled(
        self, flow_run: Any, state_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        ...

    async def _set_work_pool_template(self, work_pool: WorkPool, job_template: Dict[str, Any]) -> None:
        ...

    async def _schedule_task(
        self, __in_seconds: float, fn: Callable, *args: Any, **kwargs: Any
    ) -> None:
        ...

    async def _give_worker_labels_to_flow_run(self, flow_run_id: UUID) -> None:
        ...

    async def __aenter__(self) -> BaseWorker:
        ...

    async def __aexit__(self, *exc_info: Tuple[Type[BaseException], BaseException, traceback]) -> Any:
        ...

    def __repr__(self) -> str:
        ...

    def _event_resource(self) -> Dict[str, str]:
        ...

    def _event_related_resources(
        self, configuration: Optional[BaseJobConfiguration] = None, include_self: bool = False
    ) -> List[RelatedResource]:
        ...

    def _emit_flow_run_submitted_event(self, configuration: BaseJobConfiguration) -> Event:
        ...

    def _emit_flow_run_executed_event(
        self,
        result: BaseWorkerResult,
        configuration: BaseJobConfiguration,
        submitted_event: Event,
    ) -> None:
        ...

    async def _emit_worker_started_event(self) -> Event:
        ...

    async def _emit_worker_stopped_event(self, started_event: Event) -> None:
        ...