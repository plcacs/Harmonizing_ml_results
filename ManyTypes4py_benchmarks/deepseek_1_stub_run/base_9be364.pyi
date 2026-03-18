```python
from __future__ import annotations
import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    Union,
)
from uuid import UUID
from typing_extensions import Literal, Self, TypeVar
from pydantic import BaseModel, Field
from prefect.client.schemas.objects import (
    Integration,
    StateType,
    WorkerMetadata,
    WorkPool,
)
from prefect.client.schemas.actions import WorkPoolCreate, WorkPoolUpdate
from prefect.client.schemas.responses import DeploymentResponse, WorkerFlowRunResponse
from prefect.events import Event, RelatedResource
from prefect.settings import Settings
from prefect.types import KeyValueLabels
from prefect.types._datetime import DateTime

if TYPE_CHECKING:
    from prefect.client.schemas.objects import Flow, FlowRun

C = TypeVar("C", bound="BaseJobConfiguration")
V = TypeVar("V", bound="BaseVariables")
R = TypeVar("R", bound="BaseWorkerResult")


class BaseJobConfiguration(BaseModel):
    command: Optional[str] = ...
    env: Dict[str, Optional[str]] = ...
    labels: Dict[str, Any] = ...
    name: Optional[str] = ...
    _related_objects: Dict[str, Any] = ...

    @property
    def is_using_a_runner(self) -> bool: ...

    @classmethod
    def _coerce_command(cls, v: Any) -> Any: ...

    @classmethod
    def _coerce_env(cls, v: Any) -> Any: ...

    @staticmethod
    def _get_base_config_defaults(variables: Any) -> Any: ...

    @classmethod
    async def from_template_and_values(
        cls,
        base_job_template: Any,
        values: Any,
        client: Optional[Any] = None,
    ) -> Self: ...

    @classmethod
    def json_template(cls) -> Dict[str, str]: ...

    def prepare_for_flow_run(
        self,
        flow_run: "FlowRun",
        deployment: Optional["DeploymentResponse"] = None,
        flow: Optional["Flow"] = None,
    ) -> None: ...

    @staticmethod
    def _base_flow_run_command() -> str: ...

    @staticmethod
    def _base_flow_run_labels(flow_run: "FlowRun") -> Dict[str, str]: ...

    @classmethod
    def _base_environment(cls) -> Dict[str, str]: ...

    @staticmethod
    def _base_flow_run_environment(flow_run: "FlowRun") -> Dict[str, str]: ...

    @staticmethod
    def _base_deployment_labels(deployment: "DeploymentResponse") -> Dict[str, str]: ...

    @staticmethod
    def _base_flow_labels(flow: "Flow") -> Dict[str, str]: ...

    def _related_resources(self) -> List[RelatedResource]: ...


class BaseVariables(BaseModel):
    name: Optional[str] = ...
    env: Dict[str, Any] = ...
    labels: Dict[str, Any] = ...
    command: Optional[str] = ...

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = "#/definitions/{model}",
        schema_generator: Any = ...,
        mode: str = "validation",
    ) -> Dict[str, Any]: ...


class BaseWorkerResult(BaseModel, abc.ABC):
    def __bool__(self) -> bool: ...


class BaseWorker(abc.ABC, Generic[C, V, R]):
    job_configuration: Type[BaseJobConfiguration] = ...
    job_configuration_variables: Optional[Type[BaseVariables]] = ...
    _documentation_url: str = ...
    _logo_url: str = ...
    _description: str = ...

    def __init__(
        self,
        work_pool_name: str,
        work_queues: Optional[List[str]] = None,
        name: Optional[str] = None,
        prefetch_seconds: Optional[float] = None,
        create_pool_if_not_found: bool = True,
        limit: Optional[int] = None,
        heartbeat_interval_seconds: Optional[float] = None,
        *,
        base_job_template: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @classmethod
    def get_documentation_url(cls) -> str: ...

    @classmethod
    def get_logo_url(cls) -> str: ...

    @classmethod
    def get_description(cls) -> str: ...

    @classmethod
    def get_default_base_job_template(cls) -> Dict[str, Any]: ...

    @staticmethod
    def get_worker_class_from_type(type: str) -> Optional[Type["BaseWorker"]]: ...

    @staticmethod
    def get_all_available_worker_types() -> List[str]: ...

    def get_name_slug(self) -> str: ...

    def get_flow_run_logger(self, flow_run: "FlowRun") -> Any: ...

    async def start(
        self,
        run_once: bool = False,
        with_healthcheck: bool = False,
        printer: Callable[..., Any] = print,
    ) -> None: ...

    @abc.abstractmethod
    async def run(
        self,
        flow_run: "FlowRun",
        configuration: C,
        task_status: Optional[Any] = None,
    ) -> R: ...

    @classmethod
    def __dispatch_key__(cls) -> Optional[str]: ...

    async def setup(self) -> None: ...

    async def teardown(self, *exc_info: Any) -> None: ...

    def is_worker_still_polling(self, query_interval_seconds: float) -> bool: ...

    async def get_and_submit_flow_runs(self) -> List["FlowRun"]: ...

    async def _update_local_work_pool_info(self) -> None: ...

    async def _worker_metadata(self) -> Optional[WorkerMetadata]: ...

    async def _send_worker_heartbeat(self) -> Optional[UUID]: ...

    async def sync_with_backend(self) -> None: ...

    def _should_get_worker_id(self) -> bool: ...

    async def _get_scheduled_flow_runs(self) -> List[WorkerFlowRunResponse]: ...

    async def _submit_scheduled_flow_runs(
        self, flow_run_response: List[WorkerFlowRunResponse]
    ) -> List["FlowRun"]: ...

    async def _check_flow_run(self, flow_run: "FlowRun") -> None: ...

    async def _submit_run(self, flow_run: "FlowRun") -> None: ...

    async def _submit_run_and_capture_errors(
        self, flow_run: "FlowRun", task_status: Optional[Any] = None
    ) -> Any: ...

    def _release_limit_slot(self, flow_run_id: UUID) -> None: ...

    def get_status(self) -> Dict[str, Any]: ...

    async def _get_configuration(
        self,
        flow_run: "FlowRun",
        deployment: Optional["DeploymentResponse"] = None,
    ) -> C: ...

    async def _propose_pending_state(self, flow_run: "FlowRun") -> bool: ...

    async def _propose_failed_state(self, flow_run: "FlowRun", exc: Exception) -> None: ...

    async def _propose_crashed_state(
        self, flow_run: "FlowRun", message: str
    ) -> None: ...

    async def _mark_flow_run_as_cancelled(
        self, flow_run: "FlowRun", state_updates: Optional[Dict[str, Any]] = None
    ) -> None: ...

    async def _set_work_pool_template(
        self, work_pool: WorkPool, job_template: Dict[str, Any]
    ) -> None: ...

    async def _schedule_task(
        self, __in_seconds: float, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None: ...

    async def _give_worker_labels_to_flow_run(self, flow_run_id: UUID) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc_info: Any) -> None: ...

    def __repr__(self) -> str: ...

    def _event_resource(self) -> Dict[str, str]: ...

    def _event_related_resources(
        self, configuration: Optional[C] = None, include_self: bool = False
    ) -> List[RelatedResource]: ...

    def _emit_flow_run_submitted_event(self, configuration: C) -> Event: ...

    def _emit_flow_run_executed_event(
        self, result: R, configuration: C, submitted_event: Event
    ) -> None: ...

    async def _emit_worker_started_event(self) -> Event: ...

    async def _emit_worker_stopped_event(self, started_event: Event) -> None: ...
```