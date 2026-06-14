from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, Union
from uuid import UUID

import anyio
import anyio.abc
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import Literal, Self, TypeVar

from prefect.client.orchestration import PrefectClient
from prefect.events import Event, RelatedResource
from prefect.logging.loggers import PrefectLogAdapter
from prefect.client.schemas.objects import WorkPool, WorkerMetadata
from prefect.types import KeyValueLabels
from prefect.types._datetime import DateTime

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prefect.client.schemas.objects import Flow, FlowRun
    from prefect.client.schemas.responses import DeploymentResponse, WorkerFlowRunResponse


class BaseJobConfiguration(BaseModel):
    command: Optional[str]
    env: Dict[str, Optional[str]]
    labels: Dict[str, str]
    name: Optional[str]
    _related_objects: dict[str, Any]

    @property
    def is_using_a_runner(self) -> bool: ...

    @classmethod
    def _coerce_command(cls, v: Any) -> Optional[str]: ...

    @classmethod
    def _coerce_env(cls, v: dict[str, Any]) -> dict[str, Optional[str]]: ...

    @staticmethod
    def _get_base_config_defaults(variables: dict[str, Any]) -> dict[str, Any]: ...

    @classmethod
    async def from_template_and_values(
        cls,
        base_job_template: dict[str, Any],
        values: dict[str, Any],
        client: Optional[PrefectClient] = None,
    ) -> Self: ...

    @classmethod
    def json_template(cls) -> dict[str, str]: ...

    def prepare_for_flow_run(
        self,
        flow_run: FlowRun,
        deployment: Optional[DeploymentResponse] = None,
        flow: Optional[Flow] = None,
    ) -> None: ...

    @staticmethod
    def _base_flow_run_command() -> str: ...

    @staticmethod
    def _base_flow_run_labels(flow_run: FlowRun) -> dict[str, str]: ...

    @classmethod
    def _base_environment(cls) -> dict[str, str]: ...

    @staticmethod
    def _base_flow_run_environment(flow_run: FlowRun) -> dict[str, str]: ...

    @staticmethod
    def _base_deployment_labels(deployment: DeploymentResponse) -> dict[str, str]: ...

    @staticmethod
    def _base_flow_labels(flow: Flow) -> dict[str, str]: ...

    def _related_resources(self) -> list[RelatedResource]: ...


class BaseVariables(BaseModel):
    name: Optional[str]
    env: Dict[str, Optional[str]]
    labels: Dict[str, str]
    command: Optional[str]

    @classmethod
    def model_json_schema(
        cls,
        by_alias: bool = True,
        ref_template: str = '#/definitions/{model}',
        schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
        mode: str = 'validation',
    ) -> dict[str, Any]: ...


class BaseWorkerResult(BaseModel, abc.ABC):
    status_code: int
    identifier: str

    def __bool__(self) -> bool: ...


C = TypeVar('C', bound=BaseJobConfiguration)
V = TypeVar('V', bound=BaseVariables)
R = TypeVar('R', bound=BaseWorkerResult)


class BaseWorker(abc.ABC, Generic[C, V, R]):
    type: str
    job_configuration: type[BaseJobConfiguration]
    job_configuration_variables: Optional[type[BaseVariables]]
    _documentation_url: str
    _logo_url: str
    _description: str

    name: str
    _started_event: Optional[Event]
    backend_id: Optional[UUID]
    _logger: PrefectLogAdapter
    is_setup: bool
    _create_pool_if_not_found: bool
    _base_job_template: Optional[dict[str, Any]]
    _work_pool_name: str
    _work_queues: set[str]
    _prefetch_seconds: float
    heartbeat_interval_seconds: float
    _work_pool: Optional[WorkPool]
    _exit_stack: Any
    _runs_task_group: Optional[anyio.abc.TaskGroup]
    _client: Optional[PrefectClient]
    _last_polled_time: DateTime
    _limit: Optional[int]
    _limiter: Optional[anyio.CapacityLimiter]
    _submitting_flow_run_ids: set[UUID]
    _cancelling_flow_run_ids: set[UUID]
    _scheduled_task_scopes: set[anyio.CancelScope]
    _worker_metadata_sent: bool

    def __init__(
        self,
        work_pool_name: str,
        work_queues: Optional[list[str]] = None,
        name: Optional[str] = None,
        prefetch_seconds: Optional[float] = None,
        create_pool_if_not_found: bool = True,
        limit: Optional[int] = None,
        heartbeat_interval_seconds: Optional[float] = None,
        *,
        base_job_template: Optional[dict[str, Any]] = None,
    ) -> None: ...

    @classmethod
    def get_documentation_url(cls) -> str: ...

    @classmethod
    def get_logo_url(cls) -> str: ...

    @classmethod
    def get_description(cls) -> str: ...

    @classmethod
    def get_default_base_job_template(cls) -> dict[str, Any]: ...

    @staticmethod
    def get_worker_class_from_type(type: str) -> Optional[type[BaseWorker[Any, Any, Any]]]: ...

    @staticmethod
    def get_all_available_worker_types() -> list[str]: ...

    def get_name_slug(self) -> str: ...

    def get_flow_run_logger(self, flow_run: FlowRun) -> PrefectLogAdapter: ...

    async def start(
        self,
        run_once: bool = False,
        with_healthcheck: bool = False,
        printer: Callable[..., Any] = ...,
    ) -> None: ...

    @abc.abstractmethod
    async def run(
        self,
        flow_run: FlowRun,
        configuration: C,
        task_status: Optional[anyio.abc.TaskStatus[Any]] = None,
    ) -> R: ...

    @classmethod
    def __dispatch_key__(cls) -> Optional[str]: ...

    async def setup(self) -> None: ...

    async def teardown(self, *exc_info: Any) -> None: ...

    def is_worker_still_polling(self, query_interval_seconds: float) -> bool: ...

    async def get_and_submit_flow_runs(self) -> list[FlowRun]: ...

    async def _update_local_work_pool_info(self) -> None: ...

    async def _worker_metadata(self) -> Optional[WorkerMetadata]: ...

    async def _send_worker_heartbeat(self) -> Optional[UUID]: ...

    async def sync_with_backend(self) -> None: ...

    def _should_get_worker_id(self) -> bool: ...

    async def _get_scheduled_flow_runs(self) -> list[WorkerFlowRunResponse]: ...

    async def _submit_scheduled_flow_runs(
        self, flow_run_response: list[WorkerFlowRunResponse]
    ) -> list[FlowRun]: ...

    async def _check_flow_run(self, flow_run: FlowRun) -> None: ...

    async def _submit_run(self, flow_run: FlowRun) -> None: ...

    async def _submit_run_and_capture_errors(
        self,
        flow_run: FlowRun,
        task_status: Optional[anyio.abc.TaskStatus[Any]] = None,
    ) -> Union[R, Exception]: ...

    def _release_limit_slot(self, flow_run_id: UUID) -> None: ...

    def get_status(self) -> dict[str, Any]: ...

    async def _get_configuration(
        self,
        flow_run: FlowRun,
        deployment: Optional[DeploymentResponse] = None,
    ) -> C: ...

    async def _propose_pending_state(self, flow_run: FlowRun) -> bool: ...

    async def _propose_failed_state(self, flow_run: FlowRun, exc: Exception) -> None: ...

    async def _propose_crashed_state(self, flow_run: FlowRun, message: str) -> None: ...

    async def _mark_flow_run_as_cancelled(
        self,
        flow_run: FlowRun,
        state_updates: Optional[dict[str, Any]] = None,
    ) -> None: ...

    async def _set_work_pool_template(
        self, work_pool: WorkPool, job_template: dict[str, Any]
    ) -> None: ...

    async def _schedule_task(
        self, __in_seconds: float, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None: ...

    async def _give_worker_labels_to_flow_run(self, flow_run_id: UUID) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc_info: Any) -> None: ...

    def __repr__(self) -> str: ...

    def _event_resource(self) -> dict[str, str]: ...

    def _event_related_resources(
        self,
        configuration: Optional[BaseJobConfiguration] = None,
        include_self: bool = False,
    ) -> list[RelatedResource]: ...

    def _emit_flow_run_submitted_event(
        self, configuration: BaseJobConfiguration
    ) -> Optional[Event]: ...

    def _emit_flow_run_executed_event(
        self,
        result: R,
        configuration: BaseJobConfiguration,
        submitted_event: Optional[Event],
    ) -> None: ...

    async def _emit_worker_started_event(self) -> Optional[Event]: ...

    async def _emit_worker_stopped_event(self, started_event: Event) -> None: ...