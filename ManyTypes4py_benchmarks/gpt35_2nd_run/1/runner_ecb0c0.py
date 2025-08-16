from typing import Any, List, Optional, Union
from prefect._internal.schemas.validators import reconcile_paused_deployment, reconcile_schedules_runner
from prefect.client.base import ServerType
from prefect.client.schemas.actions import DeploymentScheduleCreate, DeploymentUpdate
from prefect.client.schemas.filters import WorkerFilter, WorkerFilterStatus
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyOptions
from prefect.client.schemas.schedules import SCHEDULE_TYPES
from prefect.deployments.schedules import create_deployment_schedule_create
from prefect.docker.docker_image import DockerImage
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.exceptions import ObjectNotFound, PrefectHTTPStatusError
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import PREFECT_DEFAULT_WORK_POOL_NAME, PREFECT_UI_URL
from prefect.types import ListOfNonEmptyStrings
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import run_coro_as_sync, sync_compatible
from prefect.utilities.callables import ParameterSchema, parameter_schema
from prefect.utilities.collections import get_from_dict, isiterable
from prefect.utilities.dockerutils import parse_image_tag
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from rich.table import Table

class DeploymentApplyError(RuntimeError):
    """
    Raised when an error occurs while applying a deployment.
    """

class RunnerDeployment(BaseModel):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(..., description='The name of the deployment.')
    flow_name: Optional[str] = Field(None, description='The name of the underlying flow; typically inferred.')
    description: Optional[str] = Field(default=None, description='An optional description of the deployment.')
    version: Optional[str] = Field(default=None, description='An optional version for the deployment.')
    tags: List[str] = Field(default_factory=list, description='One of more tags to apply to this deployment.')
    schedules: Optional[List[DeploymentScheduleCreate]] = Field(default=None, description='The schedules that should cause this deployment to run.')
    concurrency_limit: Optional[int] = Field(default=None, description='The maximum number of concurrent runs of this deployment.')
    concurrency_options: Optional[ConcurrencyOptions] = Field(default=None, description='The concurrency limit config for the deployment.')
    paused: Optional[bool] = Field(default=None, description='Whether or not the deployment is paused.')
    parameters: dict = Field(default_factory=dict)
    entrypoint: Optional[str] = Field(default=None, description='The path to the entrypoint for the workflow, relative to the `path`.')
    triggers: List[TriggerTypes] = Field(default_factory=list, description='The triggers that should cause this deployment to run.')
    enforce_parameter_schema: bool = Field(default=True, description='Whether or not the Prefect API should enforce the parameter schema for this deployment.')
    storage: Optional[RunnerStorage] = Field(default=None, description='The storage object used to retrieve flow code for this deployment.')
    work_pool_name: Optional[str] = Field(default=None, description='The name of the work pool to use for this deployment.')
    work_queue_name: Optional[str] = Field(default=None, description='The name of the work queue to use for this deployment.')
    job_variables: dict = Field(default_factory=dict, description='Job variables used to override the default values of a work pool base job template.')
    _sla: Optional[Any] = PrivateAttr(default=None)
    _entrypoint_type: EntrypointType = PrivateAttr(default=EntrypointType.FILE_PATH)
    _path: Optional[str] = PrivateAttr(default=None)
    _parameter_openapi_schema: ParameterSchema = PrivateAttr(default_factory=ParameterSchema)

    @property
    def entrypoint_type(self) -> EntrypointType:
        return self._entrypoint_type

    @property
    def full_name(self) -> str:
        return f'{self.flow_name}/{self.name}'

    @field_validator('name', mode='before')
    @classmethod
    def validate_name(cls, value: str) -> str:
        if value.endswith('.py'):
            return Path(value).stem
        return value

    @model_validator(mode='after')
    def validate_automation_names(self) -> 'RunnerDeployment':
        """Ensure that each trigger has a name for its automation if none is provided."""
        for i, trigger in enumerate(self.triggers, start=1):
            if trigger.name is None:
                trigger.name = f'{self.name}__automation_{i}'
        return self

    @model_validator(mode='before')
    @classmethod
    def reconcile_paused(cls, values: dict) -> dict:
        return reconcile_paused_deployment(values)

    @model_validator(mode='before')
    @classmethod
    def reconcile_schedules(cls, values: dict) -> dict:
        return reconcile_schedules_runner(values)

    async def _create(self, work_pool_name: Optional[str] = None, image: Optional[str] = None) -> UUID:
        ...

    async def _update(self, deployment_id: UUID, client: PrefectClient) -> UUID:
        ...

    async def _create_triggers(self, deployment_id: UUID, client: PrefectClient) -> None:
        ...

    @sync_compatible
    async def apply(self, work_pool_name: Optional[str] = None, image: Optional[str] = None) -> UUID:
        ...

    async def _create_slas(self, deployment_id: UUID, client: PrefectClient) -> None:
        ...

    @staticmethod
    def _construct_deployment_schedules(interval: Optional[Union[int, List[int], timedelta, List[timedelta]]] = None, anchor_date: Optional[datetime] = None, cron: Optional[Union[str, List[str]]] = None, rrule: Optional[Union[str, List[str]]] = None, timezone: Optional[str] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None) -> List[DeploymentScheduleCreate]:
        ...

    def _set_defaults_from_flow(self, flow: 'Flow') -> None:
        ...

    @classmethod
    def from_flow(cls, flow: 'Flow', name: str, interval: Optional[Union[int, timedelta]] = None, cron: Optional[Union[str, List[str]]] = None, rrule: Optional[Union[str, List[str]]] = None, paused: Optional[bool] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None, concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig]] = None, parameters: Optional[dict] = None, triggers: Optional[List[TriggerTypes]] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, version: Optional[str] = None, enforce_parameter_schema: bool = True, work_pool_name: Optional[str] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict] = None, entrypoint_type: EntrypointType = EntrypointType.FILE_PATH, _sla: Optional[Any] = None) -> 'RunnerDeployment':
        ...

    @classmethod
    def from_entrypoint(cls, entrypoint: str, name: str, flow_name: Optional[str] = None, interval: Optional[Union[int, timedelta]] = None, cron: Optional[Union[str, List[str]]] = None, rrule: Optional[Union[str, List[str]]] = None, paused: Optional[bool] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None, concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig]] = None, parameters: Optional[dict] = None, triggers: Optional[List[TriggerTypes]] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, version: Optional[str] = None, enforce_parameter_schema: bool = True, work_pool_name: Optional[str] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict] = None, _sla: Optional[Any] = None) -> 'RunnerDeployment':
        ...

    @classmethod
    async def afrom_storage(cls, storage: RunnerStorage, entrypoint: str, name: str, flow_name: Optional[str] = None, interval: Optional[Union[int, timedelta]] = None, cron: Optional[Union[str, List[str]]] = None, rrule: Optional[Union[str, List[str]]] = None, paused: Optional[bool] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None, concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig]] = None, parameters: Optional[dict] = None, triggers: Optional[List[TriggerTypes]] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, version: Optional[str] = None, enforce_parameter_schema: bool = True, work_pool_name: Optional[str] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict] = None, _sla: Optional[Any] = None) -> 'RunnerDeployment':
        ...

    @classmethod
    def from_storage(cls, storage: RunnerStorage, entrypoint: str, name: str, flow_name: Optional[str] = None, interval: Optional[Union[int, timedelta]] = None, cron: Optional[Union[str, List[str]]] = None, rrule: Optional[Union[str, List[str]]] = None, paused: Optional[bool] = None, schedule: Optional[Schedule] = None, schedules: Optional[List[Schedule]] = None, concurrency_limit: Optional[Union[int, ConcurrencyLimitConfig]] = None, parameters: Optional[dict] = None, triggers: Optional[List[TriggerTypes]] = None, description: Optional[str] = None, tags: Optional[List[str]] = None, version: Optional[str] = None, enforce_parameter_schema: bool = True, work_pool_name: Optional[str] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict] = None, _sla: Optional[Any] = None) -> 'RunnerDeployment':
        ...

@sync_compatible
async def deploy(*deployments: RunnerDeployment, work_pool_name: Optional[str] = None, image: Optional[str] = None, build: bool = True, push: bool = True, print_next_steps_message: bool = True, ignore_warnings: bool = False) -> List[UUID]:
    ...
