from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Literal, Self
from prefect._internal.schemas.validators import get_or_create_run_name, raise_on_name_alphanumeric_dashes_only, set_run_policy_deprecated_fields, validate_cache_key_length, validate_default_queue_id_not_none, validate_max_metadata_length, validate_message_template_variables, validate_name_present_on_nonanonymous_blocks, validate_not_negative, validate_parent_and_ref_diff, validate_schedule_max_scheduled_runs
from prefect.server.schemas import schedules, states
from prefect.server.schemas.statuses import WorkPoolStatus
from prefect.server.utilities.schemas.bases import ORMBaseModel, PrefectBaseModel
from prefect.settings import PREFECT_DEPLOYMENT_SCHEDULE_MAX_SCHEDULED_RUNS
from prefect.types import MAX_VARIABLE_NAME_LENGTH, DateTime, LaxUrl, Name, NameOrEmpty, NonEmptyishName, NonNegativeInteger, PositiveInteger, StrictVariableValue
from prefect.types._datetime import now
from prefect.utilities.collections import AutoEnum, dict_to_flatdict, flatdict_to_dict, listrepr
from prefect.utilities.names import generate_slug, obfuscate
if TYPE_CHECKING:
    from prefect.server.database import orm_models

FLOW_RUN_NOTIFICATION_TEMPLATE_KWARGS = ['flow_run_notification_policy_id', 'flow_id', 'flow_name', 'flow_run_url', 'flow_run_id', 'flow_run_name', 'flow_run_parameters', 'flow_run_state_type', 'flow_run_state_name', 'flow_run_state_timestamp', 'flow_run_state_message']
DEFAULT_BLOCK_SCHEMA_VERSION = 'non-versioned'
KeyValueLabels = dict[str, Union[StrictBool, StrictInt, StrictFloat, str]]

class Flow(ORMBaseModel):
    name: str = Field(default=..., description='The name of the flow', examples=['my-flow'])
    tags: List[str] = Field(default_factory=list, description='A list of flow tags', examples=[['tag-1', 'tag-2']])
    labels: Dict[str, Union[str, int, bool]] = Field(default_factory=dict, description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.', examples=[{'key': 'value1', 'key2': 42}])

class FlowRunPolicy(PrefectBaseModel):
    max_retries: int = Field(default=0, description='The maximum number of retries. Field is not used. Please use `retries` instead.', deprecated=True)
    retry_delay_seconds: int = Field(default=0, description='The delay between retries. Field is not used. Please use `retry_delay` instead.', deprecated=True)
    retries: Optional[int] = Field(default=None, description='The number of retries.')
    retry_delay: Optional[int] = Field(default=None, description='The delay time between retries, in seconds.')
    pause_keys: set = Field(default_factory=set, description='Tracks pauses this run has observed.')
    resuming: bool = Field(default=False, description='Indicates if this run is resuming from a pause.')
    retry_type: Optional[str] = Field(default=None, description='The type of retry this run is undergoing.')

    @model_validator(mode='before')
    def populate_deprecated_fields(cls, values):
        return set_run_policy_deprecated_fields(values)

class CreatedBy(BaseModel):
    id: Optional[str] = Field(default=None, description='The id of the creator of the object.')
    type: Optional[str] = Field(default=None, description='The type of the creator of the object.')
    display_value: Optional[str] = Field(default=None, description='The display value for the creator.')

class UpdatedBy(BaseModel):
    id: Optional[str] = Field(default=None, description='The id of the updater of the object.')
    type: Optional[str] = Field(default=None, description='The type of the updater of the object.')
    display_value: Optional[str] = Field(default=None, description='The display value for the updater.')

class ConcurrencyLimitStrategy(AutoEnum):
    ENQUEUE = 'ENQUEUE'
    CANCEL_NEW = 'CANCEL_NEW'

class ConcurrencyOptions(BaseModel):
    pass

class FlowRun(ORMBaseModel):
    name: str = Field(default_factory=lambda: generate_slug(2), description='The name of the flow run. Defaults to a random slug if not specified.', examples=['my-flow-run'])
    flow_id: str = Field(default=..., description='The id of the flow being run.')
    state_id: Optional[str] = Field(default=None, description="The id of the flow run's current state.")
    deployment_id: Optional[str] = Field(default=None, description='The id of the deployment associated with this flow run, if available.')
    deployment_version: Optional[str] = Field(default=None, description='The version of the deployment associated with this flow run.', examples=['1.0'])
    work_queue_name: Optional[str] = Field(default=None, description='The work queue that handled this flow run.')
    flow_version: Optional[str] = Field(default=None, description='The version of the flow executed in this flow run.', examples=['1.0'])
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for the flow run.')
    idempotency_key: Optional[str] = Field(default=None, description='An optional idempotency key for the flow run. Used to ensure the same flow run is not created multiple times.')
    context: Dict[str, Any] = Field(default_factory=dict, description='Additional context for the flow run.', examples=[{'my_var': 'my_value'}])
    empirical_policy: FlowRunPolicy = Field(default_factory=FlowRunPolicy)
    tags: List[str] = Field(default_factory=list, description='A list of tags on the flow run', examples=[['tag-1', 'tag-2']])
    labels: Dict[str, Union[str, int, bool]] = Field(default_factory=dict, description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.', examples=[{'key': 'value1', 'key2': 42}])
    parent_task_run_id: Optional[str] = Field(default=None, description="If the flow run is a subflow, the id of the 'dummy' task in the parent flow used to track subflow state.")
    state_type: Optional[str] = Field(default=None, description='The type of the current flow run state.')
    state_name: Optional[str] = Field(default=None, description='The name of the current flow run state.')
    run_count: int = Field(default=0, description='The number of times the flow run was executed.')
    expected_start_time: Optional[datetime.datetime] = Field(default=None, description="The flow run's expected start time.")
    next_scheduled_start_time: Optional[datetime.datetime] = Field(default=None, description='The next time the flow run is scheduled to start.')
    start_time: Optional[datetime.datetime] = Field(default=None, description='The actual start time.')
    end_time: Optional[datetime.datetime] = Field(default=None, description='The actual end time.')
    total_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='Total run time. If the flow run was executed multiple times, the time of each run will be summed.')
    estimated_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='A real-time estimate of the total run time.')
    estimated_start_time_delta: datetime.timedelta = Field(default=datetime.timedelta(0), description='The difference between actual and expected start time.')
    auto_scheduled: bool = Field(default=False, description='Whether or not the flow run was automatically scheduled.')
    infrastructure_document_id: Optional[str] = Field(default=None, description='The block document defining infrastructure to use this flow run.')
    infrastructure_pid: Optional[str] = Field(default=None, description='The id of the flow run as returned by an infrastructure block.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this flow run.')
    work_queue_id: Optional[str] = Field(default=None, description="The id of the run's work pool queue.")
    state: Optional[str] = Field(default=None, description='The current state of the flow run.')
    job_variables: Optional[Dict[str, Any]] = Field(default=None, description='Variables used as overrides in the base job template')

    @field_validator('name', mode='before')
    @classmethod
    def set_name(cls, name: str) -> str:
        return get_or_create_run_name(name)

    def __eq__(self, other: Any) -> bool:
        """
        Check for "equality" to another flow run schema

        Estimates times are rolling and will always change with repeated queries for
        a flow run so we ignore them during equality checks.
        """
        if isinstance(other, FlowRun):
            exclude_fields = {'estimated_run_time', 'estimated_start_time_delta'}
            return self.model_dump(exclude=exclude_fields) == other.model_dump(exclude=exclude_fields)
        return super().__eq__(other)

class TaskRunPolicy(PrefectBaseModel):
    max_retries: int = Field(default=0, description='The maximum number of retries. Field is not used. Please use `retries` instead.', deprecated=True)
    retry_delay_seconds: int = Field(default=0, description='The delay between retries. Field is not used. Please use `retry_delay` instead.', deprecated=True)
    retries: Optional[int] = Field(default=None, description='The number of retries.')
    retry_delay: Optional[Union[int, List[int]]] = Field(default=None, description='A delay time or list of delay times between retries, in seconds.')
    retry_jitter_factor: Optional[int] = Field(default=None, description='Determines the amount a retry should jitter')

    @model_validator(mode='before')
    def populate_deprecated_fields(cls, values):
        return set_run_policy_deprecated_fields(values)

    @field_validator('retry_delay')
    @classmethod
    def validate_configured_retry_delays(cls, v):
        if isinstance(v, list) and len(v) > 50:
            raise ValueError('Can not configure more than 50 retry delays per task.')
        return v

    @field_validator('retry_jitter_factor')
    @classmethod
    def validate_jitter_factor(cls, v):
        return validate_not_negative(v)

class TaskRunInput(PrefectBaseModel):
    model_config: ConfigDict = ConfigDict(frozen=True)

class TaskRunResult(TaskRunInput):
    input_type: str = 'task_run'

class Parameter(TaskRunInput):
    input_type: str = 'parameter'

class Constant(TaskRunInput):
    input_type: str = 'constant'

class TaskRun(ORMBaseModel):
    name: str = Field(default_factory=lambda: generate_slug(2), examples=['my-task-run'])
    flow_run_id: Optional[str] = Field(default=None, description='The flow run id of the task run.')
    task_key: str = Field(default=..., description='A unique identifier for the task being run.')
    dynamic_key: str = Field(default=..., description='A dynamic key used to differentiate between multiple runs of the same task within the same flow run.')
    cache_key: Optional[str] = Field(default=None, description='An optional cache key. If a COMPLETED state associated with this cache key is found, the cached COMPLETED state will be used instead of executing the task run.')
    cache_expiration: Optional[datetime.datetime] = Field(default=None, description='Specifies when the cached state should expire.')
    task_version: Optional[str] = Field(default=None, description='The version of the task being run.')
    empirical_policy: TaskRunPolicy = Field(default_factory=TaskRunPolicy)
    tags: List[str] = Field(default_factory=list, description='A list of tags for the task run.', examples=[['tag-1', 'tag-2']])
    labels: Dict[str, Union[str, int, bool]] = Field(default_factory=dict, description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.', examples=[{'key': 'value1', 'key2': 42}])
    state_id: Optional[str] = Field(default=None, description='The id of the current task run state.')
    task_inputs: Dict[str, Any] = Field(default_factory=dict, description='Tracks the source of inputs to a task run. Used for internal bookkeeping.')
    state_type: Optional[str] = Field(default=None, description='The type of the current task run state.')
    state_name: Optional[str] = Field(default=None, description='The name of the current task run state.')
    run_count: int = Field(default=0, description='The number of times the task run has been executed.')
    flow_run_run_count: int = Field(default=0, description='If the parent flow has retried, this indicates the flow retry this run is associated with.')
    expected_start_time: Optional[datetime.datetime] = Field(default=None, description="The task run's expected start time.")
    next_scheduled_start_time: Optional[datetime.datetime] = Field(default=None, description='The next time the task run is scheduled to start.')
    start_time: Optional[datetime.datetime] = Field(default=None, description='The actual start time.')
    end_time: Optional[datetime.datetime] = Field(default=None, description='The actual end time.')
    total_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='Total run time. If the task run was executed multiple times, the time of each run will be summed.')
    estimated_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='A real-time estimate of total run time.')
    estimated_start_time_delta: datetime.timedelta = Field(default=datetime.timedelta(0), description='The difference between actual and expected start time.')
    state: Optional[str] = Field(default=None, description='The current task run state.')

    @field_validator('name', mode='before')
    @classmethod
    def set_name(cls, name: str) -> str:
        return get_or_create_run_name(name)

    @field_validator('cache_key')
    @classmethod
    def validate_cache_key(cls, cache_key: str) -> str:
        return validate_cache_key_length(cache_key)

class DeploymentSchedule(ORMBaseModel):
    deployment_id: Optional[str] = Field(default=None, description='The deployment id associated with this schedule.')
    schedule: str = Field(default=..., description='The schedule for the deployment.')
    active: bool = Field(default=True, description='Whether or not the schedule is active.')
    max_scheduled_runs: Optional[int] = Field(default=None, description='The maximum number of scheduled runs for the schedule.')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='A dictionary of parameter value overrides.')
    slug: Optional[str] = Field(default=None, description='A unique slug for the schedule.')

    @field_validator('max_scheduled_runs')
    @classmethod
    def validate_max_scheduled_runs(cls, v: int) -> int:
        return validate_schedule_max_scheduled_runs(v, PREFECT_DEPLOYMENT_SCHEDULE_MAX_SCHEDULED_RUNS.value())

class Deployment(ORMBaseModel):
    name: str = Field(default=..., description='The name of the deployment.')
    version: Optional[str] = Field(default=None, description='An optional version for the deployment.')
    description: Optional[str] = Field(default=None, description='A description for the deployment.')
    flow_id: str = Field(default=..., description='The flow id associated with the deployment.')
    paused: bool = Field(default=False, description='Whether or not the deployment is paused.')
    schedules: List[str] = Field(default_factory=list, description='A list of schedules for the deployment.')
    concurrency_limit: Optional[str] = Field(default=None, description='The concurrency limit for the deployment.')
    concurrency_options: Optional[str] = Field(default=None, description='The concurrency options for the deployment.')
    job_variables: Dict[str, Any] = Field(default_factory=dict, description='Overrides to apply to flow run infrastructure at runtime.')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for flow runs scheduled by the deployment.')
    pull_steps: Optional[str] = Field(default=None, description='Pull steps for cloning and running this deployment.')
    tags: List[str] = Field(default_factory=list, description='A list of tags for the deployment', examples=[['tag-1', 'tag-2']])
    labels: Dict[str, Union[str, int, bool]] = Field(default_factory=dict, description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.', examples=[{'key': 'value1', 'key2': 42}])
    work_queue_name: Optional[str] = Field(default=None, description='The work queue for the deployment. If no work queue is set, work will not be scheduled.')
    last_polled: Optional[datetime.datetime] = Field(default=None, description='The last time the deployment was polled for status updates.')
    parameter_openapi_schema: Dict[str, Any] = Field(default_factory=dict, description='The parameter schema of the flow, including defaults.')
    path: Optional[str] = Field(default=None, description='The path to the working directory for the workflow, relative to remote storage or an absolute path.')
    entrypoint: Optional[str] = Field(default=None, description='The path to the entrypoint for the workflow, relative to the `path`.')
    storage_document_id: Optional[str] = Field(default=None, description='The block document defining storage used for this flow.')
    infrastructure_document_id: Optional[str] = Field(default=None, description='The block document defining infrastructure to use for flow runs.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this deployment.')
    updated_by: Optional[UpdatedBy] = Field(default=None, description='Optional information about the updater of this deployment.')
    work_queue_id: Optional[str] = Field(default=None, description='The id of the work pool queue to which this deployment is assigned.')
    enforce_parameter_schema: bool = Field(default=True, description='Whether or not the deployment should enforce the parameter schema.')

class ConcurrencyLimit(ORMBaseModel):
    tag: str = Field(default=..., description='A tag the concurrency limit is applied to.')
    concurrency_limit: int = Field(default=..., description='The concurrency limit.')
    active_slots: List[str] = Field(default_factory=list, description='A list of active run ids using a concurrency slot')

class ConcurrencyLimitV2(ORMBaseModel):
    active: bool = Field(default=True, description='Whether the concurrency limit is active.')
    name: str = Field(default=..., description='The name of the concurrency limit.')
    limit: int = Field(default=..., description='The concurrency limit.')
    active_slots: int = Field(default=0, description='The number of active slots.')
    denied_slots: int = Field(default=0, description='The number of denied slots.')
    slot_decay_per_second: int = Field(default=0, description='The decay rate for active slots when used as a rate limit.')
    avg_slot_occupancy_seconds: float = Field(default=2.0, description='The average amount of time a slot