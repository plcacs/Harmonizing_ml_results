#!/usr/bin/env python3
"""
Full schemas of Prefect REST API objects.
"""
from __future__ import annotations
import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Literal, Self

from prefect._internal.schemas.validators import (
    get_or_create_run_name,
    raise_on_name_alphanumeric_dashes_only,
    set_run_policy_deprecated_fields,
    validate_cache_key_length,
    validate_default_queue_id_not_none,
    validate_max_metadata_length,
    validate_message_template_variables,
    validate_name_present_on_nonanonymous_blocks,
    validate_not_negative,
    validate_parent_and_ref_diff,
    validate_schedule_max_scheduled_runs,
)
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

FLOW_RUN_NOTIFICATION_TEMPLATE_KWARGS: List[str] = [
    'flow_run_notification_policy_id', 'flow_id', 'flow_name', 'flow_run_url',
    'flow_run_id', 'flow_run_name', 'flow_run_parameters', 'flow_run_state_type',
    'flow_run_state_name', 'flow_run_state_timestamp', 'flow_run_state_message'
]
DEFAULT_BLOCK_SCHEMA_VERSION: str = 'non-versioned'
KeyValueLabels = dict[str, Union[StrictBool, StrictInt, StrictFloat, str]]


class Flow(ORMBaseModel):
    """An ORM representation of flow data."""
    name: str = Field(default=..., description='The name of the flow', examples=['my-flow'])
    tags: List[str] = Field(default_factory=list, description='A list of flow tags', examples=[['tag-1', 'tag-2']])
    labels: KeyValueLabels = Field(
        default_factory=dict,
        description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.',
        examples=[{'key': 'value1', 'key2': 42}],
    )

    def __eq__(self, other: object) -> bool:
        """
        Check for "equality" to another flow run schema

        Estimates times are rolling and will always change with repeated queries for
        a flow run so we ignore them during equality checks.
        """
        if isinstance(other, Flow):
            exclude_fields = {'estimated_run_time', 'estimated_start_time_delta'}
            return self.model_dump(exclude=exclude_fields) == other.model_dump(exclude=exclude_fields)
        return super().__eq__(other)


class FlowRunPolicy(PrefectBaseModel):
    """Defines of how a flow run should retry."""
    max_retries: int = Field(
        default=0,
        description='The maximum number of retries. Field is not used. Please use `retries` instead.',
        deprecated=True,
    )
    retry_delay_seconds: int = Field(
        default=0,
        description='The delay between retries. Field is not used. Please use `retry_delay` instead.',
        deprecated=True,
    )
    retries: Optional[int] = Field(default=None, description='The number of retries.')
    retry_delay: Optional[float] = Field(default=None, description='The delay time between retries, in seconds.')
    pause_keys: set = Field(default_factory=set, description='Tracks pauses this run has observed.')
    resuming: bool = Field(default=False, description='Indicates if this run is resuming from a pause.')
    retry_type: Optional[Any] = Field(default=None, description='The type of retry this run is undergoing.')

    @model_validator(mode='before')
    @classmethod
    def populate_deprecated_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return set_run_policy_deprecated_fields(values)


class CreatedBy(BaseModel):
    id: Optional[Any] = Field(default=None, description='The id of the creator of the object.')
    type: Optional[Any] = Field(default=None, description='The type of the creator of the object.')
    display_value: Optional[Any] = Field(default=None, description='The display value for the creator.')


class UpdatedBy(BaseModel):
    id: Optional[Any] = Field(default=None, description='The id of the updater of the object.')
    type: Optional[Any] = Field(default=None, description='The type of the updater of the object.')
    display_value: Optional[Any] = Field(default=None, description='The display value for the updater.')


class ConcurrencyLimitStrategy(AutoEnum):
    """
    Enumeration of concurrency collision strategies.
    """
    ENQUEUE = AutoEnum.auto()
    CANCEL_NEW = AutoEnum.auto()


class ConcurrencyOptions(BaseModel):
    """
    Class for storing the concurrency config in database.
    """
    pass


class FlowRun(ORMBaseModel):
    """An ORM representation of flow run data."""
    name: str = Field(
        default_factory=lambda: generate_slug(2),
        description='The name of the flow run. Defaults to a random slug if not specified.',
        examples=['my-flow-run'],
    )
    flow_id: Any = Field(default=..., description='The id of the flow being run.')
    state_id: Optional[Any] = Field(default=None, description="The id of the flow run's current state.")
    deployment_id: Optional[Any] = Field(default=None, description='The id of the deployment associated with this flow run, if available.')
    deployment_version: Optional[str] = Field(default=None, description='The version of the deployment associated with this flow run.', examples=['1.0'])
    work_queue_name: Optional[str] = Field(default=None, description='The work queue that handled this flow run.')
    flow_version: Optional[str] = Field(default=None, description='The version of the flow executed in this flow run.', examples=['1.0'])
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for the flow run.')
    idempotency_key: Optional[Any] = Field(default=None, description='An optional idempotency key for the flow run. Used to ensure the same flow run is not created multiple times.')
    context: Dict[str, Any] = Field(default_factory=dict, description='Additional context for the flow run.', examples=[{'my_var': 'my_value'}])
    empirical_policy: FlowRunPolicy = Field(default_factory=FlowRunPolicy)
    tags: List[str] = Field(default_factory=list, description='A list of tags on the flow run', examples=[['tag-1', 'tag-2']])
    labels: KeyValueLabels = Field(
        default_factory=dict,
        description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.',
        examples=[{'key': 'value1', 'key2': 42}],
    )
    parent_task_run_id: Optional[Any] = Field(default=None, description="If the flow run is a subflow, the id of the 'dummy' task in the parent flow used to track subflow state.")
    state_type: Optional[Any] = Field(default=None, description='The type of the current flow run state.')
    state_name: Optional[Any] = Field(default=None, description='The name of the current flow run state.')
    run_count: int = Field(default=0, description='The number of times the flow run was executed.')
    expected_start_time: Optional[datetime.datetime] = Field(default=None, description="The flow run's expected start time.")
    next_scheduled_start_time: Optional[datetime.datetime] = Field(default=None, description='The next time the flow run is scheduled to start.')
    start_time: Optional[datetime.datetime] = Field(default=None, description='The actual start time.')
    end_time: Optional[datetime.datetime] = Field(default=None, description='The actual end time.')
    total_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='Total run time. If the flow run was executed multiple times, the time of each run will be summed.')
    estimated_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='A real-time estimate of the total run time.')
    estimated_start_time_delta: datetime.timedelta = Field(default=datetime.timedelta(0), description='The difference between actual and expected start time.')
    auto_scheduled: bool = Field(default=False, description='Whether or not the flow run was automatically scheduled.')
    infrastructure_document_id: Optional[Any] = Field(default=None, description='The block document defining infrastructure to use this flow run.')
    infrastructure_pid: Optional[Any] = Field(default=None, description='The id of the flow run as returned by an infrastructure block.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this flow run.')
    work_queue_id: Optional[Any] = Field(default=None, description="The id of the run's work pool queue.")
    state: Optional[Any] = Field(default=None, description='The current state of the flow run.')
    job_variables: Optional[Dict[str, Any]] = Field(default=None, description='Variables used as overrides in the base job template')

    @field_validator('name', mode='before')
    @classmethod
    def set_name(cls, name: Any) -> str:
        return get_or_create_run_name(name)


class TaskRunPolicy(PrefectBaseModel):
    """Defines of how a task run should retry."""
    max_retries: int = Field(
        default=0,
        description='The maximum number of retries. Field is not used. Please use `retries` instead.',
        deprecated=True,
    )
    retry_delay_seconds: int = Field(
        default=0,
        description='The delay between retries. Field is not used. Please use `retry_delay` instead.',
        deprecated=True,
    )
    retries: Optional[int] = Field(default=None, description='The number of retries.')
    retry_delay: Optional[Union[float, List[float]]] = Field(default=None, description='A delay time or list of delay times between retries, in seconds.')
    retry_jitter_factor: Optional[float] = Field(default=None, description='Determines the amount a retry should jitter')

    @model_validator(mode='before')
    @classmethod
    def populate_deprecated_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return set_run_policy_deprecated_fields(values)

    @field_validator('retry_delay')
    @classmethod
    def validate_configured_retry_delays(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) > 50:
            raise ValueError('Can not configure more than 50 retry delays per task.')
        return v

    @field_validator('retry_jitter_factor')
    @classmethod
    def validate_jitter_factor(cls, v: Optional[float]) -> Optional[float]:
        return validate_not_negative(v)


class TaskRunInput(PrefectBaseModel):
    """
    Base class for classes that represent inputs to task runs, which
    could include, constants, parameters, or other task runs.
    """
    model_config = ConfigDict(frozen=True)


class TaskRunResult(TaskRunInput):
    """Represents a task run result input to another task run."""
    input_type: str = 'task_run'


class Parameter(TaskRunInput):
    """Represents a parameter input to a task run."""
    input_type: str = 'parameter'


class Constant(TaskRunInput):
    """Represents constant input value to a task run."""
    input_type: str = 'constant'


class TaskRun(ORMBaseModel):
    """An ORM representation of task run data."""
    name: str = Field(default_factory=lambda: generate_slug(2), examples=['my-task-run'])
    flow_run_id: Optional[Any] = Field(default=None, description='The flow run id of the task run.')
    task_key: Any = Field(default=..., description='A unique identifier for the task being run.')
    dynamic_key: Any = Field(default=..., description='A dynamic key used to differentiate between multiple runs of the same task within the same flow run.')
    cache_key: Optional[str] = Field(default=None, description='An optional cache key. If a COMPLETED state associated with this cache key is found, the cached COMPLETED state will be used instead of executing the task run.')
    cache_expiration: Optional[datetime.datetime] = Field(default=None, description='Specifies when the cached state should expire.')
    task_version: Optional[str] = Field(default=None, description='The version of the task being run.')
    empirical_policy: TaskRunPolicy = Field(default_factory=TaskRunPolicy)
    tags: List[str] = Field(default_factory=list, description='A list of tags for the task run.', examples=[['tag-1', 'tag-2']])
    labels: KeyValueLabels = Field(
        default_factory=dict,
        description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.',
        examples=[{'key': 'value1', 'key2': 42}],
    )
    state_id: Optional[Any] = Field(default=None, description='The id of the current task run state.')
    task_inputs: Dict[str, Any] = Field(default_factory=dict, description='Tracks the source of inputs to a task run. Used for internal bookkeeping.')
    state_type: Optional[Any] = Field(default=None, description='The type of the current task run state.')
    state_name: Optional[Any] = Field(default=None, description='The name of the current task run state.')
    run_count: int = Field(default=0, description='The number of times the task run has been executed.')
    flow_run_run_count: int = Field(default=0, description='If the parent flow has retried, this indicates the flow retry this run is associated with.')
    expected_start_time: Optional[datetime.datetime] = Field(default=None, description="The task run's expected start time.")
    next_scheduled_start_time: Optional[datetime.datetime] = Field(default=None, description='The next time the task run is scheduled to start.')
    start_time: Optional[datetime.datetime] = Field(default=None, description='The actual start time.')
    end_time: Optional[datetime.datetime] = Field(default=None, description='The actual end time.')
    total_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='Total run time. If the task run was executed multiple times, the time of each run will be summed.')
    estimated_run_time: datetime.timedelta = Field(default=datetime.timedelta(0), description='A real-time estimate of total run time.')
    estimated_start_time_delta: datetime.timedelta = Field(default=datetime.timedelta(0), description='The difference between actual and expected start time.')
    state: Optional[Any] = Field(default=None, description='The current task run state.')

    @field_validator('name', mode='before')
    @classmethod
    def set_name(cls, name: Any) -> str:
        return get_or_create_run_name(name)

    @field_validator('cache_key')
    @classmethod
    def validate_cache_key(cls, cache_key: Optional[str]) -> Optional[str]:
        return validate_cache_key_length(cache_key)


class DeploymentSchedule(ORMBaseModel):
    deployment_id: Optional[Any] = Field(default=None, description='The deployment id associated with this schedule.')
    schedule: Any = Field(default=..., description='The schedule for the deployment.')
    active: bool = Field(default=True, description='Whether or not the schedule is active.')
    max_scheduled_runs: Optional[int] = Field(default=None, description='The maximum number of scheduled runs for the schedule.')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='A dictionary of parameter value overrides.')
    slug: Optional[str] = Field(default=None, description='A unique slug for the schedule.')

    @field_validator('max_scheduled_runs')
    @classmethod
    def validate_max_scheduled_runs(cls, v: Optional[int]) -> Optional[int]:
        return validate_schedule_max_scheduled_runs(v, PREFECT_DEPLOYMENT_SCHEDULE_MAX_SCHEDULED_RUNS.value())


class Deployment(ORMBaseModel):
    """An ORM representation of deployment data."""
    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(default=..., description='The name of the deployment.')
    version: Optional[str] = Field(default=None, description='An optional version for the deployment.')
    description: Optional[str] = Field(default=None, description='A description for the deployment.')
    flow_id: Any = Field(default=..., description='The flow id associated with the deployment.')
    paused: bool = Field(default=False, description='Whether or not the deployment is paused.')
    schedules: List[DeploymentSchedule] = Field(default_factory=list, description='A list of schedules for the deployment.')
    concurrency_limit: Optional[Any] = Field(default=None, description='The concurrency limit for the deployment.')
    concurrency_options: Optional[Any] = Field(default=None, description='The concurrency options for the deployment.')
    job_variables: Dict[str, Any] = Field(default_factory=dict, description='Overrides to apply to flow run infrastructure at runtime.')
    parameters: Dict[str, Any] = Field(default_factory=dict, description='Parameters for flow runs scheduled by the deployment.')
    pull_steps: Optional[Any] = Field(default=None, description='Pull steps for cloning and running this deployment.')
    tags: List[str] = Field(default_factory=list, description='A list of tags for the deployment', examples=[['tag-1', 'tag-2']])
    labels: KeyValueLabels = Field(
        default_factory=dict,
        description='A dictionary of key-value labels. Values can be strings, numbers, or booleans.',
        examples=[{'key': 'value1', 'key2': 42}],
    )
    work_queue_name: Optional[str] = Field(default=None, description='The work queue for the deployment. If no work queue is set, work will not be scheduled.')
    last_polled: Optional[datetime.datetime] = Field(default=None, description='The last time the deployment was polled for status updates.')
    parameter_openapi_schema: Dict[str, Any] = Field(default_factory=dict, description='The parameter schema of the flow, including defaults.')
    path: Optional[str] = Field(default=None, description='The path to the working directory for the workflow, relative to remote storage or an absolute path.')
    entrypoint: Optional[str] = Field(default=None, description='The path to the entrypoint for the workflow, relative to the `path`.')
    storage_document_id: Optional[Any] = Field(default=None, description='The block document defining storage used for this flow.')
    infrastructure_document_id: Optional[Any] = Field(default=None, description='The block document defining infrastructure to use for flow runs.')
    created_by: Optional[CreatedBy] = Field(default=None, description='Optional information about the creator of this deployment.')
    updated_by: Optional[UpdatedBy] = Field(default=None, description='Optional information about the updater of this deployment.')
    work_queue_id: Optional[Any] = Field(default=None, description='The id of the work pool queue to which this deployment is assigned.')
    enforce_parameter_schema: bool = Field(default=True, description='Whether or not the deployment should enforce the parameter schema.')


class ConcurrencyLimit(ORMBaseModel):
    """An ORM representation of a concurrency limit."""
    tag: Any = Field(default=..., description='A tag the concurrency limit is applied to.')
    concurrency_limit: Any = Field(default=..., description='The concurrency limit.')
    active_slots: List[Any] = Field(default_factory=list, description='A list of active run ids using a concurrency slot')


class ConcurrencyLimitV2(ORMBaseModel):
    """An ORM representation of a v2 concurrency limit."""
    active: bool = Field(default=True, description='Whether the concurrency limit is active.')
    name: Any = Field(default=..., description='The name of the concurrency limit.')
    limit: Any = Field(default=..., description='The concurrency limit.')
    active_slots: int = Field(default=0, description='The number of active slots.')
    denied_slots: int = Field(default=0, description='The number of denied slots.')
    slot_decay_per_second: float = Field(default=0, description='The decay rate for active slots when used as a rate limit.')
    avg_slot_occupancy_seconds: float = Field(default=2.0, description='The average amount of time a slot is occupied.')


class BlockType(ORMBaseModel):
    """An ORM representation of a block type"""
    name: Any = Field(default=..., description="A block type's name")
    slug: Any = Field(default=..., description="A block type's slug")
    logo_url: Optional[str] = Field(default=None, description="Web URL for the block type's logo")
    documentation_url: Optional[str] = Field(default=None, description="Web URL for the block type's documentation")
    description: Optional[str] = Field(default=None, description="A short blurb about the corresponding block's intended use")
    code_example: Optional[str] = Field(default=None, description='A code snippet demonstrating use of the corresponding block')
    is_protected: bool = Field(default=False, description='Protected block types cannot be modified via API.')


class BlockSchema(ORMBaseModel):
    """An ORM representation of a block schema."""
    checksum: Any = Field(default=..., description="The block schema's unique checksum")
    fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="The block schema's field schema",
        json_schema_extra={'additionalProperties': True},
    )
    block_type_id: Any = Field(default=..., description='A block type ID')
    block_type: Optional[BlockType] = Field(default=None, description='The associated block type')
    capabilities: List[Any] = Field(default_factory=list, description='A list of Block capabilities')
    version: str = Field(default=DEFAULT_BLOCK_SCHEMA_VERSION, description='Human readable identifier for the block schema')


class BlockSchemaReference(ORMBaseModel):
    """An ORM representation of a block schema reference."""
    parent_block_schema_id: Any = Field(default=..., description='ID of block schema the reference is nested within')
    parent_block_schema: Optional[BlockSchema] = Field(default=None, description='The block schema the reference is nested within')
    reference_block_schema_id: Any = Field(default=..., description='ID of the nested block schema')
    reference_block_schema: Optional[BlockSchema] = Field(default=None, description='The nested block schema')
    name: Any = Field(default=..., description='The name that the reference is nested under')


class BlockDocument(ORMBaseModel):
    """An ORM representation of a block document."""
    name: Optional[str] = Field(default=None, description="The block document's name. Not required for anonymous block documents.")
    data: Dict[str, Any] = Field(default_factory=dict, description="The block document's data")
    block_schema_id: Any = Field(default=..., description='A block schema ID')
    block_schema: Optional[BlockSchema] = Field(default=None, description='The associated block schema')
    block_type_id: Any = Field(default=..., description='A block type ID')
    block_type_name: Optional[str] = Field(default=None, description="The associated block type's name")
    block_type: Optional[BlockType] = Field(default=None, description='The associated block type')
    block_document_references: Dict[str, Any] = Field(default_factory=dict, description="Record of the block document's references")
    is_anonymous: bool = Field(default=False, description='Whether the block is anonymous (anonymous blocks are usually created by Prefect automatically)')

    @model_validator(mode='before')
    def validate_name_is_present_if_not_anonymous(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return validate_name_present_on_nonanonymous_blocks(values)

    @classmethod
    async def from_orm_model(
        cls: Type[Self],
        session: AsyncSession,
        orm_block_document: Any,
        include_secrets: bool = False,
    ) -> Self:
        data: Dict[str, Any] = await orm_block_document.decrypt_data(session=session)
        if not include_secrets:
            flat_data: Dict[Any, Any] = dict_to_flatdict(data)
            for secret_field in orm_block_document.block_schema.fields.get('secret_fields', []):
                secret_key: tuple = tuple(secret_field.split('.'))
                if flat_data.get(secret_key) is not None:
                    flat_data[secret_key] = obfuscate(flat_data[secret_key])
                elif '*' in secret_key:
                    wildcard_index: int = secret_key.index('*')
                    for data_key in flat_data.keys():
                        if secret_key[0:wildcard_index] == data_key[0:wildcard_index]:
                            flat_data[data_key] = obfuscate(flat_data[data_key])
            data = flatdict_to_dict(flat_data)
        return cls(
            id=orm_block_document.id,
            created=orm_block_document.created,
            updated=orm_block_document.updated,
            name=orm_block_document.name,
            data=data,
            block_schema_id=orm_block_document.block_schema_id,
            block_schema=orm_block_document.block_schema,
            block_type_id=orm_block_document.block_type_id,
            block_type_name=orm_block_document.block_type_name,
            block_type=orm_block_document.block_type,
            is_anonymous=orm_block_document.is_anonymous,
        )


class BlockDocumentReference(ORMBaseModel):
    """An ORM representation of a block document reference."""
    parent_block_document_id: Any = Field(default=..., description='ID of block document the reference is nested within')
    parent_block_document: Optional[BlockDocument] = Field(default=None, description='The block document the reference is nested within')
    reference_block_document_id: Any = Field(default=..., description='ID of the nested block document')
    reference_block_document: Optional[BlockDocument] = Field(default=None, description='The nested block document')
    name: Any = Field(default=..., description='The name that the reference is nested under')

    @model_validator(mode='before')
    def validate_parent_and_ref_are_different(self, values: Dict[str, Any]) -> Dict[str, Any]:
        return validate_parent_and_ref_diff(values)


class Configuration(ORMBaseModel):
    """An ORM representation of account info."""
    key: Any = Field(default=..., description='Account info key')
    value: Any = Field(default=..., description='Account info')


class SavedSearchFilter(PrefectBaseModel):
    """A filter for a saved search model. Intended for use by the Prefect UI."""
    object: Any = Field(default=..., description='The object over which to filter.')
    property: Any = Field(default=..., description='The property of the object on which to filter.')
    type: Any = Field(default=..., description='The type of the property.')
    operation: Any = Field(default=..., description='The operator to apply to the object. For example, `equals`.')
    value: Any = Field(default=..., description='A JSON-compatible value for the filter.')


class SavedSearch(ORMBaseModel):
    """An ORM representation of saved search data. Represents a set of filter criteria."""
    name: Any = Field(default=..., description='The name of the saved search.')
    filters: List[Any] = Field(default_factory=list, description='The filter set for the saved search.')


class Log(ORMBaseModel):
    """An ORM representation of log data."""
    name: Any = Field(default=..., description='The logger name.')
    level: Any = Field(default=..., description='The log level.')
    message: Any = Field(default=..., description='The log message.')
    timestamp: Any = Field(default=..., description='The log timestamp.')
    flow_run_id: Optional[Any] = Field(default=None, description='The flow run ID associated with the log.')
    task_run_id: Optional[Any] = Field(default=None, description='The task run ID associated with the log.')


class QueueFilter(PrefectBaseModel):
    """Filter criteria definition for a work queue."""
    tags: Optional[List[Any]] = Field(default=None, description='Only include flow runs with these tags in the work queue.')
    deployment_ids: Optional[List[Any]] = Field(default=None, description='Only include flow runs from these deployments in the work queue.')


class WorkQueue(ORMBaseModel):
    """An ORM representation of a work queue"""
    name: Any = Field(default=..., description='The name of the work queue.')
    description: str = Field(default='', description='An optional description for the work queue.')
    is_paused: bool = Field(default=False, description='Whether or not the work queue is paused.')
    concurrency_limit: Optional[Any] = Field(default=None, description='An optional concurrency limit for the work queue.')
    priority: int = Field(default=1, description="The queue's priority. Lower values are higher priority (1 is the highest).")
    work_pool_id: Optional[Any] = Field(default=None, description='The work pool with which the queue is associated.')
    filter: Optional[Any] = Field(default=None, description='DEPRECATED: Filter criteria for the work queue.', deprecated=True)
    last_polled: Optional[datetime.datetime] = Field(default=None, description='The last time an agent polled this queue for work.')


class WorkQueueHealthPolicy(PrefectBaseModel):
    maximum_late_runs: int = Field(default=0, description='The maximum number of late runs in the work queue before it is deemed unhealthy. Defaults to `0`.')
    maximum_seconds_since_last_polled: int = Field(default=60, description='The maximum number of time in seconds elapsed since work queue has been polled before it is deemed unhealthy. Defaults to `60`.')

    def evaluate_health_status(self, late_runs_count: int, last_polled: Optional[datetime.datetime] = None) -> bool:
        """
        Given empirical information about the state of the work queue, evaluate its health status.

        Args:
            late_runs_count: the count of late runs for the work queue.
            last_polled: the last time the work queue was polled, if available.

        Returns:
            bool: whether or not the work queue is healthy.
        """
        healthy: bool = True
        if self.maximum_late_runs is not None and late_runs_count > self.maximum_late_runs:
            healthy = False
        if self.maximum_seconds_since_last_polled is not None:
            if last_polled is None or (now('UTC') - last_polled).total_seconds() > self.maximum_seconds_since_last_polled:
                healthy = False
        return healthy


class WorkQueueStatusDetail(PrefectBaseModel):
    healthy: bool = Field(..., description='Whether or not the work queue is healthy.')
    late_runs_count: int = Field(default=0, description='The number of late flow runs in the work queue.')
    last_polled: Optional[datetime.datetime] = Field(default=None, description='The last time an agent polled this queue for work.')
    health_check_policy: Any = Field(..., description='The policy used to determine whether or not the work queue is healthy.')


class FlowRunNotificationPolicy(ORMBaseModel):
    """An ORM representation of a flow run notification."""
    is_active: bool = Field(default=True, description='Whether the policy is currently active')
    state_names: Any = Field(default=..., description='The flow run states that trigger notifications')
    tags: Any = Field(default=..., description='The flow run tags that trigger notifications (set [] to disable)')
    block_document_id: Any = Field(default=..., description='The block document ID used for sending notifications')
    message_template: Optional[str] = Field(
        default=None,
        description=f'A templatable notification message. Use {{braces}} to add variables. Valid variables include: {listrepr(sorted(FLOW_RUN_NOTIFICATION_TEMPLATE_KWARGS), sep=", ")}',
        examples=['Flow run {flow_run_name} with id {flow_run_id} entered state {flow_run_state_name}.'],
    )

    @field_validator('message_template')
    @classmethod
    def validate_message_template_variables(cls, v: Optional[str]) -> Optional[str]:
        return validate_message_template_variables(v)


class Agent(ORMBaseModel):
    """An ORM representation of an agent"""
    name: str = Field(default_factory=lambda: generate_slug(2), description='The name of the agent. If a name is not provided, it will be auto-generated.')
    work_queue_id: Any = Field(default=..., description='The work queue with which the agent is associated.')
    last_activity_time: Optional[datetime.datetime] = Field(default=None, description='The last time this agent polled for work.')


class WorkPool(ORMBaseModel):
    """An ORM representation of a work pool"""
    name: Any = Field(..., description='The name of the work pool.')
    description: Optional[str] = Field(default=None, description='A description of the work pool.')
    type: Any = Field(..., description='The work pool type.')
    base_job_template: Dict[str, Any] = Field(default_factory=dict, description="The work pool's base job template.")
    is_paused: bool = Field(default=False, description='Pausing the work pool stops the delivery of all work.')
    concurrency_limit: Optional[Any] = Field(default=None, description='A concurrency limit for the work pool.')
    status: Optional[Any] = Field(default=None, description='The current status of the work pool.')
    default_queue_id: Optional[Any] = Field(None, description="The id of the pool's default queue.")

    @field_validator('default_queue_id')
    def helpful_error_for_missing_default_queue_id(self, v: Any) -> Any:
        return validate_default_queue_id_not_none(v)

    @classmethod
    def model_validate(cls, obj: Any, *, strict: Optional[bool] = None, from_attributes: bool = False, context: Optional[Dict[str, Any]] = None) -> Self:
        parsed: Self = super().model_validate(obj, strict=strict, from_attributes=from_attributes, context=context)
        if from_attributes:
            if obj.type == 'prefect-agent':
                parsed.status = None
        return parsed


class Worker(ORMBaseModel):
    """An ORM representation of a worker"""
    name: Any = Field(..., description='The name of the worker.')
    work_pool_id: Any = Field(..., description='The work pool with which the queue is associated.')
    last_heartbeat_time: Optional[datetime.datetime] = Field(default=None, description='The last time the worker process sent a heartbeat.')
    heartbeat_interval_seconds: Optional[int] = Field(default=None, description='The number of seconds to expect between heartbeats sent by the worker.')


Flow.model_rebuild()
FlowRun.model_rebuild()


class Artifact(ORMBaseModel):
    key: Optional[Any] = Field(default=None, description='An optional unique reference key for this artifact.')
    type: Optional[Any] = Field(default=None, description="An identifier that describes the shape of the data field. e.g. 'result', 'table', 'markdown'")
    description: Optional[str] = Field(default=None, description='A markdown-enabled description of the artifact.')
    data: Optional[Any] = Field(default=None, description='Data associated with the artifact, e.g. a result.; structure depends on the artifact type.')
    metadata_: Optional[Dict[str, str]] = Field(default=None, description='User-defined artifact metadata. Content must be string key and value pairs.')
    flow_run_id: Optional[Any] = Field(default=None, description='The flow run associated with the artifact.')
    task_run_id: Optional[Any] = Field(default=None, description='The task run associated with the artifact.')

    @classmethod
    def from_result(cls: Type[Artifact], data: Any) -> Artifact:
        artifact_info: Dict[str, Any] = dict()
        if isinstance(data, dict):
            artifact_key = data.pop('artifact_key', None)
            if artifact_key:
                artifact_info['key'] = artifact_key
            artifact_type = data.pop('artifact_type', None)
            if artifact_type:
                artifact_info['type'] = artifact_type
            description = data.pop('artifact_description', None)
            if description:
                artifact_info['description'] = description
        return cls(data=data, **artifact_info)

    @field_validator('metadata_')
    @classmethod
    def validate_metadata_length(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        return validate_max_metadata_length(v)


class ArtifactCollection(ORMBaseModel):
    key: Any = Field(..., description='An optional unique reference key for this artifact.')
    latest_id: Any = Field(..., description='The latest artifact ID associated with the key.')
    type: Optional[Any] = Field(default=None, description="An identifier that describes the shape of the data field. e.g. 'result', 'table', 'markdown'")
    description: Optional[str] = Field(default=None, description='A markdown-enabled description of the artifact.')
    data: Optional[Any] = Field(default=None, description='Data associated with the artifact, e.g. a result.; structure depends on the artifact type.')
    metadata_: Optional[Dict[str, str]] = Field(default=None, description='User-defined artifact metadata. Content must be string key and value pairs.')
    flow_run_id: Optional[Any] = Field(default=None, description='The flow run associated with the artifact.')
    task_run_id: Optional[Any] = Field(default=None, description='The task run associated with the artifact.')


class Variable(ORMBaseModel):
    name: str = Field(default=..., description='The name of the variable', examples=['my-variable'], max_length=MAX_VARIABLE_NAME_LENGTH)
    value: Any = Field(default=..., description='The value of the variable', examples=['my-value'])
    tags: List[str] = Field(default_factory=list, description='A list of variable tags', examples=[['tag-1', 'tag-2']])


class FlowRunInput(ORMBaseModel):
    flow_run_id: Any = Field(description='The flow run ID associated with the input.')
    key: str = Field(description='The key of the input.')
    value: Any = Field(description='The value of the input.')
    sender: Optional[Any] = Field(default=None, description='The sender of the input.')

    @field_validator('key', check_fields=False)
    @classmethod
    def validate_name_characters(cls, v: str) -> str:
        raise_on_name_alphanumeric_dashes_only(v)
        return v


class CsrfToken(ORMBaseModel):
    token: Any = Field(default=..., description='The CSRF token')
    client: Any = Field(default=..., description='The client id associated with the CSRF token')
    expiration: Any = Field(default=..., description='The expiration time of the CSRF token')